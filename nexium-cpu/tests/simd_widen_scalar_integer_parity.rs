#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]
use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;
use std::sync::{Mutex, OnceLock};
const CODE: u64 = 0x100000;
const DATA: u64 = 0x200000;
const SOURCE: u64 = 0x300000;
const CANARY: u64 = 0xface_1234_5678_9abc;
const FPCR: u64 = 0x0300_0000;
const FLAGS: u64 = 0x0800_009f;
const NZCV: u64 = 0xa000_0000;
#[derive(Clone, Copy, Debug)]
struct Case {
    kind: u32,
    size: u32,
    upper: bool,
    rn: u32,
    rm: u32,
    rd: u32,
    n: [u8; 16],
    m: [u8; 16],
}
impl Case {
    fn word(self) -> u32 {
        match self.kind {
            0 => {
                0x2e21_3800
                    | (self.size << 22)
                    | (u32::from(self.upper) << 30)
                    | (self.rn << 5)
                    | self.rd
            }
            1 => 0x5ef1_b800 | (self.rn << 5) | self.rd,
            _ => {
                (if self.kind == 2 {
                    0x5ee0_8400
                } else {
                    0x7ee0_8400
                }) | (self.rm << 16)
                    | (self.rn << 5)
                    | self.rd
            }
        }
    }
    fn apply(self, v: &mut [[u8; 16]; 32]) {
        let n = v[self.rn as usize];
        let m = v[self.rm as usize];
        let mut out = [0; 16];
        let load = |input: &[u8], start: usize, count: usize| {
            let mut b = [0; 8];
            b[..count].copy_from_slice(&input[start..start + count]);
            u64::from_le_bytes(b)
        };
        if self.kind == 0 {
            let count = 1usize << self.size;
            let start = usize::from(self.upper) * 8;
            for lane in 0..8 / count {
                let raw = load(&n, start + lane * count, count) as i128;
                let answer = (raw << (count * 8)) as u64;
                out[lane * count * 2..(lane + 1) * count * 2]
                    .copy_from_slice(&answer.to_le_bytes()[..count * 2]);
            }
        } else {
            let first = load(&n, 0, 8) as u128;
            let second = load(
                if self.kind == 1 { &n } else { &m },
                if self.kind == 1 { 8 } else { 0 },
                8,
            ) as u128;
            let answer = if self.kind == 3 {
                first.wrapping_sub(second)
            } else {
                first + second
            };
            out[..8].copy_from_slice(&(answer as u64).to_le_bytes());
        }
        v[self.rd as usize] = out;
    }
}
fn cpus() -> &'static Mutex<[(CpuBackendKind, Cpu); 2]> {
    static CPUS: OnceLock<Mutex<[(CpuBackendKind, Cpu); 2]>> = OnceLock::new();
    CPUS.get_or_init(|| {
        Mutex::new(
            [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic]
                .map(|kind| (kind, Cpu::new(kind).unwrap())),
        )
    })
}
fn seed(register: usize) -> [u8; 16] {
    std::array::from_fn(|lane| (register * 37 + lane * 11 + 0x93) as u8)
}
fn pair(a: u64, b: u64) -> [u8; 16] {
    let mut v = [0; 16];
    v[..8].copy_from_slice(&a.to_le_bytes());
    v[8..].copy_from_slice(&b.to_le_bytes());
    v
}
fn pattern(size: u32, rotation: usize) -> [u8; 16] {
    let bytes = 1usize << size;
    let bits = bytes * 8;
    let sign = 1u64 << (bits - 1);
    let mask = (1u64 << bits) - 1;
    let values = [
        0,
        1,
        mask,
        sign,
        sign - 1,
        sign + 1,
        0xaaaa_5555 & mask,
        0x0123_4567 & mask,
    ];
    let mut output = [0; 16];
    for (lane, out) in output.chunks_exact_mut(bytes).enumerate() {
        out.copy_from_slice(
            &values[(lane + rotation + if lane >= 8 / bytes { 3 } else { 0 }) % 8].to_le_bytes()
                [..bytes],
        );
    }
    output
}
fn check(cases: &[Case]) {
    let mut engines = cpus().lock().unwrap_or_else(|e| e.into_inner());
    for (backend, cpu) in engines.iter_mut() {
        let mut source = vec![0x6b; 0x1000];
        let mut data = vec![0xa5; 0x1000];
        unsafe {
            cpu.map_host(SOURCE, 0x1000, Perm::RO, source.as_mut_ptr())
                .unwrap();
            cpu.map_host(DATA, 0x1000, Perm::RW, data.as_mut_ptr())
                .unwrap();
        }
        let mut failure = None;
        let mut start = 0;
        while start < cases.len() {
            let instruction = cases[start].word();
            let end = start
                + cases[start..]
                    .iter()
                    .take_while(|c| c.word() == instruction)
                    .count();
            let mut program = vec![0xd51b_4416, 0xd51b_4437, 0xd51b_4218];
            for r in 0..32 {
                program.push(0x3dc0_0280 | ((r as u32) << 10) | r as u32);
            }
            program.push(instruction);
            for r in 0..32 {
                program.push(0x3d80_02a0 | ((r as u32) << 10) | r as u32);
            }
            program.extend([
                0xd53b_4439,
                0xd53b_441a,
                0xd53b_421b,
                0xf901_02b9,
                0xf901_06ba,
                0xf901_0abb,
                0xd400_0fc1,
            ]);
            let mut code = vec![0x5a; 0x1000];
            for (i, word) in program.iter().enumerate() {
                code[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            let original_code = code.clone();
            unsafe {
                cpu.map_host(CODE, 0x1000, Perm::RX, code.as_mut_ptr())
                    .unwrap();
            }
            for &case in &cases[start..end] {
                source.fill(0x6b);
                data.fill(0xa5);
                let mut vectors: [[u8; 16]; 32] = std::array::from_fn(seed);
                vectors[case.rn as usize] = case.n;
                if case.kind >= 2 {
                    vectors[case.rm as usize] = case.m;
                }
                for (r, v) in vectors.iter().enumerate() {
                    source[r * 16..(r + 1) * 16].copy_from_slice(v);
                }
                let original_source = source.clone();
                let mut want = data.clone();
                case.apply(&mut vectors);
                for (r, v) in vectors.iter().enumerate() {
                    want[r * 16..(r + 1) * 16].copy_from_slice(v);
                }
                for (i, status) in [FLAGS, FPCR, NZCV].into_iter().enumerate() {
                    want[512 + i * 8..520 + i * 8].copy_from_slice(&status.to_le_bytes());
                }
                let mut registers = [CANARY; 31];
                for (r, value) in [
                    (20, SOURCE),
                    (21, DATA),
                    (22, FPCR),
                    (23, FLAGS),
                    (24, NZCV),
                    (25, FLAGS),
                    (26, FPCR),
                    (27, NZCV),
                ] {
                    registers[r] = value;
                }
                for (r, value) in registers.into_iter().enumerate() {
                    cpu.set_register(r as u32, value);
                }
                cpu.set_sp(CANARY);
                cpu.set_pc(CODE);
                let event = cpu.run(10000).unwrap().event;
                let fault = cpu.take_fault();
                let actual: [u64; 31] = std::array::from_fn(|r| cpu.get_register(r as u32));
                if !matches!(event, CpuEvent::Svc(0x7e))
                    || fault.is_some()
                    || actual != registers
                    || cpu.get_sp() != CANARY
                    || data != want
                    || source != original_source
                    || code != original_code
                {
                    failure=Some(format!("{backend:?} {case:?}: event{event:?} fault{fault:?} gpr{} sp{} data{} source{} code{}",actual==registers,cpu.get_sp()==CANARY,data==want,source==original_source,code==original_code));
                    break;
                }
            }
            unsafe {
                cpu.unmap_host(CODE, 0x1000).unwrap();
            }
            if failure.is_some() {
                break;
            }
            start = end;
        }
        unsafe {
            cpu.unmap_host(SOURCE, 0x1000).unwrap();
            cpu.unmap_host(DATA, 0x1000).unwrap();
        }
        assert!(failure.is_none(), "{}", failure.unwrap_or_default());
    }
}
#[test]
fn widening_all_width_halves_aliases_randoms_and_source_guards() {
    let mut cases = Vec::new();
    let mut random = 0x91fa_a482_e613_820u64;
    for size in 0..3 {
        for upper in [false, true] {
            for (rn, rd) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31)] {
                for sample in 0..72 {
                    let n = if sample < 8 {
                        pattern(size, sample)
                    } else {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        let a = random;
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        pair(a, random)
                    };
                    cases.push(Case {
                        kind: 0,
                        size,
                        upper,
                        rn,
                        rm: 0,
                        rd,
                        n,
                        m: [0; 16],
                    });
                }
            }
        }
    }
    assert_eq!(cases.len(), 2160);
    check(&cases);
}
#[test]
fn scalar_pair_add_sub_all_aliases_modular_boundaries_and_unused_upper_bits() {
    let values = [
        0,
        1,
        2,
        3,
        1 << 31,
        (1 << 32) - 1,
        1 << 32,
        (1u64 << 63) - 1,
        1u64 << 63,
        (1u64 << 63) + 1,
        u64::MAX - 1,
        u64::MAX,
        CANARY,
    ];
    let mut cases = Vec::new();
    for kind in 1..4 {
        for (rn, rm, rd) in [
            (0, 1, 2),
            (0, 1, 0),
            (0, 1, 1),
            (0, 0, 2),
            (0, 0, 0),
            (31, 31, 31),
            (31, 0, 31),
            (0, 31, 31),
        ] {
            for &a in &values {
                for &b in &values {
                    cases.push(Case {
                        kind,
                        size: 3,
                        upper: false,
                        rn,
                        rm,
                        rd,
                        n: pair(a, b),
                        m: pair(b, !a),
                    });
                }
            }
        }
    }
    assert_eq!(cases.len(), 4056);
    check(&cases);
}
#[test]
fn exact_wonder_opcodes_and_independent_known_values() {
    let cases = [
        Case {
            kind: 0,
            size: 1,
            upper: false,
            rn: 1,
            rm: 0,
            rd: 0,
            n: pair(0xffff_8000_0001_0000, CANARY),
            m: [0; 16],
        },
        Case {
            kind: 1,
            size: 3,
            upper: false,
            rn: 0,
            rm: 0,
            rd: 0,
            n: pair(u64::MAX, 2),
            m: [0; 16],
        },
        Case {
            kind: 3,
            size: 3,
            upper: false,
            rn: 0,
            rm: 1,
            rd: 3,
            n: pair(0, CANARY),
            m: pair(1, !CANARY),
        },
    ];
    assert_eq!(cases[0].word(), 0x2e61_3820);
    assert_eq!(cases[1].word(), 0x5ef1_b800);
    assert_eq!(cases[2].word(), 0x7ee1_8403);
    for (case, literal) in cases.into_iter().zip([
        pair(0x0001_0000_0000_0000, 0xffff_0000_8000_0000),
        pair(1, 0),
        pair(u64::MAX, 0),
    ]) {
        let mut v = [[0; 16]; 32];
        v[case.rn as usize] = case.n;
        if case.kind >= 2 {
            v[case.rm as usize] = case.m;
        }
        case.apply(&mut v);
        assert_eq!(v[case.rd as usize], literal);
    }
    check(&cases);
}
