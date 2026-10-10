#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const INPUTS: usize = 0x800;
const OUTPUTS: usize = 0x2000;
const STATUS: usize = 0xb000;
const CANARY: u64 = 0xfade_1234_5678_9abc;
const FPCR: u64 = 0x0300_0000;
const NZCV: u64 = 0xa000_0000;

#[derive(Clone, Copy, Debug)]
struct Case {
    cls: bool,
    bytes: usize,
    q: bool,
    rn: u32,
    rd: u32,
    source: [u8; 16],
    accumulator: [u8; 16],
    repetitions: usize,
    literal: Option<[u8; 16]>,
}

impl Case {
    fn word(self) -> u32 {
        assert!(matches!(self.bytes, 1 | 2 | 4));
        (if self.cls { 0x0e20_4800 } else { 0x2e20_4800 })
            | (u32::from(self.q) << 30)
            | (self.bytes.trailing_zeros() << 22)
            | (self.rn << 5)
            | self.rd
    }

    fn model(self, vectors: &mut [[u8; 16]; 32]) {
        let source = vectors[self.rn as usize];
        let bits = self.bytes * 8;
        let mut result = [0; 16];
        for lane in 0..(if self.q { 16 } else { 8 }) / self.bytes {
            let start = lane * self.bytes;
            let mut raw = [0; 8];
            raw[..self.bytes].copy_from_slice(&source[start..start + self.bytes]);
            let value = u64::from_le_bytes(raw);
            let mut count = 0u64;
            let expected_bit = if self.cls {
                (value >> (bits - 1)) & 1
            } else {
                0
            };
            for bit in (0..bits - usize::from(self.cls)).rev() {
                if (value >> bit) & 1 != expected_bit {
                    break;
                }
                count += 1;
            }
            result[start..start + self.bytes].copy_from_slice(&count.to_le_bytes()[..self.bytes]);
        }
        vectors[self.rd as usize] = result;
    }
}

fn seed(register: usize) -> [u8; 16] {
    std::array::from_fn(|byte| (register * 37 + byte * 11 + 0x93) as u8)
}

fn pattern(bytes: usize, rotation: usize) -> [u8; 16] {
    let bits = bytes * 8;
    let mask = u64::MAX >> (64 - bits);
    let sign = 1u64 << (bits - 1);
    let values = [
        0,
        1,
        mask,
        sign,
        sign - 1,
        sign + 1,
        0xaaaa_5555_aaaa_5555 & mask,
        0x0123_4567_89ab_cdef & mask,
    ];
    let mut result = [0; 16];
    for (lane, output) in result.chunks_exact_mut(bytes).enumerate() {
        output.copy_from_slice(
            &values[(lane + rotation + if lane >= 8 / bytes { 3 } else { 0 }) & 7].to_le_bytes()
                [..bytes],
        );
    }
    result
}

fn check(cases: &[Case], fpsr: u64) {
    let mut backends = [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic]
        .map(|kind| (kind, Cpu::new(kind).expect("CPU initialization")));
    for cases in cases.chunks(60) {
        let mut vectors: [[u8; 16]; 32] = std::array::from_fn(seed);
        let mut initial = vec![0xa5; 0x10000];
        for (register, vector) in vectors.iter().enumerate() {
            initial[register * 16..register * 16 + 16].copy_from_slice(vector);
        }
        for (slot, case) in cases.iter().enumerate() {
            let offset = INPUTS + slot * 32;
            initial[offset..offset + 16].copy_from_slice(&case.accumulator);
            initial[offset + 16..offset + 32].copy_from_slice(&case.source);
        }
        let mut expected = initial.clone();
        let mut program = vec![0xd51b_4410, 0xd51b_4433, 0xd51b_4214];
        for register in 0..32u32 {
            program.push(0x3dc0_01a0 | (register << 10) | register);
        }
        program.extend([0x9100_0484, 0x9100_04a5, 0x9100_04c6, 0x9100_04e7]);
        for (slot, &case) in cases.iter().enumerate() {
            let first = INPUTS / 16 + slot * 2;
            program.push(0x3dc0_01a0 | ((first as u32) << 10) | case.rd);
            vectors[case.rd as usize] = case.accumulator;
            program.push(0x3dc0_01a0 | (((first + 1) as u32) << 10) | case.rn);
            vectors[case.rn as usize] = case.source;
            for _ in 0..case.repetitions {
                program.push(case.word());
                case.model(&mut vectors);
            }
            if let Some(literal) = case.literal {
                assert_eq!(
                    vectors[case.rd as usize], literal,
                    "independent oracle literal: {case:?}"
                );
            }
            for (register, vector) in vectors.iter().enumerate() {
                let index = slot * 32 + register;
                assert!(index < 4096);
                program.push(0x3d80_01c0 | ((index as u32) << 10) | register as u32);
                let offset = OUTPUTS + index * 16;
                expected[offset..offset + 16].copy_from_slice(vector);
            }
            program.extend([0xd53b_4431, 0xd53b_4412, 0xd53b_4215]);
            for (field, register) in [17u32, 18, 21].into_iter().enumerate() {
                program.push(0xf900_01e0 | (((slot * 3 + field) as u32) << 10) | register);
                let offset = STATUS + slot * 24 + field * 8;
                expected[offset..offset + 8]
                    .copy_from_slice(&[fpsr, FPCR, NZCV][field].to_le_bytes());
            }
        }
        assert!(INPUTS + cases.len() * 32 <= OUTPUTS);
        assert!(OUTPUTS + cases.len() * 512 <= STATUS);
        assert!(STATUS + cases.len() * 24 <= initial.len());
        program.extend([0x8b05_008a, 0x8b07_00cb, 0xd400_0fc1]);
        let mut encoded = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
        for (index, word) in program.iter().enumerate() {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut initial_gprs = [CANARY; 31];
        initial_gprs[4..8].copy_from_slice(&[10, 20, 30, 40]);
        initial_gprs[13] = DATA;
        initial_gprs[14] = DATA + OUTPUTS as u64;
        initial_gprs[15] = DATA + STATUS as u64;
        initial_gprs[16] = FPCR;
        initial_gprs[19] = fpsr;
        initial_gprs[20] = NZCV;
        let mut expected_gprs = initial_gprs;
        expected_gprs[4..8].copy_from_slice(&[11, 21, 31, 41]);
        expected_gprs[10] = 32;
        expected_gprs[11] = 72;
        expected_gprs[17] = fpsr;
        expected_gprs[18] = FPCR;
        expected_gprs[21] = NZCV;
        for (backend, cpu) in &mut backends {
            let mut code = encoded.clone();
            let mut actual = initial.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, actual.len() as u64, Perm::RW, actual.as_mut_ptr())
                    .unwrap();
            }
            for (register, &value) in initial_gprs.iter().enumerate() {
                cpu.set_register(register as u32, value);
            }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let mut reached_svc = false;
            for _ in 0..16 {
                match cpu
                    .run(100_000)
                    .expect("SIMD leading-count execution")
                    .event
                {
                    CpuEvent::Svc(0x7e) => {
                        reached_svc = true;
                        break;
                    }
                    CpuEvent::Running => {}
                    event => panic!("{backend:?}: {event:?}, PC={:#x}", cpu.get_pc()),
                }
            }
            assert!(
                reached_svc,
                "{backend:?}: bounded execution did not reach SVC"
            );
            for (slot, case) in cases.iter().enumerate() {
                for register in 0..32 {
                    let offset = OUTPUTS + (slot * 32 + register) * 16;
                    assert_eq!(
                        &actual[offset..offset + 16],
                        &expected[offset..offset + 16],
                        "{backend:?}: {case:?} V{register}"
                    );
                }
                let offset = STATUS + slot * 24;
                assert_eq!(
                    &actual[offset..offset + 24],
                    &expected[offset..offset + 24],
                    "{backend:?}: {case:?} sticky FPSR/FPCR/NZCV"
                );
            }
            for (register, &value) in expected_gprs.iter().enumerate() {
                assert_eq!(
                    cpu.get_register(register as u32),
                    value,
                    "{backend:?}: X{register}"
                );
            }
            assert_eq!(cpu.get_sp(), CANARY);
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, actual.len() as u64).unwrap();
            }
            assert_eq!(actual, expected, "{backend:?}: source/data/neighbor guards");
            assert_eq!(code, encoded, "{backend:?}: source code unchanged");
        }
    }
}

fn arrangements() -> [(usize, bool); 6] {
    [
        (1, false),
        (1, true),
        (2, false),
        (2, true),
        (4, false),
        (4, true),
    ]
}

fn leading_pattern(bytes: usize, leading: usize, complement: bool, q: bool) -> [u8; 16] {
    let width = bytes * 8;
    let mask = (1u64 << width) - 1;
    let mut source = [0; 16];
    for lane in 0..16 / bytes {
        let count = (leading + lane * 3) % (width + 1);
        let value = if count == width {
            0
        } else {
            1u64 << (width - count - 1)
        };
        let value = if complement { !value & mask } else { value };
        source[lane * bytes..lane * bytes + bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
    }
    if !q {
        source[8..].fill(0xe7);
    }
    source
}

#[test]
fn all_legal_formats_each_leading_bit_zero_and_high_poison_match_integer_oracle() {
    let mut cases = Vec::new();
    for (bytes, q) in arrangements() {
        for (rn, rd) in [(2, 1), (2, 2), (31, 31), (31, 0), (0, 31)] {
            for leading in 0..=bytes * 8 {
                for complement in [false, true] {
                    cases.push(Case {
                        cls: false,
                        bytes,
                        q,
                        rn,
                        rd,
                        source: leading_pattern(bytes, leading, complement, q),
                        accumulator: seed(rd as usize),
                        repetitions: 1,
                        literal: None,
                    });
                }
            }
            for rotation in 0..8 {
                cases.push(Case {
                    cls: false,
                    bytes,
                    q,
                    rn,
                    rd,
                    source: pattern(bytes, rotation),
                    accumulator: seed(rd as usize),
                    repetitions: 1,
                    literal: None,
                });
            }
        }
    }
    let signed_cases: Vec<_> = cases
        .iter()
        .map(|case| Case { cls: true, ..*case })
        .collect();
    cases.extend(signed_cases);
    assert_eq!(cases.len(), 2840);
    check(&cases, 0x93);
}

#[test]
fn all_registers_aliases_upper_clearing_and_sticky_flags_are_preserved() {
    let mut cases = Vec::new();
    for (bytes, q) in arrangements() {
        for rn in 0..32 {
            for rd in [rn, (rn + 13) & 31, 31] {
                cases.push(Case {
                    cls: false,
                    bytes,
                    q,
                    rn,
                    rd,
                    source: seed(rn as usize),
                    accumulator: seed(rd as usize),
                    repetitions: 1,
                    literal: None,
                });
            }
        }
    }
    let signed_cases: Vec<_> = cases
        .iter()
        .map(|case| Case { cls: true, ..*case })
        .collect();
    cases.extend(signed_cases);
    assert_eq!(cases.len(), 1152);
    check(&cases, 0x0800_0093);
}

#[test]
fn exact_wonder_clz_zero_one_and_signed_boundary_match_literal_counts() {
    let mut source = [0xe7; 16];
    source[..4].copy_from_slice(&1u32.to_le_bytes());
    source[4..8].copy_from_slice(&0u32.to_le_bytes());
    let mut literal = [0; 16];
    literal[..4].copy_from_slice(&31u32.to_le_bytes());
    literal[4..8].copy_from_slice(&32u32.to_le_bytes());
    let wonder = Case {
        cls: false,
        bytes: 4,
        q: false,
        rn: 0,
        rd: 0,
        source,
        accumulator: [0xa7; 16],
        repetitions: 1,
        literal: Some(literal),
    };
    assert_eq!(
        wonder.word(),
        0x2ea0_4800,
        "Wonder CLZ V0.2S,V0.2S at 0x804f038"
    );
    let mut boundary = wonder;
    boundary.source[..4].copy_from_slice(&0x8000_0000u32.to_le_bytes());
    boundary.source[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    boundary.literal = Some([0; 16]);
    let mut cases = vec![wonder, boundary];
    for (values, counts) in [
        ([u32::MAX, 0x8000_0000], [31u32, 0]),
        ([0, 0x7fff_ffff], [31, 0]),
        ([0xffff_fffe, 0xffff_fffc], [30, 29]),
    ] {
        let mut case = Case {
            cls: true,
            ..wonder
        };
        let mut expected = [0; 16];
        for lane in 0..2 {
            case.source[lane * 4..lane * 4 + 4].copy_from_slice(&values[lane].to_le_bytes());
            expected[lane * 4..lane * 4 + 4].copy_from_slice(&counts[lane].to_le_bytes());
        }
        case.literal = Some(expected);
        assert_eq!(case.word(), 0x0ea0_4800);
        cases.push(case);
    }
    check(&cases, 0x0800_0093);
}
