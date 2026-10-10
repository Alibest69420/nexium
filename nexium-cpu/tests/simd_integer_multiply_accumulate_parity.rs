#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;
use std::sync::{Mutex, OnceLock};

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const SOURCE: u64 = 0x30_0000;
const PAGE: usize = 0x1000;
const OUTPUT: usize = 0x1000;
const CONSTANTS: usize = 0x10000;
const STRIDE: usize = 800;
const CANARY: u64 = 0xface_1234_5678_9abc;
const NZCV: u64 = 0xa000_0000;
const FPCR: u32 = 0x0300_0000;
const STICKY: u32 = 0x0800_009f;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Multiply,
    Add,
    Subtract,
}

const KINDS: [Kind; 3] = [Kind::Multiply, Kind::Add, Kind::Subtract];

#[derive(Clone, Copy, Debug)]
struct Case {
    kind: Kind,
    bytes: usize,
    q: bool,
    rn: u32,
    rm: u32,
    rd: u32,
    a: [u8; 16],
    n: [u8; 16],
    m: [u8; 16],
    fpcr: u32,
    fpsr: u32,
    literal: Option<[u8; 16]>,
}

impl Case {
    fn word(self) -> u32 {
        assert!(matches!(self.bytes, 1 | 2 | 4));
        let base = match self.kind {
            Kind::Multiply => 0x0e20_9c00,
            Kind::Add => 0x0e20_9400,
            Kind::Subtract => 0x2e20_9400,
        };
        base | (u32::from(self.q) << 30)
            | (self.bytes.trailing_zeros() << 22)
            | (self.rm << 16)
            | (self.rn << 5)
            | self.rd
    }

    fn expected(self, a: [u8; 16], n: [u8; 16], m: [u8; 16]) -> [u8; 16] {
        let modulus = 1u128 << (self.bytes * 8);
        let mut result = [0; 16];
        for lane in 0..(if self.q { 16 } else { 8 }) / self.bytes {
            let start = lane * self.bytes;
            let read = |vector: [u8; 16]| {
                let mut bits = [0; 8];
                bits[..self.bytes].copy_from_slice(&vector[start..start + self.bytes]);
                u64::from_le_bytes(bits) as u128
            };
            let product = (read(n) * read(m)) % modulus;
            let value = match self.kind {
                Kind::Multiply => product,
                Kind::Add => (read(a) + product) % modulus,
                Kind::Subtract => (read(a) + modulus - product) % modulus,
            };
            result[start..start + self.bytes]
                .copy_from_slice(&(value as u64).to_le_bytes()[..self.bytes]);
        }
        result
    }
}

fn cpus() -> &'static Mutex<[(CpuBackendKind, Cpu); 2]> {
    static CPUS: OnceLock<Mutex<[(CpuBackendKind, Cpu); 2]>> = OnceLock::new();
    CPUS.get_or_init(|| {
        Mutex::new(
            [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic]
                .map(|backend| (backend, Cpu::new(backend).expect("CPU initialization"))),
        )
    })
}

fn seed(register: usize) -> [u8; 16] {
    std::array::from_fn(|byte| (register * 37 + byte * 11 + 0x93) as u8)
}

fn put(data: &mut [u8], offset: usize, value: u64) {
    data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn check(records: &[Case]) {
    let mut cpus = cpus()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    for (backend, cpu) in &mut *cpus {
        for records in records.chunks(32) {
            let mut data = vec![0xa5; 0x12000];
            let mut source = vec![0x6d; PAGE];
            let mut vectors: [[u8; 16]; 32] = std::array::from_fn(seed);

            for (register, vector) in vectors.iter().enumerate() {
                source[0x800 + register * 16..0x800 + register * 16 + 16].copy_from_slice(vector);
            }
            assert!(records.len() * 48 <= 0x800);
            assert!(OUTPUT + records.len() * STRIDE <= CONSTANTS);
            let mut expected = data.clone();
            let mut registers = [CANARY; 31];
            registers[22] = SOURCE;
            registers[24] = DATA + CONSTANTS as u64;
            registers[27] = DATA + OUTPUT as u64;
            registers[28] = SOURCE + 0x800;
            registers[29] = NZCV;
            let mut program = vec![0xd51b_421d];
            for register in 0..32 {
                program.push(0x3dc0_0000 | (register << 10) | (28 << 5) | register);
            }
            for (slot, record) in records.iter().enumerate() {
                let pos = slot * 48;
                source[pos..pos + 16].copy_from_slice(&record.a);
                source[pos + 16..pos + 32].copy_from_slice(&record.n);
                source[pos + 32..pos + 48].copy_from_slice(&record.m);
                for (which, reg, value) in [
                    (0, record.rd, record.a),
                    (1, record.rn, record.n),
                    (2, record.rm, record.m),
                ] {
                    program
                        .push(0x3dc0_0000 | (((slot * 3 + which) as u32) << 10) | (22 << 5) | reg);
                    vectors[reg as usize] = value;
                }
                for (which, reg, value) in
                    [(0, 25, record.fpcr as u64), (1, 26, record.fpsr as u64)]
                {
                    put(&mut data, CONSTANTS + (slot * 2 + which) * 8, value);
                    program
                        .push(0xf940_0000 | (((slot * 2 + which) as u32) << 10) | (24 << 5) | reg);
                    registers[reg as usize] = value;
                }
                program.extend([0xd51b_4419, 0xd51b_443a]);
                let result = record.expected(
                    vectors[record.rd as usize],
                    vectors[record.rn as usize],
                    vectors[record.rm as usize],
                );
                if let Some(literal) = record.literal {
                    assert_eq!(
                        result, literal,
                        "independent modulo oracle literal: {record:?}"
                    );
                }
                let fpsr = record.fpsr;
                vectors[record.rd as usize] = result;
                program.push(record.word());
                let offset = slot * STRIDE;
                for register in 0..31 {
                    program.push(
                        0xf900_0000
                            | (((offset / 8 + register) as u32) << 10)
                            | (27 << 5)
                            | register as u32,
                    );
                    put(
                        &mut expected,
                        OUTPUT + offset + register * 8,
                        registers[register],
                    );
                }
                program.push(0x9100_03fe);
                for (word, position, value) in [
                    (None, 248, CANARY),
                    (Some(0xd53b_441e), 256, record.fpcr as u64),
                    (Some(0xd53b_443e), 264, fpsr as u64),
                    (Some(0xd53b_421e), 272, NZCV),
                ] {
                    if let Some(word) = word {
                        program.push(word);
                    }
                    program.push(
                        0xf900_0000 | (((offset + position) as u32 / 8) << 10) | (27 << 5) | 30,
                    );
                    put(&mut expected, OUTPUT + offset + position, value);
                }
                registers[30] = NZCV;

                for (register, vector) in vectors.iter().enumerate() {
                    let position = offset + 288 + register * 16;
                    program.push(
                        0x3d80_0000 | ((position as u32 / 16) << 10) | (27 << 5) | register as u32,
                    );
                    expected[OUTPUT + position..OUTPUT + position + 16].copy_from_slice(vector);
                }
            }
            program.push(0xd400_0fc1);
            expected[CONSTANTS..].copy_from_slice(&data[CONSTANTS..]);
            let source_before = source.clone();
            let mut code = vec![0x5a; (program.len() * 4).next_multiple_of(PAGE)];
            for (index, word) in program.iter().enumerate() {
                code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            let code_before = code.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, data.len() as u64, Perm::RW, data.as_mut_ptr())
                    .unwrap();
                cpu.map_host(SOURCE, PAGE as u64, Perm::RO, source.as_mut_ptr())
                    .unwrap();
            }
            for (register, value) in [CANARY; 31].iter().enumerate() {
                cpu.set_register(register as u32, *value);
            }
            for (reg, value) in [
                (22, SOURCE),
                (24, DATA + CONSTANTS as u64),
                (27, DATA + OUTPUT as u64),
                (28, SOURCE + 0x800),
                (29, NZCV),
            ] {
                cpu.set_register(reg, value);
            }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let event = cpu
                .run(1_000_000)
                .expect("integer vector multiplication execution")
                .event;
            let fault = cpu.take_fault();
            let final_registers: [u64; 31] =
                std::array::from_fn(|register| cpu.get_register(register as u32));
            let sp = cpu.get_sp();
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, data.len() as u64).unwrap();
                cpu.unmap_host(SOURCE, PAGE as u64).unwrap();
            }
            assert!(
                matches!(event, CpuEvent::Svc(0x7e)),
                "{backend:?}: {event:?}"
            );
            assert!(
                fault.is_none(),
                "{backend:?}: mapped integer multiplication faulted"
            );
            assert_eq!(final_registers, registers, "{backend:?}: final GPR guards");
            assert_eq!(sp, CANARY, "{backend:?}: SP guard");
            assert_eq!(code, code_before, "{backend:?}: code guards");
            assert_eq!(source, source_before, "{backend:?}: RO source guards");

            for (index, (&actual, &wanted)) in data.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual,
                    wanted,
                    "{backend:?}: snapshot/guard byte{index:#x}, record{:?}",
                    records.get(index.saturating_sub(OUTPUT) / STRIDE)
                );
            }
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

fn pattern(bytes: usize, rotation: usize) -> [u8; 16] {
    let sign = 1u64 << (bytes * 8 - 1);
    let mask = (1u64 << (bytes * 8)) - 1;
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
        output.copy_from_slice(&values[(lane + rotation) & 7].to_le_bytes()[..bytes]);
    }
    result
}

fn words(values: [u32; 4]) -> [u8; 16] {
    let mut result = [0; 16];
    for (lane, value) in values.into_iter().enumerate() {
        result[lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    result
}

fn next_random(state: &mut u64) -> [u8; 16] {
    std::array::from_fn(|_| {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state as u8
    })
}

#[test]
fn every_legal_format_alias_and_boundary_or_random_product_matches() {
    let mut records = Vec::new();
    let mut random = 0x933b_bb80_4ea1_9402;
    for kind in KINDS {
        for (bytes, q) in arrangements() {
            for (rd, rn, rm) in [
                (0, 1, 2),
                (1, 1, 2),
                (2, 1, 2),
                (0, 1, 1),
                (31, 31, 2),
                (31, 0, 31),
                (31, 31, 31),
                (31, 0, 1),
            ] {
                for sample in 0..32 {
                    let (a, n, m) = if sample < 8 {
                        (
                            pattern(bytes, sample + 5),
                            pattern(bytes, sample),
                            pattern(bytes, sample + 3),
                        )
                    } else {
                        (
                            next_random(&mut random),
                            next_random(&mut random),
                            next_random(&mut random),
                        )
                    };
                    records.push(Case {
                        kind,
                        bytes,
                        q,
                        rn,
                        rm,
                        rd,
                        a,
                        n,
                        m,
                        fpcr: FPCR,
                        fpsr: if sample & 1 == 0 { 0 } else { STICKY },
                        literal: None,
                    });
                }
            }
        }
    }
    assert_eq!(records.len(), 4608);
    check(&records);
}

#[test]
fn all_65536_byte_pairs_have_exact_modular_multiply_add_and_subtract_results() {
    let mut records = Vec::new();
    for kind in KINDS {
        for q in [false, true] {
            let active = if q { 16 } else { 8 };
            for n in 0u16..256 {
                for start in (0u16..256).step_by(active) {
                    let mut n_vector = [0xa3; 16];
                    let mut m_vector = [0x5c; 16];
                    let mut accumulator = [0xff; 16];
                    for lane in 0..active {
                        n_vector[lane] = n as u8;
                        m_vector[lane] = (start + lane as u16) as u8;
                        accumulator[lane] =
                            ((n as usize * 17 + start as usize * 29 + lane * 31) & 255) as u8;
                    }
                    records.push(Case {
                        kind,
                        bytes: 1,
                        q,
                        rd: 2,
                        rn: 0,
                        rm: 1,
                        a: accumulator,
                        n: n_vector,
                        m: m_vector,
                        fpcr: FPCR,
                        fpsr: if (n + start) & 1 == 0 { 0 } else { STICKY },
                        literal: None,
                    });
                }
            }
        }
    }
    assert_eq!(records.len(), 36864);
    check(&records);
}

#[test]
fn all_registers_accumulator_aliases_q0_upper_clearing_and_flags_are_preserved() {
    let mut records = Vec::new();
    for kind in KINDS {
        for (bytes, q) in arrangements() {
            for rn in 0..32 {
                let rm = (rn + 17) & 31;
                for rd in [rn, rm, 31] {
                    records.push(Case {
                        kind,
                        bytes,
                        q,
                        rn,
                        rm,
                        rd,
                        a: seed(rd as usize),
                        n: seed(rn as usize),
                        m: seed(rm as usize),
                        fpcr: FPCR,
                        fpsr: STICKY,
                        literal: None,
                    });
                }
            }
        }
    }
    assert_eq!(records.len(), 1728);
    check(&records);
}

#[test]
fn exact_wonder_mla_and_all_aliased_overflow_literals_match() {
    let a = words([1, 1, u32::MAX, u32::MAX]);
    let n = words([u32::MAX, 0x8000_0000, 0x7fff_ffff, u32::MAX]);
    let m = words([2, u32::MAX, u32::MAX, u32::MAX]);
    let mut records = vec![
        Case {
            kind: Kind::Add,
            bytes: 4,
            q: true,
            rn: 0,
            rm: 1,
            rd: 2,
            a,
            n,
            m,
            fpcr: FPCR,
            fpsr: 0,
            literal: Some(words([u32::MAX, 0x8000_0001, 0x8000_0000, 0])),
        },
        Case {
            kind: Kind::Subtract,
            bytes: 4,
            q: true,
            rn: 0,
            rm: 1,
            rd: 2,
            a,
            n,
            m,
            fpcr: FPCR,
            fpsr: STICKY,
            literal: Some(words([3, 0x8000_0001, 0x7fff_fffe, 0xffff_fffe])),
        },
    ];
    assert_eq!(
        records[0].word(),
        0x4ea1_9402,
        "Wonder PC0x933bbb8: MLA V2.4S,V0.4S,V1.4S"
    );
    let all = words([u32::MAX, 0x8000_0000, 1, 0x7fff_ffff]);
    for (kind, literal) in [
        (Kind::Add, words([0, 0x8000_0000, 2, 0x8000_0000])),
        (
            Kind::Subtract,
            words([0xffff_fffe, 0x8000_0000, 0, 0x7fff_fffe]),
        ),
    ] {
        records.push(Case {
            kind,
            bytes: 4,
            q: true,
            rn: 31,
            rm: 31,
            rd: 31,
            a,
            n,
            m: all,
            fpcr: FPCR,
            fpsr: STICKY,
            literal: Some(literal),
        });
    }
    assert_eq!(records.len(), 4);
    check(&records);
}
