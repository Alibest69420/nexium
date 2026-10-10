#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;
use std::sync::{Mutex, OnceLock};

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const SOURCE: u64 = 0x30_0000;
const PAGE: usize = 0x1000;
const OUTPUT: usize = 0x1000;
const STRIDE: usize = 800;
const CANARY: u64 = 0xface_1234_5678_9abc;
const NZCV: u64 = 0xa000_0000;
const FPCR: u32 = 0x0700_0000;
const FPSR: u32 = 0x0800_009f;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Encrypt,
    Decrypt,
    Mix,
    InverseMix,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    kind: Kind,
    rd: u32,
    rn: u32,
    state: [u8; 16],
    key: [u8; 16],
    literal: Option<[u8; 16]>,
}

impl Case {
    fn word(self) -> u32 {
        let base = match self.kind {
            Kind::Encrypt => 0x4e28_4800,
            Kind::Decrypt => 0x4e28_5800,
            Kind::Mix => 0x4e28_6800,
            Kind::InverseMix => 0x4e28_7800,
        };
        base | (self.rn << 5) | self.rd
    }
}

fn multiply(mut left: u8, mut right: u8) -> u8 {
    let mut result = 0;
    for _ in 0..8 {
        if right & 1 != 0 {
            result ^= left;
        }
        left = (left << 1) ^ if left & 0x80 != 0 { 0x1b } else { 0 };
        right >>= 1;
    }
    result
}

fn substitution(value: u8) -> u8 {
    let mut inverse = 1;
    let mut base = value;
    let mut exponent = 254;
    while exponent != 0 {
        if exponent & 1 != 0 {
            inverse = multiply(inverse, base);
        }
        base = multiply(base, base);
        exponent >>= 1;
    }
    inverse
        ^ inverse.rotate_left(1)
        ^ inverse.rotate_left(2)
        ^ inverse.rotate_left(3)
        ^ inverse.rotate_left(4)
        ^ 0x63
}

fn oracle(kind: Kind, state: [u8; 16], key: [u8; 16]) -> [u8; 16] {
    let mut output = [0; 16];
    match kind {
        Kind::Encrypt | Kind::Decrypt => {
            let decrypt = matches!(kind, Kind::Decrypt);
            for column in 0..4 {
                for row in 0..4 {
                    let source_column = if decrypt {
                        (column + 4 - row) % 4
                    } else {
                        (column + row) % 4
                    };
                    let index = row + source_column * 4;
                    let value = state[index] ^ key[index];
                    output[row + column * 4] = if decrypt {
                        (0..=255u8)
                            .find(|&candidate| substitution(candidate) == value)
                            .unwrap()
                    } else {
                        substitution(value)
                    };
                }
            }
        }
        Kind::Mix | Kind::InverseMix => {
            let coefficients = if matches!(kind, Kind::Mix) {
                [2, 3, 1, 1]
            } else {
                [14, 11, 13, 9]
            };
            for column in 0..4 {
                for row in 0..4 {
                    for term in 0..4 {
                        output[row + column * 4] ^=
                            multiply(key[term + column * 4], coefficients[(term + 4 - row) % 4]);
                    }
                }
            }
        }
    }
    output
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

fn check(cases: &[Case]) {
    let mut cpus = cpus()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (backend, cpu) in &mut *cpus {
        for cases in cases.chunks(16) {
            let mut data = vec![0xa5; 0x10000];
            let mut expected = data.clone();
            let mut source = vec![0x6d; PAGE];
            let mut vectors: [[u8; 16]; 32] = std::array::from_fn(seed);
            for (register, vector) in vectors.iter().enumerate() {
                source[0x800 + register * 16..0x810 + register * 16].copy_from_slice(vector);
            }
            let mut registers: [u64; 31] = std::array::from_fn(|register| CANARY ^ register as u64);
            registers[19] = DATA + OUTPUT as u64;
            registers[20] = SOURCE;
            registers[21] = FPCR as u64;
            registers[22] = FPSR as u64;
            registers[23] = NZCV;
            let mut program = vec![0xd51b_4415, 0xd51b_4436, 0xd51b_4217];
            for register in 0..32 {
                program.push(0x3dc0_0000 | ((0x80 + register) << 10) | (20 << 5) | register);
            }
            for (slot, case) in cases.iter().enumerate() {
                source[slot * 32..slot * 32 + 16].copy_from_slice(&case.state);
                source[slot * 32 + 16..slot * 32 + 32].copy_from_slice(&case.key);
                program.push(0x3dc0_0000 | (((slot * 2) as u32) << 10) | (20 << 5) | case.rd);
                vectors[case.rd as usize] = case.state;
                program.push(0x3dc0_0000 | (((slot * 2 + 1) as u32) << 10) | (20 << 5) | case.rn);
                vectors[case.rn as usize] = case.key;
                let value = oracle(
                    case.kind,
                    vectors[case.rd as usize],
                    vectors[case.rn as usize],
                );
                if let Some(literal) = case.literal {
                    assert_eq!(value, literal, "independent AES literal: {case:?}");
                }
                vectors[case.rd as usize] = value;
                program.push(case.word());
                let offset = slot * STRIDE;
                for register in 0..31 {
                    program.push(
                        0xf900_0000
                            | (((offset / 8 + register) as u32) << 10)
                            | (19 << 5)
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
                    (Some(0xd53b_441e), 256, FPCR as u64),
                    (Some(0xd53b_443e), 264, FPSR as u64),
                    (Some(0xd53b_421e), 272, NZCV),
                ] {
                    if let Some(word) = word {
                        program.push(word);
                    }
                    program.push(
                        0xf900_0000 | (((offset + position) as u32 / 8) << 10) | (19 << 5) | 30,
                    );
                    put(&mut expected, OUTPUT + offset + position, value);
                }
                registers[30] = NZCV;
                for (register, vector) in vectors.iter().enumerate() {
                    let position = offset + 288 + register * 16;
                    program.push(
                        0x3d80_0000 | ((position as u32 / 16) << 10) | (19 << 5) | register as u32,
                    );
                    expected[OUTPUT + position..OUTPUT + position + 16].copy_from_slice(vector);
                }
            }
            program.push(0xd400_0fc1);
            let mut code = vec![0x5a; (program.len() * 4).next_multiple_of(PAGE)];
            for (index, word) in program.iter().enumerate() {
                code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            let code_before = code.clone();
            let source_before = source.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, data.len() as u64, Perm::RW, data.as_mut_ptr())
                    .unwrap();
                cpu.map_host(SOURCE, PAGE as u64, Perm::RO, source.as_mut_ptr())
                    .unwrap();
            }
            for (register, &value) in registers.iter().enumerate() {
                cpu.set_register(register as u32, value);
            }
            cpu.set_register(30, CANARY ^ 30);
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let mut event = CpuEvent::Running;
            for _ in 0..8 {
                event = cpu.run_with_count(100_000).event;
                if !matches!(event, CpuEvent::Running) {
                    break;
                }
            }
            assert!(
                matches!(event, CpuEvent::Svc(0x7e)),
                "{backend:?}: {event:?}"
            );
            assert!(cpu.take_fault().is_none(), "{backend:?} memory fault");
            assert_eq!(
                cpu.get_pc(),
                CODE + program.len() as u64 * 4,
                "{backend:?} PC"
            );
            assert_eq!(
                data, expected,
                "{backend:?} AES outputs/register/status/memory guards"
            );
            assert_eq!(source, source_before, "{backend:?} source guard");
            assert_eq!(code, code_before, "{backend:?} code guard");
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, data.len() as u64).unwrap();
                cpu.unmap_host(SOURCE, PAGE as u64).unwrap();
            }
        }
    }
}

#[test]
fn exact_mk8_sdk_inverse_mix_alias_matches_dynarmic_and_fips_literal() {
    let input = [
        0x04, 0x66, 0x81, 0xe5, 0xe0, 0xcb, 0x19, 0x9a, 0x48, 0xf8, 0xd3, 0x7a, 0x28, 0x06, 0x26,
        0x4c,
    ];
    let output = [
        0xd4, 0xbf, 0x5d, 0x30, 0xe0, 0xb4, 0x52, 0xae, 0xb8, 0x41, 0x11, 0xf1, 0x1e, 0x27, 0x98,
        0xe5,
    ];
    let case = Case {
        kind: Kind::InverseMix,
        rd: 1,
        rn: 1,
        state: [0xff; 16],
        key: input,
        literal: Some(output),
    };
    assert_eq!(case.word(), 0x4e28_7821);
    check(&[case]);
}

#[test]
fn every_aes_operation_register_source_alias_and_status_matches_dynarmic() {
    let mut cases = Vec::new();
    for kind in [Kind::Encrypt, Kind::Decrypt, Kind::Mix, Kind::InverseMix] {
        for rd in 0..32 {
            for rn in [rd, (rd + 7) % 32] {
                for pattern in 0..5 {
                    let state = match pattern {
                        0 => [0; 16],
                        1 => [0xff; 16],
                        _ => std::array::from_fn(|byte| {
                            (byte * 17 + rd as usize * 37 + pattern * 41) as u8
                        }),
                    };
                    let key = match pattern {
                        0 => [0; 16],
                        1 => [0xff; 16],
                        _ => std::array::from_fn(|byte| {
                            (byte * 29 + rn as usize * 11 + pattern * 53) as u8
                        }),
                    };
                    cases.push(Case {
                        kind,
                        rd,
                        rn,
                        state,
                        key,
                        literal: None,
                    });
                }
            }
        }
    }
    check(&cases);
}
