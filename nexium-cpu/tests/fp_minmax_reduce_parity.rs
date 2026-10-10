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
const FZ: u32 = 1 << 24;
const DN: u32 = 1 << 25;
const IOC: u32 = 1;
const IDC: u32 = 1 << 7;

#[derive(Clone, Copy, Debug)]
enum Reduction {
    Maximum,
    Minimum,
    MaximumNumber,
    MinimumNumber,
}

const REDUCTIONS: [Reduction; 4] = [
    Reduction::Maximum,
    Reduction::Minimum,
    Reduction::MaximumNumber,
    Reduction::MinimumNumber,
];

impl Reduction {
    fn word(self, rn: u32, rd: u32) -> u32 {
        (match self {
            Self::Maximum => 0x6e30_f800,
            Self::Minimum => 0x6eb0_f800,
            Self::MaximumNumber => 0x6e30_c800,
            Self::MinimumNumber => 0x6eb0_c800,
        }) | (rn << 5)
            | rd
    }

    fn minimum(self) -> bool {
        matches!(self, Self::Minimum | Self::MinimumNumber)
    }

    fn numeric(self) -> bool {
        matches!(self, Self::MaximumNumber | Self::MinimumNumber)
    }
}

fn pair(mut a: u32, mut b: u32, op: Reduction, fpcr: u32) -> (u32, u32) {
    let mut flags = 0;
    for value in [&mut a, &mut b] {
        let exponent = (*value >> 23) & 0xff;
        let fraction = *value & 0x7f_ffff;
        if fpcr & FZ != 0 && exponent == 0 && fraction != 0 {
            *value &= 0x8000_0000;
            flags |= IDC;
        }
    }
    let classify = |value: u32| {
        let nan = (value >> 23) & 0xff == 0xff && value & 0x7f_ffff != 0;
        (nan, nan && value & 0x40_0000 == 0)
    };
    let (a_nan, a_signaling) = classify(a);
    let (b_nan, b_signaling) = classify(b);
    let selected_nan = if a_signaling {
        Some(a)
    } else if b_signaling {
        Some(b)
    } else if op.numeric() && a_nan != b_nan {
        return (if a_nan { b } else { a }, flags);
    } else if a_nan {
        Some(a)
    } else if b_nan {
        Some(b)
    } else {
        None
    };
    if let Some(value) = selected_nan {
        if a_signaling || b_signaling {
            flags |= IOC;
        }
        return (
            if fpcr & DN != 0 {
                0x7fc0_0000
            } else {
                value | 0x40_0000
            },
            flags,
        );
    }
    let a_mag = a & 0x7fff_ffff;
    let b_mag = b & 0x7fff_ffff;
    if a_mag == 0 && b_mag == 0 {
        let negative = if op.minimum() { a | b } else { a & b };
        return (negative & 0x8000_0000, flags);
    }
    let a_negative = a >> 31 != 0;
    let b_negative = b >> 31 != 0;
    let a_less = if a_negative != b_negative {
        a_negative
    } else if a_negative {
        a_mag > b_mag
    } else {
        a_mag < b_mag
    };
    (if a_less == op.minimum() { a } else { b }, flags)
}

fn reduce(lanes: [u32; 4], op: Reduction, fpcr: u32) -> (u32, u32) {
    let (left, lf) = pair(lanes[0], lanes[1], op, fpcr);
    let (right, rf) = pair(lanes[2], lanes[3], op, fpcr);
    let (result, final_flags) = pair(left, right, op, fpcr);
    (result, lf | rf | final_flags)
}

#[derive(Clone, Copy, Debug)]
struct Record {
    op: Reduction,
    rn: u32,
    rd: u32,
    lanes: [u32; 4],
    fpcr: u32,
    fpsr: u32,
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

fn check(records: &[Record], native_only: bool) {
    let mut cpus = cpus()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (backend, cpu) in &mut *cpus {
        if native_only && *backend != CpuBackendKind::Rustarmic {
            continue;
        }
        for records in records.chunks(32) {
            let mut data = vec![0xa5; 0x12000];
            let mut source = vec![0x6d; PAGE];
            let mut vectors: [[u8; 16]; 32] = std::array::from_fn(seed);
            for (register, vector) in vectors.iter().enumerate() {
                source[0x800 + register * 16..0x800 + register * 16 + 16].copy_from_slice(vector);
            }
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
                for (index, lane) in record.lanes.iter().enumerate() {
                    source[slot * 16 + index * 4..slot * 16 + index * 4 + 4]
                        .copy_from_slice(&lane.to_le_bytes());
                }
                let input: [u8; 16] = source[slot * 16..slot * 16 + 16].try_into().unwrap();
                program.push(0x3dc0_0000 | ((slot as u32) << 10) | (22 << 5) | record.rn);
                vectors[record.rn as usize] = input;
                for (which, reg, value) in
                    [(0, 25, record.fpcr as u64), (1, 26, record.fpsr as u64)]
                {
                    put(&mut data, CONSTANTS + (slot * 2 + which) * 8, value);
                    program
                        .push(0xf940_0000 | (((slot * 2 + which) as u32) << 10) | (24 << 5) | reg);
                    registers[reg as usize] = value;
                }
                program.extend([0xd51b_4419, 0xd51b_443a]);
                let (result, flags) = reduce(record.lanes, record.op, record.fpcr);
                vectors[record.rd as usize] = [0; 16];
                vectors[record.rd as usize][..4].copy_from_slice(&result.to_le_bytes());
                program.push(record.op.word(record.rn, record.rd));
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
                    (Some(0xd53b_443e), 264, (record.fpsr | flags) as u64),
                    (Some(0xd53b_421e), 272, NZCV),
                ] {
                    if position == 264 && *backend == CpuBackendKind::Dynarmic {
                        continue;
                    }
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
            let event = cpu.run(1_000_000).expect("FP reduction execution").event;
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
            assert!(fault.is_none(), "{backend:?}: mapped reduction faulted");
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

fn expand(vectors: &[[u32; 4]], controls: &[u32]) -> Vec<Record> {
    let mut records = Vec::new();
    for op in REDUCTIONS {
        for (rn, rd) in [(0, 1), (1, 1), (31, 0), (0, 31), (31, 31), (16, 7)] {
            for &fpcr in controls {
                for (index, &lanes) in vectors.iter().enumerate() {
                    records.push(Record {
                        op,
                        rn,
                        rd,
                        lanes,
                        fpcr,
                        fpsr: [0, 0x0800_001c, IOC, IDC, 0x0800_009f][index % 5],
                    });
                }
            }
        }
    }
    records
}

#[test]
fn all_four_reductions_finite_order_signed_zeros_subnormal_edges_and_aliases() {
    let samples = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x0080_0000,
        0x8080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3f7f_ffff,
        0xbf7f_ffff,
        0x3f80_0001,
        0xbf80_0001,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
    ];
    let mut vectors = Vec::new();
    for (i, &a) in samples.iter().enumerate() {
        for (j, &b) in samples.iter().enumerate() {
            vectors.push([
                a,
                b,
                samples[(i * 5 + j * 3) % samples.len()],
                samples[(i + j * 7 + 1) % samples.len()],
            ]);
        }
    }
    let mut random = 0xcafe_babe_1234_5678u64;
    for _ in 0..128 {
        vectors.push(std::array::from_fn(|_| {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let bits = random as u32;
            if bits & 0x7f80_0000 == 0x7f80_0000 {
                bits & 0xff80_0000
            } else {
                bits
            }
        }));
    }
    assert_eq!(
        reduce([0, 0x8000_0000, 0, 0], Reduction::Minimum, 0),
        (0x8000_0000, 0)
    );
    assert_eq!(
        reduce(
            [0x8000_0000, 0, 0x8000_0000, 0x8000_0000],
            Reduction::Maximum,
            0
        ),
        (0, 0)
    );
    check(&expand(&vectors, &[0]), false);
}

#[test]
fn stagewise_nan_priority_numeric_suppression_and_default_nan_values() {
    let samples = [
        0x7fc0_0123,
        0xffc0_0456,
        0x7f80_0789,
        0xff80_0abc,
        0x3f80_0000,
        0xbf80_0000,
        0,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
    ];
    let mut vectors = Vec::new();
    for &a in &samples {
        for &b in &samples {
            for tail in [[0x4000_0000, 0xc040_0000], [0x7fc0_0123, 0x7f80_0789]] {
                let lanes = [a, b, tail[0], tail[1]];
                for rotation in 0..4 {
                    vectors.push(std::array::from_fn(|index| lanes[(index + rotation) % 4]));
                }
            }
        }
    }
    assert_eq!(
        reduce(
            [0x7f80_0789, 0x3f80_0000, 0x4000_0000, 0x4040_0000],
            Reduction::MinimumNumber,
            0
        ),
        (0x4000_0000, IOC)
    );
    assert_eq!(
        reduce(
            [0x7fc0_0123, 0xff80_0abc, 0x4000_0000, 0x4040_0000],
            Reduction::Minimum,
            0
        ),
        (0xffc0_0abc, IOC)
    );
    check(&expand(&vectors, &[0, DN]), false);
}

#[test]
fn native_dn_fz_and_exact_sticky_ioc_idc_preserve_other_status_and_control_bits() {
    let vectors = [
        [1, 0x8000_0001, 0x007f_ffff, 0x807f_ffff],
        [0x0080_0000, 0x8080_0000, 1, 0x8000_0001],
        [0x7fc0_0123, 1, 0xff80_0456, 0x8000_0001],
        [0x7f80_0789, 0xffc0_0456, 1, 0x8000_0001],
        [0x7fc0_0123, 0xffc0_0456, 0x7fc0_0789, 0xffc0_0abc],
        [0x7f80_0123, 0xff80_0456, 0x7f80_0789, 0xff80_0abc],
    ];
    assert_eq!(reduce(vectors[0], Reduction::Maximum, FZ), (0, IDC));
    assert_eq!(
        reduce(vectors[0], Reduction::Minimum, FZ),
        (0x8000_0000, IDC)
    );
    check(
        &expand(&vectors, &[0, DN, FZ, DN | FZ, FZ | (3 << 22)]),
        true,
    );
}

#[test]
fn wonder_exact_fminnmv_and_adjacent_fmaxnmv_four_single_lanes() {
    assert_eq!(Reduction::MinimumNumber.word(0, 1), 0x6eb0_c801);
    check(
        &expand(
            &[
                [0x3f80_0000, 0xc000_0000, 0x4080_0000, 0x4040_0000],
                [0x7fc0_0123, 0x7f80_0456, 0x4000_0000, 0xc040_0000],
            ],
            &[0],
        ),
        false,
    );
}
