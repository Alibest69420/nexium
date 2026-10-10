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
    Sat {
        width: u32,
        q: bool,
        scalar: bool,
        unsigned: bool,
        subtract: bool,
    },
    Compare {
        double: bool,
        q: bool,
        scalar: bool,
        greater: bool,
    },
    Carry {
        width: u32,
        subtract: bool,
        flags: bool,
        n: u64,
        m: u64,
    },
}
#[derive(Clone, Copy, Debug)]
struct Case {
    kind: Kind,
    rn: u32,
    rm: u32,
    rd: u32,
    a: [u8; 16],
    n: [u8; 16],
    m: [u8; 16],
    fpcr: u32,
    fpsr: u32,
    nzcv: u64,
}
const QC: u32 = 1 << 27;
const DN: u32 = 1 << 25;
const FZ: u32 = 1 << 24;
impl Case {
    fn word(self) -> u32 {
        let common = self.rm << 16 | self.rn << 5 | self.rd;
        match self.kind {
            Kind::Sat {
                width,
                q,
                scalar,
                unsigned,
                subtract,
            } => {
                0x0e20_0c00
                    | (scalar as u32) << 28
                    | (q as u32) << 30
                    | (unsigned as u32) << 29
                    | (subtract as u32) << 13
                    | (width.trailing_zeros() - 3) << 22
                    | common
            }
            Kind::Compare {
                double,
                q,
                scalar,
                greater,
            } => {
                0x2e20_ec00
                    | (scalar as u32) << 28
                    | (q as u32) << 30
                    | (double as u32) << 22
                    | (greater as u32) << 23
                    | common
            }
            Kind::Carry {
                width,
                subtract,
                flags,
                ..
            } => {
                0x1a00_0000
                    | ((width == 64) as u32) << 31
                    | (subtract as u32) << 30
                    | (flags as u32) << 29
                    | common
            }
        }
    }
    fn expected(self, n: [u8; 16], m: [u8; 16]) -> ([u8; 16], u32) {
        let mut output = [0; 16];
        let mut status = 0;
        let (width, count) = match self.kind {
            Kind::Sat {
                width, q, scalar, ..
            } => (
                width,
                if scalar {
                    1
                } else {
                    (if q { 128 } else { 64 }) / width
                },
            ),
            Kind::Compare {
                double, q, scalar, ..
            } => (
                if double { 64 } else { 32 },
                if scalar {
                    1
                } else if double || !q {
                    2
                } else {
                    4
                },
            ),
            Kind::Carry { .. } => unreachable!(),
        };
        for i in 0..count as usize {
            let read = |v: [u8; 16]| {
                let bytes = width as usize / 8;
                let mut raw = [0; 8];
                raw[..bytes].copy_from_slice(&v[i * bytes..(i + 1) * bytes]);
                u64::from_le_bytes(raw)
            };
            let (value, flags) = match self.kind {
                Kind::Sat {
                    unsigned, subtract, ..
                } => {
                    let modulus = 1i128 << width;
                    let signed = |x: u64| {
                        if !unsigned && x as i128 >= modulus / 2 {
                            x as i128 - modulus
                        } else {
                            x as i128
                        }
                    };
                    let a = signed(read(n));
                    let b = signed(read(m));
                    let x = if subtract { a - b } else { a + b };
                    let (lo, hi) = if unsigned {
                        (0, modulus - 1)
                    } else {
                        (-modulus / 2, modulus / 2 - 1)
                    };
                    let answer = x.clamp(lo, hi);
                    (answer as u64, if x != answer { QC } else { 0 })
                }
                Kind::Compare {
                    double, greater, ..
                } => compare(read(n), read(m), double, greater, self.fpcr),
                Kind::Carry { .. } => unreachable!(),
            };
            status |= flags;
            let bytes = width as usize / 8;
            output[i * bytes..(i + 1) * bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
        }
        (output, status)
    }
}
fn carry(n: u64, m: u64, width: u32, input: u8, subtract: bool) -> (u64, u8) {
    let modulus = 1u128 << width;
    let mask = modulus - 1;
    let a = n as u128 & mask;
    let b = if subtract {
        !m as u128 & mask
    } else {
        m as u128 & mask
    };
    let sum = a + b + u128::from(input & 2 != 0);
    let result = sum % modulus;
    let signed = |x: u128| {
        if x >= modulus / 2 {
            x as i128 - modulus as i128
        } else {
            x as i128
        }
    };
    let exact = signed(a) + signed(b) + i128::from(input & 2 != 0);
    let overflow = exact < -(modulus as i128 / 2) || exact >= modulus as i128 / 2;
    (
        result as u64,
        ((result >= modulus / 2) as u8) * 8
            | ((result == 0) as u8) * 4
            | ((sum >= modulus) as u8) * 2
            | overflow as u8,
    )
}
fn compare(n: u64, m: u64, double: bool, greater: bool, fpcr: u32) -> (u64, u32) {
    let sign = if double { 1 << 63 } else { 1 << 31 };
    let infinity = if double {
        0x7ff0_0000_0000_0000
    } else {
        0x7f80_0000
    };
    let minimum = if double {
        0x0010_0000_0000_0000
    } else {
        0x0080_0000
    };
    let mut a = n & (sign - 1);
    let mut b = m & (sign - 1);
    let mut flags = 0;
    for x in [&mut a, &mut b] {
        if fpcr & FZ != 0 && *x != 0 && *x < minimum {
            *x = 0;
            flags |= 128;
        }
    }
    if a > infinity || b > infinity {
        return (0, flags | 1);
    }
    let (a, b) = if double {
        (f64::from_bits(a), f64::from_bits(b))
    } else {
        (
            f32::from_bits(a as u32) as f64,
            f32::from_bits(b as u32) as f64,
        )
    };
    (
        if if greater { a > b } else { a >= b } {
            if double {
                u64::MAX
            } else {
                u32::MAX as u64
            }
        } else {
            0
        },
        flags,
    )
}
fn pattern(width: u32, values: &[u64], offset: usize) -> [u8; 16] {
    let mut result = [0; 16];
    let bytes = width as usize / 8;
    for (i, lane) in result.chunks_exact_mut(bytes).enumerate() {
        lane.copy_from_slice(&values[(offset + i * 3) % values.len()].to_le_bytes()[..bytes]);
    }
    result
}
fn record(
    kind: Kind,
    rn: u32,
    rm: u32,
    rd: u32,
    n: [u8; 16],
    m: [u8; 16],
    fpcr: u32,
    fpsr: u32,
    nzcv: u64,
) -> Case {
    Case {
        kind,
        rn,
        rm,
        rd,
        a: [0x93; 16],
        n,
        m,
        fpcr,
        fpsr,
        nzcv,
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
            let mut program = Vec::new();
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
                for (which, reg, value) in [
                    (0, 25, record.fpcr as u64),
                    (1, 26, record.fpsr as u64),
                    (2, 29, record.nzcv),
                ] {
                    put(&mut data, CONSTANTS + (slot * 5 + which) * 8, value);
                    program
                        .push(0xf940_0000 | (((slot * 5 + which) as u32) << 10) | (24 << 5) | reg);
                    registers[reg as usize] = value;
                }
                program.extend([0xd51b_4419, 0xd51b_443a, 0xd51b_421d]);
                if let Kind::Carry { n, m, .. } = record.kind {
                    for (which, reg, value) in [(3, record.rn, n), (4, record.rm, m)] {
                        put(&mut data, CONSTANTS + (slot * 5 + which) * 8, value);
                        program.push(
                            0xf940_0000 | (((slot * 5 + which) as u32) << 10) | (24 << 5) | reg,
                        );
                        if reg != 31 {
                            registers[reg as usize] = value;
                        }
                    }
                }
                let mut fpsr = record.fpsr;
                let mut nzcv = record.nzcv;
                match record.kind {
                    Kind::Carry {
                        width,
                        subtract,
                        flags,
                        ..
                    } => {
                        let (value, status) = carry(
                            if record.rn == 31 {
                                0
                            } else {
                                registers[record.rn as usize]
                            },
                            if record.rm == 31 {
                                0
                            } else {
                                registers[record.rm as usize]
                            },
                            width,
                            (record.nzcv >> 28) as u8,
                            subtract,
                        );
                        if record.rd != 31 {
                            registers[record.rd as usize] = value;
                        }
                        if flags {
                            nzcv = (status as u64) << 28;
                        }
                    }
                    _ => {
                        let (result, status) = record
                            .expected(vectors[record.rn as usize], vectors[record.rm as usize]);
                        vectors[record.rd as usize] = result;
                        let legacy_idc_omission = *backend == CpuBackendKind::Dynarmic
                            && matches!(record.kind, Kind::Compare { .. })
                            && [0, DN, FZ, DN | FZ].contains(&record.fpcr);
                        fpsr |= if legacy_idc_omission {
                            status & !128
                        } else {
                            status
                        };
                    }
                }
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
                    (Some(0xd53b_421e), 272, nzcv),
                ] {
                    if let Some(word) = word {
                        program.push(word);
                    }
                    program.push(
                        0xf900_0000 | (((offset + position) as u32 / 8) << 10) | (27 << 5) | 30,
                    );
                    put(&mut expected, OUTPUT + offset + position, value);
                }
                registers[30] = nzcv;

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
            let event = cpu.run(1_000_000).expect("binary status execution").event;
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
                "{backend:?}: mapped binary status operation faulted"
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

#[test]
fn carry_w_x_aliases_zero_register_exact_nzcv_and_ordinary_preservation() {
    let mut cases = Vec::new();
    for width in [32, 64] {
        let mask = if width == 64 {
            u64::MAX
        } else {
            u32::MAX as u64
        };
        let values = [
            0,
            1,
            mask,
            mask - 1,
            1 << (width - 1),
            (1 << (width - 1)) - 1,
            CANARY,
        ];
        for subtract in [false, true] {
            for flags in [false, true] {
                for (rn, rm, rd) in [
                    (0, 1, 2),
                    (0, 1, 0),
                    (0, 1, 1),
                    (31, 1, 2),
                    (0, 31, 2),
                    (0, 1, 31),
                    (15, 15, 15),
                ] {
                    for n in values {
                        for m in values {
                            for input in [0u64, 2, 13, 15] {
                                cases.push(record(
                                    Kind::Carry {
                                        width,
                                        subtract,
                                        flags,
                                        n,
                                        m,
                                    },
                                    rn,
                                    rm,
                                    rd,
                                    [0; 16],
                                    [0; 16],
                                    FPCR,
                                    STICKY,
                                    input << 28,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases.len(), 10976);
    check(&cases);
}
#[test]
fn sat_all_scalar_vector_widths_boundary_alias_and_sticky_status() {
    let mut cases = Vec::new();
    for scalar in [false, true] {
        for width in [8u32, 16, 32, 64] {
            for q in [false, true] {
                if scalar && !q || !scalar && width == 64 && !q {
                    continue;
                }
                let modulus = 1u128 << width;
                let values = [
                    0,
                    1,
                    2,
                    (modulus - 1) as u64,
                    (modulus / 2) as u64,
                    (modulus / 2 - 1) as u64,
                    (modulus / 2 + 1) as u64,
                ];
                for unsigned in [false, true] {
                    for subtract in [false, true] {
                        for (rn, rm, rd) in
                            [(0, 1, 2), (0, 1, 0), (0, 1, 1), (31, 0, 31), (1, 1, 1)]
                        {
                            for a in 0..values.len() {
                                for b in 0..values.len() {
                                    for fpsr in [0x94, QC | 0x94] {
                                        cases.push(record(
                                            Kind::Sat {
                                                width,
                                                q,
                                                scalar,
                                                unsigned,
                                                subtract,
                                            },
                                            rn,
                                            rm,
                                            rd,
                                            pattern(width, &values, a),
                                            pattern(width, &values, b),
                                            DN,
                                            fpsr,
                                            NZCV,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    check(&cases);
}
#[test]
fn facge_facgt_scalar_vector_all_special_pairs_dn_fz_aliases_and_status() {
    let mut cases = Vec::new();
    for (scalar, double, q) in [
        (false, false, false),
        (false, false, true),
        (false, true, true),
        (true, false, true),
        (true, true, true),
    ] {
        let width = if double { 64 } else { 32 };
        let values: Vec<u64> = if double {
            vec![
                0,
                0x8000_0000_0000_0000,
                1,
                0x000f_ffff_ffff_ffff,
                0x8000_0000_0000_0001,
                0x0010_0000_0000_0000,
                0x3ff0_0000_0000_0000,
                0xbff0_0000_0000_0000,
                0x7fef_ffff_ffff_ffff,
                0x7ff0_0000_0000_0000,
                0xfff0_0000_0000_0000,
                0x7ff8_0000_0000_0123,
                0xfff0_0000_0000_0456,
            ]
        } else {
            vec![
                0,
                0x8000_0000,
                1,
                0x007f_ffff,
                0x8000_0001,
                0x0080_0000,
                0x3f80_0000,
                0xbf80_0000,
                0x7f7f_ffff,
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0123,
                0xff80_0456,
            ]
        };
        for greater in [false, true] {
            for (rn, rm, rd) in [(1, 2, 0), (1, 2, 1), (1, 2, 2), (31, 0, 31), (31, 31, 31)] {
                for a in 0..values.len() {
                    for b in 0..values.len() {
                        for fpcr in [0, DN, FZ, DN | FZ] {
                            cases.push(record(
                                Kind::Compare {
                                    double,
                                    q,
                                    scalar,
                                    greater,
                                },
                                rn,
                                rm,
                                rd,
                                pattern(width, &values, a),
                                pattern(width, &values, b),
                                fpcr,
                                0x14,
                                NZCV,
                            ));
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        record(
            Kind::Compare {
                double: false,
                q: true,
                scalar: false,
                greater: false
            },
            1,
            2,
            2,
            [0; 16],
            [0; 16],
            0,
            0,
            0
        )
        .word(),
        0x6e22_ec22
    );
    check(&cases);
}

#[test]
fn absolute_compare_fz_new_idc_and_retained_old_idc_literals() {
    assert_eq!(compare(1, 0, false, false, FZ), (u32::MAX as u64, 128));
    assert_eq!(compare(1, 0, true, true, FZ), (0, 128));
    assert_eq!(compare(1, 0x7fc0_0123, false, false, FZ), (0, 129));
    let mut cases = Vec::new();
    for (scalar, double, q) in [
        (false, false, false),
        (false, false, true),
        (false, true, true),
        (true, false, true),
        (true, true, true),
    ] {
        let width = if double { 64 } else { 32 };
        for greater in [false, true] {
            for fpsr in [0x14, 0x94] {
                cases.push(record(
                    Kind::Compare {
                        scalar,
                        double,
                        q,
                        greater,
                    },
                    1,
                    2,
                    0,
                    pattern(width, &[1], 0),
                    [0; 16],
                    FZ,
                    fpsr,
                    NZCV,
                ));
            }
        }
    }
    check(&cases);
}
