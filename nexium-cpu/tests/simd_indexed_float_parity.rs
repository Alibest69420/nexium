#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const INPUTS: usize = 0x1000;
const OUTPUTS: usize = 0x4000;
const CANARY: u64 = 0xfade_1234_5678_9abc;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Mul,
    Add,
    Sub,
}

#[derive(Clone, Copy, Debug)]
struct Form {
    double: bool,
    scalar: bool,
    lanes: usize,
}

impl Form {
    fn bytes(self) -> usize {
        if self.double {
            8
        } else {
            4
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Instruction {
    kind: Kind,
    form: Form,
    rd: u32,
    rn: u32,
    rm: u32,
    index: u32,
}

impl Instruction {
    fn word(self) -> u32 {
        assert!(self.rm < 32 && self.rn < 32 && self.rd < 32);
        assert!(self.index < if self.form.double { 2 } else { 4 });
        assert!(!self.form.double || self.form.scalar || self.form.lanes == 2);
        let operation = match self.kind {
            Kind::Mul => 0x9000,
            Kind::Add => 0x1000,
            Kind::Sub => 0x5000,
        };
        let lanes = if self.form.double {
            self.index << 11
        } else {
            ((self.index & 1) << 21) | ((self.index >> 1) << 11)
        };
        (if self.form.scalar { 0x5f80_0000 } else { 0x0f80_0000 })
            | if !self.form.scalar && self.form.lanes * self.form.bytes() == 16 { 1 << 30 } else { 0 }
            | ((self.form.double as u32) << 22)
            | ((self.rm & 15) << 16)
            | ((self.rm >> 4) << 20)
            | lanes | operation | (self.rn << 5) | self.rd
    }
}

fn properties(double: bool) -> (u64, u64, u64, u64) {
    if double {
        (1 << 63, 0x7ff0_0000_0000_0000, (1 << 52) - 1, 1 << 51)
    } else {
        (1 << 31, 0x7f80_0000, (1 << 23) - 1, 1 << 22)
    }
}

fn arithmetic(kind: Kind, double: bool, a: u64, n: u64, m: u64) -> u64 {
    let (sign, inf, fraction, quiet) = properties(double);
    let n = if matches!(kind, Kind::Sub) {
        n ^ sign
    } else {
        n
    };
    let nan = |value: u64| value & inf == inf && value & fraction != 0;
    let operands: &[u64] = if matches!(kind, Kind::Mul) {
        &[n, m]
    } else {
        &[a, n, m]
    };
    for &value in operands {
        if nan(value) && value & quiet == 0 {
            return value | quiet;
        }
    }
    let zero_inf =
        ((n & !sign == 0) && (m & !sign == inf)) || ((m & !sign == 0) && (n & !sign == inf));
    if !matches!(kind, Kind::Mul) && nan(a) && zero_inf {
        return inf | quiet;
    }
    for &value in operands {
        if nan(value) {
            return value | quiet;
        }
    }
    if zero_inf {
        return inf | quiet;
    }
    let product_inf = n & !sign == inf || m & !sign == inf;
    if !matches!(kind, Kind::Mul) && product_inf && a & !sign == inf && (a ^ n ^ m) & sign != 0 {
        return inf | quiet;
    }
    if double {
        let (a, n, m) = (f64::from_bits(a), f64::from_bits(n), f64::from_bits(m));
        if matches!(kind, Kind::Mul) {
            (n * m).to_bits()
        } else {
            n.mul_add(m, a).to_bits()
        }
    } else {
        let (a, n, m) = (
            f32::from_bits(a as u32),
            f32::from_bits(n as u32),
            f32::from_bits(m as u32),
        );
        (if matches!(kind, Kind::Mul) {
            (n * m).to_bits()
        } else {
            n.mul_add(m, a).to_bits()
        }) as u64
    }
}

fn get(vector: &[u8; 16], form: Form, index: usize) -> u64 {
    let mut bits = [0; 8];
    bits[..form.bytes()].copy_from_slice(&vector[index * form.bytes()..(index + 1) * form.bytes()]);
    u64::from_le_bytes(bits)
}

fn put(vector: &mut [u8; 16], form: Form, index: usize, value: u64) {
    vector[index * form.bytes()..(index + 1) * form.bytes()]
        .copy_from_slice(&value.to_le_bytes()[..form.bytes()]);
}

fn integer_bits(double: bool, value: i32) -> u64 {
    if double {
        (value as f64).to_bits()
    } else {
        (value as f32).to_bits() as u64
    }
}

fn numbers(form: Form, register: u32) -> [u8; 16] {
    let mut vector = [0; 16];
    for index in 0..16 / form.bytes() {
        put(
            &mut vector,
            form,
            index,
            integer_bits(form.double, (register * 5 + index as u32 * 3 + 1) as i32),
        );
    }
    vector
}

fn with_bits(form: Form, register: u32, index: usize, bits: u64) -> [u8; 16] {
    let mut vector = numbers(form, register);
    put(&mut vector, form, index, bits);
    vector
}

#[derive(Debug)]
struct Record {
    label: String,
    writes: Vec<(u32, [u8; 16])>,
    operations: Vec<Instruction>,
    literal: Option<(u32, u64)>,
}

fn check(records: &[Record]) {
    let mut backends = [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic]
        .map(|backend| (backend, Cpu::new(backend).expect("CPU initialization")));
    for records in records.chunks(60) {
        let mut initial = vec![0xa5; 0x10000];
        let mut vectors: [[u8; 16]; 32] = std::array::from_fn(|register| {
            std::array::from_fn(|byte| (register * 37 + byte * 11 + 0x93) as u8)
        });
        for (register, value) in vectors.iter().enumerate() {
            initial[register * 16..register * 16 + 16].copy_from_slice(value);
        }
        let mut expected = initial.clone();
        let mut program = Vec::new();
        program.push(0xd51b_441f);
        for register in 0..32u32 {
            program.push(0x3dc0_0060 | (register << 10) | register);
        }
        for (slot, record) in records.iter().enumerate() {
            assert!(record.writes.len() <= 4);
            for (part, &(register, value)) in record.writes.iter().enumerate() {
                let input = INPUTS + slot * 64 + part * 16;
                initial[input..input + 16].copy_from_slice(&value);
                expected[input..input + 16].copy_from_slice(&value);
                program.push(0x3dc0_0020 | (((slot * 4 + part) as u32) << 10) | register);
                vectors[register as usize] = value;
            }
            for &operation in &record.operations {
                let f = operation.form;
                let m = get(&vectors[operation.rm as usize], f, operation.index as usize);
                let n = vectors[operation.rn as usize];
                let a = vectors[operation.rd as usize];
                let mut result = [0; 16];
                for index in 0..f.lanes {
                    put(
                        &mut result,
                        f,
                        index,
                        arithmetic(
                            operation.kind,
                            f.double,
                            get(&a, f, index),
                            get(&n, f, index),
                            m,
                        ),
                    );
                }
                vectors[operation.rd as usize] = result;
                program.push(operation.word());
            }
            if let Some((register, literal)) = record.literal {
                let form = record.operations.last().unwrap().form;
                assert_eq!(
                    get(&vectors[register as usize], form, 0),
                    literal,
                    "oracle literal: {}",
                    record.label
                );
            }
            for (register, value) in vectors.iter().enumerate() {
                let output = slot * 32 + register;
                program.push(0x3d80_0040 | ((output as u32) << 10) | register as u32);
                expected[OUTPUTS + output * 16..OUTPUTS + output * 16 + 16].copy_from_slice(value);
            }
        }
        program.push(0xd400_0fc1);
        let mut encoded = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
        for (index, word) in program.iter().enumerate() {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        for (backend, cpu) in &mut backends {
            let mut code = encoded.clone();
            let mut actual = initial.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, actual.len() as u64, Perm::RW, actual.as_mut_ptr())
                    .unwrap();
            }
            for register in 0..31 {
                cpu.set_register(register, CANARY);
            }
            cpu.set_register(1, DATA + INPUTS as u64);
            cpu.set_register(2, DATA + OUTPUTS as u64);
            cpu.set_register(3, DATA);
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let event = cpu.run(100_000).expect("indexed FP execution").event;
            assert!(
                matches!(event, CpuEvent::Svc(0x7e)),
                "{backend:?}: {event:?}"
            );
            for (slot, record) in records.iter().enumerate() {
                for register in 0..32 {
                    let pos = OUTPUTS + slot * 512 + register * 16;
                    assert_eq!(
                        &actual[pos..pos + 16],
                        &expected[pos..pos + 16],
                        "{backend:?}: {} V{register} {:?}",
                        record.label,
                        record.operations
                    );
                }
            }
            for register in 0..31 {
                let wanted = match register {
                    1 => DATA + INPUTS as u64,
                    2 => DATA + OUTPUTS as u64,
                    3 => DATA,
                    _ => CANARY,
                };
                assert_eq!(
                    cpu.get_register(register),
                    wanted,
                    "{backend:?}: X{register}"
                );
            }
            assert_eq!(cpu.get_sp(), CANARY, "{backend:?}: SP");
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, actual.len() as u64).unwrap();
            }
            assert_eq!(actual, expected, "{backend:?}: source/memory guards");
            assert_eq!(code, encoded, "{backend:?}: code unchanged");
        }
    }
}

fn matrix(forms: &[Form]) -> Vec<Record> {
    let mut records = Vec::new();
    for &form in forms {
        for kind in [Kind::Mul, Kind::Add, Kind::Sub] {
            for rm in 0..32u32 {
                for index in 0..if form.double { 2 } else { 4 } {
                    let other = (rm + 1) % 32;
                    for (rn, rd) in [
                        (other, (rm + 2) % 32),
                        (other, other),
                        (other, rm),
                        (rm, other),
                        (rm, rm),
                    ] {
                        records.push(Record {
                            label: "all registers/lanes, exact arithmetic, aliases".into(),
                            writes: vec![
                                (rd, numbers(form, rd)),
                                (rn, numbers(form, rn)),
                                (rm, numbers(form, rm)),
                            ],
                            operations: vec![Instruction {
                                kind,
                                form,
                                rd,
                                rn,
                                rm,
                                index,
                            }],
                            literal: None,
                        });
                    }
                }
            }
        }
    }
    records
}

#[test]
fn scalar_indexed_s_d_all_registers_lanes_and_aliases_match_dynarmic() {
    check(&matrix(&[
        Form {
            double: false,
            scalar: true,
            lanes: 1,
        },
        Form {
            double: true,
            scalar: true,
            lanes: 1,
        },
    ]));
}

#[test]
fn vector_indexed_exact_lane_selection_high_rm_and_q0_upper_clear() {
    check(&matrix(&[
        Form {
            double: false,
            scalar: false,
            lanes: 2,
        },
        Form {
            double: false,
            scalar: false,
            lanes: 4,
        },
        Form {
            double: true,
            scalar: false,
            lanes: 2,
        },
    ]));
}

#[test]
fn scalar_indexed_fused_cancellation_overflow_rescue_and_specials() {
    let mut records = Vec::new();
    for double in [false, true] {
        let form = Form {
            double,
            scalar: true,
            lanes: 1,
        };
        let (sign, inf, _, quiet) = properties(double);
        let (one, plus, minus, epsilon, maximum) = if double {
            (
                0x3ff0_0000_0000_0000,
                0x3ff0_0000_0000_0001,
                0x3fef_ffff_ffff_fffe,
                0x3970_0000_0000_0000,
                0x7fef_ffff_ffff_ffff,
            )
        } else {
            (
                0x3f80_0000,
                0x3f80_0001,
                0x3f7f_fffe,
                0x2880_0000,
                0x7f7f_ffff,
            )
        };
        let two = integer_bits(double, 2);
        let nan_a = inf | quiet | 0x123;
        let nan_n = inf | quiet | 0x456;
        let nan_m = inf | quiet | 0x789;
        for kind in [Kind::Add, Kind::Sub] {
            let subtract = matches!(kind, Kind::Sub);
            let n_sign = if subtract { sign } else { 0 };
            let values = [
                (
                    "fused cancellation",
                    one | if subtract { 0 } else { sign },
                    plus,
                    minus,
                    epsilon | if subtract { 0 } else { sign },
                ),
                (
                    "overflowing product rescued",
                    maximum | if subtract { 0 } else { sign },
                    maximum,
                    two,
                    maximum | if subtract { sign } else { 0 },
                ),
                (
                    "positive exact cancellation zero",
                    one | if subtract { 0 } else { sign },
                    one,
                    one,
                    0,
                ),
                (
                    "negative zero",
                    sign,
                    if subtract { 0 } else { sign },
                    one,
                    sign,
                ),
                ("quiet addend NaN priority", nan_a, nan_n, nan_m, nan_a),
                (
                    "signaling addend NaN priority",
                    nan_a ^ quiet,
                    nan_n ^ quiet,
                    nan_m ^ quiet,
                    nan_a,
                ),
                (
                    "signaling multiplicand beats quiet addend",
                    nan_a,
                    nan_n ^ quiet,
                    nan_m,
                    nan_n ^ n_sign,
                ),
                (
                    "signaling multiplier beats quiet addend",
                    nan_a,
                    nan_n,
                    nan_m ^ quiet,
                    nan_m,
                ),
                (
                    "multiplicand NaN sign before subtraction",
                    one,
                    nan_n,
                    one,
                    nan_n ^ n_sign,
                ),
                (
                    "quiet addend and invalid product default NaN",
                    nan_a,
                    0,
                    inf,
                    inf | quiet,
                ),
                ("invalid product default NaN", one, 0, inf, inf | quiet),
                (
                    "opposed infinities default NaN",
                    inf | if subtract { 0 } else { sign },
                    inf,
                    one,
                    inf | quiet,
                ),
            ];
            for rm in [0, 15, 16, 31] {
                for index in 0..if double { 2 } else { 4 } {
                    for &(label, a, n, m, expected) in &values {
                        let rd = (rm + 1) % 32;
                        let rn = (rm + 2) % 32;
                        records.push(Record {
                            label: label.into(),
                            writes: vec![
                                (rd, with_bits(form, rd, 0, a)),
                                (rn, with_bits(form, rn, 0, n)),
                                (rm, with_bits(form, rm, index as usize, m)),
                            ],
                            operations: vec![Instruction {
                                kind,
                                form,
                                rd,
                                rn,
                                rm,
                                index,
                            }],
                            literal: Some((rd, expected)),
                        });
                    }
                }
            }
        }
        for (n, m) in [
            (0, one),
            (sign, one),
            (0, one | sign),
            (1, one),
            (maximum, two),
            (inf, one),
            (inf | sign, one),
        ] {
            for index in 0..if double { 2 } else { 4 } {
                records.push(Record {
                    label: "FMUL finite extrema, signed zeros and infinity".into(),
                    writes: vec![
                        (3, numbers(form, 3)),
                        (1, with_bits(form, 1, 0, n)),
                        (31, with_bits(form, 31, index as usize, m)),
                    ],
                    operations: vec![Instruction {
                        kind: Kind::Mul,
                        form,
                        rd: 3,
                        rn: 1,
                        rm: 31,
                        index,
                    }],
                    literal: Some((3, arithmetic(Kind::Mul, double, 0, n, m))),
                });
            }
        }
    }
    check(&records);
}

#[test]
fn wonder_exact_scalar_indexed_multiply_pair_preserves_source_until_overwrite() {
    let form = Form {
        double: false,
        scalar: true,
        lanes: 1,
    };
    let first = Instruction {
        kind: Kind::Mul,
        form,
        rd: 2,
        rn: 3,
        rm: 0,
        index: 0,
    };
    let second = Instruction {
        kind: Kind::Mul,
        form,
        rd: 0,
        rn: 1,
        rm: 0,
        index: 0,
    };
    assert_eq!(first.word(), 0x5f80_9062);
    assert_eq!(second.word(), 0x5f80_9020);
    let mut records = Vec::new();
    for factor in [
        0,
        0x8000_0000,
        0x3f00_0000,
        0xbf00_0000,
        0x3f80_0001,
        0x4120_0000,
    ] {
        records.push(Record {
            label: "Wonder 0x8474518/0x847451c exact pair".into(),
            writes: vec![
                (2, numbers(form, 2)),
                (0, with_bits(form, 0, 0, factor)),
                (1, with_bits(form, 1, 0, 0x4100_0000)),
                (3, with_bits(form, 3, 0, 0xc0c0_0000)),
            ],
            operations: vec![first, second],
            literal: Some((0, arithmetic(Kind::Mul, false, 0, 0x4100_0000, factor))),
        });
    }
    check(&records);
}
