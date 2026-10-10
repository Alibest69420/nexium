#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const INPUTS: usize = 0x1000;
const OUTPUTS: usize = 0x6000;
const FINAL: usize = 0xf000;
const CANARY: u64 = 0xface_1234_5678_9abc;
const FZ: u32 = 1 << 24;
const DN: u32 = 1 << 25;

#[derive(Clone, Copy, Debug)]
struct Form {
    double: bool,
    scalar: bool,
    lanes: usize,
}

const FORMS: [Form; 5] = [
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
];

impl Form {
    fn bytes(self) -> usize {
        if self.double {
            8
        } else {
            4
        }
    }
    fn modifiers(self) -> u32 {
        ((self.double as u32) << 22)
            | if !self.scalar && self.lanes * self.bytes() == 16 {
                1 << 30
            } else {
                0
            }
    }
    fn estimate(self, rd: u32, rn: u32) -> u32 {
        (if self.scalar {
            0x7ea1_d800
        } else {
            0x2ea1_d800
        }) | self.modifiers()
            | (rn << 5)
            | rd
    }
    fn step(self, rd: u32, rn: u32, rm: u32) -> u32 {
        (if self.scalar {
            0x5ea0_fc00
        } else {
            0x0ea0_fc00
        }) | self.modifiers()
            | (rm << 16)
            | (rn << 5)
            | rd
    }
    fn multiply(self, rd: u32, rn: u32, rm: u32) -> u32 {
        (if self.scalar {
            0x1e20_0800
        } else {
            0x2e20_dc00
        }) | self.modifiers()
            | (rm << 16)
            | (rn << 5)
            | rd
    }
}

fn format(double: bool) -> (u32, u64, u64, i32) {
    if double {
        (52, 0x8000_0000_0000_0000, 0x7ff0_0000_0000_0000, 1023)
    } else {
        (23, 0x8000_0000, 0x7f80_0000, 127)
    }
}

fn unpack_input(bits: u64, double: bool, control: u32) -> (u64, u32) {
    let (fraction, sign, infinity, _) = format(double);
    let subnormal = bits & infinity == 0 && bits & ((1 << fraction) - 1) != 0;
    if control & FZ != 0 && subnormal {
        (bits & sign, 1 << 7)
    } else {
        (bits, 0)
    }
}

fn estimate_oracle(input: u64, double: bool, control: u32) -> (u64, u32) {
    let (fraction, sign, infinity, bias) = format(double);
    let quiet = 1 << (fraction - 1);
    let mask = (1 << fraction) - 1;
    let (input, mut flags) = unpack_input(input, double, control);
    let mantissa = input & mask;
    if input & infinity == infinity && mantissa != 0 {
        if mantissa & quiet == 0 {
            flags |= 1;
        }
        return (
            if control & DN != 0 {
                infinity | quiet
            } else {
                input | quiet
            },
            flags,
        );
    }
    if input & !sign == 0 {
        return ((input & sign) | infinity, flags | 2);
    }
    if input & sign != 0 {
        return (infinity | quiet, flags | 1);
    }
    if input == infinity {
        return (0, flags);
    }
    let field = ((input & infinity) >> fraction) as i32;
    let (mut significand, mut exponent) = if field == 0 {
        (mantissa, 1 - bias)
    } else {
        ((1 << fraction) | mantissa, field - bias)
    };
    while significand < 1 << fraction {
        significand <<= 1;
        exponent -= 1;
    }
    let scaled = significand >> (fraction - if exponent & 1 == 0 { 7 } else { 8 });
    let odd = if scaled < 256 {
        scaled * 2 + 1
    } else {
        (scaled | 1) * 2
    };
    let (mut low, mut high) = (512u64, 1024u64);
    while low < high {
        let middle = (low + high + 1) / 2;
        if odd * middle * middle < 1 << 28 {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let estimate = (low + 1) / 2;
    let output_exponent = (-(exponent + 1)) >> 1;
    (
        ((output_exponent + bias) as u64) << fraction | ((estimate & 255) << (fraction - 8)),
        flags,
    )
}

fn integer_scale_jam(value: u128, shift: i32) -> u128 {
    if value == 0 {
        0
    } else if shift >= 0 {
        value << shift
    } else {
        let right = (-shift) as u32;
        if right >= 128 {
            u128::from(value != 0)
        } else {
            (value >> right) | u128::from(value & ((1u128 << right) - 1) != 0)
        }
    }
}

fn step_oracle(n: u64, m: u64, double: bool, control: u32) -> (u64, u32) {
    let (fraction, sign, infinity, bias) = format(double);
    let mask = (1 << fraction) - 1;
    let quiet = 1 << (fraction - 1);
    let (n, nflags) = unpack_input(n ^ sign, double, control);
    let (m, mflags) = unpack_input(m, double, control);
    let mut flags = nflags | mflags;
    let nan = |v: u64| v & infinity == infinity && v & mask != 0;
    let signaling = |v: u64| nan(v) && v & quiet == 0;
    for value in [n, m] {
        if signaling(value) {
            flags |= 1;
            return (
                if control & DN != 0 {
                    infinity | quiet
                } else {
                    value | quiet
                },
                flags,
            );
        }
    }
    for value in [n, m] {
        if nan(value) {
            return (
                if control & DN != 0 {
                    infinity | quiet
                } else {
                    value | quiet
                },
                flags,
            );
        }
    }
    let ninf = n & !sign == infinity;
    let minf = m & !sign == infinity;
    let nzero = n & !sign == 0;
    let mzero = m & !sign == 0;
    if (ninf && mzero) || (minf && nzero) {
        return (
            if double {
                0x3ff8_0000_0000_0000
            } else {
                0x3fc0_0000
            },
            flags,
        );
    }
    let negative_product = (n ^ m) & sign != 0;
    if ninf || minf {
        return (infinity | if negative_product { sign } else { 0 }, flags);
    }
    let finite = |value: u64| {
        let exponent = ((value & infinity) >> fraction) as i32;
        if exponent == 0 {
            (value & mask, 1 - bias - fraction as i32)
        } else {
            (
                (1 << fraction) | (value & mask),
                exponent - bias - fraction as i32,
            )
        }
    };
    let (nmant, nexp) = finite(n);
    let (mmant, mexp) = finite(m);
    let product = u128::from(nmant) * u128::from(mmant);
    let pexp = nexp + mexp;
    let product_high = if product == 0 {
        -10000
    } else {
        127 - product.leading_zeros() as i32 + pexp
    };
    let unit = product_high.max(1) - 120;
    let three = integer_scale_jam(3, -unit);
    let scaled_product = integer_scale_jam(product, pexp - unit);
    let (magnitude, negative) = if negative_product {
        if scaled_product > three {
            (scaled_product - three, true)
        } else {
            (three - scaled_product, false)
        }
    } else {
        (three + scaled_product, false)
    };
    if magnitude == 0 {
        return (0, flags);
    }
    let high = 127 - magnitude.leading_zeros() as i32;
    let mut exponent = unit - 1 + high;
    assert!(
        exponent >= 1 - bias,
        "finite step cannot have subnormal cancellation"
    );
    let shift = high - fraction as i32;
    let mut mantissa = if shift <= 0 {
        magnitude << -shift
    } else {
        let mut rounded = magnitude >> shift;
        let remainder = magnitude & ((1u128 << shift) - 1);
        let halfway = 1u128 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        rounded
    };
    if mantissa >= 1u128 << (fraction + 1) {
        mantissa >>= 1;
        exponent += 1;
    }
    let output = if exponent > bias {
        infinity
    } else {
        ((exponent + bias) as u64) << fraction | (mantissa as u64 & mask)
    };
    (output | if negative { sign } else { 0 }, flags)
}

fn samples(double: bool) -> Vec<u64> {
    let (fraction, sign, infinity, bias) = format(double);
    let mask = (1 << fraction) - 1;
    let mut values = vec![
        0,
        sign,
        infinity,
        sign | infinity,
        infinity | 0x123,
        sign | infinity | 0x456,
        infinity | (1 << (fraction - 1)) | 0x123,
        sign | infinity | (1 << (fraction - 1)) | 0x456,
        1,
        2,
        mask,
        mask >> 1,
        (mask >> 1) + 1,
        sign | 1,
        sign | mask,
        1 << fraction,
        (1 << fraction) + 1,
        infinity - 1,
        sign | (infinity - 1),
    ];
    for exponent in [
        -bias + 1,
        -bias + 2,
        -32,
        -1,
        0,
        1,
        2,
        3,
        32,
        bias - 1,
        bias,
    ] {
        let value = ((exponent + bias) as u64) << fraction;
        values.extend([value, value | sign]);
    }
    for scaled in 128u64..512 {
        let (exponent, top) = if scaled < 256 {
            (bias, 7)
        } else {
            (bias + 1, 8)
        };
        let mantissa = (scaled << (fraction - top)) & mask;
        values.push((exponent as u64) << fraction | mantissa);
    }
    let mut seed = 0x1234_5678_9abc_def0u64;
    for _ in 0..64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        values.push(if double { seed } else { seed as u32 as u64 });
    }
    values
}

fn seed_vector(register: u32) -> [u8; 16] {
    std::array::from_fn(|byte| (register as usize * 37 + byte * 11 + 0x93) as u8)
}

fn vector(form: Form, values: &[u64], start: usize, salt: u32) -> [u8; 16] {
    let mut result = seed_vector(salt);
    for lane in 0..form.lanes {
        let value = values[(start + lane) % values.len()];
        let pos = lane * form.bytes();
        result[pos..pos + form.bytes()].copy_from_slice(&value.to_le_bytes()[..form.bytes()]);
    }
    result
}

fn lane(vector: &[u8; 16], form: Form, index: usize) -> u64 {
    let mut bits = [0; 8];
    bits[..form.bytes()].copy_from_slice(&vector[index * form.bytes()..(index + 1) * form.bytes()]);
    u64::from_le_bytes(bits)
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Estimate,
    Step,
    Normalize,
}

#[derive(Clone, Debug)]
struct Record {
    form: Form,
    operation: Operation,
    rn: u32,
    rm: u32,
    rd: u32,
    n: [u8; 16],
    m: [u8; 16],
    control: u32,
    flags: Option<u32>,
}

fn multiply_bits(n: u64, m: u64, double: bool) -> u64 {
    if double {
        (f64::from_bits(n) * f64::from_bits(m)).to_bits()
    } else {
        (f32::from_bits(n as u32) * f32::from_bits(m as u32)).to_bits() as u64
    }
}

fn check_records(records: &[Record]) {
    check_records_backends(
        records,
        &[CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic],
    );
}

fn check_records_backends(records: &[Record], backends: &[CpuBackendKind]) {
    for records in records.chunks(400) {
        let mut initial = vec![0xa5; 0x10000];
        let mut vectors: [[u8; 16]; 32] = std::array::from_fn(|i| seed_vector(i as u32));
        for (i, value) in vectors.iter().enumerate() {
            initial[i * 16..i * 16 + 16].copy_from_slice(value);
        }
        let mut expected = initial.clone();
        let mut program = Vec::new();
        for register in 0..32 {
            program.push(0x3dc0_0060 | ((register as u32) << 10) | register as u32);
        }
        let mut registers = [CANARY; 31];
        registers[1] = DATA + INPUTS as u64;
        registers[2] = DATA + OUTPUTS as u64;
        registers[3] = DATA;
        registers[6] = DATA + FINAL as u64;
        for (slot, record) in records.iter().enumerate() {
            let f = record.form;
            let rd = record.rd as usize;
            let rn = record.rn as usize;
            let rm = record.rm as usize;
            initial[INPUTS + slot * 32..INPUTS + slot * 32 + 16].copy_from_slice(&record.n);
            initial[INPUTS + slot * 32 + 16..INPUTS + slot * 32 + 32].copy_from_slice(&record.m);
            expected[INPUTS + slot * 32..INPUTS + slot * 32 + 32]
                .copy_from_slice(&initial[INPUTS + slot * 32..INPUTS + slot * 32 + 32]);
            program.push(0xd2a0_0004 | ((record.control >> 16) << 5));
            program.push(0xd51b_4404);
            program.push(0xd51b_443f);
            registers[4] = record.control as u64;
            program.push(0x3dc0_0060 | (record.rd << 10) | record.rd);
            vectors[rd] = seed_vector(record.rd);
            program.push(0x3dc0_0020 | (((slot * 2) as u32) << 10) | record.rn);
            vectors[rn] = record.n;
            if matches!(record.operation, Operation::Step) && rm != rn {
                program.push(0x3dc0_0020 | (((slot * 2 + 1) as u32) << 10) | record.rm);
                vectors[rm] = record.m;
            }
            let mut result = [0; 16];
            let mut normalize_product = [0; 16];
            let mut normalize_step = [0; 16];
            for index in 0..f.lanes {
                let n = lane(&record.n, f, index);
                let m = lane(if rn == rm { &record.n } else { &record.m }, f, index);
                let value = match record.operation {
                    Operation::Estimate => estimate_oracle(n, f.double, record.control).0,
                    Operation::Step => step_oracle(n, m, f.double, record.control).0,
                    Operation::Normalize => {
                        let mut y = estimate_oracle(n, f.double, 0).0;
                        let mut product = 0;
                        let mut step = 0;
                        for _ in 0..2 {
                            product = multiply_bits(n, y, f.double);
                            step = step_oracle(y, product, f.double, 0).0;
                            y = multiply_bits(y, step, f.double);
                        }
                        let pos = index * f.bytes();
                        normalize_product[pos..pos + f.bytes()]
                            .copy_from_slice(&product.to_le_bytes()[..f.bytes()]);
                        normalize_step[pos..pos + f.bytes()]
                            .copy_from_slice(&step.to_le_bytes()[..f.bytes()]);
                        y
                    }
                };
                let pos = index * f.bytes();
                result[pos..pos + f.bytes()].copy_from_slice(&value.to_le_bytes()[..f.bytes()]);
            }
            match record.operation {
                Operation::Estimate => program.push(f.estimate(record.rd, record.rn)),
                Operation::Step => program.push(f.step(record.rd, record.rn, record.rm)),
                Operation::Normalize => {
                    assert_eq!((record.rd, record.rn, record.rm), (6, 3, 16));
                    program.extend_from_slice(&[
                        f.estimate(6, 3),
                        f.multiply(16, 3, 6),
                        f.step(16, 6, 16),
                        f.multiply(6, 6, 16),
                        f.multiply(16, 3, 6),
                        f.step(7, 6, 16),
                        f.multiply(6, 6, 7),
                    ]);
                    vectors[16] = normalize_product;
                    vectors[7] = normalize_step;
                }
            }
            vectors[rd] = result;
            let outputs = if matches!(record.operation, Operation::Normalize) {
                [record.rd, record.rn, 16, 7]
            } else {
                [
                    record.rd,
                    record.rn,
                    if matches!(record.operation, Operation::Step) {
                        record.rm
                    } else {
                        4
                    },
                    4,
                ]
            };
            for (part, register) in outputs.into_iter().enumerate() {
                program.push(0x3d80_0040 | (((slot * 5 + part) as u32) << 10) | register);
                expected[OUTPUTS + slot * 80 + part * 16..OUTPUTS + slot * 80 + part * 16 + 16]
                    .copy_from_slice(&vectors[register as usize]);
            }
            if let Some(flags) = record.flags {
                program.push(0xd53b_4429);
                program.push(0xf900_0049 | (((slot * 10 + 8) as u32) << 10));
                expected[OUTPUTS + slot * 80 + 64..OUTPUTS + slot * 80 + 72]
                    .copy_from_slice(&(flags as u64).to_le_bytes());
                registers[9] = flags as u64;
            }
            program.push(0xd53b_4408);
            program.push(0xf900_0048 | (((slot * 10 + 9) as u32) << 10));
            expected[OUTPUTS + slot * 80 + 72..OUTPUTS + slot * 80 + 80]
                .copy_from_slice(&(record.control as u64).to_le_bytes());
            registers[8] = record.control as u64;
        }
        for register in 0..32 {
            program.push(0x3d80_00c0 | ((register as u32) << 10) | register as u32);
            expected[FINAL + register * 16..FINAL + register * 16 + 16]
                .copy_from_slice(&vectors[register]);
        }
        program.push(0xd400_0fc1);
        let mut code = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
        for (i, word) in program.iter().enumerate() {
            code[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        for &backend in backends {
            let mut mapped_code = code.clone();
            let mut actual = initial.clone();
            let mut cpu = Cpu::new(backend).expect("CPU initialization");
            unsafe {
                cpu.map_host(
                    CODE,
                    mapped_code.len() as u64,
                    Perm::RX,
                    mapped_code.as_mut_ptr(),
                )
                .unwrap();
                cpu.map_host(DATA, actual.len() as u64, Perm::RW, actual.as_mut_ptr())
                    .unwrap();
            }
            for register in 0..31 {
                cpu.set_register(register, CANARY);
            }
            for register in [1, 2, 3, 6] {
                cpu.set_register(register, registers[register as usize]);
            }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let event = cpu.run(100_000).expect("reciprocal sqrt execution").event;
            assert!(
                matches!(event, CpuEvent::Svc(0x7e)),
                "{backend:?}: {event:?}"
            );
            for (slot, record) in records.iter().enumerate() {
                let pos = OUTPUTS + slot * 80;
                assert_eq!(
                    &actual[pos..pos + 80],
                    &expected[pos..pos + 80],
                    "{backend:?}: {record:?}"
                );
            }
            for register in 0..31 {
                assert_eq!(
                    cpu.get_register(register),
                    registers[register as usize],
                    "{backend:?}: X{register}"
                );
            }
            assert_eq!(cpu.get_sp(), CANARY, "{backend:?}: SP");
            drop(cpu);
            assert_eq!(
                actual, expected,
                "{backend:?}: full source/results/vector guards"
            );
        }
    }
}

#[test]
fn reciprocal_sqrt_estimate_architectural_bits_all_formats_and_aliases() {
    let mut records = Vec::new();
    for form in FORMS {
        let values = samples(form.double);
        for (rn, rd) in [(3, 6), (3, 3), (31, 0), (0, 31)] {
            for start in (0..values.len()).step_by(form.lanes) {
                records.push(Record {
                    form,
                    operation: Operation::Estimate,
                    rn,
                    rm: 0,
                    rd,
                    n: vector(form, &values, start, rn),
                    m: seed_vector(0),
                    control: 0,
                    flags: None,
                });
            }
        }
    }
    assert_eq!(FORMS[3].estimate(6, 3), 0x6ea1_d866);
    check_records(&records);
}

fn step_pairs(double: bool) -> Vec<(u64, u64)> {
    let (fraction, sign, infinity, bias) = format(double);
    let one = (bias as u64) << fraction;
    let two = ((bias + 1) as u64) << fraction;
    let three = two | (1 << (fraction - 1));
    let quiet = infinity | (1 << (fraction - 1));
    let mut pairs = vec![
        (one, one),
        (one, three),
        (one | sign, three),
        (two, two),
        (one + 1, three - 1),
        (infinity - 1, two | sign),
        (infinity - 1, two),
        (1, ((bias * 2) as u64) << fraction),
        (1, one),
        (0, one),
        (sign, one),
        (infinity, one),
        (sign | infinity, one),
        (one, infinity),
        (quiet | 0x123, quiet | 0x456),
        (quiet | sign | 0x123, quiet | 0x456),
        (quiet | 0x123, infinity | 0x456),
        (infinity | 0x123, quiet | 0x456),
        (infinity | 0x123, sign | infinity | 0x456),
    ];
    for zero in [0, sign] {
        for inf in [infinity, infinity | sign] {
            pairs.extend([(zero, inf), (inf, zero)]);
        }
    }
    let values = samples(double);
    for i in 0..64 {
        pairs.push((
            values[(i * 17) % values.len()],
            values[(i * 29 + 3) % values.len()],
        ));
    }
    pairs
}

#[test]
fn reciprocal_sqrt_step_fused_rounding_specials_and_aliases() {
    assert_eq!(
        step_oracle(0x3f80_0001, 0x403f_ffff, false, 0).0,
        0xb37f_fffc
    );
    assert_eq!(
        step_oracle(0x3ff0_0000_0000_0001, 0x4007_ffff_ffff_ffff, true, 0).0,
        0xbc9f_ffff_ffff_fffc
    );
    assert_eq!(
        step_oracle(0x7f7f_ffff, 0xc000_0000, false, 0).0,
        0x7f7f_ffff
    );
    assert_eq!(step_oracle(1, 0x7f00_0000, false, 0).0, 0x3fbf_ffff);
    assert_eq!(
        step_oracle(0x7fef_ffff_ffff_ffff, 0xc000_0000_0000_0000, true, 0).0,
        0x7fef_ffff_ffff_ffff
    );
    assert_eq!(
        step_oracle(1, 0x7fe0_0000_0000_0000, true, 0).0,
        0x3ff7_ffff_ffff_ffff
    );
    let mut records = Vec::new();
    for form in FORMS {
        let pairs = step_pairs(form.double);
        let n: Vec<_> = pairs.iter().map(|pair| pair.0).collect();
        let m: Vec<_> = pairs.iter().map(|pair| pair.1).collect();
        for (rn, rm, rd) in [
            (6, 16, 16),
            (6, 16, 7),
            (3, 3, 3),
            (31, 0, 31),
            (0, 31, 0),
            (31, 31, 2),
        ] {
            for start in (0..pairs.len()).step_by(form.lanes) {
                records.push(Record {
                    form,
                    operation: Operation::Step,
                    rn,
                    rm,
                    rd,
                    n: vector(form, &n, start, rn),
                    m: vector(form, &m, start, rm),
                    control: 0,
                    flags: None,
                });
            }
        }
    }
    assert_eq!(FORMS[3].step(16, 6, 16), 0x4eb0_fcd0);
    assert_eq!(FORMS[3].step(7, 6, 16), 0x4eb0_fcc7);
    check_records(&records);
}

fn control_records() -> Vec<Record> {
    let mut records = Vec::new();
    for form in FORMS {
        let (_, sign, infinity, _) = format(form.double);
        for control in [0, DN, FZ, DN | FZ] {
            for operation in [Operation::Estimate, Operation::Step] {
                for n in [
                    0,
                    sign,
                    infinity,
                    sign | infinity,
                    infinity | 0x123,
                    sign | infinity | 0x123,
                    1,
                    sign | 1,
                ] {
                    let m = 0;
                    let flags = match operation {
                        Operation::Estimate => estimate_oracle(n, form.double, control).1,
                        Operation::Step => step_oracle(n, m, form.double, control).1,
                        Operation::Normalize => unreachable!(),
                    };
                    records.push(Record {
                        form,
                        operation,
                        rn: 3,
                        rm: 16,
                        rd: 6,
                        n: vector(form, &[n], 0, 3),
                        m: vector(form, &[m], 0, 16),
                        control,
                        flags: Some(flags),
                    });
                }
            }
        }
    }
    records
}

#[test]
fn reciprocal_sqrt_dn_fz_values_and_fpcr_match_dynarmic() {
    let mut records = control_records();
    for record in &mut records {
        record.flags = None;
    }
    check_records(&records);
}

#[test]
fn reciprocal_sqrt_rustarmic_architectural_ioc_dzc_idc_flags() {
    check_records_backends(&control_records(), &[CpuBackendKind::Rustarmic]);
}

#[test]
fn reciprocal_sqrt_two_refinement_normalization_chain_matches_dynarmic() {
    let mut records = Vec::new();
    for form in FORMS {
        let (fraction, _, _, bias) = format(form.double);
        let values: Vec<_> = (-16..=16)
            .map(|exponent| {
                ((exponent + bias) as u64) << fraction
                    | (0x123456789abcdefu64 & ((1 << fraction) - 1))
            })
            .collect();
        for start in (0..values.len()).step_by(form.lanes) {
            records.push(Record {
                form,
                operation: Operation::Normalize,
                rn: 3,
                rm: 16,
                rd: 6,
                n: vector(form, &values, start, 3),
                m: seed_vector(16),
                control: 0,
                flags: None,
            });
        }
    }
    check_records(&records);
}
