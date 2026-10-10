#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const INPUTS: usize = 0x1000;
const OUTPUTS: usize = 0x4000;
const STRIDE: usize = 544;
const CANARY: u64 = 0xface_1234_5678_9abc;
const FZ: u32 = 1 << 24;
const DN: u32 = 1 << 25;
const INITIAL_FLAGS: u32 = 1 << 27;

struct Diagnostics;
impl log::Log for Diagnostics {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}", record.args());
        }
    }
    fn flush(&self) {}
}
static DIAGNOSTICS: Diagnostics = Diagnostics;

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
            0x5ea1_d800
        } else {
            0x0ea1_d800
        }) | self.modifiers()
            | (rn << 5)
            | rd
    }
    fn step(self, rd: u32, rn: u32, rm: u32) -> u32 {
        (if self.scalar {
            0x5e20_fc00
        } else {
            0x0e20_fc00
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
    fn compare_zero(self, rd: u32, rn: u32) -> u32 {
        (if self.scalar {
            0x5ea0_d800
        } else {
            0x0ea0_d800
        }) | self.modifiers()
            | (rn << 5)
            | rd
    }
    fn chain(self) -> [u32; 6] {
        [
            self.estimate(17, 5),
            self.step(18, 17, 5),
            self.multiply(17, 17, 18),
            self.step(18, 17, 5),
            self.compare_zero(5, 5),
            self.multiply(16, 17, 18),
        ]
    }
}

fn format(double: bool) -> (u32, u64, u64, i32) {
    if double {
        (52, 1 << 63, 0x7ff0_0000_0000_0000, 1023)
    } else {
        (23, 1 << 31, 0x7f80_0000, 127)
    }
}

fn unpack_input(bits: u64, double: bool, control: u32) -> (u64, u32) {
    let (fraction, sign, infinity, _) = format(double);
    if control & FZ != 0 && bits & infinity == 0 && bits & ((1 << fraction) - 1) != 0 {
        (bits & sign, 1 << 7)
    } else {
        (bits, 0)
    }
}

fn estimate_oracle(input: u64, double: bool, control: u32) -> (u64, u32) {
    let (fraction, sign, infinity, bias) = format(double);
    let mask = (1 << fraction) - 1;
    let quiet = 1 << (fraction - 1);
    let (input, mut flags) = unpack_input(input, double, control);
    let negative = input & sign != 0;
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
    if input & !sign == infinity {
        return (input & sign, flags);
    }
    if input & !sign == 0 {
        return ((input & sign) | infinity, flags | 2);
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
    if exponent < (1 - bias) - 2 {
        let to_infinity = match (control >> 22) & 3 {
            0 => true,
            1 => !negative,
            2 => negative,
            3 => false,
            _ => unreachable!(),
        };
        return (
            (input & sign) | if to_infinity { infinity } else { infinity - 1 },
            flags | 0x14,
        );
    }
    if control & FZ != 0 && exponent >= bias - 1 {
        return (input & sign, flags | 8);
    }
    let scaled = significand >> (fraction - 8);
    let denominator = scaled * 2 + 1;
    let quotient = (1 << 18) / denominator;
    let remainder = (1 << 18) % denominator;
    let estimate = quotient + u64::from(remainder * 2 > denominator);
    assert!((256..512).contains(&estimate));
    let output_field = -(exponent + 1) + bias;
    let output = if output_field >= 1 {
        (output_field as u64) << fraction | ((estimate & 255) << (fraction - 8))
    } else {
        assert!(output_field >= -1);
        (estimate << (fraction - 8)) >> (1 - output_field)
    };
    (output | (input & sign), flags)
}

fn scale_jam(value: u128, shift: i32) -> u128 {
    if value == 0 {
        0
    } else if shift >= 0 {
        value << shift
    } else {
        let right = (-shift) as u32;
        if right >= 128 {
            1
        } else {
            (value >> right) | u128::from(value & ((1u128 << right) - 1) != 0)
        }
    }
}

fn step_oracle(n: u64, m: u64, double: bool, control: u32) -> (u64, u32) {
    assert_eq!(
        control & (3 << 22),
        0,
        "directed step rounding is not covered"
    );
    let (fraction, sign, infinity, bias) = format(double);
    let mask = (1 << fraction) - 1;
    let quiet = 1 << (fraction - 1);
    let (n, nflags) = unpack_input(n ^ sign, double, control);
    let (m, mflags) = unpack_input(m, double, control);
    let mut flags = nflags | mflags;
    let nan = |v: u64| v & infinity == infinity && v & mask != 0;
    for value in [n, m] {
        if nan(value) && value & quiet == 0 {
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
                    value
                },
                flags,
            );
        }
    }
    let ninf = n & !sign == infinity;
    let minf = m & !sign == infinity;
    let two = ((bias + 1) as u64) << fraction;
    if (ninf && m & !sign == 0) || (minf && n & !sign == 0) {
        return (two, flags);
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
    let scaled_two = scale_jam(2, -unit);
    let scaled_product = scale_jam(product, pexp - unit);
    let (magnitude, negative) = if negative_product {
        if scaled_product > scaled_two {
            (scaled_product - scaled_two, true)
        } else {
            (scaled_two - scaled_product, false)
        }
    } else {
        (scaled_two + scaled_product, false)
    };
    if magnitude == 0 {
        return (0, flags);
    }
    let high = 127 - magnitude.leading_zeros() as i32;
    let mut exponent = unit + high;
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
    let threshold = 1 << (fraction - 2);
    let mut positive = vec![
        0,
        infinity,
        infinity | 0x123,
        infinity | (1 << (fraction - 1)) | 0x456,
        1,
        2,
        threshold - 1,
        threshold,
        threshold + 1,
        mask >> 1,
        (mask >> 1) + 1,
        mask,
        1 << fraction,
        (1 << fraction) + 1,
        infinity - 1,
    ];
    for exponent in [
        1 - bias,
        2 - bias,
        -32,
        -1,
        0,
        1,
        2,
        32,
        bias - 2,
        bias - 1,
        bias,
    ] {
        positive.push(((exponent + bias) as u64) << fraction);
    }
    for scaled in 256u64..512 {
        let bits = (bias as u64) << fraction | ((scaled - 256) << (fraction - 8));
        positive.push(bits | ((scaled * 37) & ((1 << (fraction - 8)) - 1)));
        if scaled % 16 == 0 {
            positive.push(bits | ((1 << (fraction - 8)) - 1));
        }
    }
    let mut values = Vec::new();
    for value in positive {
        values.extend([value, value | sign]);
    }
    let mut state = 0x1234_5678_9abc_def0u64;
    for _ in 0..64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push(if double { state } else { state as u32 as u64 });
    }
    values
}

fn seed(register: u32) -> [u8; 16] {
    std::array::from_fn(|byte| (register as usize * 37 + byte * 11 + 0x93) as u8)
}

fn vector(form: Form, values: &[u64], start: usize, salt: u32) -> [u8; 16] {
    let mut result = seed(salt);
    for index in 0..form.lanes {
        let value = values[(start + index) % values.len()];
        let pos = index * form.bytes();
        result[pos..pos + form.bytes()].copy_from_slice(&value.to_le_bytes()[..form.bytes()]);
    }
    result
}

fn lane(value: &[u8; 16], form: Form, index: usize) -> u64 {
    let mut bytes = [0; 8];
    bytes[..form.bytes()].copy_from_slice(&value[index * form.bytes()..(index + 1) * form.bytes()]);
    u64::from_le_bytes(bytes)
}

fn put(value: &mut [u8; 16], form: Form, index: usize, bits: u64) {
    let pos = index * form.bytes();
    value[pos..pos + form.bytes()].copy_from_slice(&bits.to_le_bytes()[..form.bytes()]);
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Estimate,
    Step,
    Refine,
}

#[derive(Debug)]
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

fn multiply(n: u64, m: u64, double: bool) -> u64 {
    if double {
        (f64::from_bits(n) * f64::from_bits(m)).to_bits()
    } else {
        (f32::from_bits(n as u32) * f32::from_bits(m as u32)).to_bits() as u64
    }
}

fn check(records: &[Record]) {
    check_backends(
        records,
        &[CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic],
        false,
    );
}

fn check_backends(records: &[Record], backends: &[CpuBackendKind], qc_only: bool) {
    let _ = log::set_logger(&DIAGNOSTICS);
    log::set_max_level(log::LevelFilter::Warn);
    let mut cpus: Vec<_> = backends
        .iter()
        .copied()
        .map(|kind| (kind, Cpu::new(kind).expect("CPU initialization")))
        .collect();
    for records in records.chunks(60) {
        let mut vectors: [[u8; 16]; 32] = std::array::from_fn(|reg| seed(reg as u32));
        let mut initial = vec![0xa5; 0x10000];
        for (reg, value) in vectors.iter().enumerate() {
            initial[reg * 16..reg * 16 + 16].copy_from_slice(value);
        }
        let mut expected = initial.clone();
        let mut program = Vec::new();
        for reg in 0..32u32 {
            program.push(0x3dc0_0060 | (reg << 10) | reg);
        }
        let mut registers = [CANARY; 31];
        registers[1] = DATA + INPUTS as u64;
        registers[2] = DATA + OUTPUTS as u64;
        registers[3] = DATA;
        registers[7] = INITIAL_FLAGS as u64;
        for (slot, record) in records.iter().enumerate() {
            let f = record.form;
            let (rd, rn, rm) = (record.rd as usize, record.rn as usize, record.rm as usize);
            let pos = INPUTS + slot * 32;
            initial[pos..pos + 16].copy_from_slice(&record.n);
            initial[pos + 16..pos + 32].copy_from_slice(&record.m);
            expected[pos..pos + 32].copy_from_slice(&initial[pos..pos + 32]);
            assert_eq!(record.control & 0xffff, 0);
            program.push(0xd2a0_0004 | ((record.control >> 16) << 5));
            program.push(0xd51b_4404);
            program.push(0xd2a0_0007 | ((INITIAL_FLAGS >> 16) << 5));
            program.push(0xd51b_4427);
            registers[4] = record.control as u64;
            program.push(0x3dc0_0060 | (record.rd << 10) | record.rd);
            vectors[rd] = seed(record.rd);
            program.push(0x3dc0_0020 | (((slot * 2) as u32) << 10) | record.rn);
            vectors[rn] = record.n;
            if matches!(record.operation, Operation::Step) && rm != rn {
                program.push(0x3dc0_0020 | (((slot * 2 + 1) as u32) << 10) | record.rm);
                vectors[rm] = record.m;
            }
            let mut result = [0; 16];
            let mut first_refinement = [0; 16];
            let mut second_step = [0; 16];
            let mut zero_comparison = [0; 16];
            let mut flags = INITIAL_FLAGS;
            for index in 0..f.lanes {
                let n = lane(&record.n, f, index);
                let m = lane(if rn == rm { &record.n } else { &record.m }, f, index);
                let (bits, lane_flags) = match record.operation {
                    Operation::Estimate => estimate_oracle(n, f.double, record.control),
                    Operation::Step => step_oracle(n, m, f.double, record.control),
                    Operation::Refine => {
                        assert_eq!(record.control, 0);
                        let y0 = estimate_oracle(n, f.double, 0).0;
                        let s0 = step_oracle(y0, n, f.double, 0).0;
                        let y1 = multiply(y0, s0, f.double);
                        let s1 = step_oracle(y1, n, f.double, 0).0;
                        put(&mut first_refinement, f, index, y1);
                        put(&mut second_step, f, index, s1);
                        let (_, sign, _, _) = format(f.double);
                        put(
                            &mut zero_comparison,
                            f,
                            index,
                            if n & !sign == 0 {
                                if f.double {
                                    u64::MAX
                                } else {
                                    u32::MAX as u64
                                }
                            } else {
                                0
                            },
                        );
                        (multiply(y1, s1, f.double), 0)
                    }
                };
                flags |= lane_flags;
                put(&mut result, f, index, bits);
            }
            match record.operation {
                Operation::Estimate => program.push(f.estimate(record.rd, record.rn)),
                Operation::Step => program.push(f.step(record.rd, record.rn, record.rm)),
                Operation::Refine => {
                    assert_eq!((record.rd, record.rn, record.rm), (16, 5, 17));
                    program.extend_from_slice(&f.chain());
                    vectors[17] = first_refinement;
                    vectors[18] = second_step;
                    vectors[5] = zero_comparison;
                }
            }
            vectors[rd] = result;
            if let Some(wanted) = record.flags {
                assert!(!matches!(record.operation, Operation::Refine));
                assert_eq!(flags, wanted, "independent flags: {record:?}");
                program.push(0xd53b_4429);
                if qc_only {
                    program.push(0x1205_0129);
                }
                let observed_flags = if qc_only {
                    wanted & INITIAL_FLAGS
                } else {
                    wanted
                };
                program.push(0xf900_0049 | (((slot * (STRIDE / 8) + 64) as u32) << 10));
                let pos = OUTPUTS + slot * STRIDE + 512;
                expected[pos..pos + 8].copy_from_slice(&(observed_flags as u64).to_le_bytes());
                registers[9] = observed_flags as u64;
            }
            program.push(0xd53b_4408);
            program.push(0xf900_0048 | (((slot * (STRIDE / 8) + 65) as u32) << 10));
            let pos = OUTPUTS + slot * STRIDE + 520;
            expected[pos..pos + 8].copy_from_slice(&(record.control as u64).to_le_bytes());
            registers[8] = record.control as u64;
            for (reg, value) in vectors.iter().enumerate() {
                program
                    .push(0x3d80_0040 | (((slot * (STRIDE / 16) + reg) as u32) << 10) | reg as u32);
                let pos = OUTPUTS + slot * STRIDE + reg * 16;
                expected[pos..pos + 16].copy_from_slice(value);
            }
        }
        program.push(0xd400_0fc1);
        let mut encoded = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
        for (index, word) in program.iter().enumerate() {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        for (kind, cpu) in &mut cpus {
            let mut code = encoded.clone();
            let mut actual = initial.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, actual.len() as u64, Perm::RW, actual.as_mut_ptr())
                    .unwrap();
            }
            for reg in 0..31 {
                cpu.set_register(reg, CANARY);
            }
            for reg in [1, 2, 3] {
                cpu.set_register(reg, registers[reg as usize]);
            }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let event = cpu
                .run(100_000)
                .unwrap_or_else(|error| {
                    let pc = cpu.get_pc();
                    let offset = pc.checked_sub(CODE).unwrap_or(u64::MAX) as usize;
                    let word = code
                        .get(offset..offset.saturating_add(4))
                        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()));
                    panic!("{kind:?}: reciprocal execution {error:?}, PC={pc:#x}, word={word:?}");
                })
                .event;
            assert!(matches!(event, CpuEvent::Svc(0x7e)), "{kind:?}: {event:?}");
            for (slot, record) in records.iter().enumerate() {
                for reg in 0..32 {
                    let pos = OUTPUTS + slot * STRIDE + reg * 16;
                    assert_eq!(
                        &actual[pos..pos + 16],
                        &expected[pos..pos + 16],
                        "{kind:?}: {record:?} V{reg}"
                    );
                }
                let pos = OUTPUTS + slot * STRIDE + 512;
                assert_eq!(
                    &actual[pos..pos + 32],
                    &expected[pos..pos + 32],
                    "{kind:?}: {record:?} exact FPSR/FPCR/guards"
                );
            }
            for reg in 0..31 {
                assert_eq!(
                    cpu.get_register(reg),
                    registers[reg as usize],
                    "{kind:?}: X{reg}"
                );
            }
            assert_eq!(cpu.get_sp(), CANARY, "{kind:?}: SP");
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, actual.len() as u64).unwrap();
            }
            assert_eq!(actual, expected, "{kind:?}: full memory/source guards");
            assert_eq!(code, encoded, "{kind:?}: unchanged code");
        }
    }
}

#[test]
fn reciprocal_estimate_exact_quantizer_all_formats_signs_and_aliases() {
    assert_eq!(estimate_oracle(0x3f80_0000, false, 0), (0x3f7f_8000, 0));
    assert_eq!(
        estimate_oracle(0x3ff0_0000_0000_0000, true, 0),
        (0x3fef_f000_0000_0000, 0)
    );
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
                    m: seed(0),
                    control: 0,
                    flags: None,
                });
            }
        }
    }
    assert_eq!(FORMS[3].estimate(17, 5), 0x4ea1_d8b1);
    check(&records);
}

fn step_pairs(double: bool) -> Vec<(u64, u64)> {
    let (fraction, sign, infinity, bias) = format(double);
    let one = (bias as u64) << fraction;
    let two = ((bias + 1) as u64) << fraction;
    let quiet = infinity | (1 << (fraction - 1));
    let mut pairs = vec![
        (one, one),
        (one, two),
        (one | sign, two),
        (two, two),
        (one + 1, two - 1),
        (infinity - 1, two),
        (infinity - 1, two | sign),
        (1, ((bias * 2) as u64) << fraction),
        (1, one),
        (0, one),
        (sign, one),
        (infinity, one),
        (sign | infinity, one),
        (one, infinity),
        (quiet | 0x123, quiet | 0x456),
        (sign | quiet | 0x123, quiet | 0x456),
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
    for index in 0..64 {
        pairs.push((
            values[(index * 17) % values.len()],
            values[(index * 29 + 3) % values.len()],
        ));
    }
    pairs
}

#[test]
fn reciprocal_step_fused_literals_nan_priority_specials_and_aliases() {
    assert_eq!(
        step_oracle(0x3f80_0001, 0x3fff_ffff, false, 0).0,
        0xb3ff_fffe
    );
    assert_eq!(
        step_oracle(0x3ff0_0000_0000_0001, 0x3fff_ffff_ffff_ffff, true, 0).0,
        0xbcaf_ffff_ffff_fffe
    );
    assert_eq!(step_oracle(0x3f80_0000, 0x4000_0000, false, 0).0, 0);
    assert_eq!(step_oracle(0x7f80_0000, 0, false, 0), (0x4000_0000, 0));
    assert_eq!(
        step_oracle(0x7ff0_0000_0000_0000, 1 << 63, true, 0),
        (0x4000_0000_0000_0000, 0)
    );
    let mut records = Vec::new();
    for form in FORMS {
        let pairs = step_pairs(form.double);
        let n: Vec<_> = pairs.iter().map(|pair| pair.0).collect();
        let m: Vec<_> = pairs.iter().map(|pair| pair.1).collect();
        for (rn, rm, rd) in [
            (17, 5, 18),
            (17, 5, 17),
            (17, 5, 5),
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
    assert_eq!(FORMS[3].step(18, 17, 5), 0x4e25_fe32);
    check(&records);
}

fn controls(operation: Operation, with_flags: bool) -> Vec<Record> {
    let mut records = Vec::new();
    for form in FORMS {
        let (fraction, sign, infinity, bias) = format(form.double);
        let threshold = 1 << (fraction - 2);
        let one = (bias as u64) << fraction;
        let values = [
            0,
            infinity,
            infinity | 0x123,
            infinity | (1 << (fraction - 1)) | 0x456,
            1,
            threshold - 1,
            threshold,
            threshold + 1,
            (1 << fraction) - 1,
            1 << fraction,
            one,
            ((bias * 2 - 1) as u64) << fraction,
            infinity - 1,
        ];
        for base_control in [0, DN, FZ, DN | FZ] {
            let rounding_modes = if matches!(operation, Operation::Estimate) {
                4
            } else {
                1
            };
            for mode in 0..rounding_modes {
                let control = base_control | (mode << 22);
                for positive in values {
                    for n in [positive, positive | sign] {
                        let m = 0;
                        let flags = match operation {
                            Operation::Estimate => estimate_oracle(n, form.double, control).1,
                            Operation::Step => step_oracle(n, m, form.double, control).1,
                            Operation::Refine => unreachable!(),
                        } | INITIAL_FLAGS;
                        records.push(Record {
                            form,
                            operation,
                            rn: 3,
                            rm: 16,
                            rd: 6,
                            n: vector(form, &[n], 0, 3),
                            m: vector(form, &[m], 0, 16),
                            control,
                            flags: with_flags.then_some(flags),
                        });
                    }
                }
            }
        }
    }
    records
}

#[test]
fn reciprocal_estimate_dn_fz_directed_overflow_values_and_fpcr_match_dynarmic() {
    check(&controls(Operation::Estimate, false));
}

#[test]
fn reciprocal_estimate_rustarmic_architectural_flags_and_reference_qc() {
    let records = controls(Operation::Estimate, true);
    check_backends(
        &records,
        &[CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic],
        true,
    );
    check_backends(&records, &[CpuBackendKind::Rustarmic], false);
}

#[test]
fn reciprocal_step_dn_fz_values_reference_qc_and_rustarmic_architectural_flags() {
    let records = controls(Operation::Step, true);
    check_backends(
        &records,
        &[CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic],
        true,
    );
    check_backends(&records, &[CpuBackendKind::Rustarmic], false);
}

#[test]
fn wonder_exact_two_refinement_sequence_and_register_state_match_dynarmic() {
    assert_eq!(
        FORMS[3].chain(),
        [
            0x4ea1_d8b1,
            0x4e25_fe32,
            0x6e32_de31,
            0x4e25_fe32,
            0x4ea0_d8a5,
            0x6e32_de30
        ]
    );
    let mut records = Vec::new();
    for form in FORMS.into_iter().filter(|form| !form.scalar) {
        let (fraction, sign, _, bias) = format(form.double);
        let mut values = Vec::new();
        for exponent in -16..=16 {
            let positive = ((exponent + bias) as u64) << fraction
                | (0x1234_5678_9abc_defu64 & ((1 << fraction) - 1));
            values.extend([positive, positive | sign]);
        }
        for start in (0..values.len()).step_by(form.lanes) {
            records.push(Record {
                form,
                operation: Operation::Refine,
                rn: 5,
                rm: 17,
                rd: 16,
                n: vector(form, &values, start, 5),
                m: seed(17),
                control: 0,
                flags: None,
            });
        }
    }
    check(&records);
}
