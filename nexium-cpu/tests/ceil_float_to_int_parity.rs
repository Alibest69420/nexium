#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const INPUTS: usize = 0x1000;
const OUTPUTS: usize = 0x4000;
const STRIDE: usize = 768;
const CANARY: u64 = 0xface_1234_5678_9abc;

#[derive(Clone, Copy, Debug)]
enum Rounding {
    NearestEven,
    TiesAway,
    Floor,
    Ceiling,
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Round {
        double: bool,
    },
    Integer {
        double: bool,
        wide: bool,
        unsigned: bool,
        rounding: Rounding,
    },
}

impl Operation {
    fn word(self, rd: u32, rn: u32) -> u32 {
        let base = match self {
            Self::Round { double } => 0x1e24_c000 | ((double as u32) << 22),
            Self::Integer {
                double,
                wide,
                unsigned,
                rounding,
            } => {
                (match rounding {
                    Rounding::NearestEven => 0x1e20_0000,
                    Rounding::TiesAway => 0x1e24_0000,
                    Rounding::Floor => 0x1e30_0000,
                    Rounding::Ceiling => 0x1e28_0000,
                }) | ((double as u32) << 22)
                    | ((wide as u32) << 31)
                    | ((unsigned as u32) << 16)
            }
        };
        base | (rn << 5) | rd
    }
}

fn format(double: bool) -> (u32, u64, u64, i32) {
    if double {
        (52, 1 << 63, 0x7ff0_0000_0000_0000, 1023)
    } else {
        (23, 1 << 31, 0x7f80_0000, 127)
    }
}

fn ceil_bits(input: u64, double: bool) -> u64 {
    let (fraction, sign, infinity, bias) = format(double);
    let mask = (1 << fraction) - 1;
    if input & infinity == infinity {
        return if input & mask != 0 {
            input | (1 << (fraction - 1))
        } else {
            input
        };
    }
    if input & !sign == 0 {
        return input;
    }
    let exponent = ((input & infinity) >> fraction) as i32 - bias;
    if exponent >= fraction as i32 {
        return input;
    }
    if exponent < 0 {
        return if input & sign == 0 {
            (bias as u64) << fraction
        } else {
            sign
        };
    }
    let fractional_bits = fraction - exponent as u32;
    let fractional_mask = (1 << fractional_bits) - 1;
    if input & fractional_mask == 0 {
        return input;
    }
    (input & !fractional_mask)
        + if input & sign == 0 {
            1 << fractional_bits
        } else {
            0
        }
}

fn integer_round(input: u64, double: bool, wide: bool, unsigned: bool, rounding: Rounding) -> u64 {
    let (fraction, sign, infinity, bias) = format(double);
    let mask = (1 << fraction) - 1;
    let width = if wide { 64 } else { 32 };
    let minimum = if unsigned { 0 } else { -(1i128 << (width - 1)) };
    let maximum = if unsigned {
        (1i128 << width) - 1
    } else {
        (1i128 << (width - 1)) - 1
    };
    let encode = |value: i128| {
        if wide {
            value as u64
        } else {
            value as u32 as u64
        }
    };
    let negative = input & sign != 0;
    if input & infinity == infinity {
        return if input & mask != 0 {
            0
        } else {
            encode(if negative { minimum } else { maximum })
        };
    }
    if input & !sign == 0 || (unsigned && negative) {
        return 0;
    }
    let field = ((input & infinity) >> fraction) as i32;
    let exponent = if field == 0 { 1 - bias } else { field - bias };
    if exponent < 0 {
        let magnitude = input & !sign;
        let half = ((bias - 1) as u64) << fraction;
        let increment = match rounding {
            Rounding::Ceiling => !negative,
            Rounding::Floor => negative,
            Rounding::NearestEven => magnitude > half,
            Rounding::TiesAway => magnitude >= half,
        };
        return encode(if negative {
            -i128::from(increment)
        } else {
            i128::from(increment)
        });
    }
    if exponent >= width {
        return encode(if negative { minimum } else { maximum });
    }
    let significand = (input & mask) | if field == 0 { 0 } else { 1 << fraction };
    let shift = exponent - fraction as i32;
    let (integer, remainder, halfway) = if shift >= 0 {
        ((significand as u128) << shift, 0, 0)
    } else {
        let right = (-shift) as u32;
        (
            (significand >> right) as u128,
            significand & ((1 << right) - 1),
            1 << (right - 1),
        )
    };
    let increment = remainder != 0
        && match rounding {
            Rounding::Ceiling => !negative,
            Rounding::Floor => negative,
            Rounding::NearestEven => {
                remainder > halfway || (remainder == halfway && integer & 1 != 0)
            }
            Rounding::TiesAway => remainder >= halfway,
        };
    let magnitude = integer as i128 + i128::from(increment);
    let value = if negative { -magnitude } else { magnitude };
    encode(value.clamp(minimum, maximum))
}

fn seed(register: u32) -> [u8; 16] {
    std::array::from_fn(|byte| (register as usize * 37 + byte * 11 + 0x93) as u8)
}

fn input(bits: u64, double: bool, register: u32) -> [u8; 16] {
    let mut bytes = seed(register);
    let width = if double { 8 } else { 4 };
    bytes[..width].copy_from_slice(&bits.to_le_bytes()[..width]);
    bytes
}

#[derive(Debug)]
struct Record {
    operation: Operation,
    rn: u32,
    rd: u32,
    input: [u8; 16],
}

fn check(records: &[Record]) {
    let mut cpus = [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic]
        .map(|kind| (kind, Cpu::new(kind).expect("CPU initialization")));
    for (kind, cpu) in &mut cpus {
        for records in records.chunks(40) {
            let mut vectors: [[u8; 16]; 32] = std::array::from_fn(|reg| seed(reg as u32));
            let mut initial = vec![0xa5; 0x10000];
            for (reg, value) in vectors.iter().enumerate() {
                initial[reg * 16..reg * 16 + 16].copy_from_slice(value);
            }
            let mut expected = initial.clone();
            let mut registers = [CANARY; 31];
            registers[1] = DATA + INPUTS as u64;
            registers[2] = DATA + OUTPUTS as u64;
            registers[3] = DATA;
            registers[8] = 0;
            let mut program = vec![0xd51b_441f, 0xd51b_443f];
            for reg in 0..32u32 {
                program.push(0x3dc0_0060 | (reg << 10) | reg);
            }
            for (slot, record) in records.iter().enumerate() {
                let pos = INPUTS + slot * 16;
                initial[pos..pos + 16].copy_from_slice(&record.input);
                expected[pos..pos + 16].copy_from_slice(&record.input);
                if matches!(record.operation, Operation::Round { .. }) {
                    program.push(0x3dc0_0060 | (record.rd << 10) | record.rd);
                    vectors[record.rd as usize] = seed(record.rd);
                }
                program.push(0x3dc0_0020 | ((slot as u32) << 10) | record.rn);
                vectors[record.rn as usize] = record.input;
                let source = u64::from_le_bytes(record.input[..8].try_into().unwrap());
                match record.operation {
                    Operation::Round { double } => {
                        let width = if double { 8 } else { 4 };
                        let source = if double { source } else { source as u32 as u64 };
                        let mut result = [0; 16];
                        result[..width]
                            .copy_from_slice(&ceil_bits(source, double).to_le_bytes()[..width]);
                        vectors[record.rd as usize] = result;
                    }
                    Operation::Integer {
                        double,
                        wide,
                        unsigned,
                        rounding,
                    } => {
                        assert!(
                            [0, 9, 10, 16, 22, 31].contains(&record.rd),
                            "do not overwrite fixture bases"
                        );
                        if record.rd != 31 {
                            let source = if double { source } else { source as u32 as u64 };
                            registers[record.rd as usize] =
                                integer_round(source, double, wide, unsigned, rounding);
                        }
                    }
                }
                program.push(record.operation.word(record.rd, record.rn));
                program.push(0xd53b_4408);
                for (reg, value) in vectors.iter().enumerate() {
                    let offset = slot * (STRIDE / 16) + reg;
                    program.push(0x3d80_0040 | ((offset as u32) << 10) | reg as u32);
                    let pos = OUTPUTS + slot * STRIDE + reg * 16;
                    expected[pos..pos + 16].copy_from_slice(value);
                }
                for (reg, value) in registers.iter().enumerate() {
                    let offset = slot * (STRIDE / 8) + 64 + reg;
                    program.push(0xf900_0040 | ((offset as u32) << 10) | reg as u32);
                    let pos = OUTPUTS + slot * STRIDE + 512 + reg * 8;
                    expected[pos..pos + 8].copy_from_slice(&value.to_le_bytes());
                }
            }
            program.push(0xd400_0fc1);
            let mut encoded = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
            for (index, word) in program.iter().enumerate() {
                encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
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
            let mut terminal = None;
            for _ in 0..8 {
                let event = cpu.run(100_000).expect("ceil/conversion execution").event;
                if !matches!(event, CpuEvent::Running) {
                    terminal = Some(event);
                    break;
                }
            }
            assert!(
                matches!(terminal, Some(CpuEvent::Svc(0x7e))),
                "{kind:?}: {terminal:?}, pc={:#x}",
                cpu.get_pc()
            );
            for (slot, record) in records.iter().enumerate() {
                let pos = OUTPUTS + slot * STRIDE;
                let actual_record = &actual[pos..pos + STRIDE];
                let expected_record = &expected[pos..pos + STRIDE];
                if actual_record != expected_record {
                    let first = actual_record
                        .iter()
                        .zip(expected_record)
                        .position(|(a, e)| a != e)
                        .unwrap();
                    panic!("{kind:?}: {record:?}, snapshot byte{first}: actual={:02x}, expected={:02x}", actual_record[first], expected_record[first]);
                }
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
            assert_eq!(actual, expected, "{kind:?}: full source/memory guards");
            assert_eq!(code, encoded, "{kind:?}: unchanged code");
        }
    }
}

fn values(double: bool) -> Vec<u64> {
    let (fraction, sign, infinity, bias) = format(double);
    let one = (bias as u64) << fraction;
    let half = ((bias - 1) as u64) << fraction;
    let one_and_half = one | (1 << (fraction - 1));
    let mut positive = vec![
        0,
        1,
        (1 << fraction) - 1,
        1 << fraction,
        (1 << fraction) + 1,
        half - 1,
        half,
        half + 1,
        one - 1,
        one,
        one + 1,
        one_and_half - 1,
        one_and_half,
        one_and_half + 1,
        (one + (1 << fraction) + (1 << (fraction - 2))) - 1,
        one + (1 << fraction) + (1 << (fraction - 2)),
        (one + (1 << fraction) + (1 << (fraction - 2))) + 1,
        infinity - 1,
        infinity,
        infinity | 0x123,
        infinity | (1 << (fraction - 1)) | 0x456,
    ];
    for exponent in [23, 24, 31, 32, 52, 53, 63, 64] {
        let bits = ((exponent + bias) as u64) << fraction;
        positive.extend([bits - 1, bits, bits + 1]);
    }
    let mut result = Vec::new();
    for bits in positive {
        result.extend([bits, bits | sign]);
    }
    let mut state = 0x5678_1234_9abc_def0u64;
    for _ in 0..32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        result.push(if double { state } else { state as u32 as u64 });
    }
    result
}

#[test]
fn frintp_scalar_ceil_signed_zeros_payloads_precision_boundaries_and_aliases() {
    assert_eq!(ceil_bits(0xbf00_0000, false), 0x8000_0000);
    assert_eq!(ceil_bits(1, false), 0x3f80_0000);
    assert_eq!(ceil_bits(0x8000_0000_0000_0001, true), 1 << 63);
    assert_eq!(ceil_bits(0xff80_0123, false), 0xffc0_0123);
    let mut records = Vec::new();
    for double in [false, true] {
        for (rn, rd) in [(0, 0), (31, 31), (31, 0), (0, 31), (3, 17)] {
            for bits in values(double) {
                records.push(Record {
                    operation: Operation::Round { double },
                    rn,
                    rd,
                    input: input(bits, double, rn),
                });
            }
        }
    }
    check(&records);
}

#[test]
fn gpr_directed_and_nearest_rounding_saturation_widths_and_source_guards() {
    use Rounding::{Ceiling, Floor, NearestEven, TiesAway};
    assert_eq!(integer_round(0xbf00_0000, false, false, false, Ceiling), 0);
    assert_eq!(integer_round(0x3f00_0000, false, false, false, Ceiling), 1);
    assert_eq!(
        integer_round(0xbfc0_0000, false, false, false, Ceiling),
        u32::MAX as u64
    );
    assert_eq!(
        integer_round(0x43e0_0000_0000_0000, true, true, true, Ceiling),
        1 << 63
    );
    assert_eq!(
        integer_round(0x43ef_ffff_ffff_ffff, true, true, true, Ceiling),
        0xffff_ffff_ffff_f800
    );
    assert_eq!(
        integer_round(0x43f0_0000_0000_0000, true, true, true, Ceiling),
        u64::MAX
    );
    assert_eq!(
        integer_round(0x3f00_0000, false, false, false, NearestEven),
        0
    );
    assert_eq!(integer_round(0x3f00_0000, false, false, false, TiesAway), 1);
    assert_eq!(
        integer_round(0x4020_0000, false, false, false, NearestEven),
        2
    );
    assert_eq!(integer_round(0x4020_0000, false, false, false, TiesAway), 3);
    assert_eq!(
        integer_round(0xbf00_0000, false, false, false, Floor),
        u32::MAX as u64
    );
    let mut records = Vec::new();
    for double in [false, true] {
        for wide in [false, true] {
            for unsigned in [false, true] {
                for rounding in [NearestEven, TiesAway, Floor, Ceiling] {
                    for (rn, rd) in [(0, 0), (31, 10), (10, 16), (3, 31), (9, 22)] {
                        for bits in values(double) {
                            records.push(Record {
                                operation: Operation::Integer {
                                    double,
                                    wide,
                                    unsigned,
                                    rounding,
                                },
                                rn,
                                rd,
                                input: input(bits, double, rn),
                            });
                        }
                    }
                }
            }
        }
    }
    check(&records);
}

#[test]
fn wonder_exact_fcvtpu_x9_s0_encoding_and_boundary_values() {
    let operation = Operation::Integer {
        double: false,
        wide: true,
        unsigned: true,
        rounding: Rounding::Ceiling,
    };
    assert_eq!(operation.word(9, 0), 0x9e29_0009);
    assert_eq!(Operation::Round { double: false }.word(0, 0), 0x1e24_c000);
    assert_eq!(Operation::Round { double: true }.word(31, 31), 0x1e64_c3ff);
    let records: Vec<_> = [
        0xbf00_0000,
        0x3f00_0000,
        0x5f7f_ffff,
        0x5f80_0000,
        0xff80_0123,
    ]
    .into_iter()
    .map(|bits| Record {
        operation,
        rn: 0,
        rd: 9,
        input: input(bits, false, 0),
    })
    .collect();
    check(&records);
}
