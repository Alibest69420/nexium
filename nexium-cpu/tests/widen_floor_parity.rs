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
enum Operation {
    Widen { upper: bool },
    Round { double: bool },
    Integer { double: bool, wide: bool },
}

impl Operation {
    fn word(self, rd: u32, rn: u32) -> u32 {
        let base = match self {
            Self::Widen { upper } => 0x0e61_7800 | ((upper as u32) << 30),
            Self::Round { double } => 0x1e25_4000 | ((double as u32) << 22),
            Self::Integer { double, wide } => {
                0x1e30_0000 | ((double as u32) << 22) | ((wide as u32) << 31)
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

fn floor_bits(input: u64, double: bool) -> u64 {
    let (fraction, sign, infinity, bias) = format(double);
    let mask = (1 << fraction) - 1;
    let quiet = 1 << (fraction - 1);
    if input & infinity == infinity {
        return if input & mask != 0 {
            input | quiet
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
            0
        } else {
            sign | ((bias as u64) << fraction)
        };
    }
    let fractional_bits = fraction - exponent as u32;
    let fractional_mask = (1 << fractional_bits) - 1;
    if input & fractional_mask == 0 {
        return input;
    }
    (input & !fractional_mask)
        + if input & sign != 0 {
            1 << fractional_bits
        } else {
            0
        }
}

fn integer_floor(input: u64, double: bool, wide: bool) -> u64 {
    let (fraction, sign, infinity, bias) = format(double);
    let mask = (1 << fraction) - 1;
    let width = if wide { 64 } else { 32 };
    let minimum = -(1i128 << (width - 1));
    let maximum = (1i128 << (width - 1)) - 1;
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
    if input & !sign == 0 {
        return 0;
    }
    let field = ((input & infinity) >> fraction) as i32;
    let exponent = if field == 0 { 1 - bias } else { field - bias };
    if exponent < 0 {
        return encode(if negative { -1 } else { 0 });
    }
    if exponent > width - 1 {
        return encode(if negative { minimum } else { maximum });
    }
    let significand = (input & mask) | if field == 0 { 0 } else { 1 << fraction };
    let shift = exponent - fraction as i32;
    let (integer, remainder) = if shift >= 0 {
        ((significand as u128) << shift, false)
    } else {
        let right = (-shift) as u32;
        (
            (significand >> right) as u128,
            significand & ((1 << right) - 1) != 0,
        )
    };
    let value = if negative {
        -(integer as i128) - i128::from(remainder)
    } else {
        integer as i128
    };
    encode(value.clamp(minimum, maximum))
}

fn widen(input: u32) -> u64 {
    if input & 0x7f80_0000 == 0x7f80_0000 && input & 0x007f_ffff != 0 {
        ((input as u64 & 0x8000_0000) << 32)
            | 0x7ff0_0000_0000_0000
            | ((input as u64 & 0x007f_ffff) << 29)
            | (1 << 51)
    } else {
        (f32::from_bits(input) as f64).to_bits()
    }
}

fn seed(register: u32) -> [u8; 16] {
    std::array::from_fn(|byte| (register as usize * 37 + byte * 11 + 0x93) as u8)
}

fn read(vector: &[u8; 16], bytes: usize, index: usize) -> u64 {
    let mut bits = [0; 8];
    bits[..bytes].copy_from_slice(&vector[index * bytes..(index + 1) * bytes]);
    u64::from_le_bytes(bits)
}

fn put(vector: &mut [u8; 16], bytes: usize, index: usize, value: u64) {
    vector[index * bytes..(index + 1) * bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
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
            if !matches!(record.operation, Operation::Integer { .. }) {
                program.push(0x3dc0_0060 | (record.rd << 10) | record.rd);
                vectors[record.rd as usize] = seed(record.rd);
            }
            program.push(0x3dc0_0020 | ((slot as u32) << 10) | record.rn);
            vectors[record.rn as usize] = record.input;
            let source = vectors[record.rn as usize];
            let mut result = [0; 16];
            match record.operation {
                Operation::Widen { upper } => {
                    for index in 0..2 {
                        put(
                            &mut result,
                            8,
                            index,
                            widen(read(&source, 4, index + if upper { 2 } else { 0 }) as u32),
                        );
                    }
                    vectors[record.rd as usize] = result;
                }
                Operation::Round { double } => {
                    let bytes = if double { 8 } else { 4 };
                    put(
                        &mut result,
                        bytes,
                        0,
                        floor_bits(read(&source, bytes, 0), double),
                    );
                    vectors[record.rd as usize] = result;
                }
                Operation::Integer { double, wide } => {
                    assert!(
                        [0, 10, 16, 22, 31].contains(&record.rd),
                        "do not overwrite fixture bases"
                    );
                    if record.rd != 31 {
                        registers[record.rd as usize] = integer_floor(
                            read(&source, if double { 8 } else { 4 }, 0),
                            double,
                            wide,
                        );
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
            let event = cpu.run(100_000).expect("widen/floor execution").event;
            assert!(matches!(event, CpuEvent::Svc(0x7e)), "{kind:?}: {event:?}");
            for (slot, record) in records.iter().enumerate() {
                let pos = OUTPUTS + slot * STRIDE;
                assert_eq!(
                    &actual[pos..pos + STRIDE],
                    &expected[pos..pos + STRIDE],
                    "{kind:?}: {record:?} all V/GPR/guards"
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
        infinity - 1,
        infinity,
        infinity | 0x123,
        infinity | (1 << (fraction - 1)) | 0x456,
    ];
    for exponent in [23, 24, 31, 32, 52, 53, 63, 64] {
        let bits = ((exponent + bias) as u64) << fraction;
        positive.extend([bits - 1, bits, bits + 1]);
    }
    let mut values = Vec::new();
    for bits in positive {
        values.extend([bits, bits | sign]);
    }
    let mut state = 0x1234_5678_9abc_def0u64;
    for _ in 0..32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push(if double { state } else { state as u32 as u64 });
    }
    values
}

#[test]
fn fcvtl_first_second_half_exact_widening_nan_payloads_and_source_aliases() {
    assert_eq!(widen(1), 0x36a0_0000_0000_0000);
    assert_eq!(widen(0xff80_0123), 0xfff8_0024_6000_0000);
    let values = values(false);
    let mut records = Vec::new();
    for upper in [false, true] {
        for (rn, rd) in [(1, 1), (0, 0), (31, 0), (0, 31), (5, 17)] {
            for start in (0..values.len()).step_by(2) {
                let mut input = seed(rn);
                for index in 0..4 {
                    put(&mut input, 4, index, values[(start + index) % values.len()]);
                }
                records.push(Record {
                    operation: Operation::Widen { upper },
                    rn,
                    rd,
                    input,
                });
            }
        }
    }
    check(&records);
}

#[test]
fn frintm_scalar_floor_bits_signed_zeros_boundaries_nans_and_aliases() {
    assert_eq!(floor_bits(0xbf00_0000, false), 0xbf80_0000);
    assert_eq!(floor_bits(0x8000_0000, false), 0x8000_0000);
    assert_eq!(
        floor_bits(0x8000_0000_0000_0001, true),
        0xbff0_0000_0000_0000
    );
    let mut records = Vec::new();
    for double in [false, true] {
        for (rn, rd) in [(0, 0), (31, 31), (31, 0), (0, 31), (3, 7)] {
            for bits in values(double) {
                let mut input = seed(rn);
                put(&mut input, if double { 8 } else { 4 }, 0, bits);
                records.push(Record {
                    operation: Operation::Round { double },
                    rn,
                    rd,
                    input,
                });
            }
        }
    }
    check(&records);
}

#[test]
fn fcvtms_scalar_floor_saturation_w_zero_extension_xzr_and_source_guards() {
    assert_eq!(integer_floor(0xbf00_0000, false, false), 0xffff_ffff);
    assert_eq!(integer_floor(0xc3e0_0000_0000_0000, true, true), 1 << 63);
    assert_eq!(
        integer_floor(0x43df_ffff_ffff_ffff, true, true),
        0x7fff_ffff_ffff_fc00
    );
    let mut records = Vec::new();
    for double in [false, true] {
        for wide in [false, true] {
            for (rn, rd) in [(0, 0), (31, 10), (10, 16), (3, 31), (0, 22)] {
                for bits in values(double) {
                    let mut input = seed(rn);
                    put(&mut input, if double { 8 } else { 4 }, 0, bits);
                    records.push(Record {
                        operation: Operation::Integer { double, wide },
                        rn,
                        rd,
                        input,
                    });
                }
            }
        }
    }
    check(&records);
}

#[test]
fn nvn_exact_widen_round_and_integer_floor_encodings() {
    assert_eq!(Operation::Widen { upper: false }.word(1, 1), 0x0e61_7821);
    assert_eq!(Operation::Widen { upper: false }.word(0, 0), 0x0e61_7800);
    assert_eq!(Operation::Round { double: true }.word(0, 0), 0x1e65_4000);
    assert_eq!(
        Operation::Integer {
            double: true,
            wide: true
        }
        .word(22, 0),
        0x9e70_0016
    );
    check(&[
        Record {
            operation: Operation::Widen { upper: false },
            rn: 1,
            rd: 1,
            input: [
                0, 0, 0x80, 0x3f, 0, 0, 0xc0, 0xbf, 0xff, 0xff, 0xff, 0x7f, 1, 0, 0, 0,
            ],
        },
        Record {
            operation: Operation::Round { double: true },
            rn: 0,
            rd: 0,
            input: (-1.5f64)
                .to_bits()
                .to_le_bytes()
                .into_iter()
                .chain([0x93; 8])
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        },
        Record {
            operation: Operation::Integer {
                double: true,
                wide: true,
            },
            rn: 0,
            rd: 22,
            input: (-1.5f64)
                .to_bits()
                .to_le_bytes()
                .into_iter()
                .chain([0x93; 8])
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        },
    ]);
}
