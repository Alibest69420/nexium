#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]
use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;
const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const OUTPUTS: usize = 0x1000;
const STATUS: usize = 0x9000;
const CANARY: u64 = 0xfade_1234_5678_9abc;
const FPCR: u64 = 0;
const FPSR: u64 = 1 << 27;
fn expected(input: u64, double: bool, x: bool, unsigned: bool, fbits: u32) -> u64 {
    let (fraction_bits, exponent_bits, bias) = if double { (52, 11, 1023) } else { (23, 8, 127) };
    let sign = input >> (fraction_bits + exponent_bits) & 1 != 0;
    let exponent_mask = (1u64 << exponent_bits) - 1;
    let exponent = input >> fraction_bits & exponent_mask;
    let fraction = input & ((1u64 << fraction_bits) - 1);
    if exponent == exponent_mask && fraction != 0 {
        return 0;
    }
    if unsigned && sign {
        return 0;
    }
    let width = if x { 64 } else { 32 };
    let limit = if unsigned {
        (1u128 << width) - 1
    } else if sign {
        1u128 << (width - 1)
    } else {
        (1u128 << (width - 1)) - 1
    };
    let magnitude = if exponent == exponent_mask {
        limit
    } else {
        let significand = if exponent == 0 {
            fraction
        } else {
            fraction | (1u64 << fraction_bits)
        };
        let shift = (if exponent == 0 { 1 } else { exponent as i32 }) - bias - fraction_bits as i32
            + fbits as i32;
        if shift >= 0 {
            if significand == 0 {
                0
            } else if shift >= 128 || significand as u128 > limit >> shift {
                limit
            } else {
                (significand as u128) << shift
            }
        } else if shift <= -64 {
            0
        } else {
            ((significand >> -shift) as u128).min(limit)
        }
    } as u64;
    let value = if sign {
        0u64.wrapping_sub(magnitude)
    } else {
        magnitude
    };
    if x {
        value
    } else {
        value as u32 as u64
    }
}

fn inputs(double: bool, x: bool, unsigned: bool, fbits: u32) -> Vec<u64> {
    let mut values = if double {
        vec![
            0,
            0x8000_0000_0000_0000,
            1,
            0x8000_0000_0000_0001,
            0x000f_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            0x3fe0_0000_0000_0000,
            0xbfe0_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0xbff0_0000_0000_0000,
            0x400e_0000_0000_0000,
            0xc00e_0000_0000_0000,
            0x7fef_ffff_ffff_ffff,
            0xffef_ffff_ffff_ffff,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_1234_5678_9abc,
            0xfff8_1234_5678_9abc,
            0x7ff0_1234_5678_9abc,
            0xfff0_1234_5678_9abc,
        ]
    } else {
        vec![
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3f00_0000,
            0xbf00_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x4070_0000,
            0xc070_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xffc1_2345,
            0x7f81_2345,
            0xff81_2345,
        ]
    };
    let width = if x { 64 } else { 32 };
    let fraction_bits = if double { 52 } else { 23 };
    let bias = if double { 1023 } else { 127 };
    let sign = 1u64 << if double { 63 } else { 31 };
    for power in [width - u32::from(!unsigned), 63, 0] {
        let exponent = bias + power as i32 - fbits as i32;
        let boundary = (exponent as u64) << fraction_bits;
        for value in [boundary - 1, boundary, boundary + 1] {
            values.extend([value, value | sign]);
        }
    }
    let mut random = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..128 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        values.push(if double { random } else { random as u32 as u64 });
    }
    values
}

#[derive(Clone, Copy, Debug)]
struct Form {
    double: bool,
    scalar: bool,
    q: bool,
    unsigned: bool,
    fbits: u32,
    rn: u32,
    rd: u32,
}
impl Form {
    fn word(self) -> u32 {
        let width = if self.double { 64 } else { 32 };
        (if self.scalar {
            0x5f00_fc00
        } else {
            0x0f00_fc00
        }) | (u32::from(self.unsigned) << 29)
            | (u32::from(self.q && !self.scalar) << 30)
            | ((2 * width - self.fbits) << 16)
            | (self.rn << 5)
            | self.rd
    }
    fn wanted(self, source: [u64; 2]) -> [u64; 2] {
        let lanes = if self.scalar {
            1
        } else if self.double || !self.q {
            2
        } else {
            4
        };
        let mut result = [0u64; 2];
        for lane in 0..lanes {
            let bits = if self.double {
                source[lane]
            } else {
                (source[lane / 2] >> (lane % 2 * 32)) & 0xffff_ffff
            };
            let integer = expected(bits, self.double, self.double, self.unsigned, self.fbits);
            if self.double {
                result[lane] = integer
            } else {
                result[lane / 2] |= integer << (lane % 2 * 32)
            }
        }
        result
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    form: Form,
    input: u64,
}
impl Case {
    fn source(self) -> [u64; 2] {
        let input = self.input;
        if self.form.double {
            [input, input ^ (1 << 63)]
        } else {
            [
                input | ((input ^ (1 << 31)) << 32),
                ((input ^ 0x0040_0000) & 0xffff_ffff)
                    | (((input ^ 0x8040_0000) & 0xffff_ffff) << 32),
            ]
        }
    }
}
fn seed(register: usize) -> [u8; 16] {
    std::array::from_fn(|byte| (register * 37 + byte * 11 + 0x93) as u8)
}
fn check(cases: &[Case]) {
    let mut backends: Vec<_> = [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic]
        .into_iter()
        .map(|kind| (kind, Cpu::new(kind).expect("CPU initialization")))
        .collect();
    for cases in cases.chunks(60) {
        let mut vectors: [[u8; 16]; 32] = std::array::from_fn(seed);
        let mut initial = vec![0xa5; 0xa000];
        for (register, value) in vectors.iter().enumerate() {
            initial[register * 16..register * 16 + 16].copy_from_slice(value);
        }
        let mut expected = initial.clone();
        let mut program = vec![0xd51b_4404, 0xd51b_4427];
        for register in 0..32u32 {
            program.push(0x3dc0_01a0 | (register << 10) | register);
        }
        for (slot, &case) in cases.iter().enumerate() {
            for register in [case.form.rd] {
                program.push(0x3dc0_01a0 | (register << 10) | register);
                vectors[register as usize] = seed(register as usize);
            }
            let source = case.source();
            let source_offset = 0x400 + slot * 16;
            initial[source_offset..source_offset + 8].copy_from_slice(&source[0].to_le_bytes());
            initial[source_offset + 8..source_offset + 16]
                .copy_from_slice(&source[1].to_le_bytes());
            expected[source_offset..source_offset + 16]
                .copy_from_slice(&initial[source_offset..source_offset + 16]);
            program.push(0x3dc0_01a0 | (((source_offset / 16) as u32) << 10) | case.form.rn);
            vectors[case.form.rn as usize] =
                std::array::from_fn(|byte| initial[source_offset + byte]);
            program.push(case.form.word());
            let result = case.form.wanted(source);
            let result_bytes: [u8; 16] =
                std::array::from_fn(|byte| result[byte / 8].to_le_bytes()[byte % 8]);
            vectors[case.form.rd as usize] = result_bytes;
            for (register, value) in vectors.iter().enumerate() {
                let offset = slot * 32 + register;
                program.push(0x3d80_01c0 | ((offset as u32) << 10) | register as u32);
                expected[OUTPUTS + offset * 16..OUTPUTS + offset * 16 + 16].copy_from_slice(value);
            }
            program.push(0xd53b_4412);

            program.push(0xf900_01f2 | ((slot as u32 * 2) << 10));
            let status = STATUS + slot * 16;
            expected[status..status + 8].copy_from_slice(&FPCR.to_le_bytes());
        }
        program.push(0xd400_0fc1);
        let mut encoded = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
        for (index, word) in program.iter().enumerate() {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut expected_gprs = [CANARY; 31];
        expected_gprs[4] = FPCR;
        expected_gprs[7] = FPSR;
        expected_gprs[13] = DATA;
        expected_gprs[14] = DATA + OUTPUTS as u64;
        expected_gprs[15] = DATA + STATUS as u64;

        expected_gprs[18] = FPCR;
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
            for register in [4, 7, 13, 14, 15] {
                cpu.set_register(register, expected_gprs[register as usize]);
            }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let event = cpu
                .run(100_000)
                .expect("fixed SIMD conversion execution")
                .event;
            assert!(
                matches!(event, CpuEvent::Svc(0x7e)),
                "{backend:?}: {event:?}"
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
                let status = STATUS + slot * 16;
                assert_eq!(
                    &actual[status..status + 16],
                    &expected[status..status + 16],
                    "{backend:?}: {case:?} FPCR unchanged/status guard"
                );
            }
            for register in 0..31 {
                assert_eq!(
                    cpu.get_register(register),
                    expected_gprs[register as usize],
                    "{backend:?}: X{register}"
                );
            }
            assert_eq!(cpu.get_sp(), CANARY);
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, actual.len() as u64).unwrap();
            }
            assert_eq!(actual, expected, "{backend:?}: all source/data guards");
            assert_eq!(code, encoded, "{backend:?}: code unchanged");
        }
    }
}

fn matrix(unsigned: bool) {
    let mut cases = Vec::new();
    for (double, scalar, q) in [
        (false, true, false),
        (true, true, false),
        (false, false, false),
        (false, false, true),
        (true, false, true),
    ] {
        for fbits in 1..=if double { 64 } else { 32 } {
            for (rn, rd) in [(2, 1), (31, 31), (0, 31)] {
                let form = Form {
                    double,
                    scalar,
                    q,
                    unsigned,
                    fbits,
                    rn,
                    rd,
                };
                for input in inputs(double, double, unsigned, fbits).into_iter().take(42) {
                    cases.push(Case { form, input });
                }
            }
        }
    }
    check(&cases);
}
#[test]
fn signed_every_format_immediate_and_alias_matches_reference_integer_oracle() {
    matrix(false)
}
#[test]
fn unsigned_every_format_immediate_and_alias_matches_reference_integer_oracle() {
    matrix(true)
}
#[test]
fn exact_wonder_fixed_vector_double_matches_reference() {
    let form = Form {
        double: true,
        scalar: false,
        q: true,
        unsigned: false,
        fbits: 4,
        rn: 1,
        rd: 1,
    };
    assert_eq!(form.word(), 0x4f7c_fc21);
    check(
        &inputs(true, true, false, 4)
            .into_iter()
            .map(|input| Case { form, input })
            .collect::<Vec<_>>(),
    );
}
