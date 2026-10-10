#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const RESULTS: usize = 0x4000;
const SOURCES: usize = 0x8000;

#[derive(Clone, Copy, Debug)]
struct Form {
    double: bool,
    wide: bool,
    unsigned: bool,
}

impl Form {
    fn fraction_bits(self) -> u32 {
        if self.double {
            52
        } else {
            23
        }
    }

    fn exponent_bias(self) -> i32 {
        if self.double {
            1023
        } else {
            127
        }
    }

    fn exponent_max(self) -> u64 {
        if self.double {
            2047
        } else {
            255
        }
    }

    fn sign_bit(self) -> u64 {
        1 << if self.double { 63 } else { 31 }
    }

    fn width(self) -> u32 {
        if self.wide {
            64
        } else {
            32
        }
    }
}

fn expected(bits: u64, form: Form, fbits: u32) -> u64 {
    let fraction_bits = form.fraction_bits();
    let fraction = bits & ((1 << fraction_bits) - 1);
    let exponent = (bits >> fraction_bits) & form.exponent_max();
    let negative = bits & form.sign_bit() != 0;
    let mask = if form.wide { u64::MAX } else { u32::MAX as u64 };
    let limit = if form.unsigned {
        mask as u128
    } else {
        (1u128 << (form.width() - 1)) - u128::from(!negative)
    };

    if exponent == form.exponent_max() && fraction != 0 {
        return 0;
    }
    if form.unsigned && negative {
        return 0;
    }

    let magnitude = if exponent == form.exponent_max() {
        limit
    } else {
        let significand = fraction | if exponent == 0 { 0 } else { 1 << fraction_bits };
        let shift = if exponent == 0 { 1 } else { exponent as i32 }
            - form.exponent_bias()
            - fraction_bits as i32
            + fbits as i32;
        if shift >= 0 {
            let shift = shift as u32;
            if shift >= 128 || significand as u128 > (limit >> shift) {
                limit
            } else {
                (significand as u128) << shift
            }
        } else {
            let shift = (-shift) as u32;
            if shift >= 64 {
                0
            } else {
                (significand >> shift) as u128
            }
            .min(limit)
        }
    };

    if negative {
        0u64.wrapping_sub(magnitude as u64) & mask
    } else {
        magnitude as u64
    }
}

fn power_bits(exponent: i32, form: Form) -> u64 {
    let encoded = exponent + form.exponent_bias();
    assert!(encoded > 0 && encoded < form.exponent_max() as i32);
    (encoded as u64) << form.fraction_bits()
}

fn cases(form: Form, fbits: u32) -> Vec<u64> {
    let fraction_bits = form.fraction_bits();
    let fraction_mask = (1 << fraction_bits) - 1;
    let infinity = form.exponent_max() << fraction_bits;
    let quiet_nan = infinity | (1 << (fraction_bits - 1)) | 0x12345;
    let signaling_nan = infinity | 0x12345;
    let mut positives = vec![
        0,
        1,
        fraction_mask,
        1 << fraction_bits,
        infinity - 1,
        infinity,
        quiet_nan,
        signaling_nan,
        power_bits(-1, form),
        power_bits(-1, form) | (1 << (fraction_bits - 1)),
        power_bits(0, form) | (1 << (fraction_bits - 2)),
        power_bits(0, form) | (3 << (fraction_bits - 2)),
    ];
    for integer_exponent in [
        0,
        1,
        2,
        fraction_bits - 1,
        fraction_bits,
        fraction_bits + 1,
        form.width() - 2,
        form.width() - 1,
        form.width(),
        form.width() + 1,
    ] {
        let boundary = power_bits(integer_exponent as i32 - fbits as i32, form);
        positives.extend([boundary - 1, boundary, boundary + 1]);
    }
    positives.sort_unstable();
    positives.dedup();
    let mut all = Vec::with_capacity(positives.len() * 2);
    for bits in positives {
        all.extend([bits, bits | form.sign_bit()]);
    }
    all
}

fn conversion(form: Form, fbits: u32) -> u32 {
    assert!(fbits <= form.width());
    let base = match (fbits == 0, form.unsigned) {
        (false, false) => 0x1e18_0000,
        (false, true) => 0x1e19_0000,
        (true, false) => 0x1e38_0000,
        (true, true) => 0x1e39_0000,
    };
    base | if form.double { 1 << 22 } else { 0 }
        | if form.wide { 1 << 31 } else { 0 }
        | if fbits != 0 { (64 - fbits) << 10 } else { 0 }
        | (2 << 5)
        | 9
}

fn run(backend: CpuBackendKind, program: &[u32], initial: &[u8]) -> Vec<u8> {
    let mut code = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
    for (index, word) in program.iter().enumerate() {
        code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    let mut data = initial.to_vec();
    let mut cpu = Cpu::new(backend).expect("CPU initialization");
    unsafe {
        cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
            .unwrap();
        cpu.map_host(DATA, data.len() as u64, Perm::RW, data.as_mut_ptr())
            .unwrap();
    }
    cpu.set_pc(CODE);
    cpu.set_register(1, DATA);
    cpu.set_register(3, DATA + RESULTS as u64);
    cpu.set_register(4, DATA + SOURCES as u64);
    cpu.set_register(0, 0x1234_5678_9abc_def0);
    let event = cpu
        .run(100_000)
        .expect("float-to-fixed conversion execution")
        .event;
    assert!(
        matches!(event, CpuEvent::Svc(0x7e)),
        "{backend:?}: {event:?}"
    );
    assert_eq!(cpu.get_register(0), 0x1234_5678_9abc_def0);
    assert_eq!(cpu.get_register(1), DATA);
    assert_eq!(cpu.get_register(3), DATA + RESULTS as u64);
    assert_eq!(cpu.get_register(4), DATA + SOURCES as u64);
    drop(cpu);
    data
}

fn check(form: Form, fbits: u32) {
    let inputs = cases(form, fbits);
    let mut initial = vec![0xa5; 0xc000];
    let mut wanted = initial.clone();
    let mut program = Vec::new();
    for (index, &bits) in inputs.iter().enumerate() {
        let low = bits
            | if form.double {
                0
            } else {
                0xfedc_ba98_0000_0000
            };
        let start = index * 16;
        initial[start..start + 8].copy_from_slice(&low.to_le_bytes());
        initial[start + 8..start + 16].copy_from_slice(&0x0123_4567_89ab_cdefu64.to_le_bytes());
        wanted[start..start + 16].copy_from_slice(&initial[start..start + 16]);
        let result = RESULTS + index * 16;
        wanted[result..result + 8].copy_from_slice(&expected(bits, form, fbits).to_le_bytes());
        let source = SOURCES + index * 32;
        wanted[source..source + 16].copy_from_slice(&initial[start..start + 16]);
        program.push(0x9280_0009);
        program.push(0x3dc0_0022 | ((index as u32) << 10));
        program.push(conversion(form, fbits));
        program.push(0xf900_0069 | (((index * 2) as u32) << 10));
        program.push(0x3d80_0082 | (((index * 2) as u32) << 10));
    }
    assert!(inputs.len() * 16 < RESULTS && SOURCES + inputs.len() * 32 <= initial.len());
    program.push(0xd400_0fc1);
    for backend in [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic] {
        let actual = run(backend, &program, &initial);
        for (index, &bits) in inputs.iter().enumerate() {
            let result = RESULTS + index * 16;
            let value = u64::from_le_bytes(actual[result..result + 8].try_into().unwrap());
            assert_eq!(
                value,
                expected(bits, form, fbits),
                "{backend:?} {form:?} fbits={fbits} input={bits:#018x}"
            );
            assert_eq!(&actual[result + 8..result + 16], &[0xa5; 8], "result guard");
            let source = SOURCES + index * 32;
            assert_eq!(
                &actual[source..source + 16],
                &initial[index * 16..index * 16 + 16],
                "source Q2 preserved"
            );
            assert_eq!(
                &actual[source + 16..source + 32],
                &[0xa5; 16],
                "source guard"
            );
        }
        assert_eq!(
            actual, wanted,
            "{backend:?} {form:?} fbits={fbits}: all output bytes and guards"
        );
    }
}

fn fixed_matrix(double: bool, wide: bool, unsigned: bool) {
    let form = Form {
        double,
        wide,
        unsigned,
    };
    for fbits in [1, 8, 16, 31, 32] {
        check(form, fbits);
    }
    if wide {
        for fbits in [63, 64] {
            check(form, fbits);
        }
    }
}

#[test]
fn integer_oracle_known_results_and_wonder_opcode() {
    let unsigned = Form {
        double: false,
        wide: false,
        unsigned: true,
    };
    let signed = Form {
        unsigned: false,
        ..unsigned
    };
    assert_eq!(
        conversion(unsigned, 8),
        0x1e19_e049,
        "Wonder FCVTZU W9,S2,#8"
    );
    assert_eq!(expected(0x3ba0_0000, unsigned, 8), 1);
    assert_eq!(expected(0x3fa0_0000, unsigned, 8), 320);
    assert_eq!(expected(0xbfa0_0000, unsigned, 8), 0);
    assert_eq!(expected(0xbfa0_0000, signed, 8), (-320i32) as u32 as u64);
    assert_eq!(expected(0x7f80_0000, unsigned, 1), u32::MAX as u64);
    assert_eq!(expected(0xff80_0000, signed, 1), i32::MIN as u32 as u64);
    assert_eq!(expected(0x7fc1_2345, signed, 8), 0);
    assert_eq!(expected(1, unsigned, 32), 0);
    let wide = Form {
        double: true,
        wide: true,
        unsigned: true,
    };
    assert_eq!(expected(0x3fe0_0000_0000_0000, wide, 64), 1 << 63);
    assert_eq!(expected(0x3ff0_0000_0000_0000, wide, 64), u64::MAX);
    assert_eq!(
        expected(
            0xbff0_0000_0000_0000,
            Form {
                unsigned: false,
                ..wide
            },
            63
        ),
        i64::MIN as u64
    );
}

#[test]
fn fixed_fcvtzs_single_to_w_matches_dynarmic() {
    fixed_matrix(false, false, false);
}

#[test]
fn fixed_fcvtzs_single_to_x_matches_dynarmic() {
    fixed_matrix(false, true, false);
}

#[test]
fn fixed_fcvtzs_double_to_w_matches_dynarmic() {
    fixed_matrix(true, false, false);
}

#[test]
fn fixed_fcvtzs_double_to_x_matches_dynarmic() {
    fixed_matrix(true, true, false);
}

#[test]
fn fixed_fcvtzu_single_to_w_matches_dynarmic() {
    fixed_matrix(false, false, true);
}

#[test]
fn fixed_fcvtzu_single_to_x_matches_dynarmic() {
    fixed_matrix(false, true, true);
}

#[test]
fn fixed_fcvtzu_double_to_w_matches_dynarmic() {
    fixed_matrix(true, false, true);
}

#[test]
fn fixed_fcvtzu_double_to_x_matches_dynarmic() {
    fixed_matrix(true, true, true);
}

#[test]
fn ordinary_fcvtzu_single_to_w_matches_dynarmic() {
    check(
        Form {
            double: false,
            wide: false,
            unsigned: true,
        },
        0,
    );
}

#[test]
fn ordinary_fcvtzu_single_to_x_matches_dynarmic() {
    check(
        Form {
            double: false,
            wide: true,
            unsigned: true,
        },
        0,
    );
}

#[test]
fn ordinary_fcvtzu_double_to_w_matches_dynarmic() {
    check(
        Form {
            double: true,
            wide: false,
            unsigned: true,
        },
        0,
    );
}

#[test]
fn ordinary_fcvtzu_double_to_x_matches_dynarmic() {
    check(
        Form {
            double: true,
            wide: true,
            unsigned: true,
        },
        0,
    );
}
