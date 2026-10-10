#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const RESULTS: usize = 0x1000;
const SOURCES: usize = 0x2000;

struct Case {
    bits: u64,
    w: u64,
    x: u64,
}

fn case(bits: u64, w: i32, x: i64) -> Case {
    Case {
        bits,
        w: w as u32 as u64,
        x: x as u64,
    }
}

fn singles() -> Vec<Case> {
    vec![
        case(0x0000_0000, 0, 0),
        case(0x8000_0000, 0, 0),
        case(0x0000_0001, 0, 0),
        case(0x8000_0001, 0, 0),
        case(0.999f32.to_bits() as u64, 0, 0),
        case((-0.999f32).to_bits() as u64, 0, 0),
        case(3.75f32.to_bits() as u64, 3, 3),
        case((-3.75f32).to_bits() as u64, -3, -3),
        case(0x4eff_ffff, 2_147_483_520, 2_147_483_520),
        case(0x4f00_0000, i32::MAX, 2_147_483_648),
        case(0x4f00_0001, i32::MAX, 2_147_483_904),
        case(0xcf00_0000, i32::MIN, -2_147_483_648),
        case(0xcf00_0001, i32::MIN, -2_147_483_904),
        case(0x5eff_ffff, i32::MAX, 0x7fff_ff80_0000_0000),
        case(0x5f00_0000, i32::MAX, i64::MAX),
        case(0x5f00_0001, i32::MAX, i64::MAX),
        case(0xdf00_0000, i32::MIN, i64::MIN),
        case(0xdf00_0001, i32::MIN, i64::MIN),
        case(0x7f7f_ffff, i32::MAX, i64::MAX),
        case(0xff7f_ffff, i32::MIN, i64::MIN),
        case(0x7f80_0000, i32::MAX, i64::MAX),
        case(0xff80_0000, i32::MIN, i64::MIN),
        case(0x7fc1_2345, 0, 0),
        case(0xffc1_2345, 0, 0),
        case(0x7f81_2345, 0, 0),
        case(0xff81_2345, 0, 0),
    ]
}

fn doubles() -> Vec<Case> {
    vec![
        case(0.0f64.to_bits(), 0, 0),
        case((-0.0f64).to_bits(), 0, 0),
        case(1, 0, 0),
        case(0x8000_0000_0000_0001, 0, 0),
        case(0.999f64.to_bits(), 0, 0),
        case((-0.999f64).to_bits(), 0, 0),
        case(3.75f64.to_bits(), 3, 3),
        case((-3.75f64).to_bits(), -3, -3),
        case(2_147_483_647.0f64.to_bits(), i32::MAX, 2_147_483_647),
        case(2_147_483_647.75f64.to_bits(), i32::MAX, 2_147_483_647),
        case(2_147_483_648.0f64.to_bits(), i32::MAX, 2_147_483_648),
        case(2_147_483_649.0f64.to_bits(), i32::MAX, 2_147_483_649),
        case((-2_147_483_648.0f64).to_bits(), i32::MIN, -2_147_483_648),
        case((-2_147_483_648.75f64).to_bits(), i32::MIN, -2_147_483_648),
        case((-2_147_483_649.0f64).to_bits(), i32::MIN, -2_147_483_649),
        case(0x43df_ffff_ffff_ffff, i32::MAX, 0x7fff_ffff_ffff_fc00),
        case(0x43e0_0000_0000_0000, i32::MAX, i64::MAX),
        case(0x43e0_0000_0000_0001, i32::MAX, i64::MAX),
        case(0xc3e0_0000_0000_0000, i32::MIN, i64::MIN),
        case(0xc3e0_0000_0000_0001, i32::MIN, i64::MIN),
        case(f64::MAX.to_bits(), i32::MAX, i64::MAX),
        case((-f64::MAX).to_bits(), i32::MIN, i64::MIN),
        case(f64::INFINITY.to_bits(), i32::MAX, i64::MAX),
        case(f64::NEG_INFINITY.to_bits(), i32::MIN, i64::MIN),
        case(0x7ff8_1234_5678_9abc, 0, 0),
        case(0xfff8_1234_5678_9abc, 0, 0),
        case(0x7ff0_1234_5678_9abc, 0, 0),
        case(0xfff0_1234_5678_9abc, 0, 0),
    ]
}

fn run(backend: CpuBackendKind, double: bool, wide: bool, cases: &[Case]) -> Vec<u8> {
    let mut code = vec![0; 0x1000];
    let mut data = vec![0xa5; 0x4000];
    let mut program = Vec::new();
    let conversion =
        0x1e38_0000 | if double { 1 << 22 } else { 0 } | if wide { 1 << 31 } else { 0 };
    for (index, case) in cases.iter().enumerate() {
        let low = case.bits | if double { 0 } else { 0xfedc_ba98_0000_0000 };
        data[index * 16..index * 16 + 8].copy_from_slice(&low.to_le_bytes());
        data[index * 16 + 8..index * 16 + 16]
            .copy_from_slice(&0x0123_4567_89ab_cdefu64.to_le_bytes());
        program.push(0x3dc0_0020 | ((index as u32) << 10));
        program.push(conversion);
        program.push(0xf900_0040 | (((index * 2) as u32) << 10));
        program.push(0x3d80_0060 | (((index * 2) as u32) << 10));
    }
    program.push(0xd400_0fc1);
    for (index, word) in program.iter().enumerate() {
        code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    let mut cpu = Cpu::new(backend).expect("CPU initialization");
    unsafe {
        cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
            .unwrap();
        cpu.map_host(DATA, data.len() as u64, Perm::RW, data.as_mut_ptr())
            .unwrap();
    }
    cpu.set_pc(CODE);
    cpu.set_register(1, DATA);
    cpu.set_register(2, DATA + RESULTS as u64);
    cpu.set_register(3, DATA + SOURCES as u64);
    let event = cpu.run(10_000).expect("FCVTZS execution").event;
    assert!(
        matches!(event, CpuEvent::Svc(0x7e)),
        "{backend:?}: {event:?}"
    );
    assert_eq!(cpu.get_register(1), DATA);
    assert_eq!(cpu.get_register(2), DATA + RESULTS as u64);
    assert_eq!(cpu.get_register(3), DATA + SOURCES as u64);
    drop(cpu);
    data
}

fn check(double: bool, wide: bool, cases: &[Case]) {
    let reference = run(CpuBackendKind::Dynarmic, double, wide, cases);
    let actual = run(CpuBackendKind::Rustarmic, double, wide, cases);
    for (index, case) in cases.iter().enumerate() {
        let expected = if wide { case.x } else { case.w };
        let start = RESULTS + index * 16;
        let result = |data: &[u8]| u64::from_le_bytes(data[start..start + 8].try_into().unwrap());
        assert_eq!(
            result(&reference),
            expected,
            "reference input={:#018x}",
            case.bits
        );
        assert_eq!(
            result(&actual),
            expected,
            "Rustarmic input={:#018x}",
            case.bits
        );
        assert_eq!(&actual[start + 8..start + 16], &[0xa5; 8], "result canary");
        let source = SOURCES + index * 32;
        assert_eq!(
            &actual[source..source + 16],
            &actual[index * 16..index * 16 + 16],
            "source vector preserved"
        );
        assert_eq!(
            &actual[source + 16..source + 32],
            &[0xa5; 16],
            "source canary"
        );
    }
    assert_eq!(actual, reference, "FCVTZS output bytes must match Dynarmic");
}

#[test]
fn fcvtzs_single_to_w_matches_dynarmic() {
    check(false, false, &singles());
}

#[test]
fn fcvtzs_single_to_x_matches_dynarmic() {
    check(false, true, &singles());
}

#[test]
fn fcvtzs_double_to_w_matches_dynarmic() {
    check(true, false, &doubles());
}

#[test]
fn fcvtzs_double_to_x_matches_dynarmic() {
    check(true, true, &doubles());
}
