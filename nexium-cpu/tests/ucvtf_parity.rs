#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const RESULTS: usize = 0x2000;

struct Case {
    value: u64,
    single: u32,
    double: u64,
}

fn case(value: u64, single: u32, double: u64) -> Case {
    Case {
        value,
        single,
        double,
    }
}

fn word_cases() -> Vec<Case> {
    vec![
        case(0, 0, 0),
        case(1, 0x3f80_0000, 0x3ff0_0000_0000_0000),
        case(2, 0x4000_0000, 0x4000_0000_0000_0000),
        case(0x007f_ffff, 0x4aff_fffe, 0x415f_ffff_c000_0000),
        case(0x0080_0000, 0x4b00_0000, 0x4160_0000_0000_0000),
        case(0x0080_0001, 0x4b00_0001, 0x4160_0000_2000_0000),
        case(0x00ff_ffff, 0x4b7f_ffff, 0x416f_ffff_e000_0000),
        case(0x0100_0000, 0x4b80_0000, 0x4170_0000_0000_0000),
        case(0x0100_0001, 0x4b80_0000, 0x4170_0000_1000_0000),
        case(0x0100_0002, 0x4b80_0001, 0x4170_0000_2000_0000),
        case(0x0100_0003, 0x4b80_0002, 0x4170_0000_3000_0000),
        case(0x7fff_ffff, 0x4f00_0000, 0x41df_ffff_ffc0_0000),
        case(0x8000_0000, 0x4f00_0000, 0x41e0_0000_0000_0000),
        case(0x8000_0001, 0x4f00_0000, 0x41e0_0000_0020_0000),
        case(0xffff_ffff, 0x4f80_0000, 0x41ef_ffff_ffe0_0000),
    ]
}

fn doubleword_cases() -> Vec<Case> {
    let mut cases = word_cases();
    cases.extend([
        case(0x001f_ffff_ffff_ffff, 0x5a00_0000, 0x433f_ffff_ffff_ffff),
        case(0x0020_0000_0000_0000, 0x5a00_0000, 0x4340_0000_0000_0000),
        case(0x0020_0000_0000_0001, 0x5a00_0000, 0x4340_0000_0000_0000),
        case(0x0020_0000_0000_0002, 0x5a00_0000, 0x4340_0000_0000_0001),
        case(0x0020_0000_0000_0003, 0x5a00_0000, 0x4340_0000_0000_0002),
        case(0x7fff_ffbf_ffff_ffff, 0x5eff_ffff, 0x43df_ffff_f000_0000),
        case(0x7fff_ffc0_0000_0000, 0x5f00_0000, 0x43df_ffff_f000_0000),
        case(0x7fff_ffc0_0000_0001, 0x5f00_0000, 0x43df_ffff_f000_0000),
        case(0x7fff_ffff_ffff_fdff, 0x5f00_0000, 0x43df_ffff_ffff_ffff),
        case(0x7fff_ffff_ffff_fe00, 0x5f00_0000, 0x43e0_0000_0000_0000),
        case(0x7fff_ffff_ffff_fe01, 0x5f00_0000, 0x43e0_0000_0000_0000),
        case(0x7fff_ffff_ffff_ffff, 0x5f00_0000, 0x43e0_0000_0000_0000),
        case(0x8000_0000_0000_0000, 0x5f00_0000, 0x43e0_0000_0000_0000),
        case(0x8000_0000_0000_0001, 0x5f00_0000, 0x43e0_0000_0000_0000),
        case(0x8000_0000_0000_03ff, 0x5f00_0000, 0x43e0_0000_0000_0000),
        case(0x8000_0000_0000_0400, 0x5f00_0000, 0x43e0_0000_0000_0000),
        case(0x8000_0000_0000_0401, 0x5f00_0000, 0x43e0_0000_0000_0001),
        case(0x8000_007f_ffff_ffff, 0x5f00_0000, 0x43e0_0000_1000_0000),
        case(0x8000_0080_0000_0000, 0x5f00_0000, 0x43e0_0000_1000_0000),
        case(0x8000_0080_0000_0001, 0x5f00_0001, 0x43e0_0000_1000_0000),
        case(0xffff_ffff_ffff_fbff, 0x5f80_0000, 0x43ef_ffff_ffff_ffff),
        case(0xffff_ffff_ffff_fc00, 0x5f80_0000, 0x43f0_0000_0000_0000),
        case(0xffff_ffff_ffff_fc01, 0x5f80_0000, 0x43f0_0000_0000_0000),
        case(u64::MAX, 0x5f80_0000, 0x43f0_0000_0000_0000),
    ]);
    cases
}

fn run(backend: CpuBackendKind, program: &[u32], initial_data: &[u8]) -> Vec<u8> {
    let mut code = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
    for (index, word) in program.iter().enumerate() {
        code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    let mut data = initial_data.to_vec();
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
    let event = cpu.run(100_000).expect("UCVTF execution").event;
    assert!(
        matches!(event, CpuEvent::Svc(0x7e)),
        "{backend:?}: {event:?}"
    );
    assert_eq!(cpu.get_register(1), DATA);
    assert_eq!(cpu.get_register(2), DATA + RESULTS as u64);
    drop(cpu);
    data
}

fn check(wide: bool, double: bool, cases: &[Case]) {
    let mut initial = vec![0xa5; 0x4000];
    for (index, case) in cases.iter().enumerate() {
        let input = case.value | if wide { 0 } else { 0xfeed_cafe_0000_0000 };
        initial[index * 16..index * 16 + 8].copy_from_slice(&input.to_le_bytes());
        initial[index * 16 + 8..index * 16 + 16]
            .copy_from_slice(&0x0123_4567_89ab_cdefu64.to_le_bytes());
    }
    let mut expected = initial.clone();
    let mut program = Vec::new();
    let mut slot = 0;
    for rn in [8, 0, 31] {
        let conversion = 0x1e23_0000
            | if wide { 1 << 31 } else { 0 }
            | if double { 1 << 22 } else { 0 }
            | (rn << 5);
        if !wide && !double && rn == 8 {
            assert_eq!(conversion, 0x1e23_0100, "Wonder UCVTF S0, W8 opcode");
        }
        for (index, case) in cases.iter().enumerate() {
            program.push(0xf940_0020 | (((index * 2) as u32) << 10) | rn);
            program.push(0x3dc0_0020 | ((index as u32) << 10));
            program.push(conversion);
            program.push(0x3d80_0040 | (((slot * 2) as u32) << 10));
            program.push(0xf900_0040 | (((slot * 4 + 2) as u32) << 10) | rn);
            let value = if rn == 31 {
                0
            } else if double {
                case.double
            } else {
                case.single as u64
            };
            let output = RESULTS + slot * 32;
            expected[output..output + 8].copy_from_slice(&value.to_le_bytes());
            expected[output + 8..output + 16].fill(0);
            let source = if rn == 31 {
                0
            } else {
                case.value | if wide { 0 } else { 0xfeed_cafe_0000_0000 }
            };
            expected[output + 16..output + 24].copy_from_slice(&source.to_le_bytes());
            slot += 1;
        }
    }
    program.push(0xd400_0fc1);
    let reference = run(CpuBackendKind::Dynarmic, &program, &initial);
    assert_eq!(
        reference, expected,
        "default-FPCR reference rounding, scalar upper zeros and preserved GPRs"
    );
    let actual = run(CpuBackendKind::Rustarmic, &program, &initial);
    assert_eq!(
        actual, expected,
        "Rustarmic rounding, scalar upper zeros and preserved GPRs"
    );
    assert_eq!(actual, reference, "UCVTF output bytes must match Dynarmic");
}

#[test]
fn ucvtf_word_to_single_matches_dynarmic() {
    check(false, false, &word_cases());
}

#[test]
fn ucvtf_word_to_double_matches_dynarmic() {
    check(false, true, &word_cases());
}

#[test]
fn ucvtf_doubleword_to_single_matches_dynarmic() {
    check(true, false, &doubleword_cases());
}

#[test]
fn ucvtf_doubleword_to_double_matches_dynarmic() {
    check(true, true, &doubleword_cases());
}
