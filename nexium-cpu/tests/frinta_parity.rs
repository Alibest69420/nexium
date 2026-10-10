#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const RESULTS: usize = 0x1000;
const SOURCES: usize = 0x2000;

fn singles() -> Vec<(u64, u64)> {
    vec![
        (0x0000_0000, 0x0000_0000),
        (0x8000_0000, 0x8000_0000),
        (0x0000_0001, 0x0000_0000),
        (0x8000_0001, 0x8000_0000),
        (0x3eff_ffff, 0x0000_0000),
        (0x3f00_0000, 0x3f80_0000),
        (0x3f00_0001, 0x3f80_0000),
        (0xbeff_ffff, 0x8000_0000),
        (0xbf00_0000, 0xbf80_0000),
        (0xbf00_0001, 0xbf80_0000),
        (0x3fbf_ffff, 0x3f80_0000),
        (0x3fc0_0000, 0x4000_0000),
        (0x3fc0_0001, 0x4000_0000),
        (0xbfbf_ffff, 0xbf80_0000),
        (0xbfc0_0000, 0xc000_0000),
        (0xbfc0_0001, 0xc000_0000),
        (0x4020_0000, 0x4040_0000),
        (0xc020_0000, 0xc040_0000),
        (0x4a80_0001, 0x4a80_0002),
        (0x4aff_ffff, 0x4b00_0000),
        (0xcaff_ffff, 0xcb00_0000),
        (0x4b00_0000, 0x4b00_0000),
        (0x7f7f_ffff, 0x7f7f_ffff),
        (0xff7f_ffff, 0xff7f_ffff),
        (0x7f80_0000, 0x7f80_0000),
        (0xff80_0000, 0xff80_0000),
        (0x7fc1_2345, 0x7fc1_2345),
        (0xffc1_2345, 0xffc1_2345),
    ]
}

fn doubles() -> Vec<(u64, u64)> {
    vec![
        (0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
        (0x8000_0000_0000_0000, 0x8000_0000_0000_0000),
        (0x0000_0000_0000_0001, 0x0000_0000_0000_0000),
        (0x8000_0000_0000_0001, 0x8000_0000_0000_0000),
        (0x3fdf_ffff_ffff_ffff, 0x0000_0000_0000_0000),
        (0x3fe0_0000_0000_0000, 0x3ff0_0000_0000_0000),
        (0x3fe0_0000_0000_0001, 0x3ff0_0000_0000_0000),
        (0xbfdf_ffff_ffff_ffff, 0x8000_0000_0000_0000),
        (0xbfe0_0000_0000_0000, 0xbff0_0000_0000_0000),
        (0xbfe0_0000_0000_0001, 0xbff0_0000_0000_0000),
        (0x3ff7_ffff_ffff_ffff, 0x3ff0_0000_0000_0000),
        (0x3ff8_0000_0000_0000, 0x4000_0000_0000_0000),
        (0x3ff8_0000_0000_0001, 0x4000_0000_0000_0000),
        (0xbff7_ffff_ffff_ffff, 0xbff0_0000_0000_0000),
        (0xbff8_0000_0000_0000, 0xc000_0000_0000_0000),
        (0xbff8_0000_0000_0001, 0xc000_0000_0000_0000),
        (0x4004_0000_0000_0000, 0x4008_0000_0000_0000),
        (0xc004_0000_0000_0000, 0xc008_0000_0000_0000),
        (0x432f_ffff_ffff_ffff, 0x4330_0000_0000_0000),
        (0xc32f_ffff_ffff_ffff, 0xc330_0000_0000_0000),
        (0x4330_0000_0000_0000, 0x4330_0000_0000_0000),
        (0x7fef_ffff_ffff_ffff, 0x7fef_ffff_ffff_ffff),
        (0xffef_ffff_ffff_ffff, 0xffef_ffff_ffff_ffff),
        (0x7ff0_0000_0000_0000, 0x7ff0_0000_0000_0000),
        (0xfff0_0000_0000_0000, 0xfff0_0000_0000_0000),
        (0x7ff8_1234_5678_9abc, 0x7ff8_1234_5678_9abc),
        (0xfff8_1234_5678_9abc, 0xfff8_1234_5678_9abc),
    ]
}

fn run(backend: CpuBackendKind, double: bool, alias: bool, cases: &[(u64, u64)]) -> Vec<u8> {
    let mut code = vec![0; 0x1000];
    let mut data = vec![0xa5; 0x4000];
    let mut program = Vec::new();
    let rn = u32::from(!alias);
    let conversion = 0x1e26_4000 | if double { 1 << 22 } else { 0 } | (rn << 5);
    if double && alias {
        assert_eq!(conversion, 0x1e66_4000, "Wonder FRINTA D0, D0 opcode");
    }
    for (index, &(input, _)) in cases.iter().enumerate() {
        let low = input | if double { 0 } else { 0xfedc_ba98_0000_0000 };
        data[index * 16..index * 16 + 8].copy_from_slice(&low.to_le_bytes());
        data[index * 16 + 8..index * 16 + 16]
            .copy_from_slice(&0x0123_4567_89ab_cdefu64.to_le_bytes());
        program.push(0x3dc0_0020 | ((index as u32) << 10) | rn);
        program.push(conversion);
        program.push(0x3d80_0040 | (((index * 2) as u32) << 10));
        program.push(0x3d80_0060 | (((index * 2) as u32) << 10) | rn);
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
    let event = cpu.run(10_000).expect("FRINTA execution").event;
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

fn check(double: bool, cases: &[(u64, u64)]) {
    for alias in [false, true] {
        let reference = run(CpuBackendKind::Dynarmic, double, alias, cases);
        let actual = run(CpuBackendKind::Rustarmic, double, alias, cases);
        for (index, &(input, expected)) in cases.iter().enumerate() {
            let start = RESULTS + index * 32;
            let mut expected_vector = [0; 16];
            expected_vector[..8].copy_from_slice(&expected.to_le_bytes());
            assert_eq!(
                &reference[start..start + 16],
                &expected_vector,
                "reference input={input:#018x}"
            );
            assert_eq!(
                &actual[start..start + 16],
                &expected_vector,
                "Rustarmic input={input:#018x}"
            );
            assert_eq!(
                &actual[start + 16..start + 32],
                &[0xa5; 16],
                "result canary"
            );
            let source = SOURCES + index * 32;
            let source_bytes = if alias {
                &actual[start..start + 16]
            } else {
                &actual[index * 16..index * 16 + 16]
            };
            assert_eq!(&actual[source..source + 16], source_bytes, "source vector");
            assert_eq!(
                &actual[source + 16..source + 32],
                &[0xa5; 16],
                "source canary"
            );
        }
        assert_eq!(
            actual, reference,
            "FRINTA result and preserved source bytes"
        );
    }
}

#[test]
fn frinta_single_matches_dynarmic_bit_for_bit() {
    check(false, &singles());
}

#[test]
fn frinta_double_matches_dynarmic_bit_for_bit() {
    check(true, &doubles());
}
