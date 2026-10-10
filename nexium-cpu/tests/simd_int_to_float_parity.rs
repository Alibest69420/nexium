#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const RESULTS: usize = 0x4000;

fn unsigned_words() -> Vec<(u64, u64)> {
    vec![
        (0, 0),
        (1, 0x3f80_0000),
        (2, 0x4000_0000),
        (0x007f_ffff, 0x4aff_fffe),
        (0x0080_0000, 0x4b00_0000),
        (0x0080_0001, 0x4b00_0001),
        (0x00ff_ffff, 0x4b7f_ffff),
        (0x0100_0000, 0x4b80_0000),
        (0x0100_0001, 0x4b80_0000),
        (0x0100_0002, 0x4b80_0001),
        (0x0100_0003, 0x4b80_0002),
        (0x7fff_ffff, 0x4f00_0000),
        (0x8000_0000, 0x4f00_0000),
        (0x8000_0001, 0x4f00_0000),
        (0xffff_ffff, 0x4f80_0000),
    ]
}

fn signed_words() -> Vec<(u64, u64)> {
    vec![
        (0, 0),
        (1, 0x3f80_0000),
        (2, 0x4000_0000),
        (0xffff_ffff, 0xbf80_0000),
        (0xffff_fffe, 0xc000_0000),
        (0x00ff_ffff, 0x4b7f_ffff),
        (0x0100_0000, 0x4b80_0000),
        (0x0100_0001, 0x4b80_0000),
        (0x0100_0002, 0x4b80_0001),
        (0x0100_0003, 0x4b80_0002),
        (0xff00_0001, 0xcb7f_ffff),
        (0xff00_0000, 0xcb80_0000),
        (0xfeff_ffff, 0xcb80_0000),
        (0xfeff_fffe, 0xcb80_0001),
        (0xfeff_fffd, 0xcb80_0002),
        (0x7fff_ffff, 0x4f00_0000),
        (0x8000_0000, 0xcf00_0000),
        (0x8000_0001, 0xcf00_0000),
        (0x8000_0080, 0xceff_ffff),
    ]
}

fn unsigned_doublewords() -> Vec<(u64, u64)> {
    vec![
        (0, 0),
        (1, 0x3ff0_0000_0000_0000),
        (2, 0x4000_0000_0000_0000),
        (0xffff_ffff, 0x41ef_ffff_ffe0_0000),
        (0x001f_ffff_ffff_ffff, 0x433f_ffff_ffff_ffff),
        (0x0020_0000_0000_0000, 0x4340_0000_0000_0000),
        (0x0020_0000_0000_0001, 0x4340_0000_0000_0000),
        (0x0020_0000_0000_0002, 0x4340_0000_0000_0001),
        (0x0020_0000_0000_0003, 0x4340_0000_0000_0002),
        (0x7fff_ffff_ffff_fdff, 0x43df_ffff_ffff_ffff),
        (0x7fff_ffff_ffff_fe00, 0x43e0_0000_0000_0000),
        (0x7fff_ffff_ffff_ffff, 0x43e0_0000_0000_0000),
        (0x8000_0000_0000_0000, 0x43e0_0000_0000_0000),
        (0x8000_0000_0000_0001, 0x43e0_0000_0000_0000),
        (0x8000_0000_0000_03ff, 0x43e0_0000_0000_0000),
        (0x8000_0000_0000_0400, 0x43e0_0000_0000_0000),
        (0x8000_0000_0000_0401, 0x43e0_0000_0000_0001),
        (0xffff_ffff_ffff_fbff, 0x43ef_ffff_ffff_ffff),
        (0xffff_ffff_ffff_fc00, 0x43f0_0000_0000_0000),
        (0xffff_ffff_ffff_fc01, 0x43f0_0000_0000_0000),
        (0xffff_ffff_ffff_ffff, 0x43f0_0000_0000_0000),
    ]
}

fn signed_doublewords() -> Vec<(u64, u64)> {
    vec![
        (0, 0),
        (1, 0x3ff0_0000_0000_0000),
        (2, 0x4000_0000_0000_0000),
        (0xffff_ffff_ffff_ffff, 0xbff0_0000_0000_0000),
        (0xffff_ffff_ffff_fffe, 0xc000_0000_0000_0000),
        (0x001f_ffff_ffff_ffff, 0x433f_ffff_ffff_ffff),
        (0x0020_0000_0000_0000, 0x4340_0000_0000_0000),
        (0x0020_0000_0000_0001, 0x4340_0000_0000_0000),
        (0x0020_0000_0000_0002, 0x4340_0000_0000_0001),
        (0x0020_0000_0000_0003, 0x4340_0000_0000_0002),
        (0xffe0_0000_0000_0001, 0xc33f_ffff_ffff_ffff),
        (0xffe0_0000_0000_0000, 0xc340_0000_0000_0000),
        (0xffdf_ffff_ffff_ffff, 0xc340_0000_0000_0000),
        (0xffdf_ffff_ffff_fffe, 0xc340_0000_0000_0001),
        (0xffdf_ffff_ffff_fffd, 0xc340_0000_0000_0002),
        (0x7fff_ffff_ffff_fdff, 0x43df_ffff_ffff_ffff),
        (0x7fff_ffff_ffff_ffff, 0x43e0_0000_0000_0000),
        (0x8000_0000_0000_0000, 0xc3e0_0000_0000_0000),
        (0x8000_0000_0000_0001, 0xc3e0_0000_0000_0000),
        (0x8000_0000_0000_0401, 0xc3df_ffff_ffff_ffff),
    ]
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
    let event = cpu.run(100_000).expect("SIMD integer conversion").event;
    assert!(
        matches!(event, CpuEvent::Svc(0x7e)),
        "{backend:?}: {event:?}"
    );
    assert_eq!(cpu.get_register(1), DATA);
    assert_eq!(cpu.get_register(2), DATA + RESULTS as u64);
    drop(cpu);
    data
}

fn check(unsigned: bool) {
    let word_cases = if unsigned {
        unsigned_words()
    } else {
        signed_words()
    };
    let doubleword_cases = if unsigned {
        unsigned_doublewords()
    } else {
        signed_doublewords()
    };
    let mut initial = vec![0xa5; 0x10000];
    let mut expected = initial.clone();
    let mut program = Vec::new();
    let mut slot = 0;
    for (base, bytes, lanes) in [
        (0x7e21_d800, 4, 1),
        (0x7e61_d800, 8, 1),
        (0x2e21_d800, 4, 2),
        (0x6e21_d800, 4, 4),
        (0x6e61_d800, 8, 2),
    ] {
        let samples = if bytes == 4 {
            &word_cases
        } else {
            &doubleword_cases
        };
        let base = if unsigned { base } else { base & !(1 << 29) };
        for alias in [false, true] {
            let rn = u32::from(!alias);
            let conversion = base | (rn << 5);
            if unsigned && bytes == 4 && lanes == 1 && alias {
                assert_eq!(conversion, 0x7e21_d800, "Wonder UCVTF S0, S0 opcode");
            }
            for index in 0..samples.len() {
                let mut output = [0; 16];
                let input_offset = slot * 16;
                for lane in 0..lanes {
                    let (input, bits) = samples[(index + lane * 7) % samples.len()];
                    initial[input_offset + lane * bytes..input_offset + (lane + 1) * bytes]
                        .copy_from_slice(&input.to_le_bytes()[..bytes]);
                    output[lane * bytes..(lane + 1) * bytes]
                        .copy_from_slice(&bits.to_le_bytes()[..bytes]);
                }
                expected[input_offset..input_offset + 16]
                    .copy_from_slice(&initial[input_offset..input_offset + 16]);
                program.push(0x3dc0_0020 | ((slot as u32) << 10) | rn);
                program.push(conversion);
                program.push(0x3d80_0040 | (((slot * 4) as u32) << 10));
                program.push(0x3d80_0040 | (((slot * 4 + 2) as u32) << 10) | rn);
                let result_offset = RESULTS + slot * 64;
                expected[result_offset..result_offset + 16].copy_from_slice(&output);
                let source = if alias {
                    &output[..]
                } else {
                    &initial[input_offset..input_offset + 16]
                };
                expected[result_offset + 32..result_offset + 48].copy_from_slice(source);
                slot += 1;
            }
        }
    }
    assert!(slot * 16 < RESULTS && RESULTS + slot * 64 <= initial.len());
    program.push(0xd400_0fc1);
    let reference = run(CpuBackendKind::Dynarmic, &program, &initial);
    assert_eq!(
        reference, expected,
        "independent expected SIMD lanes, upper zeros, preserved sources and canaries"
    );
    let actual = run(CpuBackendKind::Rustarmic, &program, &initial);
    assert_eq!(
        actual, expected,
        "Rustarmic SIMD integer conversion results"
    );
    assert_eq!(
        actual, reference,
        "SIMD integer-to-float outputs match Dynarmic"
    );
}

#[test]
fn simd_ucvtf_scalar_and_vector_forms_match_dynarmic() {
    check(true);
}

#[test]
fn simd_scvtf_scalar_and_vector_forms_match_dynarmic() {
    check(false);
}

#[test]
fn single_vector_lanes_match_dynarmic_under_register_pressure() {
    let mut initial = vec![0xa5; 0x10000];
    let mut expected = initial.clone();
    let mut program = vec![0xd280_0029];
    let mut slot = 0;
    for unsigned in [false, true] {
        let samples = if unsigned {
            unsigned_words()
        } else {
            signed_words()
        };
        for q in [false, true] {
            let lanes = if q { 4 } else { 2 };
            for alias in [false, true] {
                let rn = u32::from(!alias);
                let conversion =
                    0x0e21_d800 | (u32::from(unsigned) << 29) | (u32::from(q) << 30) | (rn << 5);
                for index in 0..samples.len() {
                    let input_offset = slot * 16;
                    let mut output = [0; 16];
                    for lane in 0..lanes {
                        let (input, bits) = samples[(index + lane * 7) % samples.len()];
                        initial[input_offset + lane * 4..input_offset + (lane + 1) * 4]
                            .copy_from_slice(&(input as u32).to_le_bytes());
                        output[lane * 4..(lane + 1) * 4]
                            .copy_from_slice(&(bits as u32).to_le_bytes());
                    }
                    expected[input_offset..input_offset + 16]
                        .copy_from_slice(&initial[input_offset..input_offset + 16]);
                    program.extend([
                        0x3dc0_0020 | ((slot as u32) << 10) | rn,
                        0x9100_0484,
                        0x9100_04a5,
                        0x9100_04c6,
                        0x9100_04e7,
                        0x9e62_012f,
                        conversion,
                        0x3d80_0040 | (((slot * 4) as u32) << 10),
                        0x3d80_0040 | (((slot * 4 + 2) as u32) << 10) | rn,
                        0x8b05_008a,
                        0x8b07_00cb,
                    ]);
                    let result_offset = RESULTS + slot * 64;
                    expected[result_offset..result_offset + 16].copy_from_slice(&output);
                    let source = if alias {
                        &output[..]
                    } else {
                        &initial[input_offset..input_offset + 16]
                    };
                    expected[result_offset + 32..result_offset + 48].copy_from_slice(source);
                    slot += 1;
                }
            }
        }
    }
    assert!(slot * 16 < RESULTS && RESULTS + slot * 64 <= initial.len());
    program.push(0xd400_0fc1);
    let reference = run(CpuBackendKind::Dynarmic, &program, &initial);
    assert_eq!(
        reference, expected,
        "independent expected pressured SIMD results"
    );
    let actual = run(CpuBackendKind::Rustarmic, &program, &initial);
    assert_eq!(
        actual, expected,
        "spilled scalar float lanes must zero-extend"
    );
    assert_eq!(actual, reference, "pressured SIMD conversion parity");
}
