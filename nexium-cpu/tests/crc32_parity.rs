#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const RESULTS: usize = 0x4000;
const SENTINEL: usize = 0xbff0;
const SENTINEL_VALUE: u64 = 0xfedc_ba98_7654_3210;

fn reference_crc(mut crc: u32, value: u64, width: usize, castagnoli: bool) -> u32 {
    let polynomial = if castagnoli { 0x82f6_3b78 } else { 0xedb8_8320 };
    for &byte in &value.to_le_bytes()[..width] {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { polynomial } else { 0 };
        }
    }
    crc
}

fn crc_opcode(width: usize, castagnoli: bool, rd: u32, rn: u32, rm: u32) -> u32 {
    let size = match width {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => unreachable!(),
    };
    0x1ac0_4000
        | if width == 8 { 1 << 31 } else { 0 }
        | if castagnoli { 1 << 12 } else { 0 }
        | (size << 10)
        | (rm << 16)
        | (rn << 5)
        | rd
}

fn run(backend: CpuBackendKind, program: &[u32], initial_data: &[u8]) -> Vec<u8> {
    let code_size = (program.len() * 4).next_multiple_of(0x1000);
    let mut code = vec![0; code_size];
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
    cpu.set_register(3, DATA + SENTINEL as u64);
    cpu.set_register(10, u32::MAX as u64);
    let event = cpu.run(100_000).expect("CRC execution").event;
    assert!(
        matches!(event, CpuEvent::Svc(0x7e)),
        "{backend:?}: {event:?}"
    );
    assert_eq!(cpu.get_register(1), DATA);
    assert_eq!(cpu.get_register(2), DATA + RESULTS as u64);
    assert_eq!(cpu.get_register(3), DATA + SENTINEL as u64);
    drop(cpu);
    data
}

#[test]
fn crc_instruction_chains_match_standard_check_vectors() {
    for (castagnoli, expected) in [(false, 0xcbf4_3926u32), (true, 0xe306_9283)] {
        let mut initial = vec![0xa5; 0xc000];
        initial[..9].copy_from_slice(b"123456789");
        let mut program = Vec::new();
        let mut oracle = u32::MAX;
        for (index, &byte) in b"123456789".iter().enumerate() {
            program.push(0x3940_002b | ((index as u32) << 10));
            program.push(crc_opcode(1, castagnoli, 10, 10, 11));
            oracle = reference_crc(oracle, byte as u64, 1, castagnoli);
        }
        assert_eq!(oracle ^ u32::MAX, expected, "independent polynomial oracle");
        program.push(0xf900_004a);
        program.push(0xd400_0fc1);
        let mut expected_data = initial.clone();
        expected_data[RESULTS..RESULTS + 8].copy_from_slice(&(oracle as u64).to_le_bytes());
        for backend in [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic] {
            assert_eq!(
                run(backend, &program, &initial),
                expected_data,
                "{backend:?}"
            );
        }
    }
}

#[test]
fn all_crc_widths_polynomials_and_register_aliases_match_dynarmic() {
    assert_eq!(
        crc_opcode(8, false, 10, 10, 11),
        0x9acb_4d4a,
        "Wonder opcode"
    );
    let pairs = [
        (0, 0),
        (u32::MAX, u64::MAX),
        (0x1234_5678, 0x0123_4567_89ab_cdef),
        (0x89ab_cdef, 0xfedc_ba98_7654_3210),
        (0x8000_0000, 0x8000_0000_0000_0001),
        (0x0000_0001, 0x0102_0304_0506_0708),
        (0xa5a5_5a5a, 0x1337_dead_beef_cafe),
        (0x1020_4080, 0x55aa_55aa_aa55_aa55),
    ];
    let layouts = [
        (12, 10, 11),
        (10, 10, 11),
        (11, 10, 11),
        (10, 10, 10),
        (12, 31, 11),
        (12, 10, 31),
        (12, 31, 31),
        (31, 10, 11),
    ];
    let mut initial = vec![0xa5; 0xc000];
    initial[SENTINEL..SENTINEL + 8].copy_from_slice(&SENTINEL_VALUE.to_le_bytes());
    let mut expected = initial.clone();
    let mut program = Vec::new();
    let mut index = 0;
    for castagnoli in [false, true] {
        for width in [1, 2, 4, 8] {
            for &(seed, value) in &pairs {
                for &(rd, rn, rm) in &layouts {
                    let seed_register = 0xdead_beef_0000_0000 | seed as u64;
                    let input = index * 16;
                    initial[input..input + 8].copy_from_slice(&seed_register.to_le_bytes());
                    initial[input + 8..input + 16].copy_from_slice(&value.to_le_bytes());
                    expected[input..input + 16].copy_from_slice(&initial[input..input + 16]);
                    program.push(0xf940_002a | (((index * 2) as u32) << 10));
                    program.push(0xf940_002b | (((index * 2 + 1) as u32) << 10));
                    program.push(0xf940_006c);
                    program.push(crc_opcode(width, castagnoli, rd, rn, rm));
                    let source_seed = if rn == 31 { 0 } else { seed };
                    let source_value = match rm {
                        31 => 0,
                        10 => seed_register,
                        _ => value,
                    };
                    let result = reference_crc(source_seed, source_value, width, castagnoli) as u64;
                    let mut registers = [seed_register, value, SENTINEL_VALUE];
                    if rd != 31 {
                        registers[(rd - 10) as usize] = result;
                    }
                    let output = RESULTS + index * 32;
                    for (slot, &value) in registers.iter().enumerate() {
                        program.push(
                            0xf900_0040 | (((index * 4 + slot) as u32) << 10) | (10 + slot as u32),
                        );
                        expected[output + slot * 8..output + slot * 8 + 8]
                            .copy_from_slice(&value.to_le_bytes());
                    }
                    index += 1;
                }
            }
        }
    }
    assert_eq!(index, 512);
    program.push(0xd400_0fc1);
    let reference = run(CpuBackendKind::Dynarmic, &program, &initial);
    assert_eq!(
        reference, expected,
        "Dynarmic must match independent software oracle and canaries"
    );
    let actual = run(CpuBackendKind::Rustarmic, &program, &initial);
    assert_eq!(
        actual, expected,
        "Rustarmic must match independent software oracle and canaries"
    );
    assert_eq!(
        actual, reference,
        "all CRC operand widths and aliases match byte-for-byte"
    );
}
