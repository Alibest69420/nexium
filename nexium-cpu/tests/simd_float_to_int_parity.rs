#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const INPUTS: usize = 0x1000;
const RESULTS: usize = 0x10000;
const REGISTERS: usize = 0x30000;
const CANARY: u64 = 0xface_1234_5678_9abc;
const INITIAL_SP: u64 = 0x1234_5678_0000;

#[derive(Clone, Copy)]
struct Case {
    bits: u64,
    signed: u64,
    unsigned: u64,
}

fn case(bits: u64, signed: u64, unsigned: u64) -> Case {
    Case {
        bits,
        signed,
        unsigned,
    }
}

fn singles() -> Vec<Case> {
    vec![
        case(0x0000_0000, 0, 0),
        case(0x8000_0000, 0, 0),
        case(0x0000_0001, 0, 0),
        case(0x8000_0001, 0, 0),
        case(0x007f_ffff, 0, 0),
        case(0x3f7f_ffff, 0, 0),
        case(0xbf7f_ffff, 0, 0),
        case(0x3f80_0000, 1, 1),
        case(0xbf80_0000, 0xffff_ffff, 0),
        case(0x3fff_ffff, 1, 1),
        case(0xbfff_ffff, 0xffff_ffff, 0),
        case(0x4070_0000, 3, 3),
        case(0xc070_0000, 0xffff_fffd, 0),
        case(0x4b7f_ffff, 0x00ff_ffff, 0x00ff_ffff),
        case(0x4b80_0000, 0x0100_0000, 0x0100_0000),
        case(0x4b80_0001, 0x0100_0002, 0x0100_0002),
        case(0x4eff_ffff, 0x7fff_ff80, 0x7fff_ff80),
        case(0x4f00_0000, 0x7fff_ffff, 0x8000_0000),
        case(0x4f00_0001, 0x7fff_ffff, 0x8000_0100),
        case(0xceff_ffff, 0x8000_0080, 0),
        case(0xcf00_0000, 0x8000_0000, 0),
        case(0xcf00_0001, 0x8000_0000, 0),
        case(0x4f7f_ffff, 0x7fff_ffff, 0xffff_ff00),
        case(0x4f80_0000, 0x7fff_ffff, 0xffff_ffff),
        case(0x7f7f_ffff, 0x7fff_ffff, 0xffff_ffff),
        case(0xff7f_ffff, 0x8000_0000, 0),
        case(0x7f80_0000, 0x7fff_ffff, 0xffff_ffff),
        case(0xff80_0000, 0x8000_0000, 0),
        case(0x7fc1_2345, 0, 0),
        case(0xffc1_2345, 0, 0),
        case(0x7f81_2345, 0, 0),
        case(0xff81_2345, 0, 0),
    ]
}

fn doubles() -> Vec<Case> {
    vec![
        case(0x0000_0000_0000_0000, 0, 0),
        case(0x8000_0000_0000_0000, 0, 0),
        case(0x0000_0000_0000_0001, 0, 0),
        case(0x8000_0000_0000_0001, 0, 0),
        case(0x000f_ffff_ffff_ffff, 0, 0),
        case(0x3fef_ffff_ffff_ffff, 0, 0),
        case(0xbfef_ffff_ffff_ffff, 0, 0),
        case(0x3ff0_0000_0000_0000, 1, 1),
        case(0xbff0_0000_0000_0000, 0xffff_ffff_ffff_ffff, 0),
        case(0x3fff_ffff_ffff_ffff, 1, 1),
        case(0xbfff_ffff_ffff_ffff, 0xffff_ffff_ffff_ffff, 0),
        case(0x400e_0000_0000_0000, 3, 3),
        case(0xc00e_0000_0000_0000, 0xffff_ffff_ffff_fffd, 0),
        case(
            0x433f_ffff_ffff_ffff,
            0x001f_ffff_ffff_ffff,
            0x001f_ffff_ffff_ffff,
        ),
        case(
            0x4340_0000_0000_0000,
            0x0020_0000_0000_0000,
            0x0020_0000_0000_0000,
        ),
        case(
            0x4340_0000_0000_0001,
            0x0020_0000_0000_0002,
            0x0020_0000_0000_0002,
        ),
        case(
            0x43df_ffff_ffff_ffff,
            0x7fff_ffff_ffff_fc00,
            0x7fff_ffff_ffff_fc00,
        ),
        case(
            0x43e0_0000_0000_0000,
            0x7fff_ffff_ffff_ffff,
            0x8000_0000_0000_0000,
        ),
        case(
            0x43e0_0000_0000_0001,
            0x7fff_ffff_ffff_ffff,
            0x8000_0000_0000_0800,
        ),
        case(0xc3df_ffff_ffff_ffff, 0x8000_0000_0000_0400, 0),
        case(0xc3e0_0000_0000_0000, 0x8000_0000_0000_0000, 0),
        case(0xc3e0_0000_0000_0001, 0x8000_0000_0000_0000, 0),
        case(
            0x43ef_ffff_ffff_ffff,
            0x7fff_ffff_ffff_ffff,
            0xffff_ffff_ffff_f800,
        ),
        case(
            0x43f0_0000_0000_0000,
            0x7fff_ffff_ffff_ffff,
            0xffff_ffff_ffff_ffff,
        ),
        case(
            0x7fef_ffff_ffff_ffff,
            0x7fff_ffff_ffff_ffff,
            0xffff_ffff_ffff_ffff,
        ),
        case(0xffef_ffff_ffff_ffff, 0x8000_0000_0000_0000, 0),
        case(
            0x7ff0_0000_0000_0000,
            0x7fff_ffff_ffff_ffff,
            0xffff_ffff_ffff_ffff,
        ),
        case(0xfff0_0000_0000_0000, 0x8000_0000_0000_0000, 0),
        case(0x7ff8_1234_5678_9abc, 0, 0),
        case(0xfff8_1234_5678_9abc, 0, 0),
        case(0x7ff0_1234_5678_9abc, 0, 0),
        case(0xfff0_1234_5678_9abc, 0, 0),
    ]
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
    for reg in 0..31 {
        cpu.set_register(reg, CANARY);
    }
    cpu.set_pc(CODE);
    cpu.set_sp(INITIAL_SP);
    cpu.set_register(1, DATA + INPUTS as u64);
    cpu.set_register(2, DATA + RESULTS as u64);
    cpu.set_register(3, DATA);
    cpu.set_register(4, DATA + REGISTERS as u64);
    let event = cpu
        .run(100_000)
        .expect("SIMD float-to-int conversion")
        .event;
    assert!(
        matches!(event, CpuEvent::Svc(0x7e)),
        "{backend:?}: {event:?}"
    );
    for reg in 0..31 {
        assert_eq!(
            cpu.get_register(reg),
            match reg {
                1 => DATA + INPUTS as u64,
                2 => DATA + RESULTS as u64,
                3 => DATA,
                4 => DATA + REGISTERS as u64,
                _ => CANARY,
            },
            "{backend:?}: preserved X{reg}"
        );
    }
    assert_eq!(cpu.get_sp(), INITIAL_SP, "{backend:?}: preserved SP");
    drop(cpu);
    data
}

fn assert_memory(actual: &[u8], expected: &[u8], label: &str) {
    assert_eq!(actual.len(), expected.len());
    if let Some(offset) = actual.iter().zip(expected).position(|(a, e)| a != e) {
        let end = (offset + 16).min(actual.len());
        panic!(
            "{label}: first mismatch data+{offset:#x}: actual={:02x?}, expected={:02x?}",
            &actual[offset..end],
            &expected[offset..end]
        );
    }
}

fn check(unsigned: bool) {
    let singles = singles();
    let doubles = doubles();
    let mut initial = vec![0xa5; 0x40000];
    for register in 0..32 {
        for byte in 0..16 {
            initial[register * 16 + byte] = (register * 37 + byte * 11 + 3) as u8;
        }
    }
    let mut vectors = initial[..32 * 16].to_vec();
    let mut expected = initial.clone();
    let mut program = Vec::new();
    for register in 0..32 {
        program.push(0x3dc0_0060 | ((register as u32) << 10) | register as u32);
    }
    let mut records = Vec::new();
    for (base, bytes, lanes) in [
        (0x5ea1_b800, 4, 1),
        (0x5ee1_b800, 8, 1),
        (0x0ea1_b800, 4, 2),
        (0x4ea1_b800, 4, 4),
        (0x4ee1_b800, 8, 2),
    ] {
        let samples = if bytes == 4 { &singles } else { &doubles };
        let base = base | (u32::from(unsigned) << 29);
        for (rd, rn) in [(1, 0), (0, 0), (31, 1), (1, 31), (31, 31)] {
            let opcode = base | (rn << 5) | rd;
            for index in 0..samples.len() {
                let slot = records.len();
                let input_offset = INPUTS + slot * 16;
                let result_offset = RESULTS + slot * 64;
                let mut input = [0xc7; 16];
                let mut output = [0; 16];
                for lane in 0..lanes {
                    let sample = samples[(index + lane * 7) % samples.len()];
                    let bits = if unsigned {
                        sample.unsigned
                    } else {
                        sample.signed
                    };
                    input[lane * bytes..(lane + 1) * bytes]
                        .copy_from_slice(&sample.bits.to_le_bytes()[..bytes]);
                    output[lane * bytes..(lane + 1) * bytes]
                        .copy_from_slice(&bits.to_le_bytes()[..bytes]);
                }
                initial[input_offset..input_offset + 16].copy_from_slice(&input);
                expected[input_offset..input_offset + 16].copy_from_slice(&input);
                program.extend([
                    0x3dc0_0060 | (rd << 10) | rd,
                    0x3dc0_0020 | ((slot as u32) << 10) | rn,
                    opcode,
                    0x3d80_0040 | (((slot * 4) as u32) << 10) | rd,
                    0x3d80_0040 | (((slot * 4 + 2) as u32) << 10) | rn,
                ]);
                vectors[rd as usize * 16..rd as usize * 16 + 16]
                    .copy_from_slice(&initial[rd as usize * 16..rd as usize * 16 + 16]);
                vectors[rn as usize * 16..rn as usize * 16 + 16].copy_from_slice(&input);
                vectors[rd as usize * 16..rd as usize * 16 + 16].copy_from_slice(&output);
                expected[result_offset..result_offset + 16].copy_from_slice(&output);
                expected[result_offset + 32..result_offset + 48].copy_from_slice(if rd == rn {
                    &output
                } else {
                    &input
                });
                records.push((result_offset, opcode, index));
            }
        }
    }
    assert_eq!(records.len(), 800);
    assert!(INPUTS + records.len() * 16 < RESULTS && RESULTS + records.len() * 64 < REGISTERS);
    assert_eq!(0x5ea1_b800 | 1, 0x5ea1_b801, "Wonder FCVTZS S1,S0 opcode");
    for register in 0..32 {
        program.push(0x3d80_0080 | ((register as u32) << 10) | register as u32);
    }
    expected[REGISTERS..REGISTERS + 32 * 16].copy_from_slice(&vectors);
    program.push(0xd400_0fc1);
    let reference = run(CpuBackendKind::Dynarmic, &program, &initial);
    for &(offset, opcode, sample) in &records {
        assert_eq!(
            &reference[offset..offset + 16],
            &expected[offset..offset + 16],
            "reference opcode={opcode:#010x} sample={sample}"
        );
    }
    assert_memory(
        &reference,
        &expected,
        "reference upper-zero/source/canaries",
    );
    let actual = run(CpuBackendKind::Rustarmic, &program, &initial);
    for (offset, opcode, sample) in records {
        assert_eq!(
            &actual[offset..offset + 16],
            &expected[offset..offset + 16],
            "Rustarmic opcode={opcode:#010x} sample={sample}"
        );
    }
    assert_memory(&actual, &expected, "Rustarmic upper-zero/source/canaries");
    assert_memory(
        &actual,
        &reference,
        "SIMD float-to-int value parity; FPSR excluded",
    );
}

#[test]
fn simd_fcvtzs_scalar_and_vector_match_dynarmic() {
    check(false);
}
#[test]
fn simd_fcvtzu_scalar_and_vector_match_dynarmic() {
    check(true);
}
