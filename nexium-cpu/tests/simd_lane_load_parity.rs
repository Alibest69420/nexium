#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const SOURCE: u64 = 0x30_0000;
const PAGE: usize = 0x1000;
const RESULTS: usize = 0x1000;
const REGISTERS: usize = 0x10000;
const WRITEBACKS: usize = 0x11000;
const CANARY: u64 = 0xface_1234_5678_9abc;
const INITIAL_SP: u64 = 0x1234_5678_0000;
const REGISTER_OFFSET: u64 = 73;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Plain,
    Immediate,
    Register,
    SameRegister,
}

fn instruction(width: usize, lane: usize, rt: u32, rn: u32, mode: Mode) -> u32 {
    let (base, q, s, size) = match width {
        1 => (0x0d40_0000, lane >> 3, (lane >> 2) & 1, lane & 3),
        2 => (0x0d40_4000, lane >> 2, (lane >> 1) & 1, (lane & 1) << 1),
        4 => (0x0d40_8000, lane >> 1, lane & 1, 0),
        8 => (0x0d40_8400, lane, 0, 1),
        _ => unreachable!(),
    };
    let post = match mode {
        Mode::Plain => 0,
        Mode::Immediate => (1 << 23) | (31 << 16),
        Mode::Register => (1 << 23) | (10 << 16),
        Mode::SameRegister => {
            assert_ne!(rn, 31, "Rm=31 encodes immediate, not SP");
            (1 << 23) | (rn << 16)
        }
    };
    base | ((q as u32) << 30) | ((s as u32) << 12) | ((size as u32) << 10) | post | (rn << 5) | rt
}

fn run(
    backend: CpuBackendKind,
    program: &[u32],
    initial: &[u8],
    original_source: &[u8],
    final_x16: u64,
    final_writeback: u64,
    sp_base: bool,
) -> Vec<u8> {
    let mut code = vec![0; (program.len() * 4).next_multiple_of(PAGE)];
    for (index, word) in program.iter().enumerate() {
        code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    let mut data = initial.to_vec();
    let mut source = original_source.to_vec();
    let mut cpu = Cpu::new(backend).expect("CPU initialization");
    unsafe {
        cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
            .unwrap();
        cpu.map_host(DATA, data.len() as u64, Perm::RW, data.as_mut_ptr())
            .unwrap();
        cpu.map_host(SOURCE, PAGE as u64, Perm::R, source.as_mut_ptr())
            .unwrap();
    }
    for reg in 0..31 {
        cpu.set_register(reg, CANARY);
    }
    cpu.set_pc(CODE);
    cpu.set_sp(INITIAL_SP);
    cpu.set_register(1, DATA);
    cpu.set_register(2, DATA + RESULTS as u64);
    cpu.set_register(3, SOURCE);
    cpu.set_register(4, DATA + REGISTERS as u64);
    cpu.set_register(5, DATA + WRITEBACKS as u64);
    cpu.set_register(10, REGISTER_OFFSET);
    let event = cpu.run(100_000).expect("LD1 lane execution").event;
    assert!(
        matches!(event, CpuEvent::Svc(0x7e)),
        "{backend:?}: {event:?}"
    );
    for reg in 0..31 {
        let expected = match reg {
            1 => DATA,
            2 => DATA + RESULTS as u64,
            3 => SOURCE,
            4 => DATA + REGISTERS as u64,
            5 => DATA + WRITEBACKS as u64,
            9 => final_writeback,
            10 => REGISTER_OFFSET,
            16 => final_x16,
            _ => CANARY,
        };
        assert_eq!(cpu.get_register(reg), expected, "{backend:?}: X{reg}");
    }
    assert_eq!(
        cpu.get_sp(),
        if sp_base { final_writeback } else { INITIAL_SP },
        "{backend:?}: SP"
    );
    drop(cpu);
    assert_eq!(
        source, original_source,
        "{backend:?}: source memory and guards"
    );
    data
}

fn check(mode: Mode, sp_base: bool, at_page_end: bool) {
    let mut initial = vec![0xa5; 0x12000];
    for register in 0..32 {
        for byte in 0..16 {
            initial[register * 16 + byte] = (register * 37 + byte * 11 + 3) as u8;
        }
    }
    let source: Vec<_> = (0..PAGE).map(|i| (i * 29 + i / 7 + 0x91) as u8).collect();
    let mut expected = initial.clone();
    let mut vectors = initial[..32 * 16].to_vec();
    let mut program = Vec::new();
    for register in 0..32 {
        program.push(0x3dc0_0020 | ((register as u32) << 10) | register as u32);
    }
    let rn = if sp_base { 31 } else { 16 };
    let mut records = Vec::new();
    let mut final_x16 = 0;
    let mut final_writeback = 0;
    for width in [1, 2, 4, 8] {
        for lane in 0..16 / width {
            for rt in [2, 0, 31] {
                let slot = records.len();
                let input_offset = if at_page_end {
                    PAGE - width
                } else {
                    32 + slot * 32
                };
                let base = SOURCE + input_offset as u64;
                let writeback = base.wrapping_add(match mode {
                    Mode::Plain => 0,
                    Mode::Immediate => width as u64,
                    Mode::Register => REGISTER_OFFSET,
                    Mode::SameRegister => base,
                });
                let mut output = initial[rt * 16..rt * 16 + 16].to_vec();
                output[lane * width..(lane + 1) * width]
                    .copy_from_slice(&source[input_offset..input_offset + width]);
                let result_offset = RESULTS + slot * 32;
                expected[result_offset..result_offset + 16].copy_from_slice(&output);
                expected[WRITEBACKS + slot * 8..WRITEBACKS + slot * 8 + 8]
                    .copy_from_slice(&writeback.to_le_bytes());
                vectors[rt * 16..rt * 16 + 16].copy_from_slice(&output);
                program.push(0x3dc0_0020 | ((rt as u32) << 10) | rt as u32);
                program.push(0x9100_0070 | ((input_offset as u32) << 10));
                if sp_base {
                    program.push(0x9100_021f);
                }
                let opcode = instruction(width, lane, rt as u32, rn, mode);
                program.push(opcode);
                program.push(0x3d80_0040 | (((slot * 2) as u32) << 10) | rt as u32);
                program.push(0x9100_0009 | (rn << 5));
                program.push(0xf900_00a9 | ((slot as u32) << 10));
                records.push((result_offset, slot, opcode, width, lane, rt));
                final_x16 = if sp_base { base } else { writeback };
                final_writeback = writeback;
            }
        }
    }
    assert_eq!(records.len(), 90);
    assert_eq!(
        instruction(4, 0, 2, 16, Mode::Plain),
        0x0d40_8202,
        "Wonder LD1 {{V2.S}}[0], [X16] opcode"
    );
    for register in 0..32 {
        program.push(0x3d80_0080 | ((register as u32) << 10) | register as u32);
    }
    expected[REGISTERS..REGISTERS + 32 * 16].copy_from_slice(&vectors);
    program.push(0xd400_0fc1);
    let reference = run(
        CpuBackendKind::Dynarmic,
        &program,
        &initial,
        &source,
        final_x16,
        final_writeback,
        sp_base,
    );
    let actual = run(
        CpuBackendKind::Rustarmic,
        &program,
        &initial,
        &source,
        final_x16,
        final_writeback,
        sp_base,
    );
    for (offset, slot, opcode, width, lane, rt) in records {
        for (backend, bytes) in [("Dynarmic", &reference), ("Rustarmic", &actual)] {
            assert_eq!(
                &bytes[offset..offset + 16],
                &expected[offset..offset + 16],
                "{backend}: {mode:?} SP={sp_base} boundary={at_page_end} opcode={opcode:#010x} V{rt} width={width} lane={lane}"
            );
            let wb = WRITEBACKS + slot * 8;
            assert_eq!(
                &bytes[wb..wb + 8],
                &expected[wb..wb + 8],
                "{backend}: writeback opcode={opcode:#010x}"
            );
        }
    }
    assert_eq!(
        reference, expected,
        "reference unaffected lanes/registers/guards"
    );
    assert_eq!(
        actual, expected,
        "Rustarmic unaffected lanes/registers/guards"
    );
    assert_eq!(actual, reference, "LD1 lane parity");
}

#[test]
fn ld1_plain_all_lane_indices() {
    check(Mode::Plain, false, false);
}
#[test]
fn ld1_immediate_post_all_lane_indices() {
    check(Mode::Immediate, false, false);
}
#[test]
fn ld1_register_post_all_lane_indices() {
    check(Mode::Register, false, false);
}
#[test]
fn ld1_same_base_offset_register_uses_old_value() {
    check(Mode::SameRegister, false, false);
}
#[test]
fn ld1_sp_plain_preserves_other_lanes() {
    check(Mode::Plain, true, false);
}
#[test]
fn ld1_sp_immediate_post_preserves_other_lanes() {
    check(Mode::Immediate, true, false);
}
#[test]
fn ld1_sp_register_post_preserves_other_lanes() {
    check(Mode::Register, true, false);
}
#[test]
fn ld1_at_unmapped_page_boundary_reads_only_one_element() {
    check(Mode::Plain, false, true);
}
