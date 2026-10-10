#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const STRIDE: usize = 32;

fn store_lane(width: usize, lane: usize, mode: usize) -> u32 {
    let (base, q, s, size) = match width {
        1 => (0x0d00_0000, lane >> 3, (lane >> 2) & 1, lane & 3),
        2 => (0x0d00_4000, lane >> 2, (lane >> 1) & 1, (lane & 1) << 1),
        4 => (0x0d00_8000, lane >> 1, lane & 1, 0),
        8 => (0x0d00_8400, lane, 0, 1),
        _ => unreachable!(),
    };
    let post = match mode {
        0 => 0,
        1 => 0x0080_0000 | (31 << 16),
        2 => 0x0080_0000 | (9 << 16),
        _ => unreachable!(),
    };
    base | ((q as u32) << 30) | ((s as u32) << 12) | ((size as u32) << 10)
        | (8 << 5) | post
}

fn run(backend: CpuBackendKind, program: &[u32]) -> (Vec<u8>, u64) {
    let mut code = vec![0u8; 0x1000];
    for (index, word) in program.iter().enumerate() {
        code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    let mut data = vec![0xa5; 0x1000];
    for (index, value) in data[..16].iter_mut().enumerate() {
        *value = 0x10 + index as u8;
    }
    let mut cpu = Cpu::new(backend).expect("CPU initialization");
    unsafe {
        cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr()).unwrap();
        cpu.map_host(DATA, data.len() as u64, Perm::RW, data.as_mut_ptr()).unwrap();
    }
    cpu.set_pc(CODE);
    cpu.set_register(1, DATA);
    cpu.set_register(8, DATA + STRIDE as u64);
    cpu.set_register(9, 7);
    let event = cpu.run(10_000).expect("lane-store execution").event;
    assert!(matches!(event, CpuEvent::Svc(0x7e)), "{backend:?}: {event:?}");
    let address = cpu.get_register(8);
    drop(cpu);
    (data, address)
}

#[test]
fn all_st1_lane_widths_and_address_modes_match_dynarmic() {
    let mut program = vec![0x3dc0_0020];
    let mut expected = vec![0xa5; 0x1000];
    for (index, value) in expected[..16].iter_mut().enumerate() {
        *value = 0x10 + index as u8;
    }
    let mut slot = 1;
    for mode in 0..3 {
        for width in [1, 2, 4, 8] {
            for lane in 0..16 / width {
                program.push(store_lane(width, lane, mode));
                let advance = match mode { 0 => 0, 1 => width, 2 => 7, _ => unreachable!() };
                program.push(0x9100_0108 | (((STRIDE - advance) as u32) << 10));
                let offset = slot * STRIDE;
                for byte in 0..width {
                    expected[offset + byte] = 0x10 + (lane * width + byte) as u8;
                }
                slot += 1;
            }
        }
    }
    assert_eq!(store_lane(4, 2, 0), 0x4d00_8100, "Wonder startup opcode");
    program.push(0xd400_0fc1);
    let reference = run(CpuBackendKind::Dynarmic, &program);
    assert_eq!(reference.0, expected, "reference transfer widths and canaries");
    assert_eq!(reference.1, DATA + (slot * STRIDE) as u64);
    let actual = run(CpuBackendKind::Rustarmic, &program);
    assert_eq!(actual, reference, "all 90 lane stores must match byte-for-byte");
}
