#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const OUTPUT: usize = 0x1000;
const SOURCES: usize = 0x10000;
const WRITEBACKS: usize = 0x11000;
const STRIDE: usize = 128;
const GUARD: usize = 32;
const REGISTER_OFFSET: u64 = 73;
const INITIAL_SP: u64 = 0x1234_5678_0000;

#[derive(Clone, Copy, Debug)]
struct Shape {
    opcode: u32,
    registers: usize,
    interleaved: bool,
}

const SHAPES: [Shape; 7] = [
    Shape {
        opcode: 7,
        registers: 1,
        interleaved: false,
    },
    Shape {
        opcode: 10,
        registers: 2,
        interleaved: false,
    },
    Shape {
        opcode: 6,
        registers: 3,
        interleaved: false,
    },
    Shape {
        opcode: 2,
        registers: 4,
        interleaved: false,
    },
    Shape {
        opcode: 8,
        registers: 2,
        interleaved: true,
    },
    Shape {
        opcode: 4,
        registers: 3,
        interleaved: true,
    },
    Shape {
        opcode: 0,
        registers: 4,
        interleaved: true,
    },
];

#[derive(Clone, Copy, Debug)]
enum Mode {
    Plain,
    Immediate,
    Register,
    SameRegister,
}

fn instruction(shape: Shape, q: bool, element_bytes: usize, rt: u32, rn: u32, mode: Mode) -> u32 {
    let size = match element_bytes {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => unreachable!(),
    };
    assert!(
        q || size != 3 || !shape.interleaved,
        "ST2/3/4 .1D is reserved"
    );
    let post = match mode {
        Mode::Plain => 0,
        Mode::Immediate => (1 << 23) | (31 << 16),
        Mode::Register => (1 << 23) | (10 << 16),
        Mode::SameRegister => {
            assert_ne!(rn, 31, "Rm=31 selects immediate, not the SP value");
            (1 << 23) | (rn << 16)
        }
    };
    0x0c00_0000
        | if q { 1 << 30 } else { 0 }
        | post
        | (shape.opcode << 12)
        | (size << 10)
        | (rn << 5)
        | rt
}

fn stored_bytes(initial: &[u8], shape: Shape, q: bool, element_bytes: usize, rt: usize) -> Vec<u8> {
    let vector_bytes = if q { 16 } else { 8 };
    let mut result = Vec::with_capacity(vector_bytes * shape.registers);
    if shape.interleaved {
        for lane in 0..vector_bytes / element_bytes {
            for register in 0..shape.registers {
                let start = ((rt + register) % 32) * 16 + lane * element_bytes;
                result.extend_from_slice(&initial[start..start + element_bytes]);
            }
        }
    } else {
        for register in 0..shape.registers {
            let start = ((rt + register) % 32) * 16;
            result.extend_from_slice(&initial[start..start + vector_bytes]);
        }
    }
    result
}

struct Record {
    offset: usize,
    length: usize,
    writeback: u64,
    description: String,
}

fn run(
    backend: CpuBackendKind,
    program: &[u32],
    initial: &[u8],
    cases: usize,
    sp_base: bool,
    final_base: u64,
) -> Vec<u8> {
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
    cpu.set_sp(INITIAL_SP);
    cpu.set_register(0, 0x0123_4567_89ab_cdef);
    cpu.set_register(1, DATA);
    cpu.set_register(2, DATA + OUTPUT as u64);
    cpu.set_register(3, DATA + SOURCES as u64);
    cpu.set_register(5, DATA + WRITEBACKS as u64);
    cpu.set_register(10, REGISTER_OFFSET);
    let event = cpu
        .run(100_000)
        .expect("SIMD structure-store execution")
        .event;
    assert!(
        matches!(event, CpuEvent::Svc(0x7e)),
        "{backend:?}: {event:?}"
    );
    assert_eq!(cpu.get_register(0), 0x0123_4567_89ab_cdef);
    assert_eq!(cpu.get_register(1), DATA);
    assert_eq!(
        cpu.get_register(2),
        DATA + OUTPUT as u64 + (cases * STRIDE) as u64
    );
    assert_eq!(cpu.get_register(3), DATA + SOURCES as u64);
    assert_eq!(cpu.get_register(5), DATA + WRITEBACKS as u64);
    assert_eq!(cpu.get_register(10), REGISTER_OFFSET);
    assert_eq!(cpu.get_register(9), final_base);
    assert_eq!(cpu.get_sp(), if sp_base { final_base } else { INITIAL_SP });
    drop(cpu);
    data
}

fn check(sp_base: bool, mode: Mode) {
    let mut initial = vec![0xa5; 0x14000];
    for register in 0..32 {
        for byte in 0..16 {
            initial[register * 16 + byte] = (register * 37 + byte * 11 + 3) as u8;
        }
    }
    let mut wanted = initial.clone();
    let mut program = Vec::new();
    for register in 0..32 {
        program.push(0x3dc0_0020 | ((register as u32) << 10) | register as u32);
    }
    let mut records = Vec::new();
    for shape in SHAPES {
        for q in [false, true] {
            for element_bytes in [1, 2, 4, 8] {
                if !q && element_bytes == 8 && shape.interleaved {
                    continue;
                }
                for rt in [3, 31] {
                    let slot = records.len();
                    let offset = OUTPUT + slot * STRIDE + GUARD;
                    let base = DATA + offset as u64;
                    let output = stored_bytes(&initial, shape, q, element_bytes, rt);
                    wanted[offset..offset + output.len()].copy_from_slice(&output);
                    let advance = match mode {
                        Mode::Plain => 0,
                        Mode::Immediate => output.len() as u64,
                        Mode::Register => REGISTER_OFFSET,
                        Mode::SameRegister => base,
                    };
                    let writeback = base + advance;
                    let snapshot = WRITEBACKS + slot * 8;
                    wanted[snapshot..snapshot + 8].copy_from_slice(&writeback.to_le_bytes());
                    let rn = if sp_base { 31 } else { 9 };
                    program.push(0x9100_0040 | ((GUARD as u32) << 10) | rn);
                    program.push(instruction(shape, q, element_bytes, rt as u32, rn, mode));
                    if sp_base {
                        program.push(0x9100_03e9);
                    }
                    program.push(0xf900_00a9 | ((slot as u32) << 10));
                    program.push(0x9100_0042 | ((STRIDE as u32) << 10));
                    records.push(Record {
                        offset,
                        length: output.len(),
                        writeback,
                        description: format!("{shape:?} Q={q} element_bytes={element_bytes} Vt={rt} SP={sp_base} {mode:?}"),
                    });
                }
            }
        }
    }
    assert!(OUTPUT + records.len() * STRIDE < SOURCES);
    for register in 0..32 {
        let start = register * 16;
        wanted[SOURCES + start..SOURCES + start + 16].copy_from_slice(&initial[start..start + 16]);
        program.push(0x3d80_0060 | ((register as u32) << 10) | register as u32);
    }
    program.push(0xd400_0fc1);
    let final_base = records.last().unwrap().writeback;
    for backend in [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic] {
        let actual = run(
            backend,
            &program,
            &initial,
            records.len(),
            sp_base,
            final_base,
        );
        for (index, record) in records.iter().enumerate() {
            assert_eq!(
                &actual[record.offset..record.offset + record.length],
                &wanted[record.offset..record.offset + record.length],
                "{backend:?}: {}",
                record.description
            );
            assert_eq!(
                &actual[record.offset - GUARD..record.offset],
                &[0xa5; GUARD],
                "leading memory guard"
            );
            assert_eq!(
                &actual[record.offset + record.length..record.offset + STRIDE - GUARD],
                &wanted[record.offset + record.length..record.offset + STRIDE - GUARD],
                "trailing memory guard"
            );
            let start = WRITEBACKS + index * 8;
            let writeback = u64::from_le_bytes(actual[start..start + 8].try_into().unwrap());
            assert_eq!(
                writeback, record.writeback,
                "{backend:?} writeback: {}",
                record.description
            );
        }
        assert_eq!(
            &actual[SOURCES..SOURCES + 32 * 16],
            &initial[..32 * 16],
            "all full source vectors preserved, including upper halves and V31 wrap"
        );
        assert_eq!(actual, wanted, "{backend:?}: complete memory and guards");
    }
}

#[test]
fn wonder_st4_opcode_and_independent_interleaving() {
    assert_eq!(
        instruction(SHAPES[6], true, 4, 3, 9, Mode::Plain),
        0x4c00_0923
    );
    let initial: Vec<u8> = (0..512)
        .map(|byte| (byte / 16 * 32 + byte % 16) as u8)
        .collect();
    assert_eq!(
        stored_bytes(&initial, SHAPES[4], false, 2, 0),
        vec![0, 1, 32, 33, 2, 3, 34, 35, 4, 5, 36, 37, 6, 7, 38, 39]
    );
    assert_eq!(
        stored_bytes(&initial, SHAPES[1], false, 8, 0),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39]
    );
}

#[test]
fn all_struct_stores_plain_match_dynarmic() {
    check(false, Mode::Plain);
}

#[test]
fn all_struct_stores_immediate_postindex_match_dynarmic() {
    check(false, Mode::Immediate);
}

#[test]
fn all_struct_stores_register_postindex_match_dynarmic() {
    check(false, Mode::Register);
}

#[test]
fn all_struct_stores_same_base_offset_register_match_dynarmic() {
    check(false, Mode::SameRegister);
}

#[test]
fn all_struct_stores_sp_plain_match_dynarmic() {
    check(true, Mode::Plain);
}

#[test]
fn all_struct_stores_sp_immediate_postindex_match_dynarmic() {
    check(true, Mode::Immediate);
}

#[test]
fn all_struct_stores_sp_register_postindex_match_dynarmic() {
    check(true, Mode::Register);
}
