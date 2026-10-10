#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const PAGE_SOURCES: u64 = 0x1000_0000;
const BASES: usize = 0x800;
const INPUTS: usize = 0x1000;
const OUTPUTS: usize = 0x5000;
const WRITEBACKS: usize = 0x10000;
const CANARY: u64 = 0xfade_1234_5678_9abc;
const OFFSET: u64 = 73;

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

#[derive(Clone, Copy, Debug)]
struct Case {
    shape: Shape,
    q: bool,
    element: usize,
    rt: u32,
    rn: u32,
    mode: Mode,
}

impl Case {
    fn length(self) -> usize {
        self.shape.registers * if self.q { 16 } else { 8 }
    }

    fn instruction(self) -> u32 {
        let size = self.element.trailing_zeros();
        assert!(self.q || size != 3 || !self.shape.interleaved);
        let post = match self.mode {
            Mode::Plain => 0,
            Mode::Immediate => (1 << 23) | (31 << 16),
            Mode::Register => (1 << 23) | (10 << 16),
            Mode::SameRegister => {
                assert_ne!(self.rn, 31, "Rm31 means immediate, not SP");
                (1 << 23) | (self.rn << 16)
            }
        };
        0x0c40_0000
            | ((self.q as u32) << 30)
            | post
            | (self.shape.opcode << 12)
            | (size << 10)
            | (self.rn << 5)
            | self.rt
    }

    fn writeback(self, base: u64) -> u64 {
        base + match self.mode {
            Mode::Plain => 0,
            Mode::Immediate => self.length() as u64,
            Mode::Register => OFFSET,
            Mode::SameRegister => base,
        }
    }
}

fn seed(register: usize) -> [u8; 16] {
    std::array::from_fn(|byte| (register * 37 + byte * 11 + 0x93) as u8)
}

fn load_vectors(case: Case, input: &[u8], vectors: &mut [[u8; 16]; 32]) {
    assert_eq!(input.len(), case.length());
    let width = if case.q { 16 } else { 8 };
    for register in 0..case.shape.registers {
        vectors[(case.rt as usize + register) % 32] = [0; 16];
    }
    if case.shape.interleaved {
        let mut cursor = 0;
        for lane in 0..width / case.element {
            for register in 0..case.shape.registers {
                let target = &mut vectors[(case.rt as usize + register) % 32];
                target[lane * case.element..(lane + 1) * case.element]
                    .copy_from_slice(&input[cursor..cursor + case.element]);
                cursor += case.element;
            }
        }
    } else {
        for register in 0..case.shape.registers {
            vectors[(case.rt as usize + register) % 32][..width]
                .copy_from_slice(&input[register * width..(register + 1) * width]);
        }
    }
}

fn check(cases: &[Case], page_end: bool) {
    for cases in cases.chunks(80) {
        let mut vectors: [[u8; 16]; 32] = std::array::from_fn(seed);
        let mut initial = vec![0xa5; 0x12000];
        for (register, value) in vectors.iter().enumerate() {
            initial[register * 16..register * 16 + 16].copy_from_slice(value);
        }
        let mut expected = initial.clone();
        let mut page_initial = Vec::new();
        let mut program = Vec::new();
        let mut expected_gprs = [CANARY; 31];
        expected_gprs[10] = OFFSET;
        expected_gprs[12] = DATA + BASES as u64;
        expected_gprs[13] = DATA;
        expected_gprs[14] = DATA + OUTPUTS as u64;
        expected_gprs[15] = DATA + WRITEBACKS as u64;
        let mut expected_sp = CANARY;
        for register in 0..32u32 {
            program.push(0x3dc0_01a0 | (register << 10) | register);
        }
        for (slot, &case) in cases.iter().enumerate() {
            let payload: Vec<u8> = (0..case.length())
                .map(|byte| (slot * 71 + byte * 11 + 0x93) as u8)
                .collect();
            let base = if page_end {
                let mut page = vec![0x5a; 0x1000];
                page[0x1000 - payload.len()..].copy_from_slice(&payload);
                page_initial.push(page);
                PAGE_SOURCES + (slot * 0x2000 + 0x1000 - payload.len()) as u64
            } else {
                let offset = INPUTS + slot * 128 + 24 + slot % 8;
                initial[offset..offset + payload.len()].copy_from_slice(&payload);
                expected[offset..offset + payload.len()].copy_from_slice(&payload);
                DATA + offset as u64
            };
            initial[BASES + slot * 8..BASES + slot * 8 + 8].copy_from_slice(&base.to_le_bytes());
            expected[BASES + slot * 8..BASES + slot * 8 + 8].copy_from_slice(&base.to_le_bytes());
            for index in 0..case.shape.registers {
                let register = (case.rt + index as u32) % 32;
                program.push(0x3dc0_01a0 | (register << 10) | register);
                vectors[register as usize] = seed(register as usize);
            }
            let base_register = if case.rn == 31 { 16 } else { case.rn };
            program.push(0xf940_0180 | ((slot as u32) << 10) | base_register);
            expected_gprs[base_register as usize] = base;
            if case.rn == 31 {
                program.push(0x9100_021f);
            }
            program.push(case.instruction());
            load_vectors(case, &payload, &mut vectors);
            let writeback = case.writeback(base);
            if case.rn == 31 {
                expected_sp = writeback;
            } else {
                expected_gprs[case.rn as usize] = writeback;
            }
            program.push(0x9100_0010 | (case.rn << 5));
            expected_gprs[16] = writeback;
            program.push(0xf900_01f0 | ((slot as u32) << 10));
            expected[WRITEBACKS + slot * 8..WRITEBACKS + slot * 8 + 8]
                .copy_from_slice(&writeback.to_le_bytes());
            for (register, value) in vectors.iter().enumerate() {
                let offset = slot * 32 + register;
                program.push(0x3d80_01c0 | ((offset as u32) << 10) | register as u32);
                expected[OUTPUTS + offset * 16..OUTPUTS + offset * 16 + 16].copy_from_slice(value);
            }
        }
        program.push(0xd400_0fc1);
        let mut encoded = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
        for (index, word) in program.iter().enumerate() {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        for backend in [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic] {
            let mut code = encoded.clone();
            let mut actual = initial.clone();
            let mut pages = page_initial.clone();
            let mut cpu = Cpu::new(backend).expect("CPU initialization");
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, actual.len() as u64, Perm::RW, actual.as_mut_ptr())
                    .unwrap();
                for (slot, page) in pages.iter_mut().enumerate() {
                    cpu.map_host(
                        PAGE_SOURCES + (slot * 0x2000) as u64,
                        0x1000,
                        Perm::R,
                        page.as_mut_ptr(),
                    )
                    .unwrap();
                }
            }
            for register in 0..31 {
                cpu.set_register(register, CANARY);
            }
            for register in [10, 12, 13, 14, 15] {
                cpu.set_register(register, expected_gprs[register as usize]);
            }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let event = cpu.run(100_000).expect("structure load execution").event;
            assert!(
                matches!(event, CpuEvent::Svc(0x7e)),
                "{backend:?} page_end={page_end}: {event:?}"
            );
            for (slot, case) in cases.iter().enumerate() {
                let start = OUTPUTS + slot * 512;
                for register in 0..32 {
                    let pos = start + register * 16;
                    assert_eq!(
                        &actual[pos..pos + 16],
                        &expected[pos..pos + 16],
                        "{backend:?} page_end={page_end}: {case:?} V{register}"
                    );
                }
                let pos = WRITEBACKS + slot * 8;
                assert_eq!(
                    &actual[pos..pos + 8],
                    &expected[pos..pos + 8],
                    "{backend:?} writeback: {case:?}"
                );
            }
            for register in 0..31 {
                assert_eq!(
                    cpu.get_register(register),
                    expected_gprs[register as usize],
                    "{backend:?}: X{register}"
                );
            }
            assert_eq!(cpu.get_sp(), expected_sp, "{backend:?}: SP");
            drop(cpu);
            assert_eq!(
                actual, expected,
                "{backend:?}: all data/source/alignment guards"
            );
            assert_eq!(
                pages, page_initial,
                "{backend:?}: read-only page source canaries"
            );
            assert_eq!(code, encoded, "{backend:?}: code unchanged");
        }
    }
}

fn cases(mode: Mode, page_end: bool) -> Vec<Case> {
    let mut cases = Vec::new();
    for shape in SHAPES {
        for q in [false, true] {
            for element in [1, 2, 4, 8] {
                if !q && element == 8 && shape.interleaved {
                    continue;
                }
                let destinations: &[u32] = if page_end { &[31] } else { &[0, 31] };
                let bases: &[u32] = if page_end { &[0, 31] } else { &[0, 9, 21, 31] };
                for &rt in destinations {
                    for &rn in bases {
                        if matches!(mode, Mode::SameRegister) && rn == 31 {
                            continue;
                        }
                        cases.push(Case {
                            shape,
                            q,
                            element,
                            rt,
                            rn,
                            mode,
                        });
                    }
                }
            }
        }
    }
    cases
}

#[test]
fn wonder_ld2_exact_opcodes_and_independent_interleave_oracle() {
    let case = Case {
        shape: SHAPES[4],
        q: false,
        element: 4,
        rt: 0,
        rn: 21,
        mode: Mode::Plain,
    };
    assert_eq!(case.instruction(), 0x0c40_8aa0);
    let mut vectors = std::array::from_fn(seed);
    load_vectors(case, &(0u8..16).collect::<Vec<_>>(), &mut vectors);
    assert_eq!(
        vectors[0],
        [0, 1, 2, 3, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        vectors[1],
        [4, 5, 6, 7, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    check(
        &[
            case,
            Case {
                rt: 2,
                rn: 22,
                ..case
            },
            Case {
                rt: 5,
                rn: 8,
                ..case
            },
        ],
        false,
    );
}

#[test]
fn all_struct_loads_plain_match_dynarmic() {
    check(&cases(Mode::Plain, false), false);
}

#[test]
fn all_struct_loads_immediate_postindex_match_dynarmic() {
    check(&cases(Mode::Immediate, false), false);
}

#[test]
fn all_struct_loads_register_postindex_match_dynarmic() {
    check(&cases(Mode::Register, false), false);
}

#[test]
fn all_struct_loads_same_base_offset_uses_old_value() {
    check(&cases(Mode::SameRegister, false), false);
}

#[test]
fn all_struct_loads_at_unmapped_page_end_read_exact_transfer_width() {
    for mode in [
        Mode::Plain,
        Mode::Immediate,
        Mode::Register,
        Mode::SameRegister,
    ] {
        check(&cases(mode, true), true);
    }
}
