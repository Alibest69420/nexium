#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const TABLE: u64 = 0x30_0000;
const PAGE: usize = 0x1000;
const RESULTS: usize = 0x4000;
const HEADER: usize = 0x9000;
const TABLE_CENTER: usize = 0x2000;
const CANARY: u64 = 0xface_1234_5678_9abc;
const EXACT_CHAIN: [u32; 4] = [0x1b0c_292a, 0xf940_096b, 0x937e_7d4a, 0xb86a_696a];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Madd,
    Prefix,
    Lookup,
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    rn: u32,
    rm: u32,
    ra: u32,
}

impl Layout {
    fn opcode(self) -> u32 {
        0x1b00_000a | (self.rm << 16) | (self.ra << 10) | (self.rn << 5)
    }

    fn index(self, input: [u64; 3]) -> u32 {
        let register = |reg| match reg {
            9 => input[0] as u32,
            10 => input[1] as u32,
            12 => input[2] as u32,
            _ => unreachable!(),
        };
        register(self.rn)
            .wrapping_mul(register(self.rm))
            .wrapping_add(register(self.ra))
    }
}

fn inputs() -> Vec<[u64; 3]> {
    let mut inputs = vec![
        [0, 0, 0],
        [1, 0, 1],
        [0, u32::MAX as u64, 0],
        [7, (-10i32) as u32 as u64, 3],
        [(-7i32) as u32 as u64, 10, 3],
        [0x8000_0000, 0xffff_ffff, 2],
        [0xffff_ffff, 2, 0xffff_ffff],
        [0x4000_0000, (-7i32) as u32 as u64, 4],
        [0x4000_0001, 0xffff_ffff, 4],
        [0xffff_fffe, 3, 0xffff_ffff],
        [0, 0x7fff_ffff, 0],
        [0, 0x8000_0000, 0],
        [0, 0x4000_0000, 0],
        [0, 0xc000_0000, 0],
        [
            0xffff_ffff_0000_0000,
            0x1234_5678_ffff_ffff,
            0xdead_beef_0000_0000,
        ],
        [
            0xface_1234_ffff_fff8,
            0xdead_0000_0000_0015,
            0xabcd_5678_0000_0004,
        ],
    ];
    let mut seed = 0x1234_5678_9abc_def0u64;
    for _ in 0..64 {
        let mut input = [0; 3];
        for value in &mut input {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let low = ((seed % 63) as i32 - 31) as u32;
            *value = (seed & 0xffff_ffff_0000_0000) | u64::from(low);
        }
        inputs.push(input);
    }
    inputs
}

fn run(program: &[u32], initial: &[u8], expected: &[u8], table: &[u8], registers: [u64; 31]) {
    let mut code = vec![0; (program.len() * 4).next_multiple_of(PAGE)];
    for (index, word) in program.iter().enumerate() {
        code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    for backend in [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic] {
        let mut mapped_code = code.clone();
        let mut data = initial.to_vec();
        let mut mapped_table = table.to_vec();
        let mut cpu = Cpu::new(backend).expect("CPU initialization");
        unsafe {
            cpu.map_host(
                CODE,
                mapped_code.len() as u64,
                Perm::RX,
                mapped_code.as_mut_ptr(),
            )
            .unwrap();
            cpu.map_host(DATA, data.len() as u64, Perm::RW, data.as_mut_ptr())
                .unwrap();
            cpu.map_host(
                TABLE,
                mapped_table.len() as u64,
                Perm::R,
                mapped_table.as_mut_ptr(),
            )
            .unwrap();
        }
        for register in 0..31 {
            cpu.set_register(register, CANARY);
        }
        for register in [1, 2, 4] {
            cpu.set_register(register, registers[register as usize]);
        }
        cpu.set_sp(CANARY);
        cpu.set_pc(CODE);
        let event = cpu
            .run(100_000)
            .expect("Wonder index chain execution")
            .event;
        assert!(
            matches!(event, CpuEvent::Svc(0x7e)),
            "{backend:?}: {event:?}"
        );
        for register in 0..31 {
            assert_eq!(
                cpu.get_register(register),
                registers[register as usize],
                "{backend:?}: X{register}"
            );
        }
        assert_eq!(cpu.get_sp(), CANARY, "{backend:?}: SP");
        assert!(
            cpu.take_fault().is_none(),
            "{backend:?}: valid chain faulted"
        );
        drop(cpu);
        for (index, (actual, wanted)) in data.iter().zip(expected).enumerate() {
            assert_eq!(
                actual,
                wanted,
                "{backend:?}: byte {index:#x}, result record {}",
                index.saturating_sub(RESULTS) / 32
            );
        }
        assert_eq!(
            mapped_table, table,
            "{backend:?}: read-only table and guards"
        );
    }
}

fn check_chain(stage: Stage) {
    let layouts = [
        Layout {
            rn: 9,
            rm: 12,
            ra: 10,
        },
        Layout {
            rn: 10,
            rm: 12,
            ra: 9,
        },
        Layout {
            rn: 9,
            rm: 10,
            ra: 12,
        },
        Layout {
            rn: 10,
            rm: 10,
            ra: 10,
        },
        Layout {
            rn: 9,
            rm: 12,
            ra: 9,
        },
        Layout {
            rn: 9,
            rm: 12,
            ra: 12,
        },
    ];
    assert_eq!(layouts[0].opcode(), EXACT_CHAIN[0]);
    let mut table = vec![0; 0x4000];
    for index in 0..table.len() / 4 {
        let value = (index as u32).wrapping_mul(0x9e37_79b9).rotate_left(13) ^ 0x8000_a5a5;
        table[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    let mut initial = vec![0xa5; 0xa000];
    initial[HEADER + 16..HEADER + 24].copy_from_slice(&(TABLE + TABLE_CENTER as u64).to_le_bytes());
    let mut expected = initial.clone();
    let mut registers = [CANARY; 31];
    registers[1] = DATA;
    registers[2] = DATA + RESULTS as u64;
    registers[4] = DATA + HEADER as u64;
    let mut program = Vec::new();
    let mut slot = 0;
    let mut negative_lookups = 0;
    for layout in layouts {
        for input in inputs() {
            let index = layout.index(input);
            let offset = (index as i32 as i64) * 4;
            let table_offset = TABLE_CENTER as i64 + offset;
            if stage == Stage::Lookup && !(0..table.len() as i64).contains(&table_offset) {
                continue;
            }
            if stage == Stage::Lookup && offset < 0 {
                negative_lookups += 1;
            }
            for (part, register) in [(0, 9), (1, 10), (2, 12)] {
                initial[(slot * 3 + part) * 8..(slot * 3 + part + 1) * 8]
                    .copy_from_slice(&input[part].to_le_bytes());
                expected[(slot * 3 + part) * 8..(slot * 3 + part + 1) * 8]
                    .copy_from_slice(&input[part].to_le_bytes());
                program.push(0xf940_0020 | (((slot * 3 + part) as u32) << 10) | register);
            }
            program.push(0x9100_008b);
            program.push(layout.opcode());
            if stage != Stage::Madd {
                program.extend_from_slice(&EXACT_CHAIN[1..3]);
            }
            if stage == Stage::Lookup {
                program.push(EXACT_CHAIN[3]);
            }
            let value = match stage {
                Stage::Madd => u64::from(index),
                Stage::Prefix => offset as u64,
                Stage::Lookup => {
                    let pos = table_offset as usize;
                    u32::from_le_bytes(table[pos..pos + 4].try_into().unwrap()) as u64
                }
            };
            let pointer = if stage == Stage::Madd {
                registers[4]
            } else {
                TABLE + TABLE_CENTER as u64
            };
            for (part, (register, value)) in
                [(10, value), (11, pointer), (9, input[0]), (12, input[2])]
                    .into_iter()
                    .enumerate()
            {
                program.push(0xf900_0040 | (((slot * 4 + part) as u32) << 10) | register);
                expected[RESULTS + slot * 32 + part * 8..RESULTS + slot * 32 + part * 8 + 8]
                    .copy_from_slice(&value.to_le_bytes());
                registers[register as usize] = value;
            }
            slot += 1;
        }
    }
    assert!(slot * 24 < RESULTS);
    assert!(RESULTS + slot * 32 < HEADER);
    if stage == Stage::Lookup {
        assert!(negative_lookups > 100, "negative-index coverage");
        assert!(program.windows(4).any(|words| words == EXACT_CHAIN));
    }
    program.push(0xd400_0fc1);
    run(&program, &initial, &expected, &table, registers);
}

#[test]
fn wonder_madd_32_bit_overflow_zero_extension_and_source_aliases() {
    check_chain(Stage::Madd);
}

#[test]
fn wonder_index_chain_prefix_extreme_signed_indices_without_dereference() {
    check_chain(Stage::Prefix);
}

#[test]
fn wonder_exact_four_instruction_chain_positive_and_negative_table_indices() {
    check_chain(Stage::Lookup);
}

#[derive(Clone, Copy, Debug)]
enum Bitfield {
    Insert,
    Extract,
}

fn low_mask(width: u32) -> u64 {
    if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn check_bitfields(kind: Bitfield) {
    let mut initial = vec![0xa5; 0x10000];
    let mut expected = initial.clone();
    let mut registers = [CANARY; 31];
    registers[1] = DATA;
    registers[2] = DATA + RESULTS as u64;
    registers[4] = DATA + HEADER as u64;
    let mut program = Vec::new();
    let mut records = Vec::new();
    for bits in [32u32, 64] {
        for lsb in [0u32, 1, 2, 7, 16, 31, 32, 63] {
            for width in [1u32, 2, 8, 16, 31, 32, 63, 64] {
                if lsb >= bits || width > bits - lsb {
                    continue;
                }
                let field_mask = low_mask(width);
                let sign = 1u64 << (width - 1);
                for payload in [
                    0,
                    1,
                    field_mask,
                    sign,
                    sign - 1,
                    sign.wrapping_add(1) & field_mask,
                ] {
                    for rd in [10u32, 11] {
                        let slot = records.len();
                        let source = match kind {
                            Bitfield::Insert => payload | ((!field_mask) & 0xface_1234_0000_0000),
                            Bitfield::Extract => {
                                (payload << lsb) | (!(field_mask << lsb) & 0xface_1234_5678_9abc)
                            }
                        };
                        let extracted = match kind {
                            Bitfield::Insert => source & field_mask,
                            Bitfield::Extract => (source >> lsb) & field_mask,
                        };
                        let signed = if extracted & sign != 0 {
                            extracted | !field_mask
                        } else {
                            extracted
                        };
                        let result = match kind {
                            Bitfield::Insert => signed << lsb,
                            Bitfield::Extract => signed,
                        } & low_mask(bits);
                        let (immr, imms) = match kind {
                            Bitfield::Insert => ((bits - lsb) & (bits - 1), width - 1),
                            Bitfield::Extract => (lsb, lsb + width - 1),
                        };
                        let opcode = if bits == 64 { 0x9340_0000 } else { 0x1300_0000 }
                            | (immr << 16)
                            | (imms << 10)
                            | (10 << 5)
                            | rd;
                        initial[slot * 8..slot * 8 + 8].copy_from_slice(&source.to_le_bytes());
                        expected[slot * 8..slot * 8 + 8].copy_from_slice(&source.to_le_bytes());
                        program.push(0xf940_002a | ((slot as u32) << 10));
                        program.push(0xaa1f_03eb);
                        program.push(opcode);
                        let outputs = if rd == 10 {
                            [result, 0]
                        } else {
                            [source, result]
                        };
                        for (part, register) in [(0, 10), (1, 11)] {
                            program
                                .push(0xf900_0040 | (((slot * 2 + part) as u32) << 10) | register);
                            expected[RESULTS + slot * 16 + part * 8
                                ..RESULTS + slot * 16 + part * 8 + 8]
                                .copy_from_slice(&outputs[part].to_le_bytes());
                            registers[register as usize] = outputs[part];
                        }
                        records.push((kind, bits, lsb, width, rd, source, result, opcode));
                    }
                }
            }
        }
    }
    assert!(records.len() * 8 <= RESULTS);
    assert!(RESULTS + records.len() * 16 <= initial.len());
    if matches!(kind, Bitfield::Insert) {
        assert!(records.iter().any(|record| record.7 == EXACT_CHAIN[2]));
    }
    program.push(0xd400_0fc1);
    run(&program, &initial, &expected, &[0; PAGE], registers);
}

#[test]
fn sbfiz_all_inserted_sign_bit_positions_widths_and_destination_aliases() {
    check_bitfields(Bitfield::Insert);
}

#[test]
fn sbfx_sign_extension_width_boundaries_and_destination_aliases() {
    check_bitfields(Bitfield::Extract);
}
