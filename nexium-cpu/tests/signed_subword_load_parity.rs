#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;
use std::sync::{Mutex, OnceLock};

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const SOURCE: u64 = 0x30_0000;
const TABLE: u64 = 0x0b7d_3000;
const PAGE: usize = 0x1000;
const OUTPUT: usize = 0x1000;
const CONSTANTS: usize = 0x10000;
const LITERALS: usize = 0x10000;
const STRIDE: usize = 800;
const CANARY: u64 = 0xface_1234_5678_9abc;
const FPSR: u64 = 0x0800_0091;
const NZCV: u64 = 0xa000_0000;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Unsigned(u32),
    Unscaled(i32),
    Pre(i32),
    Post(i32),
    Register { option: u32, scaled: bool, rm: u32 },
}

impl Mode {
    fn writeback(self) -> bool {
        matches!(self, Self::Pre(_) | Self::Post(_))
    }

    fn word(self, bytes: usize, wide: bool, rn: u32, rt: u32) -> u32 {
        let size = bytes.trailing_zeros() << 30;
        let destination = if wide { 0 } else { 1 << 22 };
        let address = match self {
            Self::Unsigned(imm) => 0x3980_0000 | (imm << 10),
            Self::Unscaled(imm) => 0x3880_0000 | (((imm as u32) & 0x1ff) << 12),
            Self::Pre(imm) => 0x3880_0c00 | (((imm as u32) & 0x1ff) << 12),
            Self::Post(imm) => 0x3880_0400 | (((imm as u32) & 0x1ff) << 12),
            Self::Register { option, scaled, rm } => {
                0x38a0_0800 | (rm << 16) | (option << 13) | ((scaled as u32) << 12)
            }
        };
        address | size | destination | (rn << 5) | rt
    }

    fn offset(self, bytes: usize) -> u64 {
        match self {
            Self::Unsigned(imm) => u64::from(imm) * bytes as u64,
            Self::Unscaled(imm) | Self::Pre(imm) => imm as u64,
            Self::Post(_) => 0,
            Self::Register { option, scaled, rm } => {
                let extended = match (option, rm) {
                    (_, 31) => 0,
                    (2 | 3, _) => 35,
                    (6 | 7, _) => (-7i64) as u64,
                    _ => unreachable!(),
                };
                extended.wrapping_mul(if scaled { bytes as u64 } else { 1 })
            }
        }
    }

    fn addressing(self, bytes: usize, access: u64, rn: u32) -> (u64, Option<(u32, u64)>, u64) {
        if let Self::Register {
            option: 3,
            scaled: false,
            rm,
        } = self
        {
            if rm == rn && rn != 31 {
                assert_eq!(access & 1, 0);
                return (access / 2, None, access / 2);
            }
        }
        let base = access.wrapping_sub(self.offset(bytes));
        let rm = match self {
            Self::Register { option, rm, .. } if rm != 31 => Some((
                rm,
                match option {
                    2 => 0xdead_beef_0000_0023,
                    3 => 35,
                    6 => 0x1234_5678_ffff_fff9,
                    7 => (-7i64) as u64,
                    _ => unreachable!(),
                },
            )),
            _ => None,
        };
        let wb = match self {
            Self::Pre(_) => access,
            Self::Post(imm) => access.wrapping_add(imm as u64),
            _ => base,
        };
        (base, rm, wb)
    }
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Single,
    Wonder,
    Literal,
    Pair { rt2: u32 },
}

#[derive(Clone, Copy, Debug)]
struct Record {
    bytes: usize,
    wide: bool,
    rn: u32,
    rt: u32,
    mode: Mode,
    raw: u64,
    kind: Kind,
}

fn signed(raw: u64, bytes: usize, wide: bool) -> u64 {
    let bits = bytes * 8;
    let magnitude = i128::from(raw & ((1u64 << bits) - 1));
    let value = if magnitude & (1i128 << (bits - 1)) != 0 {
        magnitude - (1i128 << bits)
    } else {
        magnitude
    };
    if wide {
        value as u64
    } else {
        value as u32 as u64
    }
}

fn seed(register: usize) -> [u8; 16] {
    std::array::from_fn(|byte| (register * 37 + byte * 11 + 0x93) as u8)
}

fn constant(program: &mut Vec<u32>, data: &mut [u8], index: &mut usize, rt: u32, value: u64) {
    let offset = CONSTANTS + *index * 8;
    data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    program.push(0xf940_0000 | ((*index as u32) << 10) | (24 << 5) | rt);
    *index += 1;
}

fn put(expected: &mut [u8], offset: usize, value: u64) {
    expected[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn cpus() -> &'static Mutex<[(CpuBackendKind, Cpu); 2]> {
    static CPUS: OnceLock<Mutex<[(CpuBackendKind, Cpu); 2]>> = OnceLock::new();
    CPUS.get_or_init(|| {
        Mutex::new(
            [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic]
                .map(|backend| (backend, Cpu::new(backend).expect("CPU initialization"))),
        )
    })
}

fn check(records: &[Record], boundary: bool) {
    let mut cpus = cpus()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (backend, cpu) in &mut *cpus {
        for records in records.chunks(32) {
            let mut data = vec![0xa5; 0x12000];
            let vectors: [[u8; 16]; 32] = std::array::from_fn(seed);
            for (register, vector) in vectors.iter().enumerate() {
                data[register * 16..register * 16 + 16].copy_from_slice(vector);
            }
            let mut expected = data.clone();
            let mut source = vec![
                0x6d;
                if boundary {
                    records.len() * PAGE
                } else {
                    0x4000
                }
            ];
            let table: Vec<u8> = (0..PAGE)
                .map(|byte| (byte * 73 + byte / 5 + 0x81) as u8)
                .collect();
            let mut registers = [CANARY; 31];
            registers[24] = DATA + CONSTANTS as u64;
            registers[25] = 0;
            registers[26] = FPSR;
            registers[27] = DATA + OUTPUT as u64;
            registers[28] = DATA;
            registers[29] = NZCV;
            let mut sp = CANARY;
            let mut program = vec![0xd51b_4419, 0xd51b_443a, 0xd51b_421d];
            for register in 0..32 {
                program.push(0x3dc0_0000 | (register << 10) | (28 << 5) | register);
            }
            let mut constants = 0;
            let mut literals = Vec::new();
            for (slot, record) in records.iter().enumerate() {
                let footprint = if matches!(record.kind, Kind::Pair { .. }) {
                    8
                } else {
                    record.bytes
                };
                let source_offset = if boundary {
                    slot * PAGE + PAGE - footprint
                } else {
                    let adjustment = if record.rn == 31 {
                        record.mode.offset(record.bytes) & 15
                    } else {
                        0
                    };
                    128 + slot * 32 + adjustment as usize
                };
                source[source_offset..source_offset + footprint]
                    .copy_from_slice(&record.raw.to_le_bytes()[..footprint]);
                let access = if boundary {
                    SOURCE + slot as u64 * (PAGE as u64 * 2) + PAGE as u64 - footprint as u64
                } else {
                    SOURCE + source_offset as u64
                };
                if record.rt != 31 {
                    constant(&mut program, &mut data, &mut constants, record.rt, CANARY);
                    registers[record.rt as usize] = CANARY;
                }
                if let Kind::Pair { rt2 } = record.kind {
                    if rt2 != 31 {
                        constant(&mut program, &mut data, &mut constants, rt2, CANARY);
                        registers[rt2 as usize] = CANARY;
                    }
                }
                if !matches!(record.kind, Kind::Literal) {
                    let (base, rm, wb) = record.mode.addressing(record.bytes, access, record.rn);
                    let base_register = if record.rn == 31 { 30 } else { record.rn };
                    constant(&mut program, &mut data, &mut constants, base_register, base);
                    registers[base_register as usize] = base;
                    if record.rn == 31 {
                        program.push(0x9100_03df);
                        sp = base;
                    }
                    if let Some((rm, value)) = rm {
                        constant(&mut program, &mut data, &mut constants, rm, value);
                        registers[rm as usize] = value;
                    }
                    match record.kind {
                        Kind::Single | Kind::Wonder => program.push(record.mode.word(
                            record.bytes,
                            record.wide,
                            record.rn,
                            record.rt,
                        )),
                        Kind::Pair { rt2 } => {
                            let (base_word, imm) = match record.mode {
                                Mode::Unsigned(imm) => (0x6940_0000, (imm * 4) as i32),
                                Mode::Unscaled(imm) => (0x6940_0000, imm),
                                Mode::Pre(imm) => (0x69c0_0000, imm),
                                Mode::Post(imm) => (0x68c0_0000, imm),
                                Mode::Register { .. } => unreachable!(),
                            };
                            program.push(
                                base_word
                                    | ((((imm / 4) as u32) & 0x7f) << 15)
                                    | (rt2 << 10)
                                    | (record.rn << 5)
                                    | record.rt,
                            );
                        }
                        Kind::Literal => unreachable!(),
                    }
                    if record.mode.writeback() {
                        if record.rn == 31 {
                            sp = wb;
                        } else {
                            registers[record.rn as usize] = wb;
                        }
                    }
                } else {
                    let literal = LITERALS + slot * 8;
                    let displacement = (literal as i64 - program.len() as i64 * 4) / 4;
                    program
                        .push(0x9800_0000 | (((displacement as u32) & 0x7ffff) << 5) | record.rt);
                    literals.push((literal, record.raw as u32));
                }
                if record.rt != 31 {
                    registers[record.rt as usize] = signed(record.raw, record.bytes, record.wide);
                }
                if let Kind::Pair { rt2 } = record.kind {
                    if rt2 != 31 {
                        registers[rt2 as usize] = signed(record.raw >> 32, 4, true);
                    }
                }
                if matches!(record.kind, Kind::Wonder) {
                    assert_eq!(
                        record
                            .mode
                            .word(record.bytes, record.wide, record.rn, record.rt),
                        0x3980_02ca
                    );
                    let index = signed(record.raw, 1, true);
                    constant(&mut program, &mut data, &mut constants, 9, TABLE + 0x830);
                    constant(&mut program, &mut data, &mut constants, 11, TABLE + 0x840);
                    constant(&mut program, &mut data, &mut constants, 8, 7);
                    registers[11] = TABLE + 0x840;
                    let offset1 = (0x830u64.wrapping_add(index.wrapping_mul(8))) as usize;
                    let offset2 = (0x840u64.wrapping_add(index.wrapping_mul(8))) as usize;
                    registers[9] =
                        u64::from_le_bytes(table[offset1..offset1 + 8].try_into().unwrap());
                    registers[10] =
                        u64::from_le_bytes(table[offset2..offset2 + 8].try_into().unwrap());
                    registers[8] = registers[10].wrapping_mul(7);
                    program.extend([0xf86a_7929, 0xf86a_796a, 0x9b08_7d48]);
                }
                let offset = slot * STRIDE;
                for register in 0..31 {
                    program.push(
                        0xf900_0000
                            | (((offset / 8 + register) as u32) << 10)
                            | (27 << 5)
                            | register as u32,
                    );
                    put(
                        &mut expected,
                        OUTPUT + offset + register * 8,
                        registers[register],
                    );
                }
                program.push(0x9100_03fe);
                for (word, position, value) in [
                    (None, 248, sp),
                    (Some(0xd53b_441e), 256, 0),
                    (Some(0xd53b_443e), 264, FPSR),
                    (Some(0xd53b_421e), 272, NZCV),
                ] {
                    if let Some(word) = word {
                        program.push(word);
                    }
                    program.push(
                        0xf900_0000 | (((offset + position) as u32 / 8) << 10) | (27 << 5) | 30,
                    );
                    put(&mut expected, OUTPUT + offset + position, value);
                }
                registers[30] = NZCV;
                for (register, vector) in vectors.iter().enumerate() {
                    let position = offset + 288 + register * 16;
                    program.push(
                        0x3d80_0000 | ((position as u32 / 16) << 10) | (27 << 5) | register as u32,
                    );
                    expected[OUTPUT + position..OUTPUT + position + 16].copy_from_slice(vector);
                }
            }
            program.push(0xd400_0fc1);
            expected[CONSTANTS..].copy_from_slice(&data[CONSTANTS..]);
            let mut code = vec![
                0x5a;
                if literals.is_empty() {
                    (program.len() * 4).next_multiple_of(PAGE)
                } else {
                    0x11000
                }
            ];
            for (index, word) in program.iter().enumerate() {
                code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            for (offset, value) in literals {
                code[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            let code_before = code.clone();
            let source_before = source.clone();
            let mut mapped_table = table.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, data.len() as u64, Perm::RW, data.as_mut_ptr())
                    .unwrap();
                cpu.map_host(TABLE, PAGE as u64, Perm::RO, mapped_table.as_mut_ptr())
                    .unwrap();
                if boundary {
                    for slot in 0..records.len() {
                        cpu.map_host(
                            SOURCE + slot as u64 * PAGE as u64 * 2,
                            PAGE as u64,
                            Perm::RO,
                            source.as_mut_ptr().add(slot * PAGE),
                        )
                        .unwrap();
                    }
                } else {
                    cpu.map_host(SOURCE, source.len() as u64, Perm::RO, source.as_mut_ptr())
                        .unwrap();
                }
            }
            for register in 0..31 {
                cpu.set_register(
                    register,
                    match register {
                        24 => DATA + CONSTANTS as u64,
                        25 => 0,
                        26 => FPSR,
                        27 => DATA + OUTPUT as u64,
                        28 => DATA,
                        29 => NZCV,
                        _ => CANARY,
                    },
                );
            }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let event = cpu.run(1_000_000).expect("signed-load execution").event;
            let fault = cpu.take_fault();
            let final_registers: [u64; 31] =
                std::array::from_fn(|register| cpu.get_register(register as u32));
            let final_sp = cpu.get_sp();
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, data.len() as u64).unwrap();
                cpu.unmap_host(TABLE, PAGE as u64).unwrap();
                if boundary {
                    for slot in 0..records.len() {
                        cpu.unmap_host(SOURCE + slot as u64 * PAGE as u64 * 2, PAGE as u64)
                            .unwrap();
                    }
                } else {
                    cpu.unmap_host(SOURCE, source.len() as u64).unwrap();
                }
            }
            assert!(
                matches!(event, CpuEvent::Svc(0x7e)),
                "{backend:?} {records:?}: {event:?}"
            );
            assert!(fault.is_none(), "{backend:?}: valid signed load faulted");
            assert_eq!(final_registers, registers, "{backend:?}: final GPR guards");
            assert_eq!(final_sp, sp, "{backend:?}: final SP");
            assert_eq!(
                source, source_before,
                "{backend:?}: read-only source guards"
            );
            assert_eq!(mapped_table, table, "{backend:?}: read-only table guards");
            assert_eq!(code, code_before, "{backend:?}: code guards");
            for (index, (&actual, &wanted)) in data.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual,
                    wanted,
                    "{backend:?}: snapshot/guard byte{index:#x}, record{:?}",
                    records.get(index.saturating_sub(OUTPUT) / STRIDE)
                );
            }
        }
    }
}

fn modes(rn: u32, bytes: usize) -> Vec<Mode> {
    let mut modes = vec![
        Mode::Unsigned(0),
        Mode::Unsigned(34 / bytes as u32),
        Mode::Unsigned(4095),
        Mode::Unscaled(-256),
        Mode::Unscaled(-1),
        Mode::Unscaled(0),
        Mode::Unscaled(255),
        Mode::Pre(-19),
        Mode::Pre(21),
        Mode::Post(-19),
        Mode::Post(21),
    ];
    for option in [2, 3, 6, 7] {
        for scaled in [false, true] {
            modes.push(Mode::Register {
                option,
                scaled,
                rm: 17,
            });
        }
    }
    modes.push(Mode::Register {
        option: 3,
        scaled: false,
        rm: 31,
    });
    if rn != 31 {
        modes.push(Mode::Register {
            option: 3,
            scaled: false,
            rm: rn,
        });
    }
    modes
}

#[test]
fn signed_byte_halfword_all_addressing_modes_and_legal_aliases() {
    let mut records = Vec::new();
    for bytes in [1, 2] {
        for wide in [false, true] {
            let samples: &[u64] = if bytes == 1 {
                &[0, 1, 0x3f, 0x40, 0x7e, 0x7f, 0x80, 0x81, 0xc0, 0xfe, 0xff]
            } else {
                &[
                    0, 1, 0x7f, 0x80, 0xff, 0x100, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0xff00, 0xfffe,
                    0xffff,
                ]
            };
            for (rn, rt) in [
                (22, 10),
                (0, 0),
                (10, 10),
                (16, 17),
                (31, 23),
                (31, 31),
                (22, 31),
            ] {
                for mode in modes(rn, bytes) {
                    if mode.writeback() && rn == rt && rn != 31 {
                        continue;
                    }
                    for &raw in samples {
                        records.push(Record {
                            bytes,
                            wide,
                            rn,
                            rt,
                            mode,
                            raw,
                            kind: Kind::Single,
                        });
                    }
                }
            }
        }
    }
    check(&records, false);
}

#[test]
fn signed_byte_halfword_page_end_reads_have_exact_width() {
    let mut records = Vec::new();
    for bytes in [1, 2] {
        for wide in [false, true] {
            for rn in [0, 22] {
                for rt in [0, 10, 31] {
                    for mode in [
                        Mode::Unsigned(0),
                        Mode::Unscaled(-1),
                        Mode::Pre(16),
                        Mode::Post(-16),
                        Mode::Register {
                            option: 6,
                            scaled: true,
                            rm: 17,
                        },
                        Mode::Register {
                            option: 3,
                            scaled: false,
                            rm: 31,
                        },
                    ] {
                        if mode.writeback() && rn == rt {
                            continue;
                        }
                        for raw in [
                            0,
                            (1u64 << (bytes * 8 - 1)) - 1,
                            1 << (bytes * 8 - 1),
                            (1 << (bytes * 8)) - 1,
                        ] {
                            records.push(Record {
                                bytes,
                                wide,
                                rn,
                                rt,
                                mode,
                                raw,
                                kind: Kind::Single,
                            });
                        }
                    }
                }
            }
        }
    }
    check(&records, true);
}

#[test]
fn exact_wonder_signed_index_chain_all_byte_values() {
    assert_eq!(signed(0xff, 1, true), u64::MAX);
    assert_eq!(signed(0xff, 1, false), u32::MAX as u64);
    let records: Vec<_> = (0..=255)
        .map(|raw| Record {
            bytes: 1,
            wide: true,
            rn: 22,
            rt: 10,
            mode: Mode::Unsigned(0),
            raw,
            kind: Kind::Wonder,
        })
        .collect();
    check(&records, false);
}

#[test]
fn signed_word_loads_literal_and_pair_regressions() {
    let mut records = Vec::new();
    for raw in [0, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_fffe, 0xffff_ffff] {
        for (rn, rt) in [(22, 10), (0, 0), (31, 23), (31, 31)] {
            for mode in modes(rn, 4) {
                if mode.writeback() && rn == rt && rn != 31 {
                    continue;
                }
                records.push(Record {
                    bytes: 4,
                    wide: true,
                    rn,
                    rt,
                    mode,
                    raw,
                    kind: Kind::Single,
                });
            }
            records.push(Record {
                bytes: 4,
                wide: true,
                rn,
                rt,
                mode: Mode::Unsigned(0),
                raw,
                kind: Kind::Literal,
            });
        }
        for (rn, rt, rt2) in [(22, 9, 10), (31, 9, 10), (22, 31, 10), (31, 9, 31)] {
            for mode in [
                Mode::Unsigned(0),
                Mode::Unscaled(-256),
                Mode::Unscaled(252),
                Mode::Pre(-16),
                Mode::Post(16),
            ] {
                records.push(Record {
                    bytes: 4,
                    wide: true,
                    rn,
                    rt,
                    mode,
                    raw: raw | ((raw ^ 0xffff_ffff) << 32),
                    kind: Kind::Pair { rt2 },
                });
            }
        }
    }
    check(&records, false);
}

#[test]
fn discarded_signed_loads_to_zero_register_still_fault() {
    let mut cpus = cpus()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (backend, cpu) in &mut *cpus {
        for rt in [0, 31] {
            if *backend == CpuBackendKind::Dynarmic && rt == 31 {
                continue;
            }
            for (bytes, wide) in [(1, false), (1, true), (2, false), (2, true), (4, true)] {
                if *backend == CpuBackendKind::Dynarmic && bytes != 1 {
                    continue;
                }
                for mode in [
                    Mode::Unsigned(0),
                    Mode::Unscaled(0),
                    Mode::Pre(16),
                    Mode::Post(-16),
                    Mode::Register {
                        option: 3,
                        scaled: false,
                        rm: 31,
                    },
                ] {
                    let access = 0x50_0000;
                    let (base, _, _) = mode.addressing(bytes, access, 22);
                    let mut code = vec![0x5a; PAGE];
                    code[..4].copy_from_slice(&mode.word(bytes, wide, 22, rt).to_le_bytes());
                    code[4..8].copy_from_slice(&0x1400_0000u32.to_le_bytes());
                    let before = code.clone();
                    unsafe {
                        cpu.map_host(CODE, PAGE as u64, Perm::RX, code.as_mut_ptr())
                            .unwrap();
                    }
                    for register in 0..31 {
                        cpu.set_register(register, CANARY);
                    }
                    cpu.set_register(22, base);
                    cpu.set_sp(CANARY);
                    cpu.set_continue_on_null(false);
                    cpu.set_pc(CODE);
                    let event = cpu.run(100).expect("discarded-load fault execution").event;
                    let fault = cpu.take_fault();
                    let registers: [u64; 31] =
                        std::array::from_fn(|register| cpu.get_register(register as u32));
                    let sp = cpu.get_sp();
                    unsafe {
                        cpu.unmap_host(CODE, PAGE as u64).unwrap();
                    }
                    assert!(
                        matches!(event, CpuEvent::Exception(0x0e)),
                        "{backend:?} {bytes}/{wide} {mode:?}: {event:?}"
                    );
                    let fault = fault.unwrap_or_else(|| {
                        panic!(
                            "{backend:?} {bytes}/{wide} {mode:?}: load must report fault metadata"
                        )
                    });
                    assert_eq!(
                        (fault.pc, fault.addr, fault.size, fault.is_write),
                        (CODE, access, bytes as u32, false),
                        "{backend:?} {mode:?}"
                    );
                    for register in 0..31 {
                        if register != 22 && register != rt {
                            assert_eq!(
                                registers[register as usize], CANARY,
                                "{backend:?}: X{register}"
                            );
                        }
                    }
                    assert_eq!(sp, CANARY);
                    assert_eq!(code, before, "{backend:?}: code guards");
                }
            }
        }
    }
}
