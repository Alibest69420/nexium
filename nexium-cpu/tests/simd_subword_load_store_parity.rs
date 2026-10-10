#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const SOURCE: u64 = 0x30_0000;
const TARGET: u64 = 0x40_0000;
const PAGE: usize = 0x1000;
const SNAPSHOTS: usize = 0x1000;
const WRITEBACKS: usize = 0x5000;
const CONSTANTS: usize = 0x7000;
const FINAL_VECTORS: usize = 0xb000;
const CANARY: u64 = 0xface_1234_5678_9abc;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Unsigned(u32),
    Unscaled(i32),
    Pre(i32),
    Post(i32),
    Register { option: u32, scaled: bool, rm: u32 },
}

impl Mode {
    fn instruction(self, bytes: usize, load: bool, rn: u32, rt: u32) -> u32 {
        let width = if bytes == 2 { 1 << 30 } else { 0 };
        let access = (load as u32) << 22;
        let address = match self {
            Self::Unsigned(imm) => 0x3d00_0000 | (imm << 10),
            Self::Unscaled(imm) => 0x3c00_0000 | (((imm as u32) & 0x1ff) << 12),
            Self::Pre(imm) => 0x3c00_0c00 | (((imm as u32) & 0x1ff) << 12),
            Self::Post(imm) => 0x3c00_0400 | (((imm as u32) & 0x1ff) << 12),
            Self::Register { option, scaled, rm } => {
                0x3c20_0800 | (rm << 16) | (option << 13) | ((scaled as u32) << 12)
            }
        };
        address | width | access | (rn << 5) | rt
    }

    fn addressing(self, bytes: usize, access: u64, rn: u32) -> (u64, u64, u64) {
        match self {
            Self::Unsigned(imm) => {
                let base = access - u64::from(imm) * bytes as u64;
                (base, CANARY, base)
            }
            Self::Unscaled(imm) => {
                let base = access.wrapping_sub(imm as u64);
                (base, CANARY, base)
            }
            Self::Pre(imm) => (access.wrapping_sub(imm as u64), CANARY, access),
            Self::Post(imm) => (access, CANARY, access.wrapping_add(imm as u64)),
            Self::Register { option, scaled, rm } => {
                if rm == rn && rm != 31 {
                    assert_eq!(option, 3);
                    assert!(!scaled);
                    assert_eq!(access & 1, 0);
                    return (access / 2, CANARY, access / 2);
                }
                let (raw, extended) = match (option, rm) {
                    (_, 31) => (CANARY, 0),
                    (2, _) => (0xdead_beef_0000_0023, 35),
                    (3, _) => (35, 35),
                    (6, _) => (0x1234_5678_ffff_fff9, (-7i64) as u64),
                    (7, _) => ((-7i64) as u64, (-7i64) as u64),
                    _ => unreachable!(),
                };
                let offset = extended.wrapping_mul(if scaled { bytes as u64 } else { 1 });
                let base = access.wrapping_sub(offset);
                (base, raw, base)
            }
        }
    }
}

fn modes(bytes: usize, rn: u32, boundary: bool) -> Vec<Mode> {
    let mut modes = vec![
        Mode::Unsigned(34 / bytes as u32),
        Mode::Unsigned(0),
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
    if rn != 31 && (!boundary || bytes == 2) {
        modes.push(Mode::Register {
            option: 3,
            scaled: false,
            rm: rn,
        });
    }
    modes
}

fn load_constant(program: &mut Vec<u32>, data: &mut [u8], index: usize, rt: u32, value: u64) {
    data[CONSTANTS + index * 8..CONSTANTS + index * 8 + 8].copy_from_slice(&value.to_le_bytes());
    program.push(0xf940_01a0 | ((index as u32) << 10) | rt);
}

fn address_setup(
    program: &mut Vec<u32>,
    data: &mut [u8],
    constant: usize,
    rn: u32,
    base: u64,
    rm: u64,
    regs: &mut [u64; 31],
    sp: &mut u64,
) {
    let base_register = if rn == 31 { 16 } else { rn };
    load_constant(program, data, constant, base_register, base);
    regs[base_register as usize] = base;
    if rn == 31 {
        program.push(0x9100_021f);
        *sp = base;
    }
    load_constant(program, data, constant + 1, 17, rm);
    regs[17] = rm;
}

fn check(boundary: bool) {
    let memory_size = if boundary { PAGE } else { 0x4000 };
    let source: Vec<u8> = (0..memory_size)
        .map(|i| (i * 73 + i / 5 + 0x80) as u8)
        .collect();
    let target = vec![0xa5; memory_size];
    let mut expected_target = target.clone();
    let mut data = vec![0x5a; 0xc000];
    for register in 0..32 {
        for byte in 0..16 {
            data[register * 16 + byte] = (register * 37 + byte * 11 + 0x93) as u8;
        }
    }
    let mut expected = data.clone();
    let mut vectors = data[..512].to_vec();
    let mut program = Vec::new();
    for register in 0..32 {
        program.push(0x3dc0_0020 | ((register as u32) << 10) | register as u32);
    }
    let mut regs = [CANARY; 31];
    regs[1] = DATA;
    regs[2] = DATA + SNAPSHOTS as u64;
    regs[3] = DATA + WRITEBACKS as u64;
    regs[6] = DATA + FINAL_VECTORS as u64;
    regs[13] = DATA + CONSTANTS as u64;
    let mut sp = CANARY;
    let mut records = Vec::new();
    for bytes in [1, 2] {
        for rn in [23, 16, 31] {
            for rt in [0, 2, 16, 31] {
                for mode in modes(bytes, rn, boundary) {
                    let slot = records.len();
                    let offset = if boundary {
                        PAGE - bytes
                    } else {
                        128 + slot * 32
                    };
                    let source_address = SOURCE + offset as u64;
                    let target_address = TARGET + offset as u64;
                    let (store_base, store_rm, store_wb) =
                        mode.addressing(bytes, target_address, rn);
                    let (load_base, load_rm, load_wb) = mode.addressing(bytes, source_address, rn);
                    let store_word = mode.instruction(bytes, false, rn, rt);
                    let load_word = mode.instruction(bytes, true, rn, rt);
                    let original = data[rt as usize * 16..rt as usize * 16 + 16].to_vec();
                    program.push(0x3dc0_0020 | (rt << 10) | rt);
                    address_setup(
                        &mut program,
                        &mut data,
                        slot * 4,
                        rn,
                        store_base,
                        store_rm,
                        &mut regs,
                        &mut sp,
                    );
                    program.push(store_word);
                    program.push(0x3d80_0040 | (((slot * 2) as u32) << 10) | rt);
                    program.push(0x9100_0009 | (rn << 5));
                    program.push(0xf900_0069 | (((slot * 2) as u32) << 10));
                    expected_target[offset..offset + bytes].copy_from_slice(&original[..bytes]);
                    expected[SNAPSHOTS + slot * 32..SNAPSHOTS + slot * 32 + 16]
                        .copy_from_slice(&original);
                    expected[WRITEBACKS + slot * 16..WRITEBACKS + slot * 16 + 8]
                        .copy_from_slice(&store_wb.to_le_bytes());
                    if rn == 31 {
                        sp = store_wb;
                    } else {
                        regs[rn as usize] = store_wb;
                    }
                    address_setup(
                        &mut program,
                        &mut data,
                        slot * 4 + 2,
                        rn,
                        load_base,
                        load_rm,
                        &mut regs,
                        &mut sp,
                    );
                    program.push(load_word);
                    program.push(0x3d80_0040 | (((slot * 2 + 1) as u32) << 10) | rt);
                    program.push(0x9100_0009 | (rn << 5));
                    program.push(0xf900_0069 | (((slot * 2 + 1) as u32) << 10));
                    let mut loaded = [0u8; 16];
                    loaded[..bytes].copy_from_slice(&source[offset..offset + bytes]);
                    expected[SNAPSHOTS + slot * 32 + 16..SNAPSHOTS + slot * 32 + 32]
                        .copy_from_slice(&loaded);
                    vectors[rt as usize * 16..rt as usize * 16 + 16].copy_from_slice(&loaded);
                    expected[WRITEBACKS + slot * 16 + 8..WRITEBACKS + slot * 16 + 16]
                        .copy_from_slice(&load_wb.to_le_bytes());
                    if rn == 31 {
                        sp = load_wb;
                    } else {
                        regs[rn as usize] = load_wb;
                    }
                    regs[9] = load_wb;
                    records.push((bytes, rn, rt, mode, store_word, load_word));
                }
            }
        }
    }
    assert!(
        records.iter().any(|record| record.5 == 0x3d40_8ae0),
        "exact Wonder opcode"
    );
    assert!(128 + records.len() * 32 < 0x4000);
    assert!(CONSTANTS + records.len() * 32 < FINAL_VECTORS);
    assert!(SNAPSHOTS + records.len() * 32 <= WRITEBACKS);
    assert!(WRITEBACKS + records.len() * 16 <= CONSTANTS);
    expected[CONSTANTS..FINAL_VECTORS].copy_from_slice(&data[CONSTANTS..FINAL_VECTORS]);
    expected[FINAL_VECTORS..FINAL_VECTORS + 512].copy_from_slice(&vectors);
    for register in 0..32 {
        program.push(0x3d80_00c0 | ((register as u32) << 10) | register as u32);
    }
    program.push(0xd400_0fc1);
    let mut code = vec![0; (program.len() * 4).next_multiple_of(PAGE)];
    for (index, word) in program.iter().enumerate() {
        code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    for backend in [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic] {
        let mut mapped_code = code.clone();
        let mut actual = data.clone();
        let mut mapped_source = source.clone();
        let mut mapped_target = target.clone();
        let mut cpu = Cpu::new(backend).expect("CPU initialization");
        unsafe {
            cpu.map_host(
                CODE,
                mapped_code.len() as u64,
                Perm::RX,
                mapped_code.as_mut_ptr(),
            )
            .unwrap();
            cpu.map_host(DATA, actual.len() as u64, Perm::RW, actual.as_mut_ptr())
                .unwrap();
            cpu.map_host(
                SOURCE,
                memory_size as u64,
                Perm::R,
                mapped_source.as_mut_ptr(),
            )
            .unwrap();
            cpu.map_host(
                TARGET,
                memory_size as u64,
                Perm::RW,
                mapped_target.as_mut_ptr(),
            )
            .unwrap();
        }
        for register in 0..31 {
            cpu.set_register(register, CANARY);
        }
        for register in [1, 2, 3, 6, 13] {
            cpu.set_register(register, regs[register as usize]);
        }
        cpu.set_sp(CANARY);
        cpu.set_pc(CODE);
        let event = cpu.run(100_000).expect("subword SIMD execution").event;
        assert!(
            matches!(event, CpuEvent::Svc(0x7e)),
            "{backend:?}: {event:?}"
        );
        for register in 0..31 {
            assert_eq!(
                cpu.get_register(register),
                regs[register as usize],
                "{backend:?}: X{register}"
            );
        }
        assert_eq!(cpu.get_sp(), sp, "{backend:?}: SP/writeback");
        drop(cpu);
        for (slot, record) in records.iter().enumerate() {
            let start = SNAPSHOTS + slot * 32;
            assert_eq!(
                &actual[start..start + 32],
                &expected[start..start + 32],
                "{backend:?}: {record:?}"
            );
        }
        assert_eq!(
            actual, expected,
            "{backend:?}: complete result memory/guards"
        );
        assert_eq!(
            mapped_source, source,
            "{backend:?}: read-only source modified"
        );
        assert_eq!(
            mapped_target, expected_target,
            "{backend:?}: exact-width stores/neighbor guards"
        );
    }
}

#[test]
fn ordinary_simd_byte_half_load_store_all_address_modes_and_guards() {
    check(false);
}

#[test]
fn ordinary_simd_byte_half_page_end_accesses_have_exact_width() {
    check(true);
}
