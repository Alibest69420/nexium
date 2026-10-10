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

fn instruction(unsigned: bool, q: bool, bytes: usize, rd: u32, rn: u32, rm: u32) -> u32 {
    assert!(q || bytes != 8, "the 1D arrangement is reserved");
    0x0e20_4400
        | (u32::from(unsigned) << 29)
        | (u32::from(q) << 30)
        | (bytes.trailing_zeros() << 22)
        | (rm << 16)
        | (rn << 5)
        | rd
}

fn mask(bits: u32) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn shifted(value: u64, raw_count: u64, bits: u32, unsigned: bool) -> u64 {
    let value = value & mask(bits);
    let count = raw_count as u8 as i8;
    if count >= 0 {
        let amount = count as u32;
        if amount >= bits {
            0
        } else {
            (value << amount) & mask(bits)
        }
    } else {
        let amount = -(count as i16) as u32;
        if unsigned {
            if amount >= bits { 0 } else { value >> amount }
        } else if amount >= bits {
            if value & (1u64 << (bits - 1)) != 0 {
                mask(bits)
            } else {
                0
            }
        } else {
            let signed = ((value << (64 - bits)) as i64) >> (64 - bits);
            ((signed >> amount) as u64) & mask(bits)
        }
    }
}

fn run(backend: CpuBackendKind, program: &[u32], initial: &[u8], slots: usize) -> Vec<u8> {
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
        .expect("SIMD variable-shift execution")
        .event;
    assert!(
        matches!(event, CpuEvent::Svc(0x7e)),
        "{backend:?}: {event:?}"
    );
    for reg in 0..31 {
        assert_eq!(
            cpu.get_register(reg),
            match reg {
                1 => DATA + (INPUTS + slots * 48) as u64,
                2 => DATA + (RESULTS + slots * 128) as u64,
                3 => DATA,
                4 => DATA + REGISTERS as u64,
                _ => CANARY,
            },
            "{backend:?}: preserved X{reg}"
        );
    }
    assert_eq!(cpu.get_sp(), INITIAL_SP, "{backend:?}: SP");
    drop(cpu);
    data
}

fn check(unsigned: bool) {
    let mut initial = vec![0xa5; 0x40000];
    for register in 0..32 {
        for byte in 0..16 {
            initial[register * 16 + byte] = (register * 37 + byte * 11 + 3) as u8;
        }
    }
    let mut expected = initial.clone();
    let mut vectors = initial[..32 * 16].to_vec();
    let mut program = Vec::new();
    for register in 0..32 {
        program.push(0x3dc0_0060 | ((register as u32) << 10) | register as u32);
    }
    let mut records = Vec::new();
    for (q, bytes) in [
        (false, 1),
        (true, 1),
        (false, 2),
        (true, 2),
        (false, 4),
        (true, 4),
        (true, 8),
    ] {
        let bits = (bytes * 8) as u32;
        let counts = [
            -128,
            -(bits as i16) - 1,
            -(bits as i16),
            -(bits as i16) + 1,
            -2,
            -1,
            0,
            1,
            2,
            bits as i16 - 1,
            bits as i16,
            bits as i16 + 1,
            127,
        ];
        let values = [
            0,
            1,
            mask(bits),
            1u64 << (bits - 1),
            (1u64 << (bits - 1)) - 1,
            (1u64 << (bits - 1)) | 1,
            0xaaaa_5555_aaaa_5555 & mask(bits),
            0x0123_4567_89ab_cdef & mask(bits),
        ];
        for (rd, rn, rm) in [
            (0, 1, 3),
            (2, 2, 3),
            (3, 1, 3),
            (0, 1, 1),
            (31, 31, 3),
            (0, 1, 31),
            (31, 31, 31),
        ] {
            for index in 0..counts.len() {
                let slot = records.len();
                let input_offset = INPUTS + slot * 48;
                let result_offset = RESULTS + slot * 128;
                let mut n = [0; 16];
                let mut m = [0; 16];
                let mut output = [0; 16];
                n[8..].fill(0x97);
                m[8..].fill(0x6b);
                for lane in 0..16 / bytes {
                    let count = counts[(index + lane * 3) % counts.len()] as i8;
                    let mut value = values[(index + lane * 5) % values.len()];
                    let mut raw_count =
                        (0x80ff_1234_5678_9a00 & mask(bits) & !0xff) | count as u8 as u64;
                    if rn == rm {
                        value = (value & !0xff) | count as u8 as u64;
                        raw_count = value;
                    }
                    n[lane * bytes..(lane + 1) * bytes]
                        .copy_from_slice(&value.to_le_bytes()[..bytes]);
                    m[lane * bytes..(lane + 1) * bytes]
                        .copy_from_slice(&raw_count.to_le_bytes()[..bytes]);
                    if lane < (if q { 16 } else { 8 }) / bytes {
                        let result = shifted(value, raw_count, bits, unsigned);
                        output[lane * bytes..(lane + 1) * bytes]
                            .copy_from_slice(&result.to_le_bytes()[..bytes]);
                    }
                }
                initial[input_offset..input_offset + 16].copy_from_slice(&n);
                initial[input_offset + 16..input_offset + 32].copy_from_slice(&m);
                initial[input_offset + 32..input_offset + 48].fill(0x3c);
                expected[input_offset..input_offset + 48]
                    .copy_from_slice(&initial[input_offset..input_offset + 48]);
                let opcode = instruction(unsigned, q, bytes, rd, rn, rm);
                program.extend([
                    0x3dc0_0820 | rd,
                    0x3dc0_0020 | rn,
                    0x3dc0_0420 | rm,
                    opcode,
                    0x3d80_0040 | rd,
                    0x3d80_0840 | rn,
                    0x3d80_1040 | rm,
                    0x9100_c021,
                    0x9102_0042,
                ]);
                vectors[rd as usize * 16..rd as usize * 16 + 16].fill(0x3c);
                vectors[rn as usize * 16..rn as usize * 16 + 16].copy_from_slice(&n);
                vectors[rm as usize * 16..rm as usize * 16 + 16].copy_from_slice(&m);
                vectors[rd as usize * 16..rd as usize * 16 + 16].copy_from_slice(&output);
                expected[result_offset..result_offset + 16].copy_from_slice(&output);
                expected[result_offset + 32..result_offset + 48].copy_from_slice(if rd == rn {
                    &output
                } else {
                    &n
                });
                expected[result_offset + 64..result_offset + 80].copy_from_slice(if rd == rm {
                    &output
                } else {
                    &m
                });
                records.push((result_offset, opcode, bytes, q, rd, rn, rm, index));
            }
        }
    }
    let slots = records.len();
    assert_eq!(slots, 637);
    assert!(INPUTS + slots * 48 < RESULTS && RESULTS + slots * 128 < REGISTERS);
    assert_eq!(
        instruction(true, false, 4, 2, 2, 3),
        0x2ea3_4442,
        "Wonder USHL V2.2S,V2.2S,V3.2S opcode"
    );
    for register in 0..32 {
        program.push(0x3d80_0080 | ((register as u32) << 10) | register as u32);
    }
    expected[REGISTERS..REGISTERS + 32 * 16].copy_from_slice(&vectors);
    program.push(0xd400_0fc1);
    let reference = run(CpuBackendKind::Dynarmic, &program, &initial, slots);
    let actual = run(CpuBackendKind::Rustarmic, &program, &initial, slots);
    for (offset, opcode, bytes, q, rd, rn, rm, index) in records {
        for (backend, memory) in [("Dynarmic", &reference), ("Rustarmic", &actual)] {
            assert_eq!(
                &memory[offset..offset + 16],
                &expected[offset..offset + 16],
                "{backend}: opcode={opcode:#010x} bytes={bytes} Q={q} Rd={rd} Rn={rn} Rm={rm} sample={index}"
            );
        }
    }
    assert_eq!(
        reference, expected,
        "reference source preservation, Q=0 upper-zero and guards"
    );
    assert_eq!(
        actual, expected,
        "Rustarmic source preservation, Q=0 upper-zero and guards"
    );
    assert_eq!(
        actual, reference,
        "SIMD signed-low-byte variable-shift parity"
    );
}

#[test]
fn ushl_all_legal_arrangements_match_dynarmic() {
    check(true);
}
#[test]
fn sshl_all_legal_arrangements_match_dynarmic() {
    check(false);
}
