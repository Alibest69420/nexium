#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const OUTPUTS: usize = 0x1000;
const STATUS: usize = 0x9000;
const CANARY: u64 = 0xfade_1234_5678_9abc;
const FPCR: u64 = 0x0300_0000;
const FPSR: u64 = 0x0800_0093;

#[derive(Clone, Copy, Debug)]
struct Case {
    size: u32,
    lane: u32,
    rn: u32,
    rd: u32,
}

impl Case {
    fn word(self) -> u32 {
        let imm5 = (1 << self.size) | (self.lane << (self.size + 1));
        0x5e00_0400 | (imm5 << 16) | (self.rn << 5) | self.rd
    }
}

fn seed(register: usize) -> [u8; 16] {
    std::array::from_fn(|byte| (register * 37 + byte * 11 + 0x93) as u8)
}

fn check(cases: &[Case]) {
    let mut backends: Vec<_> = [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic]
        .into_iter()
        .map(|kind| (kind, Cpu::new(kind).expect("CPU initialization")))
        .collect();
    for cases in cases.chunks(60) {
        let mut vectors: [[u8; 16]; 32] = std::array::from_fn(seed);
        let mut initial = vec![0xa5; 0xa000];
        for (register, value) in vectors.iter().enumerate() {
            initial[register * 16..register * 16 + 16].copy_from_slice(value);
        }
        let mut expected = initial.clone();
        let mut program = vec![0xd51b_4404, 0xd51b_4427];
        for register in 0..32u32 {
            program.push(0x3dc0_01a0 | (register << 10) | register);
        }
        for (slot, &case) in cases.iter().enumerate() {
            for register in [case.rn, case.rd] {
                program.push(0x3dc0_01a0 | (register << 10) | register);
                vectors[register as usize] = seed(register as usize);
            }
            program.push(case.word());
            let bytes = 1usize << case.size;
            let source = vectors[case.rn as usize];
            let mut result = [0; 16];
            let start = case.lane as usize * bytes;
            result[..bytes].copy_from_slice(&source[start..start + bytes]);
            vectors[case.rd as usize] = result;
            for (register, value) in vectors.iter().enumerate() {
                let offset = slot * 32 + register;
                program.push(0x3d80_01c0 | ((offset as u32) << 10) | register as u32);
                expected[OUTPUTS + offset * 16..OUTPUTS + offset * 16 + 16].copy_from_slice(value);
            }
            program.extend([0xd53b_4431, 0xd53b_4412]);
            program.push(0xf900_01f1 | ((slot as u32 * 2) << 10));
            program.push(0xf900_01f2 | ((slot as u32 * 2 + 1) << 10));
            let status = STATUS + slot * 16;
            expected[status..status + 8].copy_from_slice(&FPSR.to_le_bytes());
            expected[status + 8..status + 16].copy_from_slice(&FPCR.to_le_bytes());
        }
        program.push(0xd400_0fc1);
        let mut encoded = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
        for (index, word) in program.iter().enumerate() {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut expected_gprs = [CANARY; 31];
        expected_gprs[4] = FPCR;
        expected_gprs[7] = FPSR;
        expected_gprs[13] = DATA;
        expected_gprs[14] = DATA + OUTPUTS as u64;
        expected_gprs[15] = DATA + STATUS as u64;
        expected_gprs[17] = FPSR;
        expected_gprs[18] = FPCR;
        for (backend, cpu) in &mut backends {
            let mut code = encoded.clone();
            let mut actual = initial.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, actual.len() as u64, Perm::RW, actual.as_mut_ptr())
                    .unwrap();
            }
            for register in 0..31 {
                cpu.set_register(register, CANARY);
            }
            for register in [4, 7, 13, 14, 15] {
                cpu.set_register(register, expected_gprs[register as usize]);
            }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let event = cpu.run(100_000).expect("scalar DUP execution").event;
            assert!(
                matches!(event, CpuEvent::Svc(0x7e)),
                "{backend:?}: {event:?}"
            );
            for (slot, case) in cases.iter().enumerate() {
                for register in 0..32 {
                    let offset = OUTPUTS + (slot * 32 + register) * 16;
                    assert_eq!(
                        &actual[offset..offset + 16],
                        &expected[offset..offset + 16],
                        "{backend:?}: {case:?} V{register}"
                    );
                }
                let status = STATUS + slot * 16;
                assert_eq!(
                    &actual[status..status + 16],
                    &expected[status..status + 16],
                    "{backend:?}: {case:?} FPCR/FPSR unchanged"
                );
            }
            for register in 0..31 {
                assert_eq!(
                    cpu.get_register(register),
                    expected_gprs[register as usize],
                    "{backend:?}: X{register}"
                );
            }
            assert_eq!(cpu.get_sp(), CANARY);
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, actual.len() as u64).unwrap();
            }
            assert_eq!(actual, expected, "{backend:?}: all source/data guards");
            assert_eq!(code, encoded, "{backend:?}: code unchanged");
        }
    }
}

#[test]
fn every_scalar_dup_width_lane_register_and_alias_matches_dynarmic() {
    let mut cases = Vec::new();
    for size in 0..4 {
        for lane in 0..16 >> size {
            for rn in 0..32 {
                for rd in [rn, (rn + 13) & 31, 31] {
                    cases.push(Case { size, lane, rn, rd });
                }
            }
        }
    }
    check(&cases);
}

#[test]
fn exact_wonder_scalar_mov_aliases_match_dynarmic() {
    let case = Case {
        size: 2,
        lane: 2,
        rn: 2,
        rd: 1,
    };
    assert_eq!(case.word(), 0x5e14_0441);
    check(&[
        case,
        Case {
            size: 2,
            lane: 1,
            rn: 4,
            rd: 6,
        },
        Case {
            size: 2,
            lane: 1,
            rn: 3,
            rd: 7,
        },
        Case {
            size: 2,
            lane: 2,
            rn: 0,
            rd: 17,
        },
        Case {
            size: 2,
            lane: 2,
            rn: 3,
            rd: 2,
        },
        Case {
            size: 2,
            lane: 1,
            rn: 0,
            rd: 5,
        },
        Case {
            size: 2,
            lane: 2,
            rn: 4,
            rd: 4,
        },
    ]);
}
