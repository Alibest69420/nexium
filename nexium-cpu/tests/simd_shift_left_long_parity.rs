#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const INPUTS: usize = 0x800;
const OUTPUTS: usize = 0x2000;
const STATUS: usize = 0xb000;
const CANARY: u64 = 0xfade_1234_5678_9abc;
const FPCR: u64 = 0x0300_0000;
const NZCV: u64 = 0xa000_0000;

#[derive(Clone, Copy, Debug)]
struct Case {
    unsigned: bool,
    bytes: usize,
    upper: bool,
    shift: u32,
    rn: u32,
    rd: u32,
    source: [u8; 16],
    accumulator: [u8; 16],
    repetitions: usize,
    literal: Option<[u8; 16]>,
}

impl Case {
    fn word(self) -> u32 {
        let bits = (self.bytes * 8) as u32;
        assert!(matches!(self.bytes, 1 | 2 | 4) && self.shift < bits);
        0x0f00_a400
            | (u32::from(self.unsigned) << 29)
            | (u32::from(self.upper) << 30)
            | ((bits + self.shift) << 16)
            | (self.rn << 5)
            | self.rd
    }

    fn model(self, vectors: &mut [[u8; 16]; 32]) {
        let source = vectors[self.rn as usize];
        let bits = self.bytes * 8;
        let destination_bytes = self.bytes * 2;
        let modulus = 1i128 << (bits * 2);
        let start = if self.upper { 8 } else { 0 };
        let mut result = [0; 16];
        for lane in 0..8 / self.bytes {
            let offset = start + lane * self.bytes;
            let mut raw = [0; 8];
            raw[..self.bytes].copy_from_slice(&source[offset..offset + self.bytes]);
            let raw = i128::from(u64::from_le_bytes(raw));
            let value = if !self.unsigned && raw & (1i128 << (bits - 1)) != 0 {
                raw - (1i128 << bits)
            } else {
                raw
            };
            let widened = (value << self.shift).rem_euclid(modulus) as u64;
            let out = lane * destination_bytes;
            result[out..out + destination_bytes]
                .copy_from_slice(&widened.to_le_bytes()[..destination_bytes]);
        }
        vectors[self.rd as usize] = result;
    }
}

fn seed(register: usize) -> [u8; 16] {
    std::array::from_fn(|byte| (register * 37 + byte * 11 + 0x93) as u8)
}

fn pattern(bytes: usize, rotation: usize) -> [u8; 16] {
    let bits = bytes * 8;
    let mask = u64::MAX >> (64 - bits);
    let sign = 1u64 << (bits - 1);
    let values = [
        0,
        1,
        mask,
        sign,
        sign - 1,
        sign + 1,
        0xaaaa_5555_aaaa_5555 & mask,
        0x0123_4567_89ab_cdef & mask,
    ];
    let mut result = [0; 16];
    for (lane, output) in result.chunks_exact_mut(bytes).enumerate() {
        output.copy_from_slice(
            &values[(lane + rotation + if lane >= 8 / bytes { 3 } else { 0 }) & 7].to_le_bytes()
                [..bytes],
        );
    }
    result
}

fn check(cases: &[Case], fpsr: u64) {
    let mut backends = [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic]
        .map(|kind| (kind, Cpu::new(kind).expect("CPU initialization")));
    for cases in cases.chunks(60) {
        let mut vectors: [[u8; 16]; 32] = std::array::from_fn(seed);
        let mut initial = vec![0xa5; 0x10000];
        for (register, vector) in vectors.iter().enumerate() {
            initial[register * 16..register * 16 + 16].copy_from_slice(vector);
        }
        for (slot, case) in cases.iter().enumerate() {
            let offset = INPUTS + slot * 32;
            initial[offset..offset + 16].copy_from_slice(&case.accumulator);
            initial[offset + 16..offset + 32].copy_from_slice(&case.source);
        }
        let mut expected = initial.clone();
        let mut program = vec![0xd51b_4410, 0xd51b_4433, 0xd51b_4214];
        for register in 0..32u32 {
            program.push(0x3dc0_01a0 | (register << 10) | register);
        }
        program.extend([0x9100_0484, 0x9100_04a5, 0x9100_04c6, 0x9100_04e7]);
        for (slot, &case) in cases.iter().enumerate() {
            let first = INPUTS / 16 + slot * 2;
            program.push(0x3dc0_01a0 | ((first as u32) << 10) | case.rd);
            vectors[case.rd as usize] = case.accumulator;
            program.push(0x3dc0_01a0 | (((first + 1) as u32) << 10) | case.rn);
            vectors[case.rn as usize] = case.source;
            for _ in 0..case.repetitions {
                program.push(case.word());
                case.model(&mut vectors);
            }
            if let Some(literal) = case.literal {
                assert_eq!(
                    vectors[case.rd as usize], literal,
                    "independent oracle literal: {case:?}"
                );
            }
            for (register, vector) in vectors.iter().enumerate() {
                let index = slot * 32 + register;
                assert!(index < 4096);
                program.push(0x3d80_01c0 | ((index as u32) << 10) | register as u32);
                let offset = OUTPUTS + index * 16;
                expected[offset..offset + 16].copy_from_slice(vector);
            }
            program.extend([0xd53b_4431, 0xd53b_4412, 0xd53b_4215]);
            for (field, register) in [17u32, 18, 21].into_iter().enumerate() {
                program.push(0xf900_01e0 | (((slot * 3 + field) as u32) << 10) | register);
                let offset = STATUS + slot * 24 + field * 8;
                expected[offset..offset + 8]
                    .copy_from_slice(&[fpsr, FPCR, NZCV][field].to_le_bytes());
            }
        }
        assert!(INPUTS + cases.len() * 32 <= OUTPUTS);
        assert!(OUTPUTS + cases.len() * 512 <= STATUS);
        assert!(STATUS + cases.len() * 24 <= initial.len());
        program.extend([0x8b05_008a, 0x8b07_00cb, 0xd400_0fc1]);
        let mut encoded = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
        for (index, word) in program.iter().enumerate() {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut initial_gprs = [CANARY; 31];
        initial_gprs[4..8].copy_from_slice(&[10, 20, 30, 40]);
        initial_gprs[13] = DATA;
        initial_gprs[14] = DATA + OUTPUTS as u64;
        initial_gprs[15] = DATA + STATUS as u64;
        initial_gprs[16] = FPCR;
        initial_gprs[19] = fpsr;
        initial_gprs[20] = NZCV;
        let mut expected_gprs = initial_gprs;
        expected_gprs[4..8].copy_from_slice(&[11, 21, 31, 41]);
        expected_gprs[10] = 32;
        expected_gprs[11] = 72;
        expected_gprs[17] = fpsr;
        expected_gprs[18] = FPCR;
        expected_gprs[21] = NZCV;
        for (backend, cpu) in &mut backends {
            let mut code = encoded.clone();
            let mut actual = initial.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, actual.len() as u64, Perm::RW, actual.as_mut_ptr())
                    .unwrap();
            }
            for (register, &value) in initial_gprs.iter().enumerate() {
                cpu.set_register(register as u32, value);
            }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let mut reached_svc = false;
            for _ in 0..16 {
                match cpu.run(100_000).expect("SSHLL/USHLL execution").event {
                    CpuEvent::Svc(0x7e) => {
                        reached_svc = true;
                        break;
                    }
                    CpuEvent::Running => {}
                    event => panic!("{backend:?}: {event:?}, PC={:#x}", cpu.get_pc()),
                }
            }
            assert!(
                reached_svc,
                "{backend:?}: bounded execution did not reach SVC"
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
                let offset = STATUS + slot * 24;
                assert_eq!(
                    &actual[offset..offset + 24],
                    &expected[offset..offset + 24],
                    "{backend:?}: {case:?} sticky FPSR/FPCR/NZCV"
                );
            }
            for (register, &value) in expected_gprs.iter().enumerate() {
                assert_eq!(
                    cpu.get_register(register as u32),
                    value,
                    "{backend:?}: X{register}"
                );
            }
            assert_eq!(cpu.get_sp(), CANARY);
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, actual.len() as u64).unwrap();
            }
            assert_eq!(actual, expected, "{backend:?}: source/data/neighbor guards");
            assert_eq!(code, encoded, "{backend:?}: source code unchanged");
        }
    }
}

#[test]
fn every_widening_width_shift_and_source_half_matches_independent_integer_oracle() {
    let mut cases = Vec::new();
    for unsigned in [false, true] {
        for bytes in [1, 2, 4] {
            for upper in [false, true] {
                for shift in 0..(bytes * 8) as u32 {
                    for (rn, rd) in [(2, 1), (2, 2), (31, 31), (31, 0), (0, 31)] {
                        for rotation in [0, 3, 5] {
                            cases.push(Case {
                                unsigned,
                                bytes,
                                upper,
                                shift,
                                rn,
                                rd,
                                source: pattern(bytes, rotation + shift as usize),
                                accumulator: seed(rd as usize),
                                repetitions: 1,
                                literal: None,
                            });
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases.len(), 3360);
    check(&cases, 0x93);
}

#[test]
fn all_registers_aliases_sign_extension_and_half_capture_preserve_sticky_flags() {
    let mut cases = Vec::new();
    for unsigned in [false, true] {
        for bytes in [1, 2, 4] {
            for upper in [false, true] {
                for rn in 0..32 {
                    for rd in [rn, (rn + 13) & 31, 31] {
                        for shift in [0, (bytes * 8 - 1) as u32] {
                            cases.push(Case {
                                unsigned,
                                bytes,
                                upper,
                                shift,
                                rn,
                                rd,
                                source: seed(rn as usize),
                                accumulator: seed(rd as usize),
                                repetitions: 1,
                                literal: None,
                            });
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases.len(), 2304);
    check(&cases, 0x0800_0093);
}

#[test]
fn exact_wonder_uxtl_and_signed_widen_then_shift_match_hand_written_literals() {
    let mut source = [0xe7; 16];
    source[..8].copy_from_slice(&[0, 1, 0x7f, 0x80, 0xff, 0x55, 0xaa, 0xfe]);
    let mut literal = [0; 16];
    for (lane, value) in [0u16, 1, 0x7f, 0x80, 0xff, 0x55, 0xaa, 0xfe]
        .into_iter()
        .enumerate()
    {
        literal[lane * 2..lane * 2 + 2].copy_from_slice(&value.to_le_bytes());
    }
    let wonder = Case {
        unsigned: true,
        bytes: 1,
        upper: false,
        shift: 0,
        rn: 0,
        rd: 0,
        source,
        accumulator: [0xa7; 16],
        repetitions: 1,
        literal: Some(literal),
    };
    assert_eq!(
        wonder.word(),
        0x2f08_a400,
        "Wonder UXTL V0.8H,V0.8B at 0x91a5234"
    );
    let mut cases = vec![wonder];
    for upper in [false, true] {
        let mut source = [0x6b; 16];
        let start = if upper { 8 } else { 0 };
        source[start..start + 4].copy_from_slice(&(-1i32).to_le_bytes());
        source[start + 4..start + 8].copy_from_slice(&i32::MIN.to_le_bytes());
        let mut literal = [0; 16];
        literal[..8].copy_from_slice(&(-0x8000_0000i64).to_le_bytes());
        literal[8..].copy_from_slice(&(-0x4000_0000_0000_0000i64).to_le_bytes());
        cases.push(Case {
            unsigned: false,
            bytes: 4,
            upper,
            shift: 31,
            rn: 31,
            rd: 31,
            source,
            accumulator: [0xa7; 16],
            repetitions: 1,
            literal: Some(literal),
        });
        let mut unsigned_literal = [0; 16];
        unsigned_literal[..8].copy_from_slice(&0x7fff_ffff_8000_0000u64.to_le_bytes());
        unsigned_literal[8..].copy_from_slice(&0x4000_0000_0000_0000u64.to_le_bytes());
        cases.push(Case {
            unsigned: true,
            bytes: 4,
            upper,
            shift: 31,
            rn: 31,
            rd: 0,
            source,
            accumulator: [0xa7; 16],
            repetitions: 1,
            literal: Some(unsigned_literal),
        });
    }
    check(&cases, 0x0800_0093);
}
