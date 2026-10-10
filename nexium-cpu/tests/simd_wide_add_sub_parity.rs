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
    subtract: bool,
    bytes: usize,
    upper: bool,
    rn: u32,
    rm: u32,
    rd: u32,
    wide: [u8; 16],
    narrow: [u8; 16],
    pair: bool,
    float_prefix: bool,
    literal: Option<[u8; 16]>,
}

impl Case {
    fn word(self) -> u32 {
        assert!(matches!(self.bytes, 1 | 2 | 4));
        0x0e20_1000
            | (u32::from(self.unsigned) << 29)
            | (u32::from(self.subtract) << 13)
            | (u32::from(self.upper) << 30)
            | (self.bytes.trailing_zeros() << 22)
            | (self.rm << 16)
            | (self.rn << 5)
            | self.rd
    }

    fn model(self, vectors: &mut [[u8; 16]; 32]) {
        let wide = vectors[self.rn as usize];
        let narrow = vectors[self.rm as usize];
        let width = self.bytes * 8;
        let wide_bytes = self.bytes * 2;
        let modulus = 1i128 << (width * 2);
        let start = if self.upper { 8 } else { 0 };
        let mut result = [0; 16];
        for lane in 0..8 / self.bytes {
            let mut n = [0; 8];
            let mut m = [0; 8];
            let n_offset = lane * wide_bytes;
            let m_offset = start + lane * self.bytes;
            n[..wide_bytes].copy_from_slice(&wide[n_offset..n_offset + wide_bytes]);
            m[..self.bytes].copy_from_slice(&narrow[m_offset..m_offset + self.bytes]);
            let n = i128::from(u64::from_le_bytes(n));
            let m = i128::from(u64::from_le_bytes(m));
            let extended = if !self.unsigned && m & (1i128 << (width - 1)) != 0 {
                m - (1i128 << width)
            } else {
                m
            };
            let sum = if self.subtract {
                n - extended
            } else {
                n + extended
            };
            let bits = sum.rem_euclid(modulus) as u64;
            result[n_offset..n_offset + wide_bytes]
                .copy_from_slice(&bits.to_le_bytes()[..wide_bytes]);
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
        let scale = words([
            0.5f32.to_bits(),
            2.0f32.to_bits(),
            (-0.25f32).to_bits(),
            4.0f32.to_bits(),
        ]);
        initial[0x600..0x610].copy_from_slice(&scale);
        for (register, vector) in vectors.iter().enumerate() {
            initial[register * 16..register * 16 + 16].copy_from_slice(vector);
        }
        for (slot, case) in cases.iter().enumerate() {
            let offset = INPUTS + slot * 48;
            initial[offset..offset + 16].copy_from_slice(&seed(case.rd as usize));
            initial[offset + 16..offset + 32].copy_from_slice(&case.wide);
            initial[offset + 32..offset + 48].copy_from_slice(&case.narrow);
        }
        let mut expected = initial.clone();
        let mut program = vec![0xd51b_4410, 0xd51b_4433, 0xd51b_4214];
        for register in 0..32u32 {
            program.push(0x3dc0_01a0 | (register << 10) | register);
        }
        program.extend([0x9100_0484, 0x9100_04a5, 0x9100_04c6, 0x9100_04e7]);
        for (slot, &case) in cases.iter().enumerate() {
            let first = INPUTS / 16 + slot * 3;
            program.push(0x3dc0_01a0 | ((first as u32) << 10) | case.rd);
            vectors[case.rd as usize] = seed(case.rd as usize);
            program.push(0x3dc0_01a0 | (((first + 1) as u32) << 10) | case.rn);
            vectors[case.rn as usize] = case.wide;
            program.push(0x3dc0_01a0 | (((first + 2) as u32) << 10) | case.rm);
            vectors[case.rm as usize] = case.narrow;
            program.push(case.word());
            case.model(&mut vectors);
            if case.pair {
                let next = Case {
                    rn: 1,
                    rm: 0,
                    rd: 0,
                    pair: false,
                    ..case
                };
                assert_eq!(next.word(), 0x2e60_1020);
                program.push(next.word());
                next.model(&mut vectors);
            }
            if case.float_prefix {
                assert!(matches!(case.word(), 0x2e61_1000 | 0x2e60_1020));
                let offset = if case.rn == 0 { 0 } else { 16 };
                program.extend([
                    0x3dc0_0481,
                    0x4e21_d800,
                    0x6e20_dc20,
                    0x3d80_00a0 | ((offset / 16) << 10),
                ]);
                let integer = vectors[0];
                vectors[1] = scale;
                for lane in 0..4 {
                    let start = lane * 4;
                    let value = i32::from_le_bytes(integer[start..start + 4].try_into().unwrap());
                    let factor = f32::from_bits(u32::from_le_bytes(
                        scale[start..start + 4].try_into().unwrap(),
                    ));
                    vectors[0][start..start + 4]
                        .copy_from_slice(&((value as f32) * factor).to_bits().to_le_bytes());
                }
                let output = 0x640 + offset as usize;
                expected[output..output + 16].copy_from_slice(&vectors[0]);
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
        assert!(INPUTS + cases.len() * 48 <= OUTPUTS);
        assert!(OUTPUTS + cases.len() * 512 <= STATUS);
        assert!(STATUS + cases.len() * 24 <= initial.len());
        program.extend([0x8b05_008a, 0x8b07_00cb, 0xd400_0fc1]);
        let mut encoded = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
        for (index, word) in program.iter().enumerate() {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut initial_gprs = [CANARY; 31];
        initial_gprs[4..8].copy_from_slice(&[10, 20, 30, 40]);
        if cases.iter().any(|case| case.float_prefix) {
            initial_gprs[4] = DATA + 0x5ef;
            initial_gprs[5] = DATA + 0x63f;
        }
        initial_gprs[13] = DATA;
        initial_gprs[14] = DATA + OUTPUTS as u64;
        initial_gprs[15] = DATA + STATUS as u64;
        initial_gprs[16] = FPCR;
        initial_gprs[19] = fpsr;
        initial_gprs[20] = NZCV;
        let mut expected_gprs = initial_gprs;
        for value in &mut expected_gprs[4..8] {
            *value += 1;
        }
        expected_gprs[10] = expected_gprs[4] + expected_gprs[5];
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
                match cpu
                    .run(100_000)
                    .expect("widening add/subtract execution")
                    .event
                {
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
fn all_twenty_four_forms_boundaries_wraps_source_halves_and_aliases_match() {
    let mut cases = Vec::new();
    for unsigned in [false, true] {
        for subtract in [false, true] {
            for bytes in [1, 2, 4] {
                for upper in [false, true] {
                    for (rd, rn, rm) in [
                        (0, 1, 2),
                        (1, 1, 2),
                        (2, 1, 2),
                        (0, 1, 1),
                        (31, 31, 3),
                        (0, 1, 31),
                        (31, 31, 31),
                        (31, 0, 1),
                    ] {
                        for rotation in 0..8 {
                            cases.push(Case {
                                unsigned,
                                subtract,
                                bytes,
                                upper,
                                rn,
                                rm,
                                rd,
                                wide: pattern(bytes * 2, rotation),
                                narrow: pattern(bytes, rotation + 3),
                                pair: false,
                                float_prefix: false,
                                literal: None,
                            });
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases.len(), 1536);
    check(&cases, 0x93);
}

#[test]
fn every_source_register_and_destination_alias_preserves_full_vectors_and_sticky_flags() {
    let mut cases = Vec::new();
    for unsigned in [false, true] {
        for subtract in [false, true] {
            for bytes in [1, 2, 4] {
                for upper in [false, true] {
                    for rn in 0..32 {
                        let rm = (rn + 17) & 31;
                        for rd in [rn, rm, 31] {
                            cases.push(Case {
                                unsigned,
                                subtract,
                                bytes,
                                upper,
                                rn,
                                rm,
                                rd,
                                wide: seed(rn as usize),
                                narrow: seed(rm as usize),
                                pair: false,
                                float_prefix: false,
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

fn words(values: [u32; 4]) -> [u8; 16] {
    let mut vector = [0; 16];
    for (lane, value) in values.into_iter().enumerate() {
        vector[lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    vector
}

#[test]
fn exact_wonder_uaddw_words_and_chained_destination_source_alias_match_literals() {
    let wide = words([0xffff_ffff, 0x7fff_ffff, 0x8000_0000, 1]);
    let mut narrow = [0x6b; 16];
    for (lane, value) in [0xffffu16, 0x8000, 1, 0xffff].into_iter().enumerate() {
        narrow[lane * 2..lane * 2 + 2].copy_from_slice(&value.to_le_bytes());
    }
    let literal = words([0x0000_fffe, 0x8000_7fff, 0x8000_0001, 0x0001_0000]);
    let actual_word = Case {
        unsigned: true,
        subtract: false,
        bytes: 2,
        upper: false,
        rn: 0,
        rm: 1,
        rd: 0,
        wide,
        narrow,
        pair: false,
        float_prefix: false,
        literal: Some(literal),
    };
    assert_eq!(
        actual_word.word(),
        0x2e61_1000,
        "Wonder UADDW V0.4S,V0.4S,V1.4H"
    );
    let nearby_word = Case {
        rn: 1,
        rm: 0,
        ..actual_word
    };
    assert_eq!(
        nearby_word.word(),
        0x2e60_1020,
        "Wonder UADDW V0.4S,V1.4S,V0.4H"
    );
    let pair = Case {
        pair: true,
        literal: Some(words([0x8001_fffd, 0xffff_0001, 0x6b6b_eb6a, 0x6b6b_eb6b])),
        ..actual_word
    };
    let mut cases = vec![actual_word, nearby_word, pair];
    for (unsigned, subtract, expected) in [
        (
            true,
            false,
            [0x0000_fffe, 0x8000_7fff, 0x8000_0001, 0x0001_0000],
        ),
        (false, false, [0xffff_fffe, 0x7fff_7fff, 0x8000_0001, 0]),
        (
            true,
            true,
            [0xffff_0000, 0x7fff_7fff, 0x7fff_ffff, 0xffff_0002],
        ),
        (false, true, [0, 0x8000_7fff, 0x7fff_ffff, 2]),
    ] {
        for upper in [false, true] {
            let mut selected = [0x93; 16];
            let start = if upper { 8 } else { 0 };
            selected[start..start + 8].copy_from_slice(&narrow[..8]);
            cases.push(Case {
                unsigned,
                subtract,
                bytes: 2,
                upper,
                rn: 31,
                rm: 0,
                rd: 31,
                wide,
                narrow: selected,
                pair: false,
                float_prefix: false,
                literal: Some(words(expected)),
            });
        }
    }
    check(&cases, 0x0800_0093);
}

#[test]
fn exact_wonder_uaddw_load_scale_scvtf_fmul_store_prefixes_match_finite_literals() {
    let wide = words([1, 10, 100, (-256i32) as u32]);
    let mut narrow = [0xe7; 16];
    for (lane, value) in [3u16, 5, 7, 9].into_iter().enumerate() {
        narrow[lane * 2..lane * 2 + 2].copy_from_slice(&value.to_le_bytes());
    }
    let literal = words([
        2.0f32.to_bits(),
        30.0f32.to_bits(),
        (-26.75f32).to_bits(),
        (-988.0f32).to_bits(),
    ]);
    let first = Case {
        unsigned: true,
        subtract: false,
        bytes: 2,
        upper: false,
        rn: 0,
        rm: 1,
        rd: 0,
        wide,
        narrow,
        pair: false,
        float_prefix: true,
        literal: Some(literal),
    };
    assert_eq!(first.word(), 0x2e61_1000);
    let second = Case {
        rn: 1,
        rm: 0,
        ..first
    };
    assert_eq!(second.word(), 0x2e60_1020);
    check(&[first, second], 0x0800_0093);
}
