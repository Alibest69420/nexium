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
    bytes: usize,
    q: bool,
    scalar: bool,
    rn: u32,
    rm: u32,
    rd: u32,
    n: [u8; 16],
    m: [u8; 16],
    chain: bool,
    literal: Option<[u8; 16]>,
}

impl Case {
    fn word(self) -> u32 {
        assert!(matches!(self.bytes, 1 | 2 | 4 | 8));
        assert!(if self.scalar {
            self.bytes == 8
        } else {
            self.q || self.bytes != 8
        });
        let base = if self.scalar {
            0x5ee0_8c00
        } else {
            0x0e20_8c00 | (u32::from(self.q) << 30) | (self.bytes.trailing_zeros() << 22)
        };
        base | (self.rm << 16) | (self.rn << 5) | self.rd
    }

    fn model(self, vectors: &mut [[u8; 16]; 32]) {
        let n = vectors[self.rn as usize];
        let m = vectors[self.rm as usize];
        let lanes = if self.scalar {
            1
        } else {
            (if self.q { 16 } else { 8 }) / self.bytes
        };
        let mut result = [0; 16];
        for lane in 0..lanes {
            let offset = lane * self.bytes;
            let intersects = (offset..offset + self.bytes).any(|index| n[index] & m[index] != 0);
            result[offset..offset + self.bytes].fill(if intersects { 0xff } else { 0 });
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
        output.copy_from_slice(&values[(lane + rotation) & 7].to_le_bytes()[..bytes]);
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
            let offset = INPUTS + slot * 48;
            initial[offset..offset + 16].copy_from_slice(&seed(case.rd as usize));
            initial[offset + 16..offset + 32].copy_from_slice(&case.n);
            initial[offset + 32..offset + 48].copy_from_slice(&case.m);
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
            vectors[case.rn as usize] = case.n;
            program.push(0x3dc0_01a0 | (((first + 2) as u32) << 10) | case.rm);
            vectors[case.rm as usize] = case.m;
            if case.chain {
                assert_eq!(case.word(), 0x0e60_8c00);
                program.push(0x2f08_a400);
                let source = vectors[0];
                for lane in 0..8 {
                    vectors[0][lane * 2..lane * 2 + 2]
                        .copy_from_slice(&(source[lane] as u16).to_le_bytes());
                }
            }
            program.push(case.word());
            case.model(&mut vectors);
            if case.chain {
                program.push(0x0f10_a400);
                let source = vectors[0];
                for lane in 0..4 {
                    let value =
                        i16::from_le_bytes(source[lane * 2..lane * 2 + 2].try_into().unwrap())
                            as i32;
                    vectors[0][lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
                }
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
                match cpu.run(100_000).expect("SIMD bit-test execution").event {
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

fn arrangements() -> [(usize, bool, bool); 8] {
    [
        (1, false, false),
        (1, true, false),
        (2, false, false),
        (2, true, false),
        (4, false, false),
        (4, true, false),
        (8, true, false),
        (8, true, true),
    ]
}

#[test]
fn all_legal_vector_and_scalar_formats_distinguish_intersection_from_equality() {
    let mut cases = Vec::new();
    for (bytes, q, scalar) in arrangements() {
        for (rd, rn, rm) in [
            (0, 1, 2),
            (1, 1, 2),
            (2, 1, 2),
            (0, 1, 1),
            (31, 31, 3),
            (0, 1, 31),
            (31, 31, 31),
        ] {
            for rotation in 0..8 {
                let n = pattern(bytes, rotation);
                let m = std::array::from_fn(|index| !n[index]);
                cases.push(Case {
                    bytes,
                    q,
                    scalar,
                    rd,
                    rn,
                    rm,
                    n,
                    m,
                    chain: false,
                    literal: None,
                });
            }
            for bit in 0..bytes * 8 {
                let mut n = [0; 16];
                let mut m = [0; 16];
                for lane in 0..16 / bytes {
                    let n_bits = 1u64 << bit;
                    let m_bits = if (lane + bit) % 3 == 0 {
                        n_bits
                    } else {
                        1u64 << ((bit + 1) % (bytes * 8))
                    };
                    n[lane * bytes..lane * bytes + bytes]
                        .copy_from_slice(&n_bits.to_le_bytes()[..bytes]);
                    m[lane * bytes..lane * bytes + bytes]
                        .copy_from_slice(&m_bits.to_le_bytes()[..bytes]);
                }
                cases.push(Case {
                    bytes,
                    q,
                    scalar,
                    rd,
                    rn,
                    rm,
                    n,
                    m,
                    chain: false,
                    literal: None,
                });
            }
        }
    }
    assert_eq!(cases.len(), 2128);
    check(&cases, 0x93);
}

#[test]
fn all_source_registers_aliases_upper_clearing_and_sticky_status_are_preserved() {
    let mut cases = Vec::new();
    for (bytes, q, scalar) in arrangements() {
        for rn in 0..32 {
            let rm = (rn + 17) & 31;
            for rd in [rn, rm, 31] {
                cases.push(Case {
                    bytes,
                    q,
                    scalar,
                    rn,
                    rm,
                    rd,
                    n: seed(rn as usize),
                    m: seed(rm as usize),
                    chain: false,
                    literal: None,
                });
            }
        }
    }
    assert_eq!(cases.len(), 768);
    check(&cases, 0x0800_0093);
}

#[test]
fn exact_wonder_uxtl_cmtst_sxtl_alias_chain_matches_all_nonzero_byte_masks() {
    let mut cases = Vec::new();
    for mask in 0u32..16 {
        let mut source = [0xd3; 16];
        let mut literal = [0; 16];
        for lane in 0..4 {
            let nonzero = mask & (1 << lane) != 0;
            source[lane] = if nonzero {
                [1, 0x80, 0xff, 0x55][lane]
            } else {
                0
            };
            literal[lane * 4..lane * 4 + 4].fill(if nonzero { 0xff } else { 0 });
        }
        let case = Case {
            bytes: 2,
            q: false,
            scalar: false,
            rn: 0,
            rm: 0,
            rd: 0,
            n: source,
            m: source,
            chain: true,
            literal: Some(literal),
        };
        assert_eq!(case.word(), 0x0e60_8c00, "Wonder CMTST V0.4H,V0.4H,V0.4H");
        cases.push(case);
    }
    check(&cases, 0x0800_0093);
}
