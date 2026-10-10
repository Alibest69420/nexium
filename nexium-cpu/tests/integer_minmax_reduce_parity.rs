#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const OUTPUTS: usize = 0x2000;
const STATUS: usize = 0x9000;
const CANARY: u64 = 0xfade_1234_5678_9abc;
const FPCR: u64 = 0x0300_0000;
const FPSR: u64 = 0x0800_0093;
const NZCV: u64 = 0xa000_0000;

#[derive(Clone, Copy, Debug)]
struct Case {
    signed: bool,
    maximum: bool,
    size: u32,
    q: bool,
    rn: u32,
    rd: u32,
    source: [u8; 16],
    chain: bool,
}

impl Case {
    fn word(self) -> u32 {
        0x0e30_a800 | (u32::from(!self.signed) << 29)
            | (u32::from(!self.maximum) << 16) | (u32::from(self.q) << 30)
            | (self.size << 22) | (self.rn << 5) | self.rd
    }

    fn expected(self) -> [u8; 16] {
        let width = 1usize << self.size;
        let lane_bits = width * 8;
        let lanes = (if self.q { 16 } else { 8 }) / width;
        let mut selected = 0usize;
        let value = |lane: usize| {
            let mut bits = [0; 8];
            bits[..width].copy_from_slice(&self.source[lane * width..(lane + 1) * width]);
            let unsigned = i128::from(u64::from_le_bytes(bits));
            if self.signed && unsigned & (1i128 << (lane_bits - 1)) != 0 {
                unsigned - (1i128 << lane_bits)
            } else { unsigned }
        };
        for lane in 1..lanes {
            if if self.maximum { value(lane) > value(selected) }
               else { value(lane) < value(selected) } { selected = lane; }
        }
        let mut result = [0; 16];
        result[..width].copy_from_slice(&self.source[selected * width..(selected + 1) * width]);
        result
    }
}

fn seed(register: usize) -> [u8; 16] {
    std::array::from_fn(|byte| (register * 37 + byte * 11 + 0x93) as u8)
}

fn patterns(size: u32) -> Vec<[u8; 16]> {
    let bytes = 1usize << size;
    let sign = 1u64 << (bytes * 8 - 1);
    let mask = (1u64 << (bytes * 8)) - 1;
    let bounds = [0, 1, sign - 1, sign, sign + 1, mask - 1, mask, 7];
    let mut sources = vec![[0; 16], [0xff; 16]];
    for rotation in 0..8 {
        let mut source = [0; 16];
        for (lane, bits) in source.chunks_exact_mut(bytes).enumerate() {
            bits.copy_from_slice(&bounds[(lane + rotation) & 7].to_le_bytes()[..bytes]);
        }
        sources.push(source);
    }
    for selected in 0..16 / bytes {
        let mut source = [0; 16];
        for (lane, bits) in source.chunks_exact_mut(bytes).enumerate() {
            let value = if lane == selected { sign } else { 7 };
            bits.copy_from_slice(&value.to_le_bytes()[..bytes]);
        }
        sources.push(source);
    }
    sources.push(std::array::from_fn(|byte| (byte * 101 + 0x83) as u8));
    sources
}

fn check(cases: &[Case]) {
    let mut backends: Vec<_> = [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic]
        .into_iter().map(|kind| (kind, Cpu::new(kind).expect("CPU initialization"))).collect();
    for cases in cases.chunks(40) {
        let mut vectors: [[u8; 16]; 32] = std::array::from_fn(seed);
        if cases.iter().any(|case| case.chain) {
            for lane in 0..4 {
                vectors[0][lane * 4..lane * 4 + 4].copy_from_slice(&(1u32 << lane).to_le_bytes());
            }
        }
        let mut initial = vec![0xa5; 0xa000];
        for (register, vector) in vectors.iter().enumerate() {
            initial[register * 16..register * 16 + 16].copy_from_slice(vector);
        }
        for index in 0..16 { initial[0x400 + index] = 0x80 | index as u8; }
        for (slot, case) in cases.iter().enumerate() {
            initial[0x800 + slot * 16..0x800 + slot * 16 + 16].copy_from_slice(&case.source);
        }
        let mut expected = initial.clone();
        let mut program = vec![0xd51b_4410, 0xd51b_4433, 0xd51b_4214];
        for register in 0..32u32 {
            program.push(0x3dc0_01a0 | (register << 10) | register);
        }
        program.extend([0x9100_0484, 0x9100_04a5, 0x9100_04c6, 0x9100_04e7]);
        let mut expected_x11 = CANARY;
        for (slot, &case) in cases.iter().enumerate() {
            program.push(0x3dc0_01a0 | (case.rd << 10) | case.rd);
            vectors[case.rd as usize] = initial[case.rd as usize * 16..case.rd as usize * 16 + 16]
                .try_into().unwrap();
            program.push(0x3dc0_01a0 | (((0x800 / 16 + slot) as u32) << 10) | case.rn);
            vectors[case.rn as usize] = case.source;
            program.push(case.word());
            if case.chain {
                assert_eq!(case.word(), 0x4eb1_aa53);
                program.extend([0x4e04_0673, 0x6eb3_8e53, 0x4e20_1e73,
                                0x4eb1_ba73, 0x1e26_026b, 0x38ab_498b]);
                let selected = i32::from_le_bytes(case.expected()[..4].try_into().unwrap());
                let mut mask = 0u32;
                for lane in 0..4 {
                    let value = i32::from_le_bytes(case.source[lane * 4..lane * 4 + 4].try_into().unwrap());
                    if value == selected { mask |= 1 << lane; }
                }
                vectors[19] = [0; 16];
                vectors[19][..4].copy_from_slice(&mask.to_le_bytes());
                expected_x11 = (initial[0x400 + mask as usize] as i8 as i64) as u64;
            } else { vectors[case.rd as usize] = case.expected(); }
            for (register, vector) in vectors.iter().enumerate() {
                let index = slot * 32 + register;
                program.push(0x3d80_01c0 | ((index as u32) << 10) | register as u32);
                expected[OUTPUTS + index * 16..OUTPUTS + index * 16 + 16].copy_from_slice(vector);
            }
            program.extend([0xd53b_4431, 0xd53b_4412, 0xd53b_4215]);
            for (field, register) in [17u32, 18, 21, 11].into_iter().enumerate() {
                program.push(0xf900_01e0 | (((slot * 4 + field) as u32) << 10) | register);
                let value = [FPSR, FPCR, NZCV, expected_x11][field];
                let offset = STATUS + slot * 32 + field * 8;
                expected[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            }
        }
        program.extend([0x8b05_008a, 0x8b07_00cb, 0xd400_0fc1]);
        let mut encoded = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
        for (index, word) in program.iter().enumerate() {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut initial_gprs = [CANARY; 31];
        initial_gprs[4..8].copy_from_slice(&[10, 20, 30, 40]);
        initial_gprs[12] = DATA + 0x400;
        initial_gprs[13] = DATA;
        initial_gprs[14] = DATA + OUTPUTS as u64;
        initial_gprs[15] = DATA + STATUS as u64;
        initial_gprs[16] = FPCR;
        initial_gprs[19] = FPSR;
        initial_gprs[20] = NZCV;
        let mut expected_gprs = initial_gprs;
        expected_gprs[4..8].copy_from_slice(&[11, 21, 31, 41]);
        expected_gprs[10] = 32;
        expected_gprs[11] = 72;
        expected_gprs[17] = FPSR;
        expected_gprs[18] = FPCR;
        expected_gprs[21] = NZCV;
        for (backend, cpu) in &mut backends {
            let mut code = encoded.clone();
            let mut actual = initial.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr()).unwrap();
                cpu.map_host(DATA, actual.len() as u64, Perm::RW, actual.as_mut_ptr()).unwrap();
            }
            for (register, &value) in initial_gprs.iter().enumerate() { cpu.set_register(register as u32, value); }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let mut reached_svc = false;
            for _ in 0..16 {
                match cpu.run(100_000).expect("integer reduction execution").event {
                    CpuEvent::Svc(0x7e) => { reached_svc = true; break; }
                    CpuEvent::Running => {}
                    event => panic!("{backend:?}: {event:?} PC={:#x}", cpu.get_pc()),
                }
            }
            assert!(reached_svc, "{backend:?}: bounded execution did not reach SVC");
            for (slot, case) in cases.iter().enumerate() {
                for register in 0..32 {
                    let offset = OUTPUTS + (slot * 32 + register) * 16;
                    assert_eq!(&actual[offset..offset + 16], &expected[offset..offset + 16],
                               "{backend:?}: {case:?} V{register}");
                }
                let offset = STATUS + slot * 32;
                assert_eq!(&actual[offset..offset + 32], &expected[offset..offset + 32],
                           "{backend:?}: {case:?} FPSR/FPCR/NZCV/table result");
            }
            for (register, &value) in expected_gprs.iter().enumerate() {
                assert_eq!(cpu.get_register(register as u32), value, "{backend:?}: X{register}");
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
fn all_integer_minmax_arrangements_boundaries_aliases_and_ignored_upper_lanes_match() {
    let mut cases = Vec::new();
    for signed in [false, true] {
        for maximum in [false, true] {
            for (size, q) in [(0, false), (0, true), (1, false), (1, true), (2, true)] {
                for (rn, rd) in [(2, 1), (2, 2), (31, 31), (31, 0), (0, 31)] {
                    for source in patterns(size) {
                        cases.push(Case { signed, maximum, size, q, rn, rd, source, chain: false });
                    }
                }
            }
        }
    }
    check(&cases);
}

#[test]
fn high_source_registers_and_all_destinations_keep_unrelated_state() {
    let mut cases = Vec::new();
    for rn in 0..32 {
        for rd in [rn, (rn + 13) & 31, 31] {
            for (signed, maximum) in [(true, false), (true, true), (false, false), (false, true)] {
                cases.push(Case { signed, maximum, size: 2, q: true, rn, rd,
                                  source: seed(rn as usize), chain: false });
            }
        }
    }
    check(&cases);
}

#[test]
fn exact_wonder_sminv_to_signed_table_lookup_chain_matches_for_every_tie_mask() {
    let mut cases = Vec::new();
    for mask in 1u32..16 {
        let mut source = [0; 16];
        for lane in 0..4 {
            let value = if mask & (1 << lane) != 0 { -7i32 } else { 3 };
            source[lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        let case = Case { signed: true, maximum: false, size: 2, q: true,
                          rn: 18, rd: 19, source, chain: true };
        assert_eq!(case.word(), 0x4eb1_aa53);
        cases.push(case);
    }
    assert_eq!(Case { rd: 2, rn: 1, ..cases[0] }.word(), 0x4eb1_a822);
    check(&cases);
}
