#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;
use std::sync::{Mutex, OnceLock};

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const SOURCE: u64 = 0x30_0000;
const PAGE: usize = 0x1000;
const OUTPUT: usize = 0x1000;
const STRIDE: usize = 800;
const CANARY: u64 = 0xface_1234_5678_9abc;
const NZCV: u64 = 0xa000_0000;
const FPCR: u32 = 0x0700_0000;
const FPSR: u32 = 0x0800_009f;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Hash,
    Hash2,
    Schedule0,
    Schedule1,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    kind: Kind,
    rd: u32,
    rn: u32,
    rm: u32,
    d: [u32; 4],
    n: [u32; 4],
    m: [u32; 4],
}

impl Case {
    fn word(self) -> u32 {
        let base = match self.kind {
            Kind::Hash => 0x5e00_4000,
            Kind::Hash2 => 0x5e00_5000,
            Kind::Schedule0 => 0x5e28_2800,
            Kind::Schedule1 => 0x5e00_6000,
        };
        base | (self.rn << 5)
            | self.rd
            | if matches!(self.kind, Kind::Schedule0) {
                0
            } else {
                self.rm << 16
            }
    }
}

fn oracle(kind: Kind, d: [u32; 4], n: [u32; 4], m: [u32; 4]) -> [u32; 4] {
    let small0 = |v: u32| v.rotate_right(7) ^ v.rotate_right(18) ^ (v >> 3);
    let small1 = |v: u32| v.rotate_right(17) ^ v.rotate_right(19) ^ (v >> 10);
    match kind {
        Kind::Schedule0 => {
            std::array::from_fn(|i| d[i].wrapping_add(small0(if i == 3 { n[0] } else { d[i + 1] })))
        }
        Kind::Schedule1 => {
            let mut out = [0u32; 4];
            out[0] = d[0].wrapping_add(n[1]).wrapping_add(small1(m[2]));
            out[1] = d[1].wrapping_add(n[2]).wrapping_add(small1(m[3]));
            out[2] = d[2].wrapping_add(n[3]).wrapping_add(small1(out[0]));
            out[3] = d[3].wrapping_add(m[0]).wrapping_add(small1(out[1]));
            out
        }
        Kind::Hash | Kind::Hash2 => {
            let mut s = if matches!(kind, Kind::Hash) {
                [d[0], d[1], d[2], d[3], n[0], n[1], n[2], n[3]]
            } else {
                [n[0], n[1], n[2], n[3], d[0], d[1], d[2], d[3]]
            };
            for word in m {
                let upper1 = s[4].rotate_right(6) ^ s[4].rotate_right(11) ^ s[4].rotate_right(25);
                let choice = (s[4] & s[5]) ^ ((!s[4]) & s[6]);
                let t1 = s[7]
                    .wrapping_add(upper1)
                    .wrapping_add(choice)
                    .wrapping_add(word);
                let upper0 = s[0].rotate_right(2) ^ s[0].rotate_right(13) ^ s[0].rotate_right(22);
                let majority = (s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]);
                s = [
                    t1.wrapping_add(upper0).wrapping_add(majority),
                    s[0],
                    s[1],
                    s[2],
                    s[3].wrapping_add(t1),
                    s[4],
                    s[5],
                    s[6],
                ];
            }
            let start = if matches!(kind, Kind::Hash) { 0 } else { 4 };
            s[start..start + 4].try_into().unwrap()
        }
    }
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

fn vector_words(register: usize) -> [u32; 4] {
    std::array::from_fn(|lane| 0x1537_9bdfu32.wrapping_mul((register * 4 + lane + 1) as u32))
}

fn put(data: &mut [u8], offset: usize, value: u64) {
    data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_vector(data: &mut [u8], offset: usize, vector: [u32; 4]) {
    for (lane, word) in vector.into_iter().enumerate() {
        data[offset + lane * 4..offset + lane * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
}

fn check(cases: &[Case]) {
    let mut cpus = cpus()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (backend, cpu) in &mut *cpus {
        for cases in cases.chunks(16) {
            let mut data = vec![0xa5; 0x10000];
            let mut expected = data.clone();
            let mut source = vec![0x6d; PAGE];
            let mut vectors: [[u32; 4]; 32] = std::array::from_fn(vector_words);
            for (register, vector) in vectors.iter().enumerate() {
                put_vector(&mut source, 0x800 + register * 16, *vector);
            }
            let mut registers: [u64; 31] = std::array::from_fn(|register| CANARY ^ register as u64);
            registers[19] = DATA + OUTPUT as u64;
            registers[20] = SOURCE;
            registers[21] = FPCR as u64;
            registers[22] = FPSR as u64;
            registers[23] = NZCV;
            let mut program = vec![0xd51b_4415, 0xd51b_4436, 0xd51b_4217];
            for register in 0..32 {
                program.push(0x3dc0_0000 | ((0x80 + register) << 10) | (20 << 5) | register);
            }
            for (slot, case) in cases.iter().enumerate() {
                for (input, (register, value)) in
                    [(case.rd, case.d), (case.rn, case.n), (case.rm, case.m)]
                        .into_iter()
                        .enumerate()
                {
                    if input == 2 && matches!(case.kind, Kind::Schedule0) {
                        break;
                    }
                    let index = slot * 3 + input;
                    put_vector(&mut source, index * 16, value);
                    program.push(0x3dc0_0000 | ((index as u32) << 10) | (20 << 5) | register);
                    vectors[register as usize] = value;
                }
                vectors[case.rd as usize] = oracle(
                    case.kind,
                    vectors[case.rd as usize],
                    vectors[case.rn as usize],
                    vectors[case.rm as usize],
                );
                program.push(case.word());
                let offset = slot * STRIDE;
                for register in 0..31 {
                    program.push(
                        0xf900_0000
                            | (((offset / 8 + register) as u32) << 10)
                            | (19 << 5)
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
                    (None, 248, CANARY),
                    (Some(0xd53b_441e), 256, FPCR as u64),
                    (Some(0xd53b_443e), 264, FPSR as u64),
                    (Some(0xd53b_421e), 272, NZCV),
                ] {
                    if let Some(word) = word {
                        program.push(word);
                    }
                    program.push(
                        0xf900_0000 | (((offset + position) as u32 / 8) << 10) | (19 << 5) | 30,
                    );
                    put(&mut expected, OUTPUT + offset + position, value);
                }
                registers[30] = NZCV;
                for (register, vector) in vectors.iter().enumerate() {
                    let position = offset + 288 + register * 16;
                    program.push(
                        0x3d80_0000 | ((position as u32 / 16) << 10) | (19 << 5) | register as u32,
                    );
                    put_vector(&mut expected, OUTPUT + position, *vector);
                }
            }
            program.push(0xd400_0fc1);
            let mut code = vec![0x5a; (program.len() * 4).next_multiple_of(PAGE)];
            for (index, word) in program.iter().enumerate() {
                code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            let code_before = code.clone();
            let source_before = source.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, data.len() as u64, Perm::RW, data.as_mut_ptr())
                    .unwrap();
                cpu.map_host(SOURCE, PAGE as u64, Perm::RO, source.as_mut_ptr())
                    .unwrap();
            }
            for (register, &value) in registers.iter().enumerate() {
                cpu.set_register(register as u32, value);
            }
            cpu.set_register(30, CANARY ^ 30);
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let mut event = CpuEvent::Running;
            for _ in 0..8 {
                event = cpu.run_with_count(100_000).event;
                if !matches!(event, CpuEvent::Running) {
                    break;
                }
            }
            assert!(
                matches!(event, CpuEvent::Svc(0x7e)),
                "{backend:?}: {event:?}"
            );
            assert!(cpu.take_fault().is_none(), "{backend:?} memory fault");
            assert_eq!(
                cpu.get_pc(),
                CODE + program.len() as u64 * 4,
                "{backend:?} PC"
            );
            assert_eq!(
                data, expected,
                "{backend:?} SHA256 outputs/register/status/memory guards"
            );
            assert_eq!(source, source_before, "{backend:?} source guard");
            assert_eq!(code, code_before, "{backend:?} code guard");
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, data.len() as u64).unwrap();
                cpu.unmap_host(SOURCE, PAGE as u64).unwrap();
            }
        }
    }
}

#[test]
fn first_four_sha256_abc_rounds_match_published_state_and_dynarmic() {
    let abcd = [0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a];
    let efgh = [0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19];
    let words = [0xa3ec_9318, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5];
    assert_eq!(
        oracle(Kind::Hash, abcd, efgh, words),
        [0xd550_f666, 0xc8c3_47a7, 0x5a6a_d9ad, 0x5d6a_ebcd]
    );
    assert_eq!(
        oracle(Kind::Hash2, efgh, abcd, words),
        [0x24e0_0850, 0xf929_39eb, 0x78ce_7989, 0xfa2a_4622]
    );
    check(&[
        Case {
            kind: Kind::Hash,
            rd: 0,
            rn: 1,
            rm: 2,
            d: abcd,
            n: efgh,
            m: words,
        },
        Case {
            kind: Kind::Hash2,
            rd: 31,
            rn: 29,
            rm: 30,
            d: efgh,
            n: abcd,
            m: words,
        },
    ]);
}

#[test]
fn exact_mk8_sdk_sha256_schedule_instruction_matches_dynarmic() {
    let case = Case {
        kind: Kind::Schedule0,
        rd: 0,
        rn: 1,
        rm: 31,
        d: [0x6162_6380, 0, 0, 0],
        n: [0, 0, 0, 0],
        m: [0xffff_ffff; 4],
    };
    assert_eq!(case.word(), 0x5e28_2820);
    check(&[case]);
}

#[test]
fn every_sha256_operation_register_alias_and_status_matches_dynarmic() {
    let mut cases = Vec::new();
    for kind in [Kind::Hash, Kind::Hash2, Kind::Schedule0, Kind::Schedule1] {
        for rd in 0..32 {
            let n = (rd + 7) % 32;
            let m = (rd + 13) % 32;
            for (rn, rm) in [(n, m), (rd, m), (n, rd), (n, n), (rd, rd)] {
                for pattern in 0..6 {
                    let words = |salt: u32| match pattern {
                        0 => [0; 4],
                        1 => [u32::MAX; 4],
                        2 => std::array::from_fn(|lane| 1u32 << ((lane as u32 * 7 + salt) % 32)),
                        _ => std::array::from_fn(|lane| {
                            0x1937_9bdfu32
                                .wrapping_mul(lane as u32 + salt + 1)
                                .wrapping_add(pattern * 0x1000_0100)
                        }),
                    };
                    cases.push(Case {
                        kind,
                        rd,
                        rn,
                        rm,
                        d: words(rd),
                        n: words(rn + 53),
                        m: words(rm + 79),
                    });
                }
            }
        }
    }
    check(&cases);
}
