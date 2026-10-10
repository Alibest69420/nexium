#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;
use std::sync::{Mutex, OnceLock};

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const SOURCE: u64 = 0x30_0000;
const PAGE: usize = 0x1000;
const OUTPUT: usize = 0x1000;
const CONSTANTS: usize = 0x10000;
const STRIDE: usize = 800;
const CANARY: u64 = 0xface_1234_5678_9abc;
const NZCV: u64 = 0xa000_0000;
const FPCR: u32 = 0x0300_0000;
const STICKY: u32 = 0x0800_009f;

#[derive(Clone, Copy, Debug)]
struct Case {
    signed: bool,
    subtract: bool,
    bytes: usize,
    upper: bool,
    rn: u32,
    rm: u32,
    rd: u32,
    a: [u8; 16],
    n: [u8; 16],
    m: [u8; 16],
    fpcr: u32,
    fpsr: u32,
    literal: Option<[u8; 16]>,
    chain: bool,
}
impl Case {
    fn word(self) -> u32 {
        assert!(matches!(self.bytes, 1 | 2 | 4));
        0x0e20_8000
            | (u32::from(!self.signed) << 29)
            | (u32::from(self.subtract) << 13)
            | (self.bytes.trailing_zeros() << 22)
            | (u32::from(self.upper) << 30)
            | (self.rm << 16)
            | (self.rn << 5)
            | self.rd
    }
    fn expected(self, a: [u8; 16], n: [u8; 16], m: [u8; 16]) -> [u8; 16] {
        let narrow_modulus = 1i128 << (self.bytes * 8);
        let modulus = narrow_modulus * narrow_modulus;
        let mut result = [0; 16];
        for lane in 0..8 / self.bytes {
            let source = usize::from(self.upper) * 8 + lane * self.bytes;
            let destination = lane * self.bytes * 2;
            let read = |vector: [u8; 16], offset: usize, length: usize| {
                let mut bits = [0; 8];
                bits[..length].copy_from_slice(&vector[offset..offset + length]);
                u64::from_le_bytes(bits) as i128
            };
            let extend = |value: i128| {
                if self.signed && value >= narrow_modulus / 2 {
                    value - narrow_modulus
                } else {
                    value
                }
            };
            let product = extend(read(n, source, self.bytes)) * extend(read(m, source, self.bytes));
            let accumulator = read(a, destination, self.bytes * 2);
            let value =
                (accumulator + if self.subtract { -product } else { product }).rem_euclid(modulus);
            result[destination..destination + self.bytes * 2]
                .copy_from_slice(&(value as u64).to_le_bytes()[..self.bytes * 2]);
        }
        result
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

fn seed(register: usize) -> [u8; 16] {
    std::array::from_fn(|byte| (register * 37 + byte * 11 + 0x93) as u8)
}

fn put(data: &mut [u8], offset: usize, value: u64) {
    data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn check(records: &[Case]) {
    let mut cpus = cpus()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    for (backend, cpu) in &mut *cpus {
        for records in records.chunks(32) {
            let mut data = vec![0xa5; 0x12000];
            let mut source = vec![0x6d; PAGE];
            let mut vectors: [[u8; 16]; 32] = std::array::from_fn(seed);
            if records.iter().any(|record| record.chain) {
                for rd in [1, 4, 5] {
                    vectors[rd] = longs([0, u64::MAX]);
                }
            }

            for (register, vector) in vectors.iter().enumerate() {
                source[0x800 + register * 16..0x800 + register * 16 + 16].copy_from_slice(vector);
            }
            assert!(records.len() * 48 <= 0x800);
            assert!(OUTPUT + records.len() * STRIDE <= CONSTANTS);
            let mut expected = data.clone();
            let mut registers = [CANARY; 31];
            registers[22] = SOURCE;
            registers[24] = DATA + CONSTANTS as u64;
            registers[27] = DATA + OUTPUT as u64;
            registers[28] = SOURCE + 0x800;
            registers[29] = NZCV;
            let mut program = vec![0xd51b_421d];
            for register in 0..32 {
                program.push(0x3dc0_0000 | (register << 10) | (28 << 5) | register);
            }
            for (slot, record) in records.iter().enumerate() {
                let pos = slot * 48;
                source[pos..pos + 16].copy_from_slice(&record.a);
                source[pos + 16..pos + 32].copy_from_slice(&record.n);
                source[pos + 32..pos + 48].copy_from_slice(&record.m);
                for (which, reg, value) in [
                    (0, record.rd, record.a),
                    (1, record.rn, record.n),
                    (2, record.rm, record.m),
                ] {
                    program
                        .push(0x3dc0_0000 | (((slot * 3 + which) as u32) << 10) | (22 << 5) | reg);
                    vectors[reg as usize] = value;
                }
                for (which, reg, value) in
                    [(0, 25, record.fpcr as u64), (1, 26, record.fpsr as u64)]
                {
                    put(&mut data, CONSTANTS + (slot * 2 + which) * 8, value);
                    program
                        .push(0xf940_0000 | (((slot * 2 + which) as u32) << 10) | (24 << 5) | reg);
                    registers[reg as usize] = value;
                }
                program.extend([0xd51b_4419, 0xd51b_443a]);
                let result = record.expected(
                    vectors[record.rd as usize],
                    vectors[record.rn as usize],
                    vectors[record.rm as usize],
                );
                if let Some(literal) = record.literal {
                    assert_eq!(
                        result, literal,
                        "independent modulo oracle literal: {record:?}"
                    );
                }
                let fpsr = record.fpsr;
                vectors[record.rd as usize] = result;
                program.push(record.word());
                if record.chain {
                    assert_eq!(record.word(), 0x6ea7_a202);
                    for (word, upper, subtract, rd, literal) in [
                        (
                            0x2ea7_a201,
                            false,
                            true,
                            1,
                            longs([0xffff_fffe_0000_0002, 0x8000_0000_7fff_ffff]),
                        ),
                        (
                            0x6ea7_8205,
                            true,
                            false,
                            5,
                            longs([0x0000_0000_ffff_ffff, 0x0000_0001_7fff_fffc]),
                        ),
                        (
                            0x2ea7_8204,
                            false,
                            false,
                            4,
                            longs([0x0000_0001_ffff_fffe, 0x7fff_ffff_7fff_ffff]),
                        ),
                    ] {
                        let next = Case {
                            upper,
                            subtract,
                            rd,
                            literal: None,
                            chain: false,
                            ..*record
                        };
                        assert_eq!(next.word(), word);
                        vectors[rd as usize] =
                            next.expected(vectors[rd as usize], vectors[16], vectors[7]);
                        assert_eq!(vectors[rd as usize], literal, "chain literal for V{rd}");
                        program.push(word);
                    }
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
                    (None, 248, CANARY),
                    (Some(0xd53b_441e), 256, record.fpcr as u64),
                    (Some(0xd53b_443e), 264, fpsr as u64),
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
            let source_before = source.clone();
            let mut code = vec![0x5a; (program.len() * 4).next_multiple_of(PAGE)];
            for (index, word) in program.iter().enumerate() {
                code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            let code_before = code.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, data.len() as u64, Perm::RW, data.as_mut_ptr())
                    .unwrap();
                cpu.map_host(SOURCE, PAGE as u64, Perm::RO, source.as_mut_ptr())
                    .unwrap();
            }
            for (register, value) in [CANARY; 31].iter().enumerate() {
                cpu.set_register(register as u32, *value);
            }
            for (reg, value) in [
                (22, SOURCE),
                (24, DATA + CONSTANTS as u64),
                (27, DATA + OUTPUT as u64),
                (28, SOURCE + 0x800),
                (29, NZCV),
            ] {
                cpu.set_register(reg, value);
            }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let event = cpu
                .run(1_000_000)
                .expect("widening multiply-accumulate execution")
                .event;
            let fault = cpu.take_fault();
            let final_registers: [u64; 31] =
                std::array::from_fn(|register| cpu.get_register(register as u32));
            let sp = cpu.get_sp();
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, data.len() as u64).unwrap();
                cpu.unmap_host(SOURCE, PAGE as u64).unwrap();
            }
            assert!(
                matches!(event, CpuEvent::Svc(0x7e)),
                "{backend:?}: {event:?}"
            );
            assert!(
                fault.is_none(),
                "{backend:?}: mapped widening multiply-accumulate faulted"
            );
            assert_eq!(final_registers, registers, "{backend:?}: final GPR guards");
            assert_eq!(sp, CANARY, "{backend:?}: SP guard");
            assert_eq!(code, code_before, "{backend:?}: code guards");
            assert_eq!(source, source_before, "{backend:?}: RO source guards");

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

fn forms() -> Vec<(bool, bool, usize, bool)> {
    let mut result = Vec::new();
    for signed in [false, true] {
        for subtract in [false, true] {
            for bytes in [1, 2, 4] {
                for upper in [false, true] {
                    result.push((signed, subtract, bytes, upper));
                }
            }
        }
    }
    assert_eq!(result.len(), 24);
    result
}
fn pattern(bytes: usize, rotation: usize) -> [u8; 16] {
    let sign = 1u64 << (bytes * 8 - 1);
    let mask = u64::MAX >> (64 - bytes * 8);
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
fn words(values: [u32; 4]) -> [u8; 16] {
    let mut result = [0; 16];
    for (lane, value) in values.into_iter().enumerate() {
        result[lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    result
}
fn longs(values: [u64; 2]) -> [u8; 16] {
    let mut result = [0; 16];
    result[..8].copy_from_slice(&values[0].to_le_bytes());
    result[8..].copy_from_slice(&values[1].to_le_bytes());
    result
}
fn next_random(state: &mut u64) -> [u8; 16] {
    std::array::from_fn(|_| {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state as u8
    })
}

#[test]
fn all_24_formats_original_source_halves_accumulator_aliases_and_wrap_match() {
    let mut records = Vec::new();
    let mut random = 0x08c3_09b8_6ea7_a202;
    for (signed, subtract, bytes, upper) in forms() {
        for (rd, rn, rm) in [
            (0, 1, 2),
            (1, 1, 2),
            (2, 1, 2),
            (0, 1, 1),
            (31, 31, 2),
            (31, 0, 31),
            (31, 31, 31),
            (31, 0, 1),
        ] {
            for sample in 0..32 {
                let (a, n, m) = if sample < 8 {
                    (
                        pattern(bytes * 2, sample + 5),
                        pattern(bytes, sample),
                        pattern(bytes, sample + 3),
                    )
                } else {
                    (
                        next_random(&mut random),
                        next_random(&mut random),
                        next_random(&mut random),
                    )
                };
                records.push(Case {
                    signed,
                    subtract,
                    bytes,
                    upper,
                    rn,
                    rm,
                    rd,
                    a,
                    n,
                    m,
                    fpcr: FPCR,
                    fpsr: if sample & 1 == 0 { 0 } else { STICKY },
                    literal: None,
                    chain: false,
                });
            }
        }
    }
    assert_eq!(records.len(), 6144);
    check(&records);
}
#[test]
fn every_byte_pair_signed_unsigned_add_sub_and_lower_upper_product_matches() {
    let mut records = Vec::new();
    for signed in [false, true] {
        for subtract in [false, true] {
            for upper in [false, true] {
                for n in 0u16..256 {
                    for start in (0u16..256).step_by(8) {
                        let mut n_vector = [0xa3; 16];
                        let mut m_vector = [0x5c; 16];
                        let mut accumulator = [0xff; 16];
                        for lane in 0..8 {
                            let source = usize::from(upper) * 8 + lane;
                            n_vector[source] = n as u8;
                            m_vector[source] = (start + lane as u16) as u8;
                            let a = n
                                .wrapping_mul(257)
                                .wrapping_add((start + lane as u16).wrapping_mul(129))
                                .wrapping_add(lane as u16 * 37);
                            accumulator[lane * 2..lane * 2 + 2].copy_from_slice(&a.to_le_bytes());
                        }
                        records.push(Case {
                            signed,
                            subtract,
                            bytes: 1,
                            upper,
                            rn: 0,
                            rm: 1,
                            rd: 2,
                            a: accumulator,
                            n: n_vector,
                            m: m_vector,
                            fpcr: FPCR,
                            fpsr: if (n + start) & 1 == 0 { 0 } else { STICKY },
                            literal: None,
                            chain: false,
                        });
                    }
                }
            }
        }
    }
    assert_eq!(records.len(), 65536);
    check(&records);
}
#[test]
fn all_registers_wide_destination_aliases_and_full_flags_state_are_preserved() {
    let mut records = Vec::new();
    for (signed, subtract, bytes, upper) in forms() {
        for rn in 0..32 {
            let rm = (rn + 17) & 31;
            for rd in [rn, rm, 31] {
                records.push(Case {
                    signed,
                    subtract,
                    bytes,
                    upper,
                    rn,
                    rm,
                    rd,
                    a: seed(rd as usize),
                    n: seed(rn as usize),
                    m: seed(rm as usize),
                    fpcr: FPCR,
                    fpsr: STICKY,
                    literal: None,
                    chain: false,
                });
            }
        }
    }
    assert_eq!(records.len(), 2304);
    check(&records);
}
#[test]
fn exact_wonder_four_instruction_chain_and_signed_product_literals_match() {
    let chain = Case {
        signed: false,
        subtract: true,
        bytes: 4,
        upper: true,
        rn: 16,
        rm: 7,
        rd: 2,
        a: longs([0, u64::MAX]),
        n: words([u32::MAX, 0x8000_0000, 1, 0x7fff_ffff]),
        m: words([2, u32::MAX, u32::MAX, 3]),
        fpcr: FPCR,
        fpsr: STICKY,
        literal: Some(longs([0xffff_ffff_0000_0001, 0xffff_fffe_8000_0002])),
        chain: true,
    };
    assert_eq!(chain.word(), 0x6ea7_a202);
    let mut records = vec![chain];
    for (subtract, literal) in [
        (false, longs([1, 0x7fff_ffff])),
        (true, longs([u64::MAX, 0xffff_ffff_7fff_ffff])),
    ] {
        records.push(Case {
            signed: true,
            subtract,
            bytes: 4,
            upper: false,
            rn: 1,
            rm: 2,
            rd: 0,
            a: longs([0, u64::MAX]),
            n: words([u32::MAX, 0x8000_0000, 0x1234_5678, 0x7fff_ffff]),
            m: words([u32::MAX, u32::MAX, 0x9abc_def0, 1]),
            fpcr: FPCR,
            fpsr: STICKY,
            literal: Some(literal),
            chain: false,
        });
    }
    assert_eq!(records.len(), 3);
    check(&records);
}
