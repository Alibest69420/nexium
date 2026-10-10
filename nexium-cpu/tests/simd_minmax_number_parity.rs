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
const FZ: u32 = 1 << 24;
const DN: u32 = 1 << 25;
const IOC: u32 = 1;
const IDC: u32 = 1 << 7;

fn format(double: bool) -> (u32, u32, u64, u64, u64, u64) {
    let (bits, fraction, exponent) = if double {
        (64, 52, 0x7ffu64)
    } else {
        (32, 23, 0xffu64)
    };
    let sign = 1u64 << (bits - 1);
    let fraction_mask = (1u64 << fraction) - 1;
    let exponent_mask = exponent << fraction;
    (
        bits,
        fraction,
        sign,
        fraction_mask,
        exponent_mask,
        1u64 << (fraction - 1),
    )
}

fn pair(mut n: u64, mut m: u64, double: bool, minimum: bool, fpcr: u32) -> (u64, u32) {
    let (_, _, sign, fraction, exponent, quiet) = format(double);
    let mut flags = 0;
    for value in [&mut n, &mut m] {
        if fpcr & FZ != 0 && *value & exponent == 0 && *value & fraction != 0 {
            *value &= sign;
            flags |= IDC;
        }
    }
    let classify = |value: u64| {
        let nan = value & exponent == exponent && value & fraction != 0;
        (nan, nan && value & quiet == 0)
    };
    let (n_nan, n_signaling) = classify(n);
    let (m_nan, m_signaling) = classify(m);
    let selected = if n_signaling {
        Some(n)
    } else if m_signaling {
        Some(m)
    } else if n_nan != m_nan {
        return (if n_nan { m } else { n }, flags);
    } else if n_nan {
        Some(n)
    } else {
        None
    };
    if let Some(value) = selected {
        if n_signaling || m_signaling {
            flags |= IOC;
        }
        return (
            if fpcr & DN != 0 {
                exponent | quiet
            } else {
                value | quiet
            },
            flags,
        );
    }
    let n_mag = n & !sign;
    let m_mag = m & !sign;
    if n_mag == 0 && m_mag == 0 {
        return ((if minimum { n | m } else { n & m }) & sign, flags);
    }
    let n_negative = n & sign != 0;
    let m_negative = m & sign != 0;
    let less = if n_negative != m_negative {
        n_negative
    } else if n_negative {
        n_mag > m_mag
    } else {
        n_mag < m_mag
    };
    (if less == minimum { n } else { m }, flags)
}

#[derive(Clone, Copy, Debug)]
struct Case {
    minimum: bool,
    double: bool,
    q: bool,
    rn: u32,
    rm: u32,
    rd: u32,
    n: [u8; 16],
    m: [u8; 16],
    fpcr: u32,
    fpsr: u32,
    chain: bool,
}

impl Case {
    fn word(self) -> u32 {
        assert!(self.q || !self.double);
        0x0e20_c400
            | (u32::from(self.minimum) << 23)
            | (u32::from(self.double) << 22)
            | (u32::from(self.q) << 30)
            | (self.rm << 16)
            | (self.rn << 5)
            | self.rd
    }

    fn expected(self, n: [u8; 16], m: [u8; 16]) -> ([u8; 16], u32) {
        let bytes = if self.double { 8 } else { 4 };
        let mut result = [0; 16];
        let mut flags = 0;
        for lane in 0..(if self.q { 16 } else { 8 }) / bytes {
            let start = lane * bytes;
            let read = |vector: [u8; 16]| {
                let mut bits = [0; 8];
                bits[..bytes].copy_from_slice(&vector[start..start + bytes]);
                u64::from_le_bytes(bits)
            };
            let (value, raised) = pair(read(n), read(m), self.double, self.minimum, self.fpcr);
            result[start..start + bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
            flags |= raised;
        }
        (result, flags)
    }
}

fn words32(values: [u32; 4]) -> [u8; 16] {
    let mut result = [0; 16];
    for (lane, value) in values.into_iter().enumerate() {
        result[lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    result
}

fn floats(values: [f32; 4]) -> [u8; 16] {
    words32(values.map(f32::to_bits))
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

fn reference_fpsr(record: Case, n: [u8; 16], m: [u8; 16], raised: u32) -> u32 {
    if record.fpcr & !(DN | FZ) != 0 {
        return record.fpsr | raised;
    }
    let (_, _, _, fraction, exponent, quiet) = format(record.double);
    let bytes = if record.double { 8 } else { 4 };
    let mut quiet_nan = false;
    for vector in [n, m] {
        for lane in 0..(if record.q { 16 } else { 8 }) / bytes {
            let start = lane * bytes;
            let mut bits = [0; 8];
            bits[..bytes].copy_from_slice(&vector[start..start + bytes]);
            let value = u64::from_le_bytes(bits);
            if value & exponent == exponent && value & fraction != 0 && value & quiet != 0 {
                quiet_nan = true;
            }
        }
    }
    record.fpsr
        | (raised & !(if record.fpcr & FZ != 0 { IDC } else { 0 }))
        | if quiet_nan { IOC } else { 0 }
}

fn check(records: &[Case], native_only: bool) {
    let mut cpus = cpus()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut reference_flag_count = 0;
    let mut reference_flag_examples = Vec::new();
    let mut reference_flag_groups = Vec::new();
    for (backend, cpu) in &mut *cpus {
        if native_only && *backend != CpuBackendKind::Rustarmic {
            continue;
        }
        for records in records.chunks(32) {
            let mut data = vec![0xa5; 0x12000];
            let mut source = vec![0x6d; PAGE];
            let mut vectors: [[u8; 16]; 32] = std::array::from_fn(seed);
            if records.iter().any(|record| record.chain) {
                vectors[3] = floats([0.5, 0.5, 0.5, 0.5]);
                vectors[4] = floats([-0.5, -0.5, -0.5, -0.5]);
                vectors[6] = floats([-8192.0, -8192.0, -8192.0, -8192.0]);
            }
            for (register, vector) in vectors.iter().enumerate() {
                source[0x800 + register * 16..0x800 + register * 16 + 16].copy_from_slice(vector);
            }
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
                let pos = slot * 32;
                source[pos..pos + 16].copy_from_slice(&record.n);
                source[pos + 16..pos + 32].copy_from_slice(&record.m);
                program.push(0x3dc0_0000 | (((slot * 2) as u32) << 10) | (22 << 5) | record.rn);
                vectors[record.rn as usize] = record.n;
                program.push(0x3dc0_0000 | (((slot * 2 + 1) as u32) << 10) | (22 << 5) | record.rm);
                vectors[record.rm as usize] = record.m;
                for (which, reg, value) in
                    [(0, 25, record.fpcr as u64), (1, 26, record.fpsr as u64)]
                {
                    put(&mut data, CONSTANTS + (slot * 2 + which) * 8, value);
                    program
                        .push(0xf940_0000 | (((slot * 2 + which) as u32) << 10) | (24 << 5) | reg);
                    registers[reg as usize] = value;
                }
                program.extend([0xd51b_4419, 0xd51b_443a]);
                let n = vectors[record.rn as usize];
                let m = vectors[record.rm as usize];
                let (result, flags) = record.expected(n, m);
                let fpsr = if *backend == CpuBackendKind::Dynarmic {
                    reference_fpsr(*record, n, m, flags)
                } else {
                    record.fpsr | flags
                };
                vectors[record.rd as usize] = result;
                program.push(record.word());
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
                if record.chain {
                    assert_eq!(record.word(), 0x4ea5_c400);
                    program.extend([
                        0x4e26_f400,
                        0x4ea0_e801,
                        0x6e63_1c81,
                        0x4e21_d400,
                        0x4ea1_b800,
                    ]);
                    let first = vectors[0];
                    let mut final_vector = [0; 16];
                    for lane in 0..4 {
                        let start = lane * 4;
                        let value = f32::from_bits(u32::from_le_bytes(
                            first[start..start + 4].try_into().unwrap(),
                        ));
                        let clamped = value.max(-8192.0);
                        let bias = if clamped < 0.0 { -0.5f32 } else { 0.5f32 };
                        vectors[1][start..start + 4].copy_from_slice(&bias.to_bits().to_le_bytes());
                        final_vector[start..start + 4]
                            .copy_from_slice(&((clamped + bias) as i32).to_le_bytes());
                    }
                    assert_eq!(
                        final_vector,
                        words32([8191, (-8192i32) as u32, 2, (-2i32) as u32]),
                        "chain literal"
                    );
                    vectors[0] = final_vector;
                }
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
                .expect("SIMD minmax-number execution")
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
            assert!(fault.is_none(), "{backend:?}: mapped minmax-number faulted");
            assert_eq!(final_registers, registers, "{backend:?}: final GPR guards");
            assert_eq!(sp, CANARY, "{backend:?}: SP guard");
            assert_eq!(code, code_before, "{backend:?}: code guards");
            assert_eq!(source, source_before, "{backend:?}: RO source guards");
            if *backend == CpuBackendKind::Dynarmic {
                for (slot, record) in records.iter().enumerate() {
                    let position = OUTPUT + slot * STRIDE + 264;
                    let actual =
                        u64::from_le_bytes(data[position..position + 8].try_into().unwrap());
                    let wanted =
                        u64::from_le_bytes(expected[position..position + 8].try_into().unwrap());
                    if actual != wanted {
                        reference_flag_count += 1;
                        let key = (
                            record.minimum,
                            record.double,
                            record.q,
                            record.fpcr,
                            actual ^ wanted,
                        );
                        if let Some((_, count)) = reference_flag_groups
                            .iter_mut()
                            .find(|(previous, _)| *previous == key)
                        {
                            *count += 1;
                        } else {
                            reference_flag_groups.push((key, 1usize));
                            if reference_flag_examples.len() < 16 {
                                let word = record.word();
                                let n_low = u64::from_le_bytes(record.n[..8].try_into().unwrap());
                                let n_high = u64::from_le_bytes(record.n[8..].try_into().unwrap());
                                let m_low = u64::from_le_bytes(record.m[..8].try_into().unwrap());
                                let m_high = u64::from_le_bytes(record.m[8..].try_into().unwrap());
                                reference_flag_examples.push((
                                    word,
                                    record.fpcr,
                                    record.fpsr,
                                    n_low,
                                    n_high,
                                    m_low,
                                    m_high,
                                    actual,
                                    wanted,
                                ));
                            }
                        }
                    }
                }
            }
            for (index, (&actual, &wanted)) in data.iter().zip(&expected).enumerate() {
                if *backend == CpuBackendKind::Dynarmic
                    && index >= OUTPUT
                    && index < OUTPUT + records.len() * STRIDE
                    && (264..272).contains(&((index - OUTPUT) % STRIDE))
                {
                    continue;
                }
                assert_eq!(
                    actual,
                    wanted,
                    "{backend:?}: snapshot/guard byte{index:#x}, record{:?}",
                    records.get(index.saturating_sub(OUTPUT) / STRIDE)
                );
            }
        }
    }
    assert_eq!(
        reference_flag_count, 0,
        "reference FPSR mismatches; form/control/xor counts: {reference_flag_groups:x?}; \
         first 16 examples(word,FPCR,oldFPSR,Nlow,Nhigh,Mlow,Mhigh,actual,wanted): \
         {reference_flag_examples:x?}"
    );
}

fn vector(double: bool, values: [u64; 4]) -> [u8; 16] {
    let bytes = if double { 8 } else { 4 };
    let mut result = [0xe7; 16];
    for lane in 0..16 / bytes {
        result[lane * bytes..lane * bytes + bytes]
            .copy_from_slice(&values[lane].to_le_bytes()[..bytes]);
    }
    result
}

fn expand(double: bool, pairs: &[(u64, u64)], controls: &[u32]) -> Vec<Case> {
    let mut records = Vec::new();
    for q in if double {
        vec![true]
    } else {
        vec![false, true]
    } {
        for minimum in [false, true] {
            for (rn, rm, rd) in [
                (0, 5, 0),
                (1, 2, 3),
                (1, 2, 1),
                (1, 2, 2),
                (1, 1, 31),
                (31, 31, 31),
                (31, 0, 31),
            ] {
                for &fpcr in controls {
                    for (index, &(n, m)) in pairs.iter().enumerate() {
                        records.push(Case {
                            minimum,
                            double,
                            q,
                            rn,
                            rm,
                            rd,
                            n: vector(double, [n, m, n, m]),
                            m: vector(double, [m, n, m, n]),
                            fpcr,
                            fpsr: [0, 0x0800_001c, IOC, IDC, 0x0800_009f][index % 5],
                            chain: false,
                        });
                    }
                }
            }
        }
    }
    records
}

#[test]
fn all_formats_finite_infinity_subnormals_signed_zeros_and_aliases_match() {
    for double in [false, true] {
        let (_, fraction_bits, sign, fraction, exponent, _) = format(double);
        let one = if double {
            0x3ff0_0000_0000_0000
        } else {
            0x3f80_0000
        };
        let values = [
            0,
            sign,
            1,
            sign | 1,
            fraction,
            sign | fraction,
            1 << fraction_bits,
            sign | (1 << fraction_bits),
            one,
            sign | one,
            one - 1,
            sign | (one - 1),
            exponent - 1,
            sign | (exponent - 1),
            exponent,
            sign | exponent,
        ];
        let mut pairs: Vec<_> = values
            .iter()
            .flat_map(|&a| values.iter().map(move |&b| (a, b)))
            .collect();
        let mut random = 0x933e_bc80_4ea5_c400u64;
        for _ in 0..64 {
            let mut next = || {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let value = random & (u64::MAX >> if double { 0 } else { 32 });
                if value & exponent == exponent {
                    value & !fraction
                } else {
                    value
                }
            };
            pairs.push((next(), next()));
        }
        check(&expand(double, &pairs, &[0]), false);
    }
}

#[test]
fn nan_payload_sign_signaling_priority_numeric_preference_and_dn_match() {
    let mut records = Vec::new();
    for double in [false, true] {
        let (_, _, sign, _, exponent, quiet) = format(double);
        let one = if double {
            0x3ff0_0000_0000_0000
        } else {
            0x3f80_0000
        };
        let values = [
            exponent | quiet | 0x123,
            sign | exponent | quiet | 0x456,
            exponent | 0x789,
            sign | exponent | 0xabc,
            one,
            sign | one,
            0,
            sign,
            exponent,
            sign | exponent,
        ];
        let pairs: Vec<_> = values
            .iter()
            .flat_map(|&n| values.iter().map(move |&m| (n, m)))
            .collect();
        assert_eq!(
            pair(values[0], values[3], double, true, 0),
            (values[3] | quiet, IOC)
        );
        assert_eq!(pair(values[0], one, double, true, DN), (one, 0));
        records.extend(expand(double, &pairs, &[0, DN]));
    }
    check(&records, false);
}

#[test]
fn dn_fz_and_exact_sticky_ioc_idc_preserve_status_and_controls() {
    let mut records = Vec::new();
    for double in [false, true] {
        let (_, _, sign, fraction, exponent, quiet) = format(double);
        let values = [
            0,
            sign,
            1,
            sign | 1,
            fraction,
            sign | fraction,
            exponent | quiet | 0x123,
            exponent | 0x789,
        ];
        let pairs: Vec<_> = values
            .iter()
            .flat_map(|&n| values.iter().map(move |&m| (n, m)))
            .collect();
        records.extend(expand(double, &pairs, &[0, DN, FZ, FZ | DN]));
    }
    check(&records, false);
}

#[test]
fn exact_wonder_clamp_sign_bias_and_truncate_prefix_matches_finite_literal() {
    let record = Case {
        minimum: true,
        double: false,
        q: true,
        rn: 0,
        rm: 5,
        rd: 0,
        n: floats([8192.0, -8193.0, 1.75, -1.75]),
        m: floats([8191.0; 4]),
        fpcr: 0,
        fpsr: 0,
        chain: true,
    };
    assert_eq!(record.word(), 0x4ea5_c400);
    check(&[record], false);
}
