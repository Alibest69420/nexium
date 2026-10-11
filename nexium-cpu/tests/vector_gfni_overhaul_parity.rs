#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;
use std::sync::{Mutex, OnceLock};

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const INPUT: usize = 0x800;
const OUTPUT: usize = 0x2000;
const STATUS: usize = 0xa000;
const CONTROL: usize = 0xb000;
const CANARY: u64 = 0xfade_1234_5678_9abc;
const NZCV: u64 = 0xa000_0000;
const FZ: u32 = 1 << 24;
const DN: u32 = 1 << 25;

#[derive(Clone, Copy, Debug)]
enum Operation {
    Rbit,
    Insert {
        bits: usize,
        shift: usize,
        right: bool,
        scalar: bool,
    },
    Ext {
        offset: usize,
    },
    Pair {
        double: bool,
        minimum: bool,
    },
}

#[derive(Clone, Copy, Debug)]
struct Case {
    op: Operation,
    q: bool,
    rd: u32,
    rn: u32,
    rm: u32,
    old: [u8; 16],
    n: [u8; 16],
    m: [u8; 16],
    fpcr: u32,
    fpsr: u32,
}

impl Case {
    fn word(self) -> u32 {
        let q = u32::from(self.q) << 30;
        let regs = (self.rn << 5) | self.rd;
        match self.op {
            Operation::Rbit => 0x2e60_5800 | q | regs,
            Operation::Insert {
                bits,
                shift,
                right,
                scalar,
            } => {
                let immediate = if right {
                    bits * 2 - shift
                } else {
                    bits + shift
                };
                (if right { 0x2f00_4400 } else { 0x2f00_5400 })
                    | (if scalar { 0x5000_0000 } else { q })
                    | ((immediate as u32) << 16)
                    | regs
            }
            Operation::Ext { offset } => {
                0x2e00_0000 | q | (self.rm << 16) | ((offset as u32) << 11) | regs
            }
            Operation::Pair { double, minimum } => {
                0x2e20_c400
                    | q
                    | (u32::from(double) << 22)
                    | (u32::from(minimum) << 23)
                    | (self.rm << 16)
                    | regs
            }
        }
    }

    fn expected(self, old: [u8; 16], n: [u8; 16], m: [u8; 16]) -> ([u8; 16], u32) {
        let mut result = [0; 16];
        let mut flags = 0;
        let active = if self.q { 16 } else { 8 };
        match self.op {
            Operation::Rbit => {
                for i in 0..active {
                    result[i] = n[i].reverse_bits();
                }
            }
            Operation::Ext { offset } => {
                for i in 0..active {
                    result[i] = if i + offset < active {
                        n[i + offset]
                    } else {
                        m[i + offset - active]
                    };
                }
            }
            Operation::Insert {
                bits,
                shift,
                right,
                scalar,
            } => {
                let bytes = bits / 8;
                for lane in 0..(if scalar { 8 } else { active }) / bytes {
                    for bit in 0..bits {
                        let source_bit = if right {
                            bit as isize + shift as isize
                        } else {
                            bit as isize - shift as isize
                        };
                        let (vector, selected) = if (0..bits as isize).contains(&source_bit) {
                            (&n, source_bit as usize)
                        } else {
                            (&old, bit)
                        };
                        let set = (vector[lane * bytes + selected / 8] >> (selected % 8)) & 1;
                        result[lane * bytes + bit / 8] |= set << (bit % 8);
                    }
                }
            }
            Operation::Pair { double, minimum } => {
                let bytes = if double { 8 } else { 4 };
                let pairs = active / bytes / 2;
                for (source, vector) in [n, m].into_iter().enumerate() {
                    for lane in 0..pairs {
                        let read = |index: usize| {
                            let mut word = [0; 8];
                            word[..bytes]
                                .copy_from_slice(&vector[index * bytes..(index + 1) * bytes]);
                            u64::from_le_bytes(word)
                        };
                        let (value, raised) = pair(
                            read(lane * 2),
                            read(lane * 2 + 1),
                            double,
                            minimum,
                            self.fpcr,
                        );
                        let at = (source * pairs + lane) * bytes;
                        result[at..at + bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
                        flags |= raised;
                    }
                }
            }
        }
        (result, self.fpsr | flags)
    }
}

fn pair(mut n: u64, mut m: u64, double: bool, minimum: bool, fpcr: u32) -> (u64, u32) {
    let (sign, fraction, exponent, quiet) = if double {
        (1u64 << 63, (1u64 << 52) - 1, 0x7ffu64 << 52, 1u64 << 51)
    } else {
        (1u64 << 31, (1u64 << 23) - 1, 0xffu64 << 23, 1u64 << 22)
    };
    let mut flags = 0;
    for value in [&mut n, &mut m] {
        if fpcr & FZ != 0 && *value & exponent == 0 && *value & fraction != 0 {
            *value &= sign;
            flags |= 0x80;
        }
    }
    let nan = |value: u64| value & exponent == exponent && value & fraction != 0;
    let sn = nan(n) && n & quiet == 0;
    let sm = nan(m) && m & quiet == 0;
    let selected = if sn {
        Some(n)
    } else if sm {
        Some(m)
    } else if nan(n) != nan(m) {
        return (if nan(n) { m } else { n }, flags);
    } else if nan(n) {
        Some(n)
    } else {
        None
    };
    if let Some(value) = selected {
        flags |= u32::from(sn || sm);
        return (
            if fpcr & DN != 0 {
                exponent | quiet
            } else {
                value | quiet
            },
            flags,
        );
    }
    if n & !sign == 0 && m & !sign == 0 {
        return ((if minimum { n | m } else { n & m }) & sign, flags);
    }
    let less = if (n ^ m) & sign != 0 {
        n & sign != 0
    } else if n & sign != 0 {
        n & !sign > m & !sign
    } else {
        n < m
    };
    (if less == minimum { n } else { m }, flags)
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

fn check(cases: &[Case]) {
    check_backend(cases, None);
}

fn check_backend(cases: &[Case], selected: Option<CpuBackendKind>) {
    let mut cpus = cpus()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (backend, cpu) in &mut *cpus {
        if selected.is_some_and(|kind| kind != *backend) {
            continue;
        }
        for cases in cases.chunks(32) {
            let mut data = vec![0xa5; 0x12000];
            let mut vectors: [[u8; 16]; 32] = std::array::from_fn(|reg| {
                std::array::from_fn(|byte| (reg * 37 + byte * 11 + 0x93) as u8)
            });
            for (reg, vector) in vectors.iter().enumerate() {
                data[reg * 16..reg * 16 + 16].copy_from_slice(vector);
            }
            for (slot, case) in cases.iter().enumerate() {
                for (index, vector) in [case.old, case.n, case.m].into_iter().enumerate() {
                    let offset = INPUT + (slot * 3 + index) * 16;
                    data[offset..offset + 16].copy_from_slice(&vector);
                }
                data[CONTROL + slot * 16..CONTROL + slot * 16 + 8]
                    .copy_from_slice(&(case.fpcr as u64).to_le_bytes());
                data[CONTROL + slot * 16 + 8..CONTROL + slot * 16 + 16]
                    .copy_from_slice(&(case.fpsr as u64).to_le_bytes());
            }
            let mut expected = data.clone();
            let mut program = vec![0xd51b_4214];
            for reg in 0..32 {
                program.push(0x3dc0_01a0 | (reg << 10) | reg);
            }
            let mut registers = [CANARY; 31];
            for (reg, value) in [
                (13, DATA),
                (14, DATA + OUTPUT as u64),
                (15, DATA + STATUS as u64),
                (20, NZCV),
                (22, DATA + CONTROL as u64),
            ] {
                registers[reg] = value;
            }
            for (slot, case) in cases.iter().enumerate() {
                for (field, reg) in [(0, 16), (1, 19)] {
                    program.push(0xf940_02c0 | (((slot * 2 + field) as u32) << 10) | reg);
                }
                registers[16] = case.fpcr as u64;
                registers[19] = case.fpsr as u64;
                program.extend([0xd51b_4410, 0xd51b_4433]);
                for (field, reg, value) in [(0, case.rd, case.old), (1, case.rn, case.n)] {
                    program
                        .push(0x3dc0_01a0 | (((INPUT / 16 + slot * 3 + field) as u32) << 10) | reg);
                    vectors[reg as usize] = value;
                }
                if matches!(case.op, Operation::Ext { .. } | Operation::Pair { .. }) {
                    program
                        .push(0x3dc0_01a0 | (((INPUT / 16 + slot * 3 + 2) as u32) << 10) | case.rm);
                    vectors[case.rm as usize] = case.m;
                }
                let (result, fpsr) = case.expected(
                    vectors[case.rd as usize],
                    vectors[case.rn as usize],
                    vectors[case.rm as usize],
                );
                vectors[case.rd as usize] = result;
                program.push(case.word());
                for (reg, vector) in vectors.iter().enumerate() {
                    let offset = OUTPUT + (slot * 32 + reg) * 16;
                    program.push(0x3d80_01c0 | (((slot * 32 + reg) as u32) << 10) | reg as u32);
                    expected[offset..offset + 16].copy_from_slice(vector);
                }
                program.extend([0xd53b_4431, 0xd53b_4412, 0xd53b_4215]);
                for (field, reg, value) in [
                    (0, 17, fpsr as u64),
                    (1, 18, case.fpcr as u64),
                    (2, 21, NZCV),
                ] {
                    let offset = STATUS + (slot * 3 + field) * 8;
                    program.push(0xf900_01e0 | (((slot * 3 + field) as u32) << 10) | reg as u32);
                    expected[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
                    registers[reg] = value;
                }
            }
            program.push(0xd400_0fc1);
            let mut code = vec![0x5a; (program.len() * 4).next_multiple_of(0x1000)];
            for (index, word) in program.into_iter().enumerate() {
                code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            let before = code.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, data.len() as u64, Perm::RW, data.as_mut_ptr())
                    .unwrap();
            }
            for (reg, value) in [CANARY; 31].into_iter().enumerate() {
                cpu.set_register(reg as u32, value);
            }
            for (reg, value) in [
                (13, DATA),
                (14, DATA + OUTPUT as u64),
                (15, DATA + STATUS as u64),
                (20, NZCV),
                (22, DATA + CONTROL as u64),
            ] {
                cpu.set_register(reg, value);
            }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let run = cpu.run(1_000_000);
            let fault = cpu.take_fault();
            let actual_registers: [u64; 31] =
                std::array::from_fn(|reg| cpu.get_register(reg as u32));
            let sp = cpu.get_sp();
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, data.len() as u64).unwrap();
            }
            assert!(
                matches!(run.expect("vector execution").event, CpuEvent::Svc(0x7e)),
                "{backend:?}"
            );
            assert!(fault.is_none(), "{backend:?}: {fault:?}");
            assert_eq!(
                actual_registers,
                registers,
                "{backend:?}: GPR guards, last {:?}",
                cases.last()
            );
            assert_eq!(sp, CANARY, "{backend:?}: SP");
            assert_eq!(code, before, "{backend:?}: code");
            if let Some(offset) = data
                .iter()
                .zip(&expected)
                .position(|(actual, wanted)| actual != wanted)
            {
                let (slot, field) = if (OUTPUT..OUTPUT + cases.len() * 512).contains(&offset) {
                    ((offset - OUTPUT) / 512, "vector")
                } else if (STATUS..STATUS + cases.len() * 24).contains(&offset) {
                    ((offset - STATUS) / 24, "FP status")
                } else {
                    (0, "memory guard")
                };
                let at = offset & !7;
                let actual = u64::from_le_bytes(data[at..at + 8].try_into().unwrap());
                let wanted = u64::from_le_bytes(expected[at..at + 8].try_into().unwrap());
                panic!(
                    "{backend:?}: {field}, byte offset {offset:#x}, slot {slot}, actual {actual:#018x}, expected {wanted:#018x}, word {:#010x}, case {:?}",
                    cases[slot].word(),
                    cases[slot]
                );
            }
        }
    }
}

fn case(op: Operation, q: bool, rd: u32, rn: u32, rm: u32) -> Case {
    Case {
        op,
        q,
        rd,
        rn,
        rm,
        old: std::array::from_fn(|i| (i * 29 + 0xa5) as u8),
        n: std::array::from_fn(|i| (i * 37 + 0x53) as u8),
        m: std::array::from_fn(|i| (i * 41 + 0xb7) as u8),
        fpcr: 0,
        fpsr: 0x0800_0000,
    }
}

#[test]
fn rbit_all_bytes_both_lengths_and_aliases() {
    let mut cases = Vec::new();
    for q in [false, true] {
        for value in 0u16..256 {
            for (rd, rn) in [(0, 1), (1, 1), (31, 0), (31, 31)] {
                let mut record = case(Operation::Rbit, q, rd, rn, 2);
                record.n = std::array::from_fn(|i| (value as u8).wrapping_add((i * 17) as u8));
                cases.push(record);
            }
        }
    }
    check(&cases);
}

#[test]
fn shift_inserts_every_arrangement_boundaries_and_aliases() {
    let mut cases = Vec::new();
    for (q, bits, scalar) in [
        (false, 8, false),
        (true, 8, false),
        (false, 16, false),
        (true, 16, false),
        (false, 32, false),
        (true, 32, false),
        (true, 64, false),
        (false, 64, true),
    ] {
        for right in [false, true] {
            let shifts = if right {
                [1, bits - 1, bits]
            } else {
                [0, 1, bits - 1]
            };
            for shift in shifts {
                for (rd, rn) in [(0, 1), (1, 1), (31, 0), (31, 31)] {
                    cases.push(case(
                        Operation::Insert {
                            bits,
                            shift,
                            right,
                            scalar,
                        },
                        q,
                        rd,
                        rn,
                        2,
                    ));
                }
            }
        }
    }
    check(&cases);
}

#[test]
fn ext_q0_poisoned_unused_source_half() {
    let mut cases = Vec::new();
    for (rd, rn, rm) in [(0, 1, 2), (1, 1, 2), (2, 1, 2), (31, 1, 31)] {
        let mut record = case(Operation::Ext { offset: 1 }, false, rd, rn, rm);
        record.n = [
            0, 1, 2, 3, 4, 5, 6, 7, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
        ];
        record.m = [
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb,
            0xbb, 0xbb,
        ];
        assert_eq!(
            record.expected(record.old, record.n, record.m).0,
            [1, 2, 3, 4, 5, 6, 7, 0x10, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        cases.push(record);
    }
    assert_eq!(cases[0].word(), 0x2e02_0820);
    check(&cases);
}

fn lanes(values: &[u64], double: bool) -> [u8; 16] {
    let bytes = if double { 8 } else { 4 };
    let mut result = [0; 16];
    for (lane, value) in values.iter().enumerate() {
        result[lane * bytes..lane * bytes + bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
    }
    result
}

fn pairwise_number_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (double, q) in [(false, false), (false, true), (true, true)] {
        let (sign, qnan, snan, one, two) = if double {
            (
                1u64 << 63,
                0x7ff8_0000_0000_0123,
                0x7ff0_0000_0000_0456,
                1f64.to_bits(),
                2f64.to_bits(),
            )
        } else {
            (
                1u64 << 31,
                0x7fc0_0123,
                0x7f80_0456,
                1f32.to_bits() as u64,
                2f32.to_bits() as u64,
            )
        };
        let patterns = [
            (qnan, one, snan, two),
            (snan, qnan, qnan | sign, snan | sign),
            (0, sign, 1, sign | 1),
            (qnan, qnan | sign, two, one),
        ];
        for minimum in [false, true] {
            for fpcr in [0, DN, FZ, DN | FZ] {
                for (a, b, c, d) in patterns {
                    for (rd, rn, rm) in [(0, 1, 2), (1, 1, 2), (2, 1, 2), (31, 31, 31)] {
                        let mut record = case(Operation::Pair { double, minimum }, q, rd, rn, rm);
                        record.fpcr = fpcr;
                        let n = [a, b, c, d];
                        let m = if double { [c, d, b, a] } else { [d, c, b, a] };
                        let count = if double { 2 } else { 4 };
                        record.n = lanes(&n[..count], double);
                        record.m = lanes(&m[..count], double);
                        cases.push(record);
                    }
                }
            }
        }
        for minimum in [false, true] {
            let mut record = case(Operation::Pair { double, minimum }, q, 0, 1, 2);
            let encode = |x: f64| {
                if double {
                    x.to_bits()
                } else {
                    (x as f32).to_bits() as u64
                }
            };
            let n = [encode(8.0), encode(3.0), encode(7.0), encode(2.0)];
            let m = [encode(6.0), encode(1.0), encode(5.0), encode(4.0)];
            let count = if double { 2 } else { 4 };
            record.n = lanes(&n[..count], double);
            record.m = lanes(&m[..count], double);
            let wanted = if double {
                if minimum {
                    lanes(&[encode(3.0), encode(1.0)], true)
                } else {
                    lanes(&[encode(8.0), encode(6.0)], true)
                }
            } else if q {
                if minimum {
                    lanes(&[encode(3.0), encode(2.0), encode(1.0), encode(4.0)], false)
                } else {
                    lanes(&[encode(8.0), encode(7.0), encode(6.0), encode(5.0)], false)
                }
            } else if minimum {
                lanes(&[encode(3.0), encode(1.0)], false)
            } else {
                lanes(&[encode(8.0), encode(6.0)], false)
            };
            assert_eq!(record.expected(record.old, record.n, record.m).0, wanted);
            cases.push(record);
        }
    }
    cases
}

#[test]
fn pairwise_number_grouping_nan_zero_dn_fz_dynarmic() {
    let cases: Vec<_> = pairwise_number_cases()
        .into_iter()
        .filter(|case| !dynarmic_fz_consumes_denormal(*case))
        .collect();
    assert_eq!(cases.len(), 342);
    check_backend(&cases, Some(CpuBackendKind::Dynarmic));
}

#[test]
fn pairwise_number_grouping_nan_zero_dn_fz_rustarmic() {
    let cases = pairwise_number_cases();
    assert_eq!(cases.len(), 390);
    check_backend(&cases, Some(CpuBackendKind::Rustarmic));
}

fn dynarmic_fz_consumes_denormal(case: Case) -> bool {
    let n = if case.rn == case.rm { case.m } else { case.n };
    case.fpcr & FZ != 0 && (case.expected(case.old, n, case.m).1 ^ case.fpsr) & 0x80 != 0
}

#[test]
#[ignore = "Dynarmic FZ path drops FPSR.IDC"]
fn dynarmic_fz_consumed_denormal_preserves_architectural_idc() {
    let cases: Vec<_> = pairwise_number_cases()
        .into_iter()
        .filter(|case| dynarmic_fz_consumes_denormal(*case))
        .collect();
    assert_eq!(cases.len(), 48);
    check_backend(&cases, Some(CpuBackendKind::Dynarmic));
}
