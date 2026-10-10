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
const FPSR: u64 = 0x0800_009f;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Add,
    Maximum,
    Minimum,
    MaximumNumber,
    MinimumNumber,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    kind: Kind,
    double: bool,
    rd: u32,
    rn: u32,
    a: u64,
    b: u64,
    fpcr: u32,
    literal: Option<u64>,
}

impl Case {
    fn word(self) -> u32 {
        let base = match self.kind {
            Kind::Add => 0x7e30_d800,
            Kind::Maximum => 0x7e30_f800,
            Kind::Minimum => 0x7eb0_f800,
            Kind::MaximumNumber => 0x7e30_c800,
            Kind::MinimumNumber => 0x7eb0_c800,
        };
        base | (u32::from(self.double) << 22) | (self.rn << 5) | self.rd
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

fn put(data: &mut [u8], offset: usize, value: u64) {
    data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn check(cases: &[Case]) {
    let mut cpus = cpus()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for cases in cases.chunks(16) {
        let mut reference = None;
        for (backend, cpu) in &mut *cpus {
            let mut data = vec![0xa5; 0x10000];
            let mut source = vec![0x6d; PAGE];
            for register in 0..32 {
                for byte in 0..16 {
                    source[0x800 + register * 16 + byte] = (register * 37 + byte * 11 + 0x93) as u8;
                }
            }
            let mut registers: [u64; 31] = std::array::from_fn(|register| CANARY ^ register as u64);
            registers[19] = DATA + OUTPUT as u64;
            registers[20] = SOURCE;
            registers[22] = FPSR;
            registers[23] = NZCV;
            let initial = registers;
            let mut prefixes = Vec::new();
            let mut program = vec![0xd51b_4436, 0xd51b_4217];
            for register in 0..32 {
                program.push(0x3dc0_0000 | ((0x80 + register) << 10) | (20 << 5) | register);
            }
            for (slot, case) in cases.iter().enumerate() {
                if case.double {
                    put(&mut source, slot * 16, case.a);
                    put(&mut source, slot * 16 + 8, case.b);
                } else {
                    source[slot * 16..slot * 16 + 4]
                        .copy_from_slice(&(case.a as u32).to_le_bytes());
                    source[slot * 16 + 4..slot * 16 + 8]
                        .copy_from_slice(&(case.b as u32).to_le_bytes());
                }
                put(&mut source, 0x700 + slot * 8, case.fpcr as u64);
                program.push(0xf940_0000 | ((0xe0 + slot as u32) << 10) | (20 << 5) | 21);
                registers[21] = case.fpcr as u64;
                program.push(0xd51b_4415);
                program.push(0x3dc0_0000 | ((slot as u32) << 10) | (20 << 5) | case.rn);
                program.push(case.word());
                prefixes.push(registers);
                let offset = slot * STRIDE;
                for register in 0..31 {
                    program.push(
                        0xf900_0000
                            | (((offset / 8 + register) as u32) << 10)
                            | (19 << 5)
                            | register as u32,
                    );
                }
                program.push(0x9100_03fe);
                for (word, position) in [
                    (None, 248),
                    (Some(0xd53b_441e), 256),
                    (Some(0xd53b_443e), 264),
                    (Some(0xd53b_421e), 272),
                ] {
                    if let Some(word) = word {
                        program.push(word);
                    }
                    program.push(
                        0xf900_0000 | (((offset + position) as u32 / 8) << 10) | (19 << 5) | 30,
                    );
                }
                registers[30] = NZCV;
                for register in 0..32 {
                    program.push(
                        0x3d80_0000
                            | (((offset + 288 + register * 16) as u32 / 16) << 10)
                            | (19 << 5)
                            | register as u32,
                    );
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
            for (register, &value) in initial.iter().enumerate() {
                cpu.set_register(register as u32, value);
            }
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let mut event = CpuEvent::Running;
            for _ in 0..8 {
                event = cpu.run_with_count(100_000).event;
                if !matches!(event, CpuEvent::Running) {
                    break;
                }
            }
            let fault = cpu.take_fault();
            let pc = cpu.get_pc();
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
                "{backend:?}: {event:?} {:?}",
                cases[0]
            );
            assert!(fault.is_none(), "{backend:?}: {fault:?}");
            assert_eq!(pc, CODE + program.len() as u64 * 4, "{backend:?} PC");
            assert_eq!(final_registers, registers, "{backend:?} GPRs");
            assert_eq!(sp, CANARY, "{backend:?} SP");
            assert_eq!(source, source_before, "{backend:?} source guards");
            assert_eq!(code, code_before, "{backend:?} code guards");
            assert!(
                data[..OUTPUT]
                    .iter()
                    .chain(data[OUTPUT + cases.len() * STRIDE..].iter())
                    .all(|&byte| byte == 0xa5),
                "{backend:?} data guards"
            );
            for (slot, case) in cases.iter().enumerate() {
                let offset = OUTPUT + slot * STRIDE;
                for (register, value) in prefixes[slot].iter().enumerate() {
                    assert_eq!(
                        &data[offset + register * 8..offset + register * 8 + 8],
                        &value.to_le_bytes(),
                        "{backend:?} GPR{register}: {case:?}"
                    );
                }
                for (position, value) in [
                    (248, CANARY),
                    (256, case.fpcr as u64),
                    (264, FPSR),
                    (272, NZCV),
                ] {
                    assert_eq!(
                        &data[offset + position..offset + position + 8],
                        &value.to_le_bytes(),
                        "{backend:?} status{position}: {case:?}"
                    );
                }
                let width = if case.double { 8 } else { 4 };
                let destination = offset + 288 + case.rd as usize * 16;
                if let Some(value) = case.literal {
                    assert_eq!(
                        &data[destination..destination + width],
                        &value.to_le_bytes()[..width],
                        "{backend:?} independent literal: {case:?}"
                    );
                }
                assert!(
                    data[destination + width..destination + 16]
                        .iter()
                        .all(|&byte| byte == 0),
                    "{backend:?} scalar upper bits: {case:?}"
                );
            }
            if let Some(expected) = &reference {
                for (index, (&actual, &wanted)) in data.iter().zip(expected).enumerate() {
                    assert_eq!(
                        actual,
                        wanted,
                        "{backend:?} byte{index:#x} {:?}",
                        cases.get(index.saturating_sub(OUTPUT) / STRIDE)
                    );
                }
            } else {
                reference = Some(data);
            }
        }
    }
}

#[test]
fn exact_mk8_scalar_pairwise_add_source_alias() {
    let case = Case {
        kind: Kind::Add,
        double: false,
        rd: 5,
        rn: 5,
        a: 1.25f32.to_bits() as u64,
        b: 2.5f32.to_bits() as u64,
        fpcr: 0,
        literal: Some(3.75f32.to_bits() as u64),
    };
    assert_eq!(case.word(), 0x7e30_d8a5);
    check(&[case]);
}

#[test]
fn all_scalar_pairwise_operations_registers_aliases_and_fp_controls_match_dynarmic() {
    let mut cases = Vec::new();
    for double in [false, true] {
        let samples: &[u64] = if double {
            &[
                0,
                0x8000_0000_0000_0000,
                1,
                0x000f_ffff_ffff_ffff,
                0x3ff0_0000_0000_0000,
                0xbff0_0000_0000_0000,
                0x3ff0_0000_0000_0001,
                0x7fef_ffff_ffff_ffff,
                0x7ff0_0000_0000_0000,
                0xfff0_0000_0000_0000,
                0x7ff8_0000_0000_0123,
                0xfff0_0000_0000_0456,
            ]
        } else {
            &[
                0,
                0x8000_0000,
                1,
                0x007f_ffff,
                0x3f80_0000,
                0xbf80_0000,
                0x3f80_0001,
                0x7f7f_ffff,
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0123,
                0xff80_0456,
            ]
        };
        for kind in [
            Kind::Add,
            Kind::Maximum,
            Kind::Minimum,
            Kind::MaximumNumber,
            Kind::MinimumNumber,
        ] {
            for rd in 0..32 {
                for rn in [rd, (rd + 7) % 32] {
                    for fpcr in [0, 1 << 24, 1 << 25, 3 << 24, 1 << 22, 2 << 22, 3 << 22] {
                        for (index, &a) in samples.iter().enumerate() {
                            let b = samples[(index * 7 + rd as usize + 3) % samples.len()];
                            cases.push(Case {
                                kind,
                                double,
                                rd,
                                rn,
                                a,
                                b,
                                fpcr,
                                literal: None,
                            });
                        }
                    }
                }
            }
        }
    }
    check(&cases);
}
