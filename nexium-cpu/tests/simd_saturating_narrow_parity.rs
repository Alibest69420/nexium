#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const INPUTS: usize = 0x1000;
const OUTPUTS: usize = 0x4000;
const STRIDE: usize = 544;
const QC: u32 = 1 << 27;
const OTHER_FLAGS: u32 = 0x93;
const CANARY: u64 = 0xfade_1234_5678_9abc;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Signed,
    Unsigned,
    SignedToUnsigned,
}

#[derive(Clone, Copy, Debug)]
struct Instruction {
    kind: Kind,
    size: u32,
    scalar: bool,
    upper: bool,
    rn: u32,
    rd: u32,
}

impl Instruction {
    fn destination_bytes(self) -> usize {
        1 << self.size
    }
    fn source_bytes(self) -> usize {
        self.destination_bytes() * 2
    }
    fn lanes(self) -> usize {
        if self.scalar {
            1
        } else {
            8 / self.destination_bytes()
        }
    }
    fn word(self) -> u32 {
        assert!(self.size < 3 && !(self.scalar && self.upper));
        let base = match self.kind {
            Kind::Signed => 0x0e21_4800,
            Kind::Unsigned => 0x2e21_4800,
            Kind::SignedToUnsigned => 0x2e21_2800,
        };
        base | if self.scalar {
            0x5000_0000
        } else {
            (self.upper as u32) << 30
        } | (self.size << 22)
            | (self.rn << 5)
            | self.rd
    }
}

fn seed(register: u32) -> [u8; 16] {
    std::array::from_fn(|byte| (register as usize * 37 + byte * 11 + 0x93) as u8)
}

fn read_integer(vector: &[u8; 16], bytes: usize, index: usize) -> u64 {
    let mut value = [0; 8];
    value[..bytes].copy_from_slice(&vector[index * bytes..(index + 1) * bytes]);
    u64::from_le_bytes(value)
}

fn put_integer(vector: &mut [u8; 16], bytes: usize, index: usize, value: u64) {
    vector[index * bytes..(index + 1) * bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
}

fn saturate(value: u64, operation: Instruction) -> (u64, bool) {
    let source_bits = operation.source_bytes() * 8;
    let destination_bits = operation.destination_bytes() * 8;
    let source_mask = u64::MAX >> (64 - source_bits);
    let raw = value & source_mask;
    let input = match operation.kind {
        Kind::Unsigned => raw as i128,
        _ => {
            if raw & (1u64 << (source_bits - 1)) != 0 {
                raw as i128 - (1i128 << source_bits)
            } else {
                raw as i128
            }
        }
    };
    let (minimum, maximum) = match operation.kind {
        Kind::Signed => (
            -(1i128 << (destination_bits - 1)),
            (1i128 << (destination_bits - 1)) - 1,
        ),
        _ => (0, (1i128 << destination_bits) - 1),
    };
    let bounded = input.clamp(minimum, maximum);
    (
        bounded as u64 & ((1u64 << destination_bits) - 1),
        bounded != input,
    )
}

fn model(operation: Instruction, vectors: &mut [[u8; 16]; 32], flags: &mut u32) {
    let input = vectors[operation.rn as usize];
    let mut result = if operation.upper {
        vectors[operation.rd as usize]
    } else {
        [0; 16]
    };
    let offset = if operation.upper {
        8 / operation.destination_bytes()
    } else {
        0
    };
    for index in 0..operation.lanes() {
        let (value, saturated) = saturate(
            read_integer(&input, operation.source_bytes(), index),
            operation,
        );
        if saturated {
            *flags |= QC;
        }
        put_integer(
            &mut result,
            operation.destination_bytes(),
            offset + index,
            value,
        );
    }
    vectors[operation.rd as usize] = result;
}

#[derive(Debug)]
struct Record {
    label: String,
    writes: Vec<(u32, [u8; 16])>,
    operations: Vec<Instruction>,
    initial_flags: u32,
    literal: Option<(u32, [u8; 16], Vec<u32>)>,
}

fn check(records: &[Record]) {
    let mut backends = [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic]
        .map(|backend| (backend, Cpu::new(backend).expect("CPU initialization")));
    for records in records.chunks(60) {
        let mut vectors: [[u8; 16]; 32] = std::array::from_fn(|register| seed(register as u32));
        let mut initial = vec![0xa5; 0x10000];
        for (register, value) in vectors.iter().enumerate() {
            initial[register * 16..register * 16 + 16].copy_from_slice(value);
        }
        let mut expected = initial.clone();
        let mut program = vec![0xd51b_441f];
        for register in 0..32u32 {
            program.push(0x3dc0_0060 | (register << 10) | register);
        }
        let mut expected_last_flags = 0;
        for (slot, record) in records.iter().enumerate() {
            assert!(
                record.writes.len() <= 4
                    && !record.operations.is_empty()
                    && record.operations.len() <= 3
            );
            for (part, &(register, value)) in record.writes.iter().enumerate() {
                let pos = INPUTS + slot * 64 + part * 16;
                initial[pos..pos + 16].copy_from_slice(&value);
                expected[pos..pos + 16].copy_from_slice(&value);
                program.push(0x3dc0_0020 | (((slot * 4 + part) as u32) << 10) | register);
                vectors[register as usize] = value;
            }
            program.push(0xd280_0004 | ((record.initial_flags & 0xffff) << 5));
            program.push(0xf2a0_0004 | ((record.initial_flags >> 16) << 5));
            program.push(0xd51b_4424);
            let mut flags = record.initial_flags;
            let mut record_flags = Vec::new();
            for (index, &operation) in record.operations.iter().enumerate() {
                model(operation, &mut vectors, &mut flags);
                program.push(operation.word());
                program.push(0xd53b_4425);
                program.push(0xf900_0045 | (((slot * (STRIDE / 8) + 64 + index) as u32) << 10));
                let pos = OUTPUTS + slot * STRIDE + 512 + index * 8;
                expected[pos..pos + 8].copy_from_slice(&(flags as u64).to_le_bytes());
                record_flags.push(flags);
            }
            expected_last_flags = flags;
            if let Some((register, vector, literal_flags)) = &record.literal {
                assert_eq!(
                    &vectors[*register as usize], vector,
                    "oracle literal: {}",
                    record.label
                );
                assert_eq!(
                    &record_flags, literal_flags,
                    "oracle flags: {}",
                    record.label
                );
            }
            for (register, value) in vectors.iter().enumerate() {
                let output = slot * (STRIDE / 16) + register;
                program.push(0x3d80_0040 | ((output as u32) << 10) | register as u32);
                let pos = OUTPUTS + slot * STRIDE + register * 16;
                expected[pos..pos + 16].copy_from_slice(value);
            }
        }
        program.push(0xd400_0fc1);
        let mut encoded = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
        for (index, word) in program.iter().enumerate() {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
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
            cpu.set_register(1, DATA + INPUTS as u64);
            cpu.set_register(2, DATA + OUTPUTS as u64);
            cpu.set_register(3, DATA);
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let event = cpu.run(100_000).expect("saturating narrow execution").event;
            assert!(
                matches!(event, CpuEvent::Svc(0x7e)),
                "{backend:?}: {event:?}"
            );
            for (slot, record) in records.iter().enumerate() {
                for register in 0..32 {
                    let pos = OUTPUTS + slot * STRIDE + register * 16;
                    assert_eq!(
                        &actual[pos..pos + 16],
                        &expected[pos..pos + 16],
                        "{backend:?}: {} V{register} {:?}",
                        record.label,
                        record.operations
                    );
                }
                let pos = OUTPUTS + slot * STRIDE + 512;
                assert_eq!(
                    &actual[pos..pos + 32],
                    &expected[pos..pos + 32],
                    "{backend:?}: {} exact sticky FPSR/guards {:?}",
                    record.label,
                    record.operations
                );
            }
            for register in 0..31 {
                let wanted = match register {
                    1 => DATA + INPUTS as u64,
                    2 => DATA + OUTPUTS as u64,
                    3 => DATA,
                    4 => records.last().unwrap().initial_flags as u64,
                    5 => expected_last_flags as u64,
                    _ => CANARY,
                };
                assert_eq!(
                    cpu.get_register(register),
                    wanted,
                    "{backend:?}: X{register}"
                );
            }
            assert_eq!(cpu.get_sp(), CANARY, "{backend:?}: SP");
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, actual.len() as u64).unwrap();
            }
            assert_eq!(actual, expected, "{backend:?}: source/memory guards");
            assert_eq!(code, encoded, "{backend:?}: code unchanged");
        }
    }
}

fn edge_values(operation: Instruction) -> Vec<u64> {
    let destination_bits = operation.destination_bytes() * 8;
    let source_bits = operation.source_bytes() * 8;
    let signed_min = -(1i128 << (destination_bits - 1));
    let signed_max = (1i128 << (destination_bits - 1)) - 1;
    let unsigned_max = (1i128 << destination_bits) - 1;
    let source_mask = u64::MAX >> (64 - source_bits);
    let mut values: Vec<_> = [
        0,
        1,
        -1,
        signed_min - 1,
        signed_min,
        signed_min + 1,
        signed_max - 1,
        signed_max,
        signed_max + 1,
        unsigned_max - 1,
        unsigned_max,
        unsigned_max + 1,
        -(1i128 << (source_bits - 1)),
        (1i128 << (source_bits - 1)) - 1,
        (1i128 << (source_bits - 1)),
        (1i128 << source_bits) - 1,
    ]
    .into_iter()
    .map(|value| value as u64 & source_mask)
    .collect();
    let mut state = 0x1234_5678_9abc_def0u64;
    for _ in 0..32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push(state & source_mask);
    }
    values
}

#[test]
fn all_saturating_narrow_formats_edges_aliases_and_exact_qc_match_dynarmic() {
    let mut records = Vec::new();
    for kind in [Kind::Signed, Kind::Unsigned, Kind::SignedToUnsigned] {
        for size in 0..3 {
            for (scalar, upper) in [(true, false), (false, false), (false, true)] {
                for (rn, rd) in [(3, 6), (3, 3), (31, 0), (0, 31)] {
                    let operation = Instruction {
                        kind,
                        size,
                        scalar,
                        upper,
                        rn,
                        rd,
                    };
                    let values = edge_values(operation);
                    for start in (0..values.len()).step_by(operation.lanes()) {
                        let mut source = seed(rn);
                        for index in 0..operation.lanes() {
                            put_integer(
                                &mut source,
                                operation.source_bytes(),
                                index,
                                values[(start + index) % values.len()],
                            );
                        }
                        records.push(Record {
                            label:
                                "all widths, mathematical saturation, aliases, upper half and QC"
                                    .into(),
                            writes: vec![(rd, seed(rd)), (rn, source)],
                            operations: vec![operation],
                            initial_flags: [0, QC, OTHER_FLAGS, QC | OTHER_FLAGS]
                                [(start / operation.lanes()) % 4],
                            literal: None,
                        });
                    }
                }
            }
        }
    }
    check(&records);
}

#[test]
fn saturation_in_each_active_lane_sets_qc_but_unused_scalar_lanes_do_not() {
    let mut records = Vec::new();
    for kind in [Kind::Signed, Kind::Unsigned, Kind::SignedToUnsigned] {
        for size in 0..3 {
            for (scalar, upper) in [(true, false), (false, false), (false, true)] {
                let operation = Instruction {
                    kind,
                    size,
                    scalar,
                    upper,
                    rn: 31,
                    rd: 0,
                };
                let overflow = if matches!(kind, Kind::SignedToUnsigned) {
                    u64::MAX
                } else {
                    (1u64 << (operation.destination_bytes() * 8)) + 1
                };
                for active in 0..operation.lanes() {
                    let mut source = [0; 16];
                    put_integer(&mut source, operation.source_bytes(), active, overflow);
                    records.push(Record {
                        label: format!("only lane {active} saturates"),
                        writes: vec![(0, seed(0)), (31, source)],
                        operations: vec![operation],
                        initial_flags: OTHER_FLAGS,
                        literal: None,
                    });
                }
                if scalar {
                    let mut source = [0xff; 16];
                    put_integer(&mut source, operation.source_bytes(), 0, 0);
                    records.push(Record {
                        label: "unused scalar upper lanes cannot set QC".into(),
                        writes: vec![(0, seed(0)), (31, source)],
                        operations: vec![operation],
                        initial_flags: OTHER_FLAGS,
                        literal: Some((0, [0; 16], vec![OTHER_FLAGS])),
                    });
                }
            }
        }
    }
    check(&records);
}

#[test]
fn qc_is_sticky_across_saturating_then_exact_operations_and_explicitly_resettable() {
    let mut records = Vec::new();
    for kind in [Kind::Signed, Kind::Unsigned, Kind::SignedToUnsigned] {
        for size in 0..3 {
            for (scalar, upper) in [(true, false), (false, false), (false, true)] {
                let operation = Instruction {
                    kind,
                    size,
                    scalar,
                    upper,
                    rn: 1,
                    rd: 4,
                };
                let mut overflow = [0; 16];
                put_integer(
                    &mut overflow,
                    operation.source_bytes(),
                    0,
                    1u64 << (operation.destination_bytes() * 8),
                );
                records.push(Record {
                    label: "sticky QC after one overflow and two exact operations".into(),
                    writes: vec![(1, overflow), (2, [0; 16]), (4, [0; 16]), (5, [0; 16])],
                    operations: vec![
                        operation,
                        Instruction {
                            rn: 2,
                            rd: 5,
                            ..operation
                        },
                        Instruction {
                            rn: 5,
                            rd: 5,
                            ..operation
                        },
                    ],
                    initial_flags: OTHER_FLAGS,
                    literal: Some((5, [0; 16], vec![QC | OTHER_FLAGS; 3])),
                });
                records.push(Record {
                    label: "explicit MSR FPSR clear after previous saturation".into(),
                    writes: vec![(1, [0; 16]), (4, [0; 16])],
                    operations: vec![operation],
                    initial_flags: 0,
                    literal: Some((4, [0; 16], vec![0])),
                });
            }
        }
    }
    check(&records);
}

#[test]
fn wonder_sqxtn_then_sqxtun_exact_alias_chain_and_qc_match_dynarmic() {
    let first = Instruction {
        kind: Kind::Signed,
        size: 1,
        scalar: false,
        upper: false,
        rn: 1,
        rd: 1,
    };
    let second = Instruction {
        kind: Kind::SignedToUnsigned,
        size: 0,
        ..first
    };
    assert_eq!(first.word(), 0x0e61_4821);
    assert_eq!(second.word(), 0x2e21_2821);
    let inputs = [
        (
            [-1i32, 0, 255, 256],
            [0, 0, 255, 255, 0, 0, 0, 0],
            vec![0, QC],
        ),
        (
            [-40000, 32767, 65535, -65536],
            [0, 255, 255, 0, 0, 0, 0, 0],
            vec![QC, QC],
        ),
        ([0, 1, 127, 255], [0, 1, 127, 255, 0, 0, 0, 0], vec![0, 0]),
    ];
    let mut records = Vec::new();
    for (values, bytes, flags) in inputs {
        let mut source = [0; 16];
        for (index, value) in values.into_iter().enumerate() {
            put_integer(&mut source, 4, index, value as u32 as u64);
        }
        let mut expected = [0; 16];
        expected[..8].copy_from_slice(&bytes);
        records.push(Record {
            label: "Wonder 925298c/9252998 SQXTN→SQXTUN".into(),
            writes: vec![(1, source)],
            operations: vec![first, second],
            initial_flags: 0,
            literal: Some((1, expected, flags)),
        });
    }
    check(&records);
}
