#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const RESULTS: usize = 0x2000;
const CANARY: u64 = 0xface_1234_5678_9abc;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Fmadd,
    Fmsub,
    Fnmadd,
    Fnmsub,
}

impl Kind {
    fn index(self) -> usize {
        match self {
            Self::Fmadd => 0,
            Self::Fmsub => 1,
            Self::Fnmadd => 2,
            Self::Fnmsub => 3,
        }
    }

    fn opcode(self, double: bool, rd: u32) -> u32 {
        let variant = match self {
            Self::Fmadd => 0,
            Self::Fmsub => 1 << 15,
            Self::Fnmadd => 1 << 21,
            Self::Fnmsub => (1 << 21) | (1 << 15),
        };
        0x1f02_0c20 | (u32::from(double) << 22) | variant | rd
    }
}

struct Case {
    name: &'static str,
    n: u64,
    m: u64,
    a: [u64; 4],
    expected: [u64; 4],
}

fn cases(double: bool) -> Vec<Case> {
    let (one, two, three, five, eleven, plus, minus, epsilon, sign, maximum, nan, infinity) =
        if double {
            (
                0x3ff0_0000_0000_0000,
                0x4000_0000_0000_0000,
                0x4008_0000_0000_0000,
                0x4014_0000_0000_0000,
                0x4026_0000_0000_0000,
                0x3ff0_0000_0000_0001,
                0x3fef_ffff_ffff_fffe,
                0x3970_0000_0000_0000,
                0x8000_0000_0000_0000,
                0x7fef_ffff_ffff_ffff,
                0x7ff8_1234_5678_9abc,
                0x7ff0_0000_0000_0000,
            )
        } else {
            (
                0x3f80_0000,
                0x4000_0000,
                0x4040_0000,
                0x40a0_0000,
                0x4130_0000,
                0x3f80_0001,
                0x3f7f_fffe,
                0x2880_0000,
                0x8000_0000,
                0x7f7f_ffff,
                0x7fc1_2345,
                0x7f80_0000,
            )
        };
    let quiet_bit = if double { 1 << 51 } else { 1 << 22 };
    let nan_n = nan;
    let nan_m = nan ^ 0x55;
    let nan_a = nan ^ 0xaa;
    let snan_n = nan_n & !quiet_bit;
    let snan_m = nan_m & !quiet_bit;
    let snan_a = nan_a & !quiet_bit;
    let default_nan = if double {
        0x7ff8_0000_0000_0000
    } else {
        0x7fc0_0000
    };
    vec![
        Case {
            name: "fused cancellation",
            n: plus,
            m: minus,
            a: [one | sign, one, one | sign, one],
            expected: [epsilon | sign, epsilon, epsilon, epsilon | sign],
        },
        Case {
            name: "overflowing product rescued by addend",
            n: maximum,
            m: two,
            a: [maximum | sign, maximum, maximum | sign, maximum],
            expected: [maximum, maximum | sign, maximum | sign, maximum],
        },
        Case {
            name: "ordinary arithmetic",
            n: two,
            m: three,
            a: [five; 4],
            expected: [eleven, one | sign, eleven | sign, one],
        },
        Case {
            name: "exact cancellation has positive zero",
            n: one,
            m: one,
            a: [one | sign, one, one | sign, one],
            expected: [0; 4],
        },
        Case {
            name: "positive zero inputs",
            n: 0,
            m: one,
            a: [0; 4],
            expected: [0, 0, sign, 0],
        },
        Case {
            name: "negative zero product and addend",
            n: sign,
            m: one,
            a: [sign; 4],
            expected: [sign, 0, 0, 0],
        },
        Case {
            name: "quiet NaN in multiplicand",
            n: nan,
            m: one,
            a: [one; 4],
            expected: [nan, nan | sign, nan | sign, nan],
        },
        Case {
            name: "quiet NaN in multiplier",
            n: one,
            m: nan,
            a: [one; 4],
            expected: [nan; 4],
        },
        Case {
            name: "quiet NaN in addend",
            n: one,
            m: one,
            a: [nan; 4],
            expected: [nan, nan, nan | sign, nan | sign],
        },
        Case {
            name: "infinite product",
            n: infinity,
            m: one,
            a: [one; 4],
            expected: [infinity, infinity | sign, infinity | sign, infinity],
        },
        Case {
            name: "signaling NaN in multiplicand is quieted",
            n: snan_n,
            m: one,
            a: [one; 4],
            expected: [nan_n, nan_n | sign, nan_n | sign, nan_n],
        },
        Case {
            name: "signaling NaN in multiplier is quieted",
            n: one,
            m: snan_m,
            a: [one; 4],
            expected: [nan_m; 4],
        },
        Case {
            name: "signaling NaN in addend is quieted",
            n: one,
            m: one,
            a: [snan_a; 4],
            expected: [nan_a, nan_a, nan_a | sign, nan_a | sign],
        },
        Case {
            name: "addend wins three quiet NaNs",
            n: nan_n,
            m: nan_m,
            a: [nan_a; 4],
            expected: [nan_a, nan_a, nan_a | sign, nan_a | sign],
        },
        Case {
            name: "addend wins three signaling NaNs",
            n: snan_n,
            m: snan_m,
            a: [snan_a; 4],
            expected: [nan_a, nan_a, nan_a | sign, nan_a | sign],
        },
        Case {
            name: "signaling multiplicand beats quiet addend and multiplier",
            n: snan_n,
            m: nan_m,
            a: [nan_a; 4],
            expected: [nan_n, nan_n | sign, nan_n | sign, nan_n],
        },
        Case {
            name: "signaling multiplier beats quiet addend and multiplicand",
            n: nan_n,
            m: snan_m,
            a: [nan_a; 4],
            expected: [nan_m; 4],
        },
        Case {
            name: "multiplicand wins two signaling NaNs",
            n: snan_n,
            m: snan_m,
            a: [one; 4],
            expected: [nan_n, nan_n | sign, nan_n | sign, nan_n],
        },
        Case {
            name: "multiplicand wins two quiet NaNs",
            n: nan_n,
            m: nan_m,
            a: [one; 4],
            expected: [nan_n, nan_n | sign, nan_n | sign, nan_n],
        },
        Case {
            name: "invalid zero times infinity produces positive default NaN",
            n: 0,
            m: infinity,
            a: [one; 4],
            expected: [default_nan; 4],
        },
        Case {
            name: "opposite infinities produce positive default NaN",
            n: infinity,
            m: one,
            a: [infinity | sign, infinity, infinity | sign, infinity],
            expected: [default_nan; 4],
        },
        Case {
            name: "invalid product overrides quiet NaN addend payload",
            n: 0,
            m: infinity,
            a: [nan_a; 4],
            expected: [default_nan; 4],
        },
        Case {
            name: "signaling NaN addend precedes invalid product",
            n: 0,
            m: infinity,
            a: [snan_a; 4],
            expected: [nan_a, nan_a, nan_a | sign, nan_a | sign],
        },
    ]
}

fn run(backend: CpuBackendKind, program: &[u32], initial: &[u8]) -> Vec<u8> {
    let mut code = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
    for (index, word) in program.iter().enumerate() {
        code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    let mut data = initial.to_vec();
    let mut cpu = Cpu::new(backend).expect("CPU initialization");
    unsafe {
        cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
            .unwrap();
        cpu.map_host(DATA, data.len() as u64, Perm::RW, data.as_mut_ptr())
            .unwrap();
    }
    for reg in 0..31 {
        cpu.set_register(reg, CANARY);
    }
    cpu.set_pc(CODE);
    cpu.set_register(1, DATA);
    cpu.set_register(2, DATA + RESULTS as u64);
    let event = cpu.run(100_000).expect("scalar fused arithmetic").event;
    assert!(
        matches!(event, CpuEvent::Svc(0x7e)),
        "{backend:?}: {event:?}"
    );
    for reg in 0..31 {
        assert_eq!(
            cpu.get_register(reg),
            match reg {
                1 => DATA,
                2 => DATA + RESULTS as u64,
                _ => CANARY,
            },
            "{backend:?}: preserved X{reg}"
        );
    }
    drop(cpu);
    data
}

fn check(kind: Kind, double: bool) {
    let cases = cases(double);
    let mut initial = vec![0xa5; 0x10000];
    let mut expected = initial.clone();
    let mut program = Vec::new();
    let mut diagnostics = Vec::new();
    let mut slot = 0;
    for rd in [0, 1, 2, 3, 31] {
        for case in &cases {
            let inputs = [case.n, case.m, case.a[kind.index()]];
            let input_offset = slot * 48;
            let result_offset = RESULTS + slot * 128;
            let mut output = [0; 16];
            output[..8].copy_from_slice(&case.expected[kind.index()].to_le_bytes());
            for (lane, bits) in inputs.iter().enumerate() {
                let offset = input_offset + lane * 16;
                let low = bits | if double { 0 } else { 0xdead_beef_0000_0000 };
                initial[offset..offset + 8].copy_from_slice(&low.to_le_bytes());
                initial[offset + 8..offset + 16].copy_from_slice(&CANARY.to_le_bytes());
            }
            expected[input_offset..input_offset + 48]
                .copy_from_slice(&initial[input_offset..input_offset + 48]);
            program.push(0x3dc0_0020 | (((slot * 3) as u32) << 10) | rd);
            for rn in 1..=3 {
                program.push(0x3dc0_0020 | (((slot * 3 + rn - 1) as u32) << 10) | rn as u32);
            }
            let opcode = kind.opcode(double, rd);
            program.push(opcode);
            program.push(0x3d80_0040 | (((slot * 8) as u32) << 10) | rd);
            expected[result_offset..result_offset + 16].copy_from_slice(&output);
            for rn in 1..=3 {
                program.push(0x3d80_0040 | (((slot * 8 + rn * 2) as u32) << 10) | rn as u32);
                let source = if rd == rn as u32 {
                    &output[..]
                } else {
                    let offset = input_offset + (rn - 1) * 16;
                    &initial[offset..offset + 16]
                };
                let offset = result_offset + rn * 32;
                expected[offset..offset + 16].copy_from_slice(source);
            }
            diagnostics.push((
                result_offset,
                opcode,
                case.name,
                case.expected[kind.index()],
            ));
            slot += 1;
        }
    }
    assert!(slot * 48 < RESULTS && RESULTS + slot * 128 <= initial.len());
    program.push(0xd400_0fc1);
    let reference = run(CpuBackendKind::Dynarmic, &program, &initial);
    for &(offset, opcode, name, expected_bits) in &diagnostics {
        let reference_bits = u64::from_le_bytes(reference[offset..offset + 8].try_into().unwrap());
        assert_eq!(
            reference_bits, expected_bits,
            "reference {kind:?} opcode={opcode:#010x}: {name}; reference={reference_bits:#018x}, expected={expected_bits:#018x}"
        );
    }
    assert_memory_eq(
        &reference,
        &expected,
        "reference sources, upper-zero and canaries",
    );
    let actual = run(CpuBackendKind::Rustarmic, &program, &initial);
    for (offset, opcode, name, expected_bits) in diagnostics {
        let actual_bits = u64::from_le_bytes(actual[offset..offset + 8].try_into().unwrap());
        assert_eq!(
            actual_bits, expected_bits,
            "{kind:?} opcode={opcode:#010x}: {name}; actual={actual_bits:#018x}, expected={expected_bits:#018x}"
        );
    }
    assert_memory_eq(
        &actual,
        &expected,
        "FMA result upper-zero, sources and canaries",
    );
    assert_memory_eq(
        &actual,
        &reference,
        "scalar fused value parity; FPSR excluded",
    );
}

fn assert_memory_eq(actual: &[u8], expected: &[u8], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: memory length");
    if let Some(offset) = actual.iter().zip(expected).position(|(a, e)| a != e) {
        let end = (offset + 16).min(actual.len());
        panic!(
            "{label}: first mismatch at data+{offset:#x}: actual={:02x?}, expected={:02x?}",
            &actual[offset..end],
            &expected[offset..end]
        );
    }
}

#[test]
fn fmadd_single_matches_dynarmic() {
    check(Kind::Fmadd, false);
}
#[test]
fn fmadd_double_matches_dynarmic() {
    check(Kind::Fmadd, true);
}
#[test]
fn fmsub_single_matches_dynarmic() {
    check(Kind::Fmsub, false);
}
#[test]
fn fmsub_double_matches_dynarmic() {
    check(Kind::Fmsub, true);
}
#[test]
fn fnmadd_single_matches_dynarmic() {
    check(Kind::Fnmadd, false);
}
#[test]
fn fnmadd_double_matches_dynarmic() {
    check(Kind::Fnmadd, true);
}
#[test]
fn fnmsub_single_matches_dynarmic() {
    check(Kind::Fnmsub, false);
}
#[test]
fn fnmsub_double_matches_dynarmic() {
    check(Kind::Fnmsub, true);
}
