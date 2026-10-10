#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const INPUTS: usize = 0x1000;
const OUTPUTS: usize = 0x4000;
const CANARY: u64 = 0xface_1234_5678_9abc;

#[derive(Clone, Copy, Debug)]
struct Form {
    double: bool,
    scalar: bool,
    lanes: usize,
}

const FORMS: [Form; 5] = [
    Form {
        double: false,
        scalar: true,
        lanes: 1,
    },
    Form {
        double: true,
        scalar: true,
        lanes: 1,
    },
    Form {
        double: false,
        scalar: false,
        lanes: 2,
    },
    Form {
        double: false,
        scalar: false,
        lanes: 4,
    },
    Form {
        double: true,
        scalar: false,
        lanes: 2,
    },
];

impl Form {
    fn bytes(self) -> usize {
        if self.double {
            8
        } else {
            4
        }
    }
    fn instruction(self, rd: u32, rn: u32, rm: u32) -> u32 {
        (if self.scalar {
            0x7ea0_d400
        } else {
            0x2ea0_d400
        }) | ((self.double as u32) << 22)
            | if !self.scalar && self.lanes * self.bytes() == 16 {
                1 << 30
            } else {
                0
            }
            | (rm << 16)
            | (rn << 5)
            | rd
    }
}

fn format(double: bool) -> (u32, u64, u64, u64) {
    if double {
        (52, 1 << 63, 0x7ff0_0000_0000_0000, 1023)
    } else {
        (23, 1 << 31, 0x7f80_0000, 127)
    }
}

fn absolute_difference(n: u64, m: u64, double: bool) -> u64 {
    let (fraction, sign, infinity, _) = format(double);
    let mask = (1 << fraction) - 1;
    let quiet = 1 << (fraction - 1);
    let nan = |value: u64| value & infinity == infinity && value & mask != 0;
    for value in [n, m] {
        if nan(value) && value & quiet == 0 {
            return (value | quiet) & !sign;
        }
    }
    for value in [n, m] {
        if nan(value) {
            return value & !sign;
        }
    }
    let ninf = n & !sign == infinity;
    let minf = m & !sign == infinity;
    if ninf && minf && (n ^ m) & sign == 0 {
        return infinity | quiet;
    }
    if ninf || minf {
        return infinity;
    }
    if double {
        (f64::from_bits(n) - f64::from_bits(m)).to_bits() & !sign
    } else {
        (f32::from_bits(n as u32) - f32::from_bits(m as u32)).to_bits() as u64 & !sign
    }
}

fn seed(register: u32) -> [u8; 16] {
    std::array::from_fn(|byte| (register as usize * 37 + byte * 11 + 0x93) as u8)
}

fn vector(form: Form, values: &[u64], start: usize, register: u32) -> [u8; 16] {
    let mut result = seed(register);
    for index in 0..form.lanes {
        let offset = index * form.bytes();
        result[offset..offset + form.bytes()]
            .copy_from_slice(&values[(start + index) % values.len()].to_le_bytes()[..form.bytes()]);
    }
    result
}

fn lane(value: &[u8; 16], form: Form, index: usize) -> u64 {
    let mut bits = [0; 8];
    bits[..form.bytes()].copy_from_slice(&value[index * form.bytes()..(index + 1) * form.bytes()]);
    u64::from_le_bytes(bits)
}

#[derive(Debug)]
struct Record {
    form: Form,
    rn: u32,
    rm: u32,
    rd: u32,
    n: [u8; 16],
    m: [u8; 16],
}

fn check(records: &[Record]) {
    let mut backends = [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic]
        .map(|kind| (kind, Cpu::new(kind).expect("CPU initialization")));
    for records in records.chunks(60) {
        let mut vectors: [[u8; 16]; 32] = std::array::from_fn(|reg| seed(reg as u32));
        let mut initial = vec![0xa5; 0x10000];
        for (reg, value) in vectors.iter().enumerate() {
            initial[reg * 16..reg * 16 + 16].copy_from_slice(value);
        }
        let mut expected = initial.clone();
        let mut program = vec![0xd51b_441f, 0xd51b_443f];
        for reg in 0..32u32 {
            program.push(0x3dc0_0060 | (reg << 10) | reg);
        }
        for (slot, record) in records.iter().enumerate() {
            let f = record.form;
            let (rd, rn, rm) = (record.rd as usize, record.rn as usize, record.rm as usize);
            let pos = INPUTS + slot * 32;
            initial[pos..pos + 16].copy_from_slice(&record.n);
            initial[pos + 16..pos + 32].copy_from_slice(&record.m);
            expected[pos..pos + 32].copy_from_slice(&initial[pos..pos + 32]);
            program.push(0x3dc0_0060 | (record.rd << 10) | record.rd);
            vectors[rd] = seed(record.rd);
            program.push(0x3dc0_0020 | (((slot * 2) as u32) << 10) | record.rn);
            vectors[rn] = record.n;
            if rn != rm {
                program.push(0x3dc0_0020 | (((slot * 2 + 1) as u32) << 10) | record.rm);
                vectors[rm] = record.m;
            }
            let n = vectors[rn];
            let m = vectors[rm];
            let mut result = [0; 16];
            for index in 0..f.lanes {
                let bits = absolute_difference(lane(&n, f, index), lane(&m, f, index), f.double);
                let pos = index * f.bytes();
                result[pos..pos + f.bytes()].copy_from_slice(&bits.to_le_bytes()[..f.bytes()]);
            }
            vectors[rd] = result;
            program.push(f.instruction(record.rd, record.rn, record.rm));
            for (reg, value) in vectors.iter().enumerate() {
                let offset = slot * 32 + reg;
                program.push(0x3d80_0040 | ((offset as u32) << 10) | reg as u32);
                let pos = OUTPUTS + offset * 16;
                expected[pos..pos + 16].copy_from_slice(value);
            }
        }
        program.extend([0xd53b_4408, 0xd400_0fc1]);
        let mut encoded = vec![0; (program.len() * 4).next_multiple_of(0x1000)];
        for (index, word) in program.iter().enumerate() {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        for (kind, cpu) in &mut backends {
            let mut code = encoded.clone();
            let mut actual = initial.clone();
            unsafe {
                cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
                cpu.map_host(DATA, actual.len() as u64, Perm::RW, actual.as_mut_ptr())
                    .unwrap();
            }
            for reg in 0..31 {
                cpu.set_register(reg, CANARY);
            }
            cpu.set_register(1, DATA + INPUTS as u64);
            cpu.set_register(2, DATA + OUTPUTS as u64);
            cpu.set_register(3, DATA);
            cpu.set_sp(CANARY);
            cpu.set_pc(CODE);
            let event = cpu.run(100_000).expect("FABD execution").event;
            assert!(matches!(event, CpuEvent::Svc(0x7e)), "{kind:?}: {event:?}");
            for (slot, record) in records.iter().enumerate() {
                for reg in 0..32 {
                    let pos = OUTPUTS + (slot * 32 + reg) * 16;
                    assert_eq!(
                        &actual[pos..pos + 16],
                        &expected[pos..pos + 16],
                        "{kind:?}: {record:?} V{reg}"
                    );
                }
            }
            for reg in 0..31 {
                let wanted = match reg {
                    1 => DATA + INPUTS as u64,
                    2 => DATA + OUTPUTS as u64,
                    3 => DATA,
                    8 => 0,
                    _ => CANARY,
                };
                assert_eq!(cpu.get_register(reg), wanted, "{kind:?}: X{reg}");
            }
            assert_eq!(cpu.get_sp(), CANARY, "{kind:?}: SP");
            unsafe {
                cpu.unmap_host(CODE, code.len() as u64).unwrap();
                cpu.unmap_host(DATA, actual.len() as u64).unwrap();
            }
            assert_eq!(actual, expected, "{kind:?}: source/memory guards");
            assert_eq!(code, encoded, "{kind:?}: code unchanged");
        }
    }
}

fn finite_pairs(double: bool) -> Vec<(u64, u64)> {
    let (fraction, sign, infinity, bias) = format(double);
    let one = bias << fraction;
    let two = (bias + 1) << fraction;
    let half = (bias - 1) << fraction;
    let max_subnormal = (1 << fraction) - 1;
    let mut values = Vec::new();
    for value in [
        0,
        1,
        2,
        max_subnormal,
        1 << fraction,
        (1 << fraction) + 1,
        half,
        one - 1,
        one,
        one + 1,
        two,
        infinity - 1,
        infinity - (1 << fraction),
    ] {
        values.extend([value, value | sign]);
    }
    let mut state = 0x1234_5678_9abc_def0u64;
    for _ in 0..32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = state & (sign | ((1 << fraction) - 1))
            | ((state.rotate_left(17) % (bias * 2 + 1)) << fraction);
        values.push(value);
    }
    let mut pairs = Vec::new();
    for (index, &value) in values.iter().enumerate() {
        let other = values[(index * 17 + 3) % values.len()];
        pairs.extend([(value, value), (value, 0), (value, other), (other, value)]);
    }
    pairs
}

fn records(pairs: fn(bool) -> Vec<(u64, u64)>) -> Vec<Record> {
    let mut records = Vec::new();
    for form in FORMS {
        let pairs = pairs(form.double);
        let n: Vec<_> = pairs.iter().map(|pair| pair.0).collect();
        let m: Vec<_> = pairs.iter().map(|pair| pair.1).collect();
        for (rn, rm, rd) in [
            (1, 2, 1),
            (1, 2, 2),
            (3, 7, 6),
            (3, 3, 3),
            (31, 0, 31),
            (0, 31, 0),
            (31, 31, 2),
        ] {
            for start in (0..pairs.len()).step_by(form.lanes) {
                records.push(Record {
                    form,
                    rn,
                    rm,
                    rd,
                    n: vector(form, &n, start, rn),
                    m: vector(form, &m, start, rm),
                });
            }
        }
    }
    records
}

#[test]
fn fabd_finite_rounding_subnormals_signed_zeros_all_formats_and_aliases() {
    assert_eq!(
        absolute_difference(0x3f80_0000, 0xc000_0000, false),
        0x4040_0000
    );
    assert_eq!(absolute_difference(0x8000_0000, 0, false), 0);
    assert_eq!(
        absolute_difference(0x3ff0_0000_0000_0001, 0x3ff0_0000_0000_0000, true),
        0x3cb0_0000_0000_0000
    );
    check(&records(finite_pairs));
}

fn special_pairs(double: bool) -> Vec<(u64, u64)> {
    let (fraction, sign, infinity, bias) = format(double);
    let one = bias << fraction;
    let quiet = infinity | (1 << (fraction - 1));
    let mut pairs = Vec::new();
    let values = [
        0,
        sign,
        infinity,
        sign | infinity,
        one,
        sign | one,
        infinity | 0x123,
        sign | infinity | 0x456,
        quiet | 0x123,
        sign | quiet | 0x456,
    ];
    for n in values {
        for m in values {
            pairs.push((n, m));
        }
    }
    pairs
}

#[test]
fn fabd_nan_signaling_priority_payloads_and_infinity_invalid_results() {
    assert_eq!(
        absolute_difference(0x7fc0_0123, 0xff80_0456, false),
        0x7fc0_0456
    );
    assert_eq!(
        absolute_difference(0x7ff8_0000_0000_0123, 0xfff0_0000_0000_0456, true),
        0x7ff8_0000_0000_0456
    );
    assert_eq!(
        absolute_difference(0x7f80_0000, 0x7f80_0000, false),
        0x7fc0_0000
    );
    assert_eq!(
        absolute_difference(0xfff0_0000_0000_0000, 0x7ff0_0000_0000_0000, true),
        0x7ff0_0000_0000_0000
    );
    check(&records(special_pairs));
}

#[test]
fn wonder_exact_fabd_two_single_lanes_alias_destination() {
    let form = FORMS[2];
    assert_eq!(form.instruction(1, 1, 2), 0x2ea2_d421);
    check(&[Record {
        form,
        rn: 1,
        rm: 2,
        rd: 1,
        n: vector(form, &[0x3f80_0000, 0xc000_0000], 0, 1),
        m: vector(form, &[0x4080_0000, 0x4040_0000], 0, 2),
    }]);
}
