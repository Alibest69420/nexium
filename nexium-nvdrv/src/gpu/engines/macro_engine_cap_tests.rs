use super::*;

const DRAW_INDEXED: [u32; 23] = [
    0x0000_0501,
    0x0346_c215,
    0x0000_0401,
    0x017d_c061,
    0x0111_8351,
    0x0000_1841,
    0x0543_4021,
    0x0015_1410,
    0x0000_1a31,
    0x0002_e007,
    0x0000_1041,
    0x0161_8021,
    0x0000_0841,
    0x017e_0021,
    0x0000_2841,
    0x0161_4071,
    0xffff_e411,
    0x0000_4211,
    0xfffe_6017,
    0xd080_8912,
    0x0111_8071,
    0x0143_40f1,
    0x0143_8071,
];
const PARAMS: [u32; 6] = [3, 6, u32::MAX, 17, 33, 41];

fn decode() -> Vec<DecodedOpcode> {
    DRAW_INDEXED
        .iter()
        .copied()
        .map(Opcode)
        .map(DecodedOpcode::from)
        .collect()
}

#[test]
fn finite_draw_indexed_1024_instances_hits_3073_write_cutoff_in_both_interpreters() {
    assert_eq!(macro_hash(&DRAW_INDEXED), 0x771b_b18c_6244_4da0);
    let registers = |method| {
        if method == REG_DRAW_INSTANCE_COUNT {
            1024
        } else {
            0
        }
    };
    let mut legacy = Interpreter::new(&DRAW_INDEXED, &PARAMS, &registers);
    let decoded = decode();
    let mut fast = FastInterpreter::new(&decoded, &PARAMS, &registers, Vec::new());
    legacy.run();
    fast.run();
    assert_eq!(legacy.cap_writes, Some(3073));
    assert_eq!(fast.cap_writes, Some(3073));
    assert_eq!((legacy.steps_remaining, fast.steps_remaining), (0, 0));
    assert!(legacy.writes.is_empty());
    assert!(fast.writes.is_empty());
    assert_eq!(legacy.pc, fast.pc);
    assert_eq!(legacy.pc, 16 * 4);
    assert_eq!(legacy.registers[4], 2);
    assert_eq!(fast.registers[4], 2);
    assert_eq!(legacy.next_param, PARAMS.len());
    assert_eq!(fast.next_param, PARAMS.len());
}

#[test]
fn finite_draw_indexed_1022_instances_exits_and_preserves_exact_writes() {
    let registers = |method| {
        if method == REG_DRAW_INSTANCE_COUNT {
            1022
        } else {
            0
        }
    };
    let mut legacy = Interpreter::new(&DRAW_INDEXED, &PARAMS, &registers);
    let decoded = decode();
    let mut fast = FastInterpreter::new(&decoded, &PARAMS, &registers, Vec::new());
    legacy.run();
    fast.run();
    assert_eq!(legacy.cap_writes, None);
    assert_eq!(fast.cap_writes, None);
    assert_eq!(legacy.writes, fast.writes);
    assert_eq!(legacy.writes.len(), 1022 * 3 + 7);
    assert_eq!(
        legacy
            .writes
            .iter()
            .filter(|(m, _)| *m == REG_DRAW_BEGIN)
            .count(),
        1022
    );
    assert_eq!(
        legacy
            .writes
            .iter()
            .filter(|(m, _)| *m == REG_DRAW_END)
            .count(),
        1022
    );
    assert_eq!(
        &legacy.writes[..4],
        &[
            (REG_INDEX_FIRST, 17),
            (REG_VERTEX_ID_BASE, 33),
            (REG_GLOBAL_BASE_VERTEX, 33),
            (REG_GLOBAL_BASE_INSTANCE, 41)
        ]
    );
    assert_eq!(
        &legacy.writes[legacy.writes.len() - 3..],
        &[
            (REG_VERTEX_ID_BASE, 0),
            (REG_GLOBAL_BASE_VERTEX, 0),
            (REG_GLOBAL_BASE_INSTANCE, 0)
        ]
    );
    assert_eq!(legacy.steps_remaining, fast.steps_remaining);
    assert_eq!(legacy.registers[4], 0);
}

fn full_draw_indexed_oracle(instances: u32) -> Vec<(u32, u32)> {
    assert!(instances > 0);
    let mut result = vec![
        (REG_INDEX_FIRST, 17),
        (REG_VERTEX_ID_BASE, 33),
        (REG_GLOBAL_BASE_VERTEX, 33),
        (REG_GLOBAL_BASE_INSTANCE, 41),
    ];
    for instance in 0..instances {
        let topology = if instance == 0 { 3 } else { 3 | (1 << 26) };
        result.extend([
            (REG_DRAW_BEGIN, topology),
            (REG_INDEX_COUNT, 6),
            (REG_DRAW_END, 0),
        ]);
    }
    result.extend([
        (REG_VERTEX_ID_BASE, 0),
        (REG_GLOBAL_BASE_VERTEX, 0),
        (REG_GLOBAL_BASE_INSTANCE, 0),
    ]);
    result
}

#[test]
fn independent_full_write_oracle_agrees_with_existing_draw_hle() {
    for instances in [1, 2, 128] {
        let out = hle_macro(
            HleMacro::DrawIndexed {
                shader_parameters: false,
            },
            &PARAMS,
            &|method| {
                if method == REG_DRAW_INSTANCE_COUNT {
                    instances
                } else {
                    0
                }
            },
            Vec::new(),
        )
        .unwrap();
        assert_eq!(out.writes, full_draw_indexed_oracle(instances));
    }
}

#[test]
fn exact_draw_indexed_extended_budget_completes_both_interpreters_through_ceiling() {
    for instances in [1022, 1023, 1024, 4096] {
        let registers = |method| {
            if method == REG_DRAW_INSTANCE_COUNT {
                instances
            } else {
                0
            }
        };
        let (budget, extended) = lle_step_budget(
            &DRAW_INDEXED,
            macro_hash(&DRAW_INDEXED),
            &PARAMS,
            &registers,
        );
        assert_eq!(budget, 8192.max(instances * 8 + 14));
        assert_eq!(
            extended,
            if instances > 1022 {
                Some(instances)
            } else {
                None
            }
        );
        let mut legacy =
            Interpreter::new(&DRAW_INDEXED, &PARAMS, &registers).with_step_budget(budget);
        let decoded = decode();
        let mut fast = FastInterpreter::new(&decoded, &PARAMS, &registers, Vec::new())
            .with_step_budget(budget);
        legacy.run();
        fast.run();
        assert_eq!(legacy.cap_writes, None);
        assert_eq!(fast.cap_writes, None);
        assert_eq!(legacy.writes, full_draw_indexed_oracle(instances));
        assert_eq!(fast.writes, legacy.writes);
        assert_eq!(budget - legacy.steps_remaining, instances * 8 + 14);
        assert_eq!(legacy.steps_remaining, fast.steps_remaining);
        assert_eq!(legacy.registers[4], 0);
        assert_eq!(fast.registers[4], 0);
    }
}

#[test]
fn draw_indexed_hle_limit_falls_back_to_complete_scoped_lle() {
    for fast_lle in [false, true] {
        let mut engine = MacroEngine::new();
        engine.fast_lle = fast_lle;
        engine.set_instruction_ptr(0);
        for word in DRAW_INDEXED {
            engine.upload_instruction(word);
        }
        engine.set_start_address_ptr(0);
        engine.bind_macro_entry(0);
        let output = engine
            .on_macro_methods(MACRO_REGISTERS_START, &PARAMS, true, &|method| {
                if method == REG_DRAW_INSTANCE_COUNT {
                    1024
                } else {
                    0
                }
            })
            .unwrap();
        assert_eq!(output.hash, 0x771b_b18c_6244_4da0);
        assert_eq!(output.entry, 0);
        assert!(!output.hle);
        assert_eq!(output.writes, full_draw_indexed_oracle(1024));
    }
}

#[test]
fn changed_unknown_huge_and_incomplete_inputs_retain_original_budget() {
    let hash = macro_hash(&DRAW_INDEXED);
    for instances in [0, 1, 1022, 4097, u32::MAX] {
        let reader = |method| {
            if method == REG_DRAW_INSTANCE_COUNT {
                instances
            } else {
                0
            }
        };
        assert_eq!(
            lle_step_budget(&DRAW_INDEXED, hash, &PARAMS, &reader),
            (8192, None)
        );
    }
    let reader = |method| {
        if method == REG_DRAW_INSTANCE_COUNT {
            u32::MAX
        } else {
            0
        }
    };
    let mut masked = PARAMS;
    masked[2] = 1024;
    assert_eq!(
        lle_step_budget(&DRAW_INDEXED, hash, &masked, &reader),
        (8206, Some(1024))
    );
    assert_eq!(
        lle_step_budget(&DRAW_INDEXED, hash ^ 1, &PARAMS, &reader),
        (8192, None)
    );
    assert_eq!(
        lle_step_budget(&DRAW_INDEXED, hash, &PARAMS[..5], &reader),
        (8192, None)
    );
    let mut extra = PARAMS.to_vec();
    extra.push(0);
    assert_eq!(
        lle_step_budget(&DRAW_INDEXED, hash, &extra, &reader),
        (8192, None)
    );
    let mut changed = DRAW_INDEXED;
    changed[0] ^= 0x100;
    assert_eq!(
        lle_step_budget(&changed, hash, &PARAMS, &reader),
        (8192, None)
    );
    for instances in [4097, u32::MAX] {
        let reader = |method| {
            if method == REG_DRAW_INSTANCE_COUNT {
                instances
            } else {
                0
            }
        };
        let (budget, _) = lle_step_budget(&DRAW_INDEXED, hash, &PARAMS, &reader);
        let mut legacy = Interpreter::new(&DRAW_INDEXED, &PARAMS, &reader).with_step_budget(budget);
        let decoded = decode();
        let mut fast =
            FastInterpreter::new(&decoded, &PARAMS, &reader, Vec::new()).with_step_budget(budget);
        legacy.run();
        fast.run();
        assert_eq!(legacy.cap_writes, Some(3073));
        assert_eq!(fast.cap_writes, Some(3073));
        assert!(legacy.writes.is_empty());
        assert!(fast.writes.is_empty());
    }
}

#[test]
fn unrecognized_infinite_loop_still_stops_at_original_budget() {
    let code = [0x0000_0027];
    let (budget, extended) = lle_step_budget(&code, macro_hash(&code), &PARAMS, &|_| 1024);
    assert_eq!((budget, extended), (8192, None));
    let mut legacy = Interpreter::new(&code, &PARAMS, &|_| 0).with_step_budget(budget);
    let decoded: Vec<_> = code
        .into_iter()
        .map(Opcode)
        .map(DecodedOpcode::from)
        .collect();
    let mut fast =
        FastInterpreter::new(&decoded, &PARAMS, &|_| 0, Vec::new()).with_step_budget(budget);
    legacy.run();
    fast.run();
    assert_eq!((legacy.steps_remaining, fast.steps_remaining), (0, 0));
    assert_eq!(legacy.cap_writes, Some(0));
    assert_eq!(fast.cap_writes, Some(0));
    assert!(legacy.writes.is_empty());
    assert!(fast.writes.is_empty());
    assert_eq!((legacy.pc, fast.pc), (0, 0));
}
