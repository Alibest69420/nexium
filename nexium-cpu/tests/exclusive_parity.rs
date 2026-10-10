#![cfg(feature = "backend-rustarmic")]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;
use std::sync::{Arc, Barrier};

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const ALIAS: u64 = 0x30_0000;
const PAGE: usize = 0x1000;
const OFFSET: usize = 0x80;
const CANARY: u64 = 0xface_1234_5678_9abc;
const INITIAL: u64 = 0x8877_6655_4433_2211;
const DESIRED: u64 = 0x1234_5678_a1b2_c3d4;
const LOAD: usize = 0;
const STORE: usize = 0x20;
const LOAD_STORE: usize = 0x40;
const PLAIN_STORE: usize = 0x60;
const CLEAR_STORE: usize = 0x80;
const SVC: u32 = 0xd400_0fc1;

#[repr(align(64))]
struct AlignedPage([u8; PAGE]);

fn page(initial: u64) -> Box<AlignedPage> {
    let mut backing = Box::new(AlignedPage([0xa5; PAGE]));
    backing.0[OFFSET..OFFSET + 8].copy_from_slice(&initial.to_le_bytes());
    backing
}

fn size_bits(bytes: usize) -> u32 {
    match bytes {
        1 => 0,
        2 => 1 << 30,
        4 => 2 << 30,
        8 => 3 << 30,
        _ => unreachable!(),
    }
}

fn load_exclusive(bytes: usize, acquire: bool) -> u32 {
    0x085f_7c01 | size_bits(bytes) | ((acquire as u32) << 15)
}

fn store_exclusive(bytes: usize, release: bool) -> u32 {
    0x0802_7c03 | size_bits(bytes) | ((release as u32) << 15)
}

fn plain_store(bytes: usize) -> u32 {
    0x3900_0003 | size_bits(bytes)
}

fn mask(bytes: usize) -> u64 {
    if bytes == 8 {
        u64::MAX
    } else {
        (1u64 << (bytes * 8)) - 1
    }
}

struct Fixture {
    cpu: Cpu,
    code: Box<AlignedPage>,
}

impl Fixture {
    fn new(
        backend: CpuBackendKind,
        backing: *mut u8,
        va: u64,
        perm: Perm,
        bytes: usize,
        acquire: bool,
        release: bool,
    ) -> Self {
        let mut code = Box::new(AlignedPage([0; PAGE]));
        let load = load_exclusive(bytes, acquire);
        let store = store_exclusive(bytes, release);
        for (offset, words) in [
            (LOAD, vec![load, SVC]),
            (STORE, vec![store, SVC]),
            (LOAD_STORE, vec![load, store, SVC]),
            (PLAIN_STORE, vec![plain_store(bytes), SVC]),
            (CLEAR_STORE, vec![0xd503_3f5f, store, SVC]),
        ] {
            for (index, word) in words.into_iter().enumerate() {
                let start = offset + index * 4;
                code.0[start..start + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let mut cpu = Cpu::new(backend).expect("CPU initialization");
        unsafe {
            cpu.map_host(CODE, PAGE as u64, Perm::RX, code.0.as_mut_ptr())
                .unwrap();
            cpu.map_host(va, PAGE as u64, perm, backing).unwrap();
        }
        for reg in 0..31 {
            cpu.set_register(reg, CANARY);
        }
        cpu.set_register(0, va + OFFSET as u64);
        cpu.set_register(3, DESIRED);
        cpu.set_sp(CANARY);
        Self { cpu, code }
    }

    fn run(&mut self, offset: usize) -> CpuEvent {
        self.cpu.set_pc(CODE + offset as u64);
        self.cpu.run(1_000).expect("exclusive execution").event
    }

    fn run_budget(&mut self, budget: u64) -> CpuEvent {
        self.cpu.run_with_count(budget).event
    }

    fn svc(&mut self, offset: usize) {
        let event = self.run(offset);
        assert!(matches!(event, CpuEvent::Svc(0x7e)), "{event:?}");
    }

    fn patch_code(&mut self, offset: usize, words: &[u32]) {
        for (index, word) in words.iter().enumerate() {
            let start = offset + index * 4;
            self.code.0[start..start + 4].copy_from_slice(&word.to_le_bytes());
        }
        self.cpu
            .invalidate_range(CODE + offset as u64, (words.len() * 4) as u64);
    }

    fn guards(&self, address: u64, loaded: u64, status: u64, desired: u64) {
        for reg in 0..31 {
            let expected = match reg {
                0 => address,
                1 => loaded,
                2 => status,
                3 => desired,
                _ => CANARY,
            };
            assert_eq!(self.cpu.get_register(reg), expected, "X{reg}");
        }
        assert_eq!(self.cpu.get_sp(), CANARY, "SP");
    }
}

fn backends() -> Vec<CpuBackendKind> {
    let mut backends = vec![CpuBackendKind::Rustarmic];
    #[cfg(feature = "backend-dynarmic")]
    backends.insert(0, CpuBackendKind::Dynarmic);
    backends
}

#[test]
fn single_core_exclusive_widths_ordering_zero_extension_and_guards() {
    for backend in backends() {
        for bytes in [1, 2, 4, 8] {
            for acquire in [false, true] {
                for release in [false, true] {
                    let mut backing = page(INITIAL);
                    let mut fixture = Fixture::new(
                        backend,
                        backing.0.as_mut_ptr(),
                        DATA,
                        Perm::RW,
                        bytes,
                        acquire,
                        release,
                    );
                    let mut expected = backing.0;
                    expected[OFFSET..OFFSET + bytes]
                        .copy_from_slice(&DESIRED.to_le_bytes()[..bytes]);
                    fixture.svc(LOAD_STORE);
                    fixture.guards(DATA + OFFSET as u64, INITIAL & mask(bytes), 0, DESIRED);
                    assert_eq!(
                        backing.0, expected,
                        "{backend:?}: {bytes}-byte store guards"
                    );
                    fixture.svc(STORE);
                    fixture.guards(DATA + OFFSET as u64, INITIAL & mask(bytes), 1, DESIRED);
                    assert_eq!(backing.0, expected, "{backend:?}: second STXR wrote memory");
                }
            }
        }
    }
}

#[test]
fn rustarmic_clrex_missing_reservation_address_and_size_mismatch_fail() {
    for bytes in [1, 2, 4, 8] {
        let mut backing = page(INITIAL);
        let original = backing.0;
        let mut fixture = Fixture::new(
            CpuBackendKind::Rustarmic,
            backing.0.as_mut_ptr(),
            DATA,
            Perm::RW,
            bytes,
            false,
            false,
        );
        fixture.svc(STORE);
        assert_eq!(fixture.cpu.get_register(2), 1, "missing reservation");
        fixture.svc(LOAD);
        fixture.svc(CLEAR_STORE);
        assert_eq!(fixture.cpu.get_register(2), 1, "CLREX");
        fixture.svc(LOAD);
        fixture.cpu.set_register(0, DATA + OFFSET as u64 + 16);
        fixture.svc(STORE);
        assert_eq!(fixture.cpu.get_register(2), 1, "different address");
        fixture.cpu.set_register(0, DATA + OFFSET as u64);
        fixture.svc(STORE);
        assert_eq!(
            fixture.cpu.get_register(2),
            1,
            "failed attempt consumes reservation"
        );
        fixture.svc(LOAD);
        let different_size = if bytes == 8 { 4 } else { bytes * 2 };
        fixture.patch_code(STORE, &[store_exclusive(different_size, false)]);
        fixture.svc(STORE);
        assert_eq!(fixture.cpu.get_register(2), 1, "different exclusive size");
        assert_eq!(backing.0, original, "failed STXR changed memory");
    }
}

#[test]
fn rustarmic_interleaved_shared_backing_stale_exclusive_store_fails() {
    for bytes in [1, 2, 4, 8] {
        for acquire in [false, true] {
            for release in [false, true] {
                for alias in [false, true] {
                    let mut backing = page(INITIAL);
                    let second_va = if alias { ALIAS } else { DATA };
                    let mut first = Fixture::new(
                        CpuBackendKind::Rustarmic,
                        backing.0.as_mut_ptr(),
                        DATA,
                        Perm::RW,
                        bytes,
                        acquire,
                        release,
                    );
                    let mut second = Fixture::new(
                        CpuBackendKind::Rustarmic,
                        backing.0.as_mut_ptr(),
                        second_va,
                        Perm::RW,
                        bytes,
                        acquire,
                        release,
                    );
                    first.cpu.set_core_id(0);
                    second.cpu.set_core_id(1);
                    second.cpu.set_register(3, DESIRED ^ mask(bytes));
                    first.svc(LOAD);
                    second.svc(LOAD);
                    first.svc(STORE);
                    second.svc(STORE);
                    first.guards(DATA + OFFSET as u64, INITIAL & mask(bytes), 0, DESIRED);
                    second.guards(
                        second_va + OFFSET as u64,
                        INITIAL & mask(bytes),
                        1,
                        DESIRED ^ mask(bytes),
                    );
                    let mut expected = page(INITIAL).0;
                    expected[OFFSET..OFFSET + bytes]
                        .copy_from_slice(&DESIRED.to_le_bytes()[..bytes]);
                    assert_eq!(backing.0, expected, "bytes={bytes} alias={alias}");
                    second.svc(LOAD_STORE);
                    assert_eq!(second.cpu.get_register(2), 0, "fresh reservation");
                }
            }
        }
    }
}

#[test]
fn rustarmic_competing_ordinary_store_changes_expected_value() {
    for bytes in [1, 2, 4, 8] {
        let mut backing = page(INITIAL);
        let mut first = Fixture::new(
            CpuBackendKind::Rustarmic,
            backing.0.as_mut_ptr(),
            DATA,
            Perm::RW,
            bytes,
            false,
            false,
        );
        let mut writer = Fixture::new(
            CpuBackendKind::Rustarmic,
            backing.0.as_mut_ptr(),
            ALIAS,
            Perm::RW,
            bytes,
            false,
            false,
        );
        first.svc(LOAD);
        writer.svc(PLAIN_STORE);
        first.cpu.set_register(3, 0);
        first.svc(STORE);
        assert_eq!(first.cpu.get_register(2), 1, "{bytes}-byte stale store");
        let mut expected = page(INITIAL).0;
        expected[OFFSET..OFFSET + bytes].copy_from_slice(&DESIRED.to_le_bytes()[..bytes]);
        assert_eq!(backing.0, expected, "stale store overwrote ordinary write");
    }
}

#[test]
fn rustarmic_context_restore_and_reset_do_not_resurrect_reservations() {
    for bytes in [1, 2, 4, 8] {
        let mut backing = page(INITIAL);
        let original = backing.0;
        let mut fixture = Fixture::new(
            CpuBackendKind::Rustarmic,
            backing.0.as_mut_ptr(),
            DATA,
            Perm::RW,
            bytes,
            false,
            false,
        );
        fixture.svc(LOAD);
        let mut saved = None;
        fixture.cpu.save_thread_context(&mut saved).unwrap();
        fixture
            .cpu
            .restore_thread_context(saved.as_ref().unwrap())
            .unwrap();
        fixture.svc(STORE);
        assert_eq!(
            fixture.cpu.get_register(2),
            1,
            "restore must clear reservation"
        );
        fixture.svc(LOAD);
        fixture.cpu.reset_thread_context().unwrap();
        fixture.svc(STORE);
        assert_eq!(
            fixture.cpu.get_register(2),
            1,
            "reset must clear reservation"
        );
        fixture
            .cpu
            .restore_thread_context(saved.as_ref().unwrap())
            .unwrap();
        fixture.svc(STORE);
        assert_eq!(
            fixture.cpu.get_register(2),
            1,
            "saved context must not carry reservation"
        );
        assert_eq!(backing.0, original);
    }
}

#[test]
fn rustarmic_exclusive_memory_permissions_fault_without_modifying_backing() {
    for bytes in [1, 2, 4, 8] {
        for perm in [Perm::R, Perm::W] {
            let mut backing = page(INITIAL);
            let original = backing.0;
            let mut fixture = Fixture::new(
                CpuBackendKind::Rustarmic,
                backing.0.as_mut_ptr(),
                DATA,
                perm,
                bytes,
                false,
                false,
            );
            let is_write = perm == Perm::R;
            if is_write {
                fixture.svc(LOAD);
            }
            let event = fixture.run(if is_write { STORE } else { LOAD });
            assert!(matches!(event, CpuEvent::Exception(0x0e)), "{event:?}");
            let fault = fixture
                .cpu
                .take_fault()
                .expect("exclusive permission fault");
            assert_eq!(fault.addr, DATA + OFFSET as u64);
            assert_eq!(fault.size, bytes as u32);
            assert_eq!(fault.is_write, is_write);
            assert_eq!(backing.0, original, "permission fault modified memory");
        }
    }
}

#[test]
fn rustarmic_two_host_threads_exclusive_increment_has_no_lost_updates() {
    for bytes in [4, 8] {
        const ITERATIONS: u64 = 20_000;
        let mut backing = page(0);
        let original = backing.0;
        let mut workers = Vec::new();
        let barrier = Arc::new(Barrier::new(2));
        for core_id in 0..2 {
            let mut fixture = Fixture::new(
                CpuBackendKind::Rustarmic,
                backing.0.as_mut_ptr(),
                DATA,
                Perm::RW,
                bytes,
                true,
                true,
            );
            let words = [
                load_exclusive(bytes, true),
                if bytes == 8 { 0x9100_0421 } else { 0x1100_0421 },
                store_exclusive(bytes, true) - 2,
                0x35ff_ffa2,
                0xf100_0484,
                0x54ff_ff61,
                SVC,
            ];
            fixture.patch_code(0, &words);
            let mut mapped_program = vec![0; words.len() * 4];
            fixture.cpu.read_bytes(CODE, &mut mapped_program).unwrap();
            let expected_program: Vec<u8> =
                words.iter().flat_map(|word| word.to_le_bytes()).collect();
            assert_eq!(
                mapped_program, expected_program,
                "mapped stress code before run"
            );
            fixture.cpu.set_register(4, ITERATIONS);
            fixture.cpu.set_core_id(core_id);
            workers.push(fixture);
        }
        let handles: Vec<_> = workers
            .into_iter()
            .map(|mut fixture| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    fixture.cpu.set_pc(CODE);
                    barrier.wait();
                    for _ in 0..16 {
                        match fixture.run_budget(1_000_000) {
                            CpuEvent::Svc(0x7e) => {
                                assert_eq!(
                                    fixture.cpu.get_register(4),
                                    0,
                                    "successful increment count"
                                );
                                assert_eq!(fixture.cpu.get_register(2), 0, "last STXR status");
                                return;
                            }
                            CpuEvent::Running => {}
                            event => {
                                let pc = fixture.cpu.get_pc();
                                let mut instruction = [0u8; 4];
                                fixture.cpu.read_bytes(pc, &mut instruction).unwrap();
                                panic!(
                                    "unexpected concurrent exclusive event {event:?} pc={pc:#x} word={:#010x} x0={:#x} x1={} w2={} x3={:#x} x4={}",
                                    u32::from_le_bytes(instruction),
                                    fixture.cpu.get_register(0),
                                    fixture.cpu.get_register(1),
                                    fixture.cpu.get_register(2),
                                    fixture.cpu.get_register(3),
                                    fixture.cpu.get_register(4),
                                );
                            }
                        }
                    }
                    panic!("exclusive increment loop did not terminate");
                })
            })
            .collect();
        let joined: Vec<_> = handles.into_iter().map(|handle| handle.join()).collect();
        for result in joined {
            result.expect("exclusive worker panicked");
        }
        let mut expected = original;
        expected[OFFSET..OFFSET + bytes].copy_from_slice(&(2 * ITERATIONS).to_le_bytes()[..bytes]);
        assert_eq!(
            backing.0, expected,
            "{bytes}-byte shared counter lost updates"
        );
    }
}

#[test]
#[ignore = "Known CAS-only gap: a same-value write does not invalidate another CPU's reservation"]
fn rustarmic_same_value_competing_exclusive_store_invalidates_reservation() {
    let mut backing = page(INITIAL);
    let mut first = Fixture::new(
        CpuBackendKind::Rustarmic,
        backing.0.as_mut_ptr(),
        DATA,
        Perm::RW,
        8,
        false,
        false,
    );
    let mut second = Fixture::new(
        CpuBackendKind::Rustarmic,
        backing.0.as_mut_ptr(),
        DATA,
        Perm::RW,
        8,
        false,
        false,
    );
    first.svc(LOAD);
    second.svc(LOAD);
    second.cpu.set_register(3, INITIAL);
    second.svc(STORE);
    assert_eq!(second.cpu.get_register(2), 0);
    first.svc(STORE);
    assert_eq!(
        first.cpu.get_register(2),
        1,
        "same-value competing write must invalidate"
    );
    assert_eq!(backing.0, page(INITIAL).0);
}

#[test]
#[ignore = "Known CAS-only gap: expected-value comparison cannot detect ABA write history"]
fn rustarmic_ordinary_store_aba_invalidates_reservation() {
    let mut backing = page(INITIAL);
    let mut first = Fixture::new(
        CpuBackendKind::Rustarmic,
        backing.0.as_mut_ptr(),
        DATA,
        Perm::RW,
        8,
        false,
        false,
    );
    let mut writer = Fixture::new(
        CpuBackendKind::Rustarmic,
        backing.0.as_mut_ptr(),
        ALIAS,
        Perm::RW,
        8,
        false,
        false,
    );
    first.svc(LOAD);
    writer.svc(PLAIN_STORE);
    writer.cpu.set_register(3, INITIAL);
    writer.svc(PLAIN_STORE);
    first.svc(STORE);
    assert_eq!(first.cpu.get_register(2), 1, "ABA write must invalidate");
    assert_eq!(backing.0, page(INITIAL).0);
}
