#![cfg(feature = "backend-rustarmic")]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const PAGE: usize = 0x1000;
const CANARY: u64 = 0xface_1234_5678_9abc;

struct Fixture {
    cpu: Cpu,
    _code: Box<[u8; PAGE]>,
}

impl Fixture {
    fn new(backend: CpuBackendKind) -> Self {
        let mut code = Box::new([0; PAGE]);
        for (index, word) in [
            0x9100_0400u32,
            0x9100_0c21,
            0xd400_0fc1,
            0xd420_0020,
        ]
        .iter()
        .enumerate()
        {
            code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut cpu = Cpu::new(backend).expect("CPU initialization");
        unsafe {
            cpu.map_host(CODE, PAGE as u64, Perm::RX, code.as_mut_ptr())
                .unwrap();
        }
        for register in 0..31 {
            cpu.set_register(register, CANARY);
        }
        cpu.set_register(0, 0);
        cpu.set_register(1, 0);
        cpu.set_sp(CANARY);
        cpu.set_pc(CODE);
        Self { cpu, _code: code }
    }

    fn guards(&self, x0: u64, x1: u64) {
        for register in 0..31 {
            let expected = match register {
                0 => x0,
                1 => x1,
                _ => CANARY,
            };
            assert_eq!(self.cpu.get_register(register), expected, "X{register}");
        }
        assert_eq!(self.cpu.get_sp(), CANARY, "SP");
    }
}

fn pre_run_halt_and_resume(backend: CpuBackendKind, repetitions: u64) {
    let mut fixture = Fixture::new(backend);
    let halt = fixture.cpu.halt_handle();
    for iteration in 0..repetitions {
        fixture.cpu.set_pc(CODE);
        halt.halt();
        halt.clone().halt();
        halt.halt();
        let stopped = fixture.cpu.run_with_count(64);
        assert!(
            matches!(stopped.event, CpuEvent::Running),
            "{backend:?}: host halt must end a slice, not interrupt the guest: {:?}",
            stopped.event
        );
        assert_eq!(fixture.cpu.get_pc(), CODE, "{backend:?}: halted PC");
        fixture.guards(iteration, iteration * 3);
        if backend == CpuBackendKind::Rustarmic {
            assert_eq!(stopped.retired, 0, "pre-run halt retired an instruction");
        }
        let resumed = fixture.cpu.run_with_count(64);
        assert!(
            matches!(resumed.event, CpuEvent::Svc(0x7e)),
            "{backend:?}: next slice failed to resume: {:?}",
            resumed.event
        );
        fixture.guards(iteration + 1, (iteration + 1) * 3);
        if backend == CpuBackendKind::Rustarmic {
            assert_eq!(resumed.retired, 3, "resumed ADD/ADD/SVC retirement count");
        }
        assert!(
            fixture.cpu.take_fault().is_none(),
            "host halt became a fault"
        );
    }
}

#[test]
fn rustarmic_pre_run_host_halt_returns_running_without_guest_progress() {
    pre_run_halt_and_resume(CpuBackendKind::Rustarmic, 1);
}

#[test]
fn rustarmic_repeated_host_halts_clear_stop_flags_and_resume_warmed_code() {
    pre_run_halt_and_resume(CpuBackendKind::Rustarmic, 32);
}

#[cfg(feature = "backend-dynarmic")]
#[test]
fn dynarmic_pre_run_host_halt_returns_running_without_guest_progress() {
    pre_run_halt_and_resume(CpuBackendKind::Dynarmic, 1);
}

#[cfg(feature = "backend-dynarmic")]
#[test]
fn dynarmic_repeated_host_halts_resume_warmed_code() {
    pre_run_halt_and_resume(CpuBackendKind::Dynarmic, 32);
}
