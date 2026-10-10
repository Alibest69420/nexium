use nexium_cpu::{Cpu, CpuBackendKind, GuestIsa};

#[cfg(feature = "backend-dynarmic")]
#[test]
fn dynarmic_reports_its_constructed_backend_kind() {
    let cpu = Cpu::new_with_isa(CpuBackendKind::Dynarmic, GuestIsa::AArch64).unwrap();
    assert_eq!(cpu.backend_kind(), CpuBackendKind::Dynarmic);
    assert_eq!(cpu.isa(), GuestIsa::AArch64);
}

#[cfg(feature = "backend-rustarmic")]
#[test]
fn rustarmic_reports_its_constructed_backend_kind() {
    let cpu = Cpu::new_with_isa(CpuBackendKind::Rustarmic, GuestIsa::AArch64).unwrap();
    assert_eq!(cpu.backend_kind(), CpuBackendKind::Rustarmic);
    assert_eq!(cpu.isa(), GuestIsa::AArch64);
}

#[cfg(all(feature = "backend-dynarmic", feature = "backend-rustarmic"))]
#[test]
fn aarch32_rustarmic_request_reports_actual_dynarmic_fallback() {
    let cpu = Cpu::new_with_isa(CpuBackendKind::Rustarmic, GuestIsa::AArch32).unwrap();
    assert_eq!(cpu.backend_kind(), CpuBackendKind::Dynarmic);
    assert_eq!(cpu.isa(), GuestIsa::AArch32);
}
