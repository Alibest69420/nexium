#![cfg(all(feature = "backend-dynarmic", any(windows, target_os = "linux")))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent, GuestIsa};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const PAGE: usize = 0x1000;
const MIB: u64 = 1024 * 1024;
const WARMUP: usize = 2;
const CONSTRUCTIONS: usize = 32;
const PRIVATE_BUDGET: u64 = 32 * MIB;
const VIRTUAL_BUDGET: u64 = 1024 * MIB;

#[derive(Clone, Copy)]
struct Usage {
    private: u64,
    virtual_size: u64,
}

#[cfg(windows)]
fn usage() -> Usage {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCountersEx {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    #[repr(C)]
    #[derive(Default)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCountersEx,
            cb: u32,
        ) -> i32;
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }

    let mut counters = ProcessMemoryCountersEx {
        cb: std::mem::size_of::<ProcessMemoryCountersEx>() as u32,
        ..Default::default()
    };
    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        ..Default::default()
    };
    unsafe {
        assert_ne!(
            K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb),
            0
        );
        assert_ne!(GlobalMemoryStatusEx(&mut status), 0);
    }
    Usage {
        private: counters.private_usage as u64,
        virtual_size: status.total_virtual - status.avail_virtual,
    }
}

#[cfg(target_os = "linux")]
fn usage() -> Usage {
    let status = std::fs::read_to_string("/proc/self/status").expect("read process status");
    let field = |key: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(key))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or_else(|| panic!("missing {key} in /proc/self/status"))
            * 1024
    };
    Usage {
        private: field("RssAnon:"),
        virtual_size: field("VmSize:"),
    }
}

fn code_page(words: &[u32]) -> Box<[u8; PAGE]> {
    let mut page = Box::new([0; PAGE]);
    for (index, word) in words.iter().enumerate() {
        page[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    page
}

fn construct_run_drop(isa: GuestIsa, code: &mut [u8; PAGE]) {
    let mut cpu = Cpu::new_with_isa(CpuBackendKind::Dynarmic, isa).expect("CPU initialization");
    unsafe {
        cpu.map_host(CODE, PAGE as u64, Perm::RX, code.as_mut_ptr())
            .expect("map code");
    }
    cpu.reset_thread_context().expect("reset thread context");
    cpu.set_pc(CODE);
    let event = cpu.run_event(64);
    assert!(matches!(event, CpuEvent::Svc(0)), "{isa:?}: {event:?}");
    assert_eq!(cpu.get_register(0), 42, "{isa:?}");
    unsafe {
        cpu.unmap_host(CODE, PAGE as u64).expect("unmap code");
    }
    let halt = cpu.halt_handle();
    drop(cpu);
    std::thread::spawn(move || {
        halt.halt();
        let _ = halt.peek_pc_lr_sp();
    })
    .join()
    .expect("halt handle thread");
}

#[test]
fn dropping_dynarmic_cpus_releases_their_memory() {
    let mut a64 = code_page(&[0xd280_0540, 0xd400_0001]);
    let mut a32 = code_page(&[0xe3a0_002a, 0xef00_0000]);
    let mut cycle = |iteration: usize| {
        if iteration % 2 == 0 {
            construct_run_drop(GuestIsa::AArch64, &mut a64);
        } else {
            construct_run_drop(GuestIsa::AArch32, &mut a32);
        }
    };
    for iteration in 0..WARMUP {
        cycle(iteration);
    }
    let baseline = usage();
    for iteration in 0..CONSTRUCTIONS {
        cycle(iteration);
        let now = usage();
        let private = now.private.saturating_sub(baseline.private);
        let virtual_size = now.virtual_size.saturating_sub(baseline.virtual_size);
        assert!(
            private <= PRIVATE_BUDGET && virtual_size <= VIRTUAL_BUDGET,
            "Dynarmic CPU memory grew after {} constructions: private +{} MiB, virtual +{} MiB",
            iteration + 1,
            private / MIB,
            virtual_size / MIB
        );
    }
}
