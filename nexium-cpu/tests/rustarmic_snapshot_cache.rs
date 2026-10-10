#![cfg(feature = "backend-rustarmic")]

use nexium_cpu::{rustarmic::RustarmicCpu, CpuEvent};
use nexium_memory::Perm;
use std::time::Instant;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const SVC: u32 = 0xd400_0fc1;
const VALUE: u64 = 0x1234_5678_9abc_def0;

fn code(words: &[u32]) -> Box<[u8; 0x1000]> {
    let mut bytes = Box::new([0u8; 0x1000]);
    for (index, word) in words.iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn slice(cpu: &mut RustarmicCpu) -> (CpuEvent, u64) {
    cpu.set_pc(CODE);
    cpu.run_with_count(1000)
}

fn svc_at(cpu: &mut RustarmicCpu, pc: u64) {
    cpu.set_pc(pc);
    for _ in 0..8 {
        let (event, _) = cpu.run_with_count(1000);
        if matches!(event, CpuEvent::Svc(0x7e)) {
            return;
        }
        assert!(matches!(event, CpuEvent::Running), "{event:?}");
    }
    panic!("did not reach SVC");
}

#[test]
fn stable_mapping_snapshots_read_live_data_across_short_slices() {
    let mut cpu = RustarmicCpu::new().unwrap();
    let mut program = code(&[0xf940_0020, SVC]);
    let mut value = Box::new(VALUE);
    unsafe {
        cpu.map_host(CODE, 0x1000, Perm::RX, program.as_mut_ptr())
            .unwrap();
        cpu.map_host(DATA, 8, Perm::RW, (&mut *value as *mut u64).cast())
            .unwrap();
    }
    cpu.set_register(1, DATA);
    for index in 0..64 {
        *value = VALUE ^ index;
        let (event, retired) = slice(&mut cpu);
        assert!(matches!(event, CpuEvent::Svc(0x7e)), "{event:?}");
        assert_eq!(retired, 2);
        assert_eq!(cpu.get_register(0), *value);
        assert!(cpu.take_fault().is_none());
    }
    assert_eq!(&program[..4], &0xf940_0020u32.to_le_bytes());
    drop(cpu);
}

#[test]
fn warmed_region_replacement_and_permission_changes_refresh_pointer_and_cache() {
    let mut cpu = RustarmicCpu::new().unwrap();
    let mut program = code(&[0xf940_0020, SVC]);
    let mut old = Box::new([11u64, 12]);
    let mut replacement = Box::new([21u64, 22]);
    unsafe {
        cpu.map_host(CODE, 0x1000, Perm::RX, program.as_mut_ptr())
            .unwrap();
        cpu.map_host(DATA, 16, Perm::RW, old.as_mut_ptr().cast())
            .unwrap();
    }
    cpu.set_register(1, DATA);
    for _ in 0..4 {
        svc_at(&mut cpu, CODE);
        assert_eq!(cpu.get_register(0), 11);
    }
    unsafe {
        cpu.unmap_host(DATA, 16).unwrap();
        cpu.map_host(DATA, 8, Perm::RO, replacement.as_mut_ptr().cast())
            .unwrap();
    }
    for _ in 0..4 {
        svc_at(&mut cpu, CODE);
        assert_eq!(cpu.get_register(0), 21);
    }
    cpu.set_register(1, DATA + 8);
    assert!(matches!(slice(&mut cpu).0, CpuEvent::Exception(0x0e)));
    let fault = cpu.take_fault().unwrap();
    assert_eq!((fault.pc, fault.addr, fault.size), (CODE, DATA + 8, 8));
    unsafe {
        cpu.unmap_host(DATA, 8).unwrap();
        cpu.map_host(DATA, 16, Perm::W, replacement.as_mut_ptr().cast())
            .unwrap();
    }
    cpu.set_register(1, DATA);
    assert!(matches!(slice(&mut cpu).0, CpuEvent::Exception(0x0e)));
    let fault = cpu.take_fault().unwrap();
    assert_eq!(
        (fault.pc, fault.addr, fault.size, fault.is_write),
        (CODE, DATA, 8, false)
    );
    assert_eq!(*old, [11, 12]);
    assert_eq!(*replacement, [21, 22]);
    drop(cpu);
}

#[test]
fn adding_an_overlap_after_warming_disables_mru_first_match_bypass() {
    let mut cpu = RustarmicCpu::new().unwrap();
    let mut program = code(&[0xf940_0020, SVC]);
    let mut earlier = Box::new([11u64, 12]);
    let mut later = Box::new([21u64, 22, 23]);
    unsafe {
        cpu.map_host(CODE, 0x1000, Perm::RX, program.as_mut_ptr())
            .unwrap();
        cpu.map_host(DATA + 8, 16, Perm::RO, earlier.as_mut_ptr().cast())
            .unwrap();
    }
    cpu.set_register(1, DATA + 8);
    svc_at(&mut cpu, CODE);
    assert_eq!(cpu.get_register(0), 11);
    unsafe {
        cpu.map_host(DATA, 24, Perm::RO, later.as_mut_ptr().cast())
            .unwrap();
    }
    for _ in 0..4 {
        cpu.set_register(1, DATA);
        svc_at(&mut cpu, CODE);
        assert_eq!(cpu.get_register(0), 21);
        cpu.set_register(1, DATA + 8);
        svc_at(&mut cpu, CODE);
        assert_eq!(cpu.get_register(0), 11, "earlier overlap wins each slice");
    }
    assert_eq!(*earlier, [11, 12]);
    assert_eq!(*later, [21, 22, 23]);
    drop(cpu);
}

#[test]
fn executable_range_added_after_snapshot_warmup_keeps_smc_invalidation() {
    let mut cpu = RustarmicCpu::new().unwrap();
    let mut program = code(&[0xb900_0022, SVC]);
    let mut target = code(&[0x5280_0020, SVC]);
    let mut initial_data = Box::new(0u32);
    unsafe {
        cpu.map_host(CODE, 0x1000, Perm::RX, program.as_mut_ptr())
            .unwrap();
        cpu.map_host(DATA, 4, Perm::RW, (&mut *initial_data as *mut u32).cast())
            .unwrap();
    }
    cpu.set_register(1, DATA);
    cpu.set_register(2, 41);
    svc_at(&mut cpu, CODE);
    assert_eq!(*initial_data, 41);
    unsafe {
        cpu.unmap_host(DATA, 4).unwrap();
        cpu.map_host(DATA, 0x1000, Perm::RWX, target.as_mut_ptr())
            .unwrap();
    }
    svc_at(&mut cpu, DATA);
    assert_eq!(cpu.get_register(0), 1);
    cpu.set_register(2, 0x5280_00e0);
    svc_at(&mut cpu, CODE);
    for _ in 0..4 {
        svc_at(&mut cpu, DATA);
        assert_eq!(
            cpu.get_register(0),
            7,
            "new executable range invalidated cached code"
        );
    }
    assert_eq!(*initial_data, 41);
    assert_eq!(&target[..4], &0x5280_00e0u32.to_le_bytes());
    assert_eq!(&target[4..8], &SVC.to_le_bytes());
    assert!(target[8..].iter().all(|&byte| byte == 0));
    drop(cpu);
}

#[test]
#[ignore = "CPU-only matched short-slice measurement; run idle with --ignored --nocapture"]
fn short_ldr_svc_snapshot_benchmark() {
    let runs = 2_000_000u64;
    let trials = 5;
    for region_count in [2, 66] {
        let mut cpu = RustarmicCpu::new().unwrap();
        let mut program = code(&[0xf940_0020, SVC]);
        let mut value = Box::new(VALUE);
        let mut padding = vec![0u8; (region_count - 2) * 0x1000];
        unsafe {
            cpu.map_host(CODE, 0x1000, Perm::RX, program.as_mut_ptr())
                .unwrap();
            for index in 0..region_count - 2 {
                cpu.map_host(
                    0x40_0000 + index as u64 * 0x1000,
                    0x1000,
                    Perm::RW,
                    padding.as_mut_ptr().add(index * 0x1000),
                )
                .unwrap();
            }
            cpu.map_host(DATA, 8, Perm::RW, (&mut *value as *mut u64).cast())
                .unwrap();
        }
        cpu.set_register(1, DATA);
        for _ in 0..250_000 {
            let (event, retired) = slice(&mut cpu);
            assert!(matches!(event, CpuEvent::Svc(0x7e)), "{event:?}");
            assert_eq!(retired, 2);
        }
        let mut elapsed_ms = Vec::new();
        for trial in 0..trials {
            let started = Instant::now();
            let mut total_retired = 0;
            for _ in 0..runs {
                let (event, retired) = slice(&mut cpu);
                assert!(matches!(event, CpuEvent::Svc(0x7e)), "{event:?}");
                total_retired += retired;
            }
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(total_retired, runs * 2);
            assert_eq!(cpu.get_register(0), VALUE);
            assert_eq!(*value, VALUE);
            println!(
                "regions={region_count} trial={} runs={runs} retired={total_retired} elapsed_ms={elapsed:.6} ns_per_run={:.3}",
                trial + 1,
                elapsed * 1_000_000.0 / runs as f64
            );
            elapsed_ms.push(elapsed);
        }
        let mean = elapsed_ms.iter().sum::<f64>() / elapsed_ms.len() as f64;
        elapsed_ms.sort_by(f64::total_cmp);
        println!(
            "regions={region_count} mean_ms={mean:.6} median_ms={:.6}",
            elapsed_ms[elapsed_ms.len() / 2]
        );
        assert_eq!(&program[..4], &0xf940_0020u32.to_le_bytes());
        assert_eq!(&program[4..8], &SVC.to_le_bytes());
        assert!(padding.iter().all(|&byte| byte == 0));
        drop(cpu);
    }
}
