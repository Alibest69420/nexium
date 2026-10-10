#![cfg(feature = "backend-rustarmic")]

use nexium_cpu::{rustarmic::RustarmicCpu, CpuEvent};
use nexium_memory::Perm;
use std::time::Instant;

const CODE: u64 = 0x100000;
const DATA: u64 = 0x200000;
const OUTPUT: u64 = 0x300000;
const SVC: u32 = 0xd4000fc1;

fn program(words: &[u32]) -> (RustarmicCpu, Vec<u8>) {
    let mut code = vec![0u8; 0x1000];
    for (index, word) in words.iter().enumerate() {
        code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    let mut cpu = RustarmicCpu::new().unwrap();
    unsafe { cpu.map_host(CODE, code.len() as u64, Perm::RX, code.as_mut_ptr()).unwrap(); }
    (cpu, code)
}

fn execute(cpu: &mut RustarmicCpu, pc: u64) -> CpuEvent {
    cpu.set_pc(pc);
    for _ in 0..8 {
        let event = cpu.run_with_count(10000).0;
        if !matches!(event, CpuEvent::Running) {
            return event;
        }
    }
    panic!("guest did not reach a terminal event at pc={:#x}", cpu.get_pc());
}

fn assert_svc(event: CpuEvent) {
    assert!(matches!(event, CpuEvent::Svc(0x7e)), "{event:?}");
}

#[test]
fn scalar_and_vector_accesses_cross_adjacent_mapping_boundaries() {
    let (mut cpu, code) = program(&[0xf9400020, 0xf9000040, 0x3dc00020, 0x3d800040, SVC]);
    let mut first = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let mut second = [9u8, 10, 11, 12, 13, 14, 15, 16];
    let mut output = [0u8; 16];
    unsafe {
        cpu.map_host(DATA, 8, Perm::RW, first.as_mut_ptr()).unwrap();
        cpu.map_host(DATA + 8, 8, Perm::RW, second.as_mut_ptr()).unwrap();
        cpu.map_host(OUTPUT, 16, Perm::RW, output.as_mut_ptr()).unwrap();
    }
    cpu.set_register(1, DATA);
    cpu.set_register(2, OUTPUT);
    assert_svc(execute(&mut cpu, CODE));
    assert_eq!(cpu.get_register(0), u64::from_le_bytes(first));
    assert_eq!(output, std::array::from_fn(|index| index as u8 + 1));
    drop(cpu);
    drop(code);
}

#[test]
fn mappings_are_refreshed_between_runs_and_unmapping_faults() {
    let (mut cpu, code) = program(&[0xf9400020, SVC]);
    let mut first = 17u64.to_le_bytes();
    let mut second = 29u64.to_le_bytes();
    cpu.set_register(1, DATA);
    unsafe { cpu.map_host(DATA, 8, Perm::RW, first.as_mut_ptr()).unwrap(); }
    assert_svc(execute(&mut cpu, CODE));
    assert_eq!(cpu.get_register(0), 17);
    unsafe {
        cpu.unmap_host(DATA, 8).unwrap();
        cpu.map_host(DATA, 8, Perm::RW, second.as_mut_ptr()).unwrap();
    }
    assert_svc(execute(&mut cpu, CODE));
    assert_eq!(cpu.get_register(0), 29);
    unsafe { cpu.unmap_host(DATA, 8).unwrap(); }
    assert!(matches!(execute(&mut cpu, CODE), CpuEvent::Exception(0x0e)));
    let fault = cpu.take_fault().unwrap();
    assert_eq!((fault.pc, fault.addr, fault.size, fault.is_write), (CODE, DATA, 8, false));
    drop(cpu);
    drop(code);
}

#[test]
fn memory_fault_is_delivered_once_and_a_repaired_mapping_runs_immediately() {
    let (mut cpu, code) = program(&[0xf9400020, SVC]);
    let mut value = 0x1234_5678_9abc_def0u64;
    cpu.set_register(1, DATA);
    for _ in 0..2 {
        cpu.set_pc(CODE);
        let (event, retired) = cpu.run_with_count(10000);
        assert!(matches!(event, CpuEvent::Exception(0x0e)), "{event:?}");
        assert!(retired > 0, "a fresh run must execute the faulting load");
        let fault = cpu.take_fault().expect("each actual access has a fresh snapshot");
        assert_eq!((fault.pc, fault.addr, fault.size, fault.is_write), (CODE, DATA, 8, false));
        assert!(cpu.take_fault().is_none());
    }
    unsafe {
        cpu.map_host(DATA, 8, Perm::RW, (&mut value as *mut u64).cast()).unwrap();
    }
    cpu.set_pc(CODE);
    let (event, retired) = cpu.run_with_count(10000);
    assert_svc(event);
    assert!(retired > 0, "no residual callback halt or duplicate exception");
    assert_eq!(cpu.get_register(0), value);
    assert!(cpu.take_fault().is_none());
    assert_eq!(&code[..4], &0xf9400020u32.to_le_bytes());
    assert_eq!(&code[4..8], &SVC.to_le_bytes());
    drop(cpu);
    drop(code);
}

#[test]
fn permission_changes_are_observed_on_the_next_run() {
    let (mut cpu, code) = program(&[0xf9000020, SVC]);
    let mut data = [0u8; 8];
    cpu.set_register(0, 41);
    cpu.set_register(1, DATA);
    unsafe { cpu.map_host(DATA, 8, Perm::RW, data.as_mut_ptr()).unwrap(); }
    assert_svc(execute(&mut cpu, CODE));
    assert_eq!(u64::from_le_bytes(data), 41);
    unsafe {
        cpu.unmap_host(DATA, 8).unwrap();
        cpu.map_host(DATA, 8, Perm::RO, data.as_mut_ptr()).unwrap();
    }
    cpu.set_register(0, 99);
    assert!(matches!(execute(&mut cpu, CODE), CpuEvent::Exception(0x0e)));
    let fault = cpu.take_fault().unwrap();
    assert_eq!((fault.pc, fault.addr, fault.size, fault.is_write), (CODE, DATA, 8, true));
    assert_eq!(u64::from_le_bytes(data), 41);
    drop(cpu);
    drop(code);
}

#[test]
fn reading_write_only_memory_preserves_fault_details() {
    let (mut cpu, code) = program(&[0xf9400020, SVC]);
    let mut data = [0u8; 8];
    unsafe { cpu.map_host(DATA, 8, Perm::W, data.as_mut_ptr()).unwrap(); }
    cpu.set_register(1, DATA);
    assert!(matches!(execute(&mut cpu, CODE), CpuEvent::Exception(0x0e)));
    let fault = cpu.take_fault().unwrap();
    assert_eq!((fault.pc, fault.addr, fault.size, fault.is_write), (CODE, DATA, 8, false));
    drop(cpu);
    drop(code);
}

#[test]
fn first_fault_is_not_replaced_by_a_secondary_null_read_in_the_same_block() {
    let (mut cpu, code) = program(&[
        0xf9000028,
        0xf9400028,
        0xf9401d09,
        SVC,
    ]);
    cpu.set_register(1, DATA);
    cpu.set_register(8, 0x12345678);
    assert!(matches!(execute(&mut cpu, CODE), CpuEvent::Exception(0x0e)));
    assert_eq!(cpu.get_pc(), CODE, "terminal event identifies the first fault");
    let fault = cpu.take_fault().unwrap();
    assert_eq!((fault.pc, fault.addr, fault.size, fault.is_write), (CODE, DATA, 8, true));
    assert_eq!(fault.regs[8], 0x12345678, "snapshot precedes subsequent block mutations");
    assert_eq!(fault.value, 0x12345678);
    drop(cpu);
    drop(code);
}

#[test]
fn independent_cores_keep_independent_mapping_snapshots() {
    let (mut first_cpu, first_code) = program(&[0xf9400020, SVC]);
    let (mut second_cpu, second_code) = program(&[0xf9400020, SVC]);
    let mut first_data = 11u64.to_le_bytes();
    let mut second_data = 23u64.to_le_bytes();
    unsafe {
        first_cpu.map_host(DATA, 8, Perm::RW, first_data.as_mut_ptr()).unwrap();
        second_cpu.map_host(DATA, 8, Perm::RW, second_data.as_mut_ptr()).unwrap();
    }
    first_cpu.set_register(1, DATA);
    second_cpu.set_register(1, DATA);
    for _ in 0..2 {
        assert_svc(execute(&mut first_cpu, CODE));
        assert_svc(execute(&mut second_cpu, CODE));
        assert_eq!(first_cpu.get_register(0), 11);
        assert_eq!(second_cpu.get_register(0), 23);
    }
    drop(first_cpu);
    drop(second_cpu);
    drop(first_code);
    drop(second_code);
}

#[test]
fn executable_writes_invalidate_cached_code_before_the_next_run() {
    let (mut cpu, mut code) = program(&[0xb9000022, SVC]);
    code[0x100..0x104].copy_from_slice(&0x52800020u32.to_le_bytes());
    code[0x104..0x108].copy_from_slice(&SVC.to_le_bytes());
    unsafe {
        cpu.unmap_host(CODE, code.len() as u64).unwrap();
        cpu.map_host(CODE, code.len() as u64, Perm::RWX, code.as_mut_ptr()).unwrap();
    }
    assert_svc(execute(&mut cpu, CODE + 0x100));
    assert_eq!(cpu.get_register(0), 1);
    cpu.set_register(1, CODE + 0x100);
    cpu.set_register(2, 0x528000e0);
    let mut expected_code = code.clone();
    expected_code[0x100..0x104].copy_from_slice(&0x528000e0u32.to_le_bytes());
    assert_svc(execute(&mut cpu, CODE));
    assert_eq!(code, expected_code, "only the requested instruction was patched");
    assert_svc(execute(&mut cpu, CODE + 0x100));
    assert_eq!(cpu.get_register(0), 7);
    assert_eq!(code, expected_code, "resumption preserved code and neighboring bytes");
    drop(cpu);
    drop(code);
}

#[test]
#[ignore = "CPU-only performance measurement; run with --ignored --nocapture"]
fn hot_memory_hook_benchmark() {
    let (mut cpu, code) = program(&[
        0xf9400020,
        0x91000400,
        0xf9000020,
        0xf1000442,
        0x54ffff81,
        SVC,
    ]);
    let mut data = [0u8; 8];
    unsafe { cpu.map_host(DATA, 8, Perm::RW, data.as_mut_ptr()).unwrap(); }
    let extra_regions = std::env::var("NEXIUM_BENCH_EXTRA_REGIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
        .min(1024);
    let mut padding = vec![0u8; extra_regions * 0x1000];
    for index in 0..extra_regions {
        unsafe {
            cpu.map_host(0x400000 + index as u64 * 0x1000, 0x1000, Perm::RW,
                padding.as_mut_ptr().add(index * 0x1000)).unwrap();
        }
    }
    if std::env::var("NEXIUM_BENCH_DATA_LAST").as_deref() == Ok("1") {
        unsafe {
            cpu.unmap_host(DATA, 8).unwrap();
            cpu.map_host(DATA, 8, Perm::RW, data.as_mut_ptr()).unwrap();
        }
    }
    println!("mapped_regions={}", extra_regions + 2);
    cpu.set_register(1, DATA);
    let loops = 5_000_000;
    for window in 0..6 {
        cpu.set_pc(CODE);
        cpu.set_register(2, loops);
        let started = Instant::now();
        let (event, retired) = cpu.run_with_count(loops * 5 + 1000);
        let elapsed = started.elapsed();
        assert_svc(event);
        assert_eq!(retired, loops * 5 + 1);
        if window > 0 {
            println!("window={window} retired={retired} elapsed_ms={:.3} mips={:.3}",
                elapsed.as_secs_f64() * 1000.0, retired as f64 / elapsed.as_secs_f64() / 1_000_000.0);
        }
    }
    assert_eq!(u64::from_le_bytes(data), loops * 6);
    drop(cpu);
    drop(code);
}
