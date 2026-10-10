#![cfg(feature = "backend-rustarmic")]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::{AddressSpace, Perm, PAGE_SIZE};
use nexium_runner::boot::sync_host_region_changes;
use std::sync::{mpsc, Arc};
use std::time::Duration;

const SVC: u32 = 0xd4000fc1;
const WAIT: Duration = Duration::from_secs(15);
const BLOCKED: Duration = Duration::from_millis(50);

#[derive(Clone, Copy, Debug)]
enum Mutation {
    Map,
    SplitUnmap,
    Protect,
}

impl Mutation {
    fn index(self) -> u64 {
        match self {
            Self::Map => 0,
            Self::SplitUnmap => 1,
            Self::Protect => 2,
        }
    }

    fn changes(self) -> usize {
        match self {
            Self::SplitUnmap => 3,
            Self::Map | Self::Protect => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Command {
    RunPinnedSlice,
    Release,
    CheckNextSlice,
}

#[derive(Debug)]
enum Report {
    Ready(u64),
    Ran,
    Released,
    Checked(u64),
}

struct Reader {
    commands: mpsc::Sender<Command>,
    reports: mpsc::Receiver<Report>,
}

fn receive<T>(receiver: &mpsc::Receiver<T>) -> T {
    receiver
        .recv_timeout(WAIT)
        .expect("worker did not report within the bounded wait")
}

fn expect_blocked(receiver: &mpsc::Receiver<()>, phase: &str) {
    assert!(
        matches!(
            receiver.recv_timeout(BLOCKED),
            Err(mpsc::RecvTimeoutError::Timeout)
        ),
        "mapping writer finished while {phase} still held an execution guard"
    );
}

fn instruction_image() -> Vec<u8> {
    let mut bytes = vec![0xc7; PAGE_SIZE as usize];
    for (offset, word) in [
        (0usize, 0xf9400020u32),
        (4, SVC),
        (0x10, 0xf9000020),
        (0x14, SVC),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn page_image(value: u64) -> Vec<u8> {
    let mut bytes = vec![0xa5; PAGE_SIZE as usize];
    bytes[..8].copy_from_slice(&value.to_le_bytes());
    bytes
}

fn assert_bytes(cpu: &Cpu, address: u64, expected: &[u8]) {
    let mut actual = vec![0; expected.len()];
    cpu.read_bytes(address, &mut actual).unwrap();
    assert_eq!(actual, expected, "mapping/backing bytes at {address:#x}");
}

fn execute_access(cpu: &mut Cpu, code: u64, address: u64, store: Option<u64>) -> CpuEvent {
    for register in 0..31 {
        cpu.set_register(register, 0x8123_4567_8900_0000 | u64::from(register));
    }
    let initial = store.unwrap_or(0xfedc_ba98_7654_3210);
    cpu.set_register(0, initial);
    cpu.set_register(1, address);
    cpu.set_sp(0x1234_5678_9000);
    cpu.set_pc(code + if store.is_some() { 0x10 } else { 0 });
    for _ in 0..8 {
        let event = cpu.run_with_count(1_000).event;
        if matches!(event, CpuEvent::Running) {
            continue;
        }
        for register in 2..31 {
            assert_eq!(
                cpu.get_register(register),
                0x8123_4567_8900_0000 | u64::from(register)
            );
        }
        assert_eq!(cpu.get_register(1), address);
        assert_eq!(cpu.get_sp(), 0x1234_5678_9000);
        if store.is_some() {
            assert_eq!(cpu.get_register(0), initial);
        }
        return event;
    }
    panic!("access did not terminate at pc={:#x}", cpu.get_pc());
}

fn assert_load(cpu: &mut Cpu, code: u64, address: u64, expected: u64) {
    let event = execute_access(cpu, code, address, None);
    assert!(
        matches!(event, CpuEvent::Svc(0x7e)),
        "load at {address:#x}: {event:?}"
    );
    assert_eq!(cpu.get_register(0), expected);
    assert!(cpu.take_fault().is_none());
}

fn assert_fault(cpu: &mut Cpu, code: u64, address: u64, store: Option<u64>) {
    let event = execute_access(cpu, code, address, store);
    assert!(
        matches!(event, CpuEvent::Exception(0x0e)),
        "fault at {address:#x}: {event:?}"
    );
    let fault = cpu.take_fault().expect("mapping fault detail");
    assert_eq!(
        (fault.pc, fault.addr, fault.size, fault.is_write),
        (
            code + if store.is_some() { 0x10 } else { 0 },
            address,
            8,
            store.is_some()
        )
    );
    if let Some(value) = store {
        assert_eq!(fault.value, value);
    }
}

fn check_next_snapshot(
    cpu: &mut Cpu,
    space: &AddressSpace,
    mutation: Mutation,
    code: u64,
    data: u64,
    new_map: u64,
    code_bytes: &[u8],
) {
    assert_load(cpu, code, data, 11);
    assert_bytes(cpu, data, &page_image(11));
    match mutation {
        Mutation::Map => {
            assert_load(cpu, code, new_map, 29);
            assert_bytes(cpu, new_map, &page_image(29));
        }
        Mutation::SplitUnmap => {
            assert_load(cpu, code, data + PAGE_SIZE * 2, 23);
            assert_bytes(cpu, data + PAGE_SIZE * 2, &page_image(23));
            assert!(space.host_region_at(data + PAGE_SIZE).is_none());
            let mut byte = [0];
            assert!(cpu.read_bytes(data + PAGE_SIZE, &mut byte).is_err());
            assert_fault(cpu, code, data + PAGE_SIZE, None);
        }
        Mutation::Protect => {
            assert_eq!(space.host_region_at(data).unwrap().perm, Perm::RO);
            assert_fault(cpu, code, data, Some(99));
            assert_bytes(cpu, data, &page_image(11));
        }
    }
    assert_bytes(cpu, code, code_bytes);
}

fn publication_case(mutation: Mutation) {
    let code = (1u64 << 41) + mutation.index() * 0x10_0000;
    let data = code + 0x1_0000;
    let new_map = data + PAGE_SIZE * 4;
    let space = Arc::new(AddressSpace::new());
    let code_bytes = Arc::new(instruction_image());
    space
        .map(code, PAGE_SIZE, Perm::RX, "publication-code")
        .unwrap();
    space.write(code, &code_bytes).unwrap();
    let pages = if matches!(mutation, Mutation::SplitUnmap) {
        3
    } else {
        1
    };
    space
        .map(data, PAGE_SIZE * pages, Perm::RW, "publication-data")
        .unwrap();
    for (page, value) in [11, 17, 23].into_iter().take(pages as usize).enumerate() {
        space
            .write(data + PAGE_SIZE * page as u64, &page_image(value))
            .unwrap();
    }
    let initial_generation = space.generation();

    std::thread::scope(|scope| {
        let mut readers = Vec::new();
        for core in 0..2 {
            let reader_space = Arc::clone(&space);
            let reader_code = Arc::clone(&code_bytes);
            let (commands, command_rx) = mpsc::channel();
            let (report_tx, reports) = mpsc::channel();
            scope.spawn(move || {
                let mut cpu = Cpu::new(CpuBackendKind::Rustarmic).unwrap();
                cpu.set_core_id(core as u64);
                let mut cursor = 0;
                let execution = reader_space.execution_guard();
                assert_eq!(
                    sync_host_region_changes(&mut cpu, &reader_space, &mut cursor, core).unwrap(),
                    2
                );
                report_tx.send(Report::Ready(cursor)).unwrap();
                assert!(matches!(receive(&command_rx), Command::RunPinnedSlice));
                assert_eq!(reader_space.generation(), initial_generation);
                assert_load(&mut cpu, code, data, 11);
                if matches!(mutation, Mutation::SplitUnmap) {
                    assert_load(&mut cpu, code, data + PAGE_SIZE, 17);
                    assert_load(&mut cpu, code, data + PAGE_SIZE * 2, 23);
                }
                assert_eq!(reader_space.host_region_at(data).unwrap().perm, Perm::RW);
                assert_bytes(&cpu, code, &reader_code);
                report_tx.send(Report::Ran).unwrap();
                assert!(matches!(receive(&command_rx), Command::Release));
                drop(execution);
                report_tx.send(Report::Released).unwrap();

                assert!(matches!(receive(&command_rx), Command::CheckNextSlice));
                let _next_execution = reader_space.execution_guard();
                assert_eq!(
                    sync_host_region_changes(&mut cpu, &reader_space, &mut cursor, core).unwrap(),
                    mutation.changes()
                );
                assert_eq!(cursor, reader_space.generation());
                assert_eq!(cursor, initial_generation + mutation.changes() as u64);
                check_next_snapshot(
                    &mut cpu,
                    &reader_space,
                    mutation,
                    code,
                    data,
                    new_map,
                    &reader_code,
                );
                drop(cpu);
                report_tx.send(Report::Checked(cursor)).unwrap();
            });
            readers.push(Reader { commands, reports });
        }

        for reader in &readers {
            assert!(
                matches!(receive(&reader.reports), Report::Ready(g) if g == initial_generation)
            );
        }
        let writer_space = Arc::clone(&space);
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let writer = scope.spawn(move || {
            started_tx.send(()).unwrap();
            match mutation {
                Mutation::Map => {
                    writer_space
                        .map(new_map, PAGE_SIZE, Perm::RW, "new-publication-data")
                        .unwrap();
                    writer_space.write(new_map, &page_image(29)).unwrap();
                }
                Mutation::SplitUnmap => {
                    assert_eq!(
                        writer_space.unmap(data + PAGE_SIZE, PAGE_SIZE).unwrap(),
                        PAGE_SIZE
                    );
                }
                Mutation::Protect => writer_space.protect(data, PAGE_SIZE, Perm::RO).unwrap(),
            }
            done_tx.send(()).unwrap();
        });
        receive(&started_rx);
        expect_blocked(&done_rx, "both cores");
        for reader in &readers {
            reader.commands.send(Command::RunPinnedSlice).unwrap();
        }
        for reader in &readers {
            assert!(matches!(receive(&reader.reports), Report::Ran));
        }
        readers[0].commands.send(Command::Release).unwrap();
        assert!(matches!(receive(&readers[0].reports), Report::Released));
        expect_blocked(&done_rx, "the second core");
        readers[1].commands.send(Command::Release).unwrap();
        assert!(matches!(receive(&readers[1].reports), Report::Released));
        receive(&done_rx);
        writer.join().unwrap();
        for reader in &readers {
            reader.commands.send(Command::CheckNextSlice).unwrap();
        }
        for reader in &readers {
            assert!(
                matches!(receive(&reader.reports), Report::Checked(g) if g == initial_generation + mutation.changes() as u64)
            );
        }
    });

    let mut actual = vec![0; PAGE_SIZE as usize];
    space.read(data, &mut actual).unwrap();
    assert_eq!(actual, page_image(11));
    space.read(code, &mut actual).unwrap();
    assert_eq!(actual, *code_bytes);
}

#[test]
fn mapping_writer_waits_for_parallel_slices_then_both_cores_see_new_mapping() {
    publication_case(Mutation::Map);
}

#[test]
fn partial_unmap_waits_for_parallel_slices_then_both_cores_keep_both_survivors() {
    publication_case(Mutation::SplitUnmap);
}

#[test]
fn protection_writer_waits_for_parallel_slices_then_both_cores_observe_permissions() {
    publication_case(Mutation::Protect);
}
