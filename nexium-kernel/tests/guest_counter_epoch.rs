#![cfg(feature = "backend-rustarmic")]


use nexium_cpu::{Cpu, CpuEvent};
use nexium_kernel::{
    kernel::{cpu_local::set_current_cpu, svc},
    Kernel,
};
use nexium_memory::{AddressSpace, Perm};
use std::sync::Arc;
use std::time::Instant;

const CODE: u64 = 0x10_0000;
const CANARY: u64 = 0x9f37_51a8_d026_be40;

struct Fixture {
    cpu: Cpu,
    code: Box<[u8; 0x1000]>,
}

impl Fixture {
    fn new(core: u64) -> Self {
        let mut code = Box::new([0; 0x1000]);
        for (index, word) in [
            0xd53b_e02au32,
            0xd400_03c1,
            0xd53b_e04b,
            0xd400_0fc1,
        ]
        .iter()
        .enumerate()
        {
            code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut cpu = Cpu::new_rustarmic().expect("CPU initialization");
        cpu.set_core_id(core);
        unsafe {
            cpu.map_host(CODE, 0x1000, Perm::RX, code.as_mut_ptr())
                .unwrap();
        }
        Self { cpu, code }
    }
}

#[test]
fn real_get_system_tick_svc_is_bracketed_by_mrs_on_both_cpu_cores() {
    let space = Arc::new(AddressSpace::new());
    let mut kernel = Kernel::new(
        space, CODE, 0x1000, 0x20_0000, 0x1000, 0x30_0000, 0x1000, 0x40_0000, 0x40_1000,
    );
    let mut fixtures = [Fixture::new(0), Fixture::new(1)];
    let mut previous = fixtures[0].cpu.guest_cntpct().expect("MRS clock accessor");
    for _ in 0..16 {
        for (core, fixture) in fixtures.iter_mut().enumerate() {
            let code_before = fixture.code.clone();
            for register in 0..31 {
                fixture
                    .cpu
                    .set_register(register, CANARY ^ u64::from(register));
            }
            fixture.cpu.set_sp(CANARY);
            fixture.cpu.set_pc(CODE);
            let before = fixture.cpu.guest_cntpct().unwrap();
            assert!(matches!(
                fixture.cpu.run_with_count(64).event,
                CpuEvent::Svc(0x1e)
            ));
            let mrs_before = fixture.cpu.get_register(10);
            let regs_before: [u64; 31] =
                std::array::from_fn(|r| fixture.cpu.get_register(r as u32));
            let svc_before = fixture.cpu.guest_cntpct().unwrap();
            let pc_before = fixture.cpu.get_pc();
            let result = {
                let _guard = set_current_cpu(&mut fixture.cpu, core);
                svc::dispatch(&mut kernel, 0x1e)
            };
            assert_eq!(result, 0, "GetSystemTick status");
            let tick = fixture.cpu.get_register(0);
            let svc_after = fixture.cpu.guest_cntpct().unwrap();
            assert!(
                svc_before <= tick && tick <= svc_after,
                "SVC used a different epoch: {svc_before} <= {tick} <= {svc_after}"
            );
            assert_eq!(fixture.cpu.get_pc(), pc_before, "SVC moved guest PC");
            for (register, value) in regs_before.into_iter().enumerate().skip(1) {
                assert_eq!(
                    fixture.cpu.get_register(register as u32),
                    value,
                    "SVC changed X{register}"
                );
            }
            assert!(matches!(
                fixture.cpu.run_with_count(64).event,
                CpuEvent::Svc(0x7e)
            ));
            let mrs_after = fixture.cpu.get_register(11);
            let after = fixture.cpu.guest_cntpct().unwrap();
            assert!(previous <= before && before <= mrs_before && mrs_before <= tick
                && tick <= mrs_after && mrs_after <= after,
                "core{core}: {previous} <= {before} <= {mrs_before} <= {tick} <= {mrs_after} <= {after}");
            assert_eq!(fixture.cpu.get_sp(), CANARY);
            for register in 1..31 {
                if register != 10 && register != 11 {
                    assert_eq!(
                        fixture.cpu.get_register(register),
                        CANARY ^ u64::from(register)
                    );
                }
            }
            assert_eq!(&fixture.code[..], &code_before[..], "instruction guard");
            assert!(fixture.cpu.take_fault().is_none());
            previous = after;
        }
    }
}

fn counter_ns(ticks: u64) -> u128 {
    u128::from(ticks) * 1_000_000_000 / 19_200_000
}

fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn long(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[test]
fn real_binder_buffer_history_uses_the_mrs_epoch_with_measured_sampler_bounds() {
    use nexium_ipc::{HipcHeader, IpcCtx, CMIF_IN_MAGIC};
    use nexium_kernel::kernel::{handles::HandleType, session::Session};

    const REGION: u64 = 0x50_0000;
    const INPUT: u64 = REGION + 0x100;
    const REPLY: u64 = REGION + 0x200;
    const TLS: u64 = REGION + 0x1000;
    const BINDER: u32 = 256;

    let space = Arc::new(AddressSpace::new());
    space
        .map(REGION, 0x2000, Perm::RW, "guest_clock_history_test")
        .unwrap();
    space.write(REGION, &[0xa5; 0x2000]).unwrap();
    let mut kernel = Kernel::new(
        Arc::clone(&space),
        CODE,
        0x1000,
        0x20_0000,
        0x1000,
        0x30_0000,
        0x1000,
        TLS,
        TLS + 0x1000,
    );
    let mut fixture = Fixture::new(0);
    fixture.cpu.guest_cntpct().expect("shared MRS clock");
    let history = kernel.nvdrv.with_bufferqueue(BINDER, |queue| {
        queue.set_preallocated(
            0,
            nexium_nvdrv::GraphicBuffer {
                width: 1,
                height: 1,
                stride: 1,
                size: 4,
                ..Default::default()
            },
        );
        assert_eq!(queue.try_dequeue(), Some(0));
        assert!(queue.queue_and_acquire(0));
        assert!(queue.release(0));
        queue.history(1)[0]
    });
    let session = kernel.handles.create_handle(HandleType::Session);
    kernel
        .sessions
        .insert(session, Session::new(session, "IHOSBinderDriver".into()));

    let parcel_words = [12u32, 16, 0, 28, 0, 0, 1];
    let parcel: Vec<u8> = parcel_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    space.write(INPUT, &parcel).unwrap();
    let mut ipc = [0u8; 0x100];
    let header = HipcHeader::new()
        .with_command_type(4)
        .with_num_send_buffers(1)
        .with_num_recv_buffers(1)
        .with_num_data_words(7);
    ipc[..8].copy_from_slice(&header.to_le_bytes());
    for (offset, address, size) in [(8, INPUT, parcel.len() as u32), (20, REPLY, 0x100)] {
        ipc[offset..offset + 4].copy_from_slice(&size.to_le_bytes());
        ipc[offset + 4..offset + 8].copy_from_slice(&(address as u32).to_le_bytes());
    }
    ipc[0x20..0x24].copy_from_slice(&CMIF_IN_MAGIC.to_le_bytes());
    ipc[0x30..0x34].copy_from_slice(&BINDER.to_le_bytes());
    ipc[0x34..0x38].copy_from_slice(&17u32.to_le_bytes());
    let parsed = IpcCtx::parse(ipc.to_vec(), false).unwrap();
    assert_eq!(parsed.send_buffers[0].addr, INPUT);
    assert_eq!(parsed.recv_buffers[0].addr, REPLY);
    assert_eq!(parsed.cmif_in_data_len, 12);
    space.write(TLS, &ipc).unwrap();
    fixture.cpu.set_tpidrro_el0(TLS);

    let host_before = Instant::now();
    fixture.cpu.set_pc(CODE);
    assert!(matches!(
        fixture.cpu.run_with_count(64).event,
        CpuEvent::Svc(0x1e)
    ));
    let ticks_before = fixture.cpu.get_register(10);
    fixture.cpu.set_register(0, u64::from(session));
    {
        let _guard = set_current_cpu(&mut fixture.cpu, 0);
        assert_eq!(svc::dispatch(&mut kernel, 0x21), 0);
    }
    assert_eq!(fixture.cpu.get_register(0), 0);
    assert!(matches!(
        fixture.cpu.run_with_count(64).event,
        CpuEvent::Svc(0x7e)
    ));
    let ticks_after = fixture.cpu.get_register(11);
    let host_after = Instant::now();

    let mut reply = [0; 0x100];
    space.read(REPLY, &mut reply).unwrap();
    assert_eq!(word(&reply, 0), 36, "parcel payload bytes");
    assert_eq!(word(&reply, 4), 16, "parcel payload offset");
    assert_eq!(word(&reply, 16), 0, "history status");
    assert_eq!(word(&reply, 20), 1, "history count");
    assert_eq!(long(&reply, 24), history.frame_number);
    assert_eq!(word(&reply, 48), 3, "recorded acquired history state");
    for (offset, at) in [(32, history.queued_at), (40, history.presented_at.unwrap())] {
        let oldest_age = host_after.saturating_duration_since(at).as_nanos();
        let youngest_age = host_before.saturating_duration_since(at).as_nanos();
        let lower = counter_ns(ticks_before).saturating_sub(oldest_age);
        let upper = counter_ns(ticks_after).saturating_sub(youngest_age);
        let actual = u128::from(long(&reply, offset));
        assert!(
            lower <= actual && actual <= upper,
            "history used another epoch at byte{offset}: {lower} <= {actual} <= {upper}"
        );
    }
    assert!(long(&reply, 32) <= long(&reply, 40), "history ordering");
    assert!(
        reply[52..].iter().all(|byte| *byte == 0xa5),
        "reply neighbor guard"
    );
    let mut input_after = vec![0; parcel.len()];
    space.read(INPUT, &mut input_after).unwrap();
    assert_eq!(input_after, parcel, "read-only request parcel");
    let mut left_guard = [0; 8];
    space.read(REPLY - 8, &mut left_guard).unwrap();
    assert_eq!(left_guard, [0xa5; 8]);
    assert!(fixture.cpu.take_fault().is_none());
}
