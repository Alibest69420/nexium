#![cfg(feature = "backend-rustarmic")]


use nexium_cpu::{Cpu, CpuEvent};
use nexium_memory::Perm;
use rustarmic::arch::sysreg;
use std::sync::{Arc, Barrier};

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const SAMPLE: usize = 0x400;
const INSPECT: usize = 0x500;
const OUTPUT: usize = 0x400;
const FPCR: u64 = 0x0340_0000;
const FPSR: u64 = 0x0800_009f;
const NZCV: u64 = 0xb000_0000;

fn mrs(register: u32, system_register: u16) -> u32 {
    0xd520_0000 | (u32::from(system_register) << 5) | register
}

fn msr(register: u32, system_register: u16) -> u32 {
    0xd500_0000 | (u32::from(system_register) << 5) | register
}

fn canary(register: u32) -> u64 {
    0xfacd_8537_1942_6000 ^ u64::from(register).wrapping_mul(0x1020_3040_5060_7081)
}

struct Fixture {
    cpu: Cpu,
    code: Box<[u8; 0x1000]>,
    data: Box<[u8; 0x1000]>,
    initial_code: Box<[u8; 0x1000]>,
    initial_data: Box<[u8; 0x1000]>,
}

impl Fixture {
    fn new(core: u64) -> Self {
        let mut code = Box::new([0x55; 0x1000]);
        let mut data = Box::new([0xa5; 0x1000]);
        for (index, byte) in data[..512].iter_mut().enumerate() {
            *byte = (index as u8)
                .wrapping_mul(37)
                .wrapping_add((index / 16) as u8);
        }
        let mut seed: Vec<u32> = (0..32)
            .map(|register| 0x3dc0_0140 | (register << 10) | register)
            .collect();
        seed.extend([
            msr(16, sysreg::FPCR),
            msr(17, sysreg::FPSR),
            msr(18, sysreg::NZCV),
            0xd400_0fa1,
        ]);
        let sample = [
            mrs(20, sysreg::CNTPCT_EL0),
            mrs(21, sysreg::CNTVCT_EL0),
            mrs(22, sysreg::CNTFRQ_EL0),
            0xd400_03c1,
        ];
        let mut inspect: Vec<u32> = (0..32)
            .map(|register| 0x3d80_0140 | (register << 10) | register)
            .collect();
        for (index, system_register) in [sysreg::FPCR, sysreg::FPSR, sysreg::NZCV]
            .into_iter()
            .enumerate()
        {
            inspect.push(mrs(8, system_register));
            inspect.push(0xf900_0148 | (((512 / 8 + index) as u32) << 10));
        }
        inspect.push(0xd400_0fc1);
        for (offset, words) in [
            (0, &seed[..]),
            (SAMPLE, &sample[..]),
            (INSPECT, &inspect[..]),
        ] {
            for (index, word) in words.iter().enumerate() {
                code[offset + index * 4..offset + index * 4 + 4]
                    .copy_from_slice(&word.to_le_bytes());
            }
        }
        let mut cpu = Cpu::new_rustarmic().expect("CPU initialization");
        cpu.set_core_id(core);
        unsafe {
            cpu.map_host(CODE, 0x1000, Perm::RX, code.as_mut_ptr())
                .unwrap();
            cpu.map_host(DATA, 0x1000, Perm::RW, data.as_mut_ptr())
                .unwrap();
        }
        cpu.set_register(10, DATA);
        cpu.set_register(16, FPCR);
        cpu.set_register(17, FPSR);
        cpu.set_register(18, NZCV);
        cpu.set_pc(CODE);
        assert!(matches!(cpu.run_with_count(256).event, CpuEvent::Svc(0x7d)));
        for register in 0..31 {
            cpu.set_register(register, canary(register));
        }
        cpu.set_sp(canary(31));
        cpu.set_tpidrro_el0(canary(32));
        Self {
            initial_code: code.clone(),
            initial_data: data.clone(),
            cpu,
            code,
            data,
        }
    }

    fn sample(&mut self) -> u64 {
        self.cpu.set_pc(CODE + SAMPLE as u64);
        let before = self
            .cpu
            .guest_cntpct()
            .expect("Rustarmic must expose its MRS clock");
        assert!(matches!(
            self.cpu.run_with_count(256).event,
            CpuEvent::Svc(0x1e)
        ));
        let physical = self.cpu.get_register(20);
        let virtual_counter = self.cpu.get_register(21);
        let after = self.cpu.guest_cntpct().unwrap();
        assert!(
            before <= physical && physical <= virtual_counter && virtual_counter <= after,
            "different counter epochs: {before} <= {physical} <= {virtual_counter} <= {after}"
        );
        assert_eq!(self.cpu.get_register(22), 19_200_000);
        let registers: [u64; 31] =
            std::array::from_fn(|register| self.cpu.get_register(register as u32));
        let pc = self.cpu.get_pc();
        for _ in 0..64 {
            let count = self.cpu.guest_cntpct().unwrap();
            assert!(count >= after);
        }
        assert_eq!(self.cpu.get_pc(), pc, "read-only clock moved PC");
        for (register, value) in registers.into_iter().enumerate() {
            assert_eq!(
                self.cpu.get_register(register as u32),
                value,
                "read-only clock changed X{register}"
            );
            if !(20..=22).contains(&register) {
                assert_eq!(value, canary(register as u32), "MRS clobbered X{register}");
            }
        }
        assert_eq!(self.cpu.get_sp(), canary(31));
        assert_eq!(self.cpu.get_tpidrro_el0(), canary(32));
        assert!(self.cpu.take_fault().is_none());

        self.cpu.set_register(10, DATA + OUTPUT as u64);
        self.cpu.set_pc(CODE + INSPECT as u64);
        assert!(matches!(
            self.cpu.run_with_count(256).event,
            CpuEvent::Svc(0x7e)
        ));
        let mut expected = self.initial_data.clone();
        expected[OUTPUT..OUTPUT + 512].copy_from_slice(&self.initial_data[..512]);
        for (index, status) in [FPCR, FPSR, NZCV].into_iter().enumerate() {
            expected[OUTPUT + 512 + index * 8..OUTPUT + 520 + index * 8]
                .copy_from_slice(&status.to_le_bytes());
        }
        assert_eq!(
            &self.data[..],
            &expected[..],
            "vectors/status/memory guards"
        );
        assert_eq!(&self.code[..], &self.initial_code[..], "instruction guard");
        for (register, value) in registers.into_iter().enumerate() {
            self.cpu.set_register(register as u32, value);
        }
        after
    }
}

#[test]
fn two_cpus_share_the_exact_mrs_epoch_and_host_reads_preserve_all_state() {
    let start = rustarmic::CpuContext::default_cntpct();
    let mut first = Fixture::new(0);
    let mut second = Fixture::new(1);
    let mut previous = start;
    for _ in 0..16 {
        let first_count = first.sample();
        let second_count = second.sample();
        assert!(previous <= first_count && first_count <= second_count);
        previous = second_count;
    }
}

#[test]
fn counters_remain_in_the_same_epoch_when_cpus_run_on_different_host_threads() {
    let first = Fixture::new(0);
    let second = Fixture::new(1);
    let start = rustarmic::CpuContext::default_cntpct();
    let barrier = Arc::new(Barrier::new(2));
    let threads: Vec<_> = [first, second]
        .into_iter()
        .map(|mut fixture| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let mut previous = start;
                for _ in 0..16 {
                    let count = fixture.sample();
                    assert!(count >= previous);
                    previous = count;
                }
                previous
            })
        })
        .collect();
    for thread in threads {
        let count = thread.join().expect("counter worker panicked");
        assert!(count <= rustarmic::CpuContext::default_cntpct());
    }
}
