use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaitBackend {
    Umwait,
    Mwaitx,
    Os,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaitOutcome {
    Changed,
    TimedOut,
}

#[repr(align(128))]
struct WaitWord(AtomicU32);

#[derive(Default)]
struct Registration {
    #[cfg(windows)]
    waiters: Vec<std::sync::Arc<windows_wait::Waiter>>,
}

pub struct Futex {
    word: WaitWord,
    sleepers: AtomicU32,
    registration: Mutex<Registration>,
    wake: Condvar,
    backend: WaitBackend,
}

struct Sleeping<'a>(&'a AtomicU32);

impl<'a> Sleeping<'a> {
    fn enter(count: &'a AtomicU32) -> Self {
        count.fetch_add(1, Ordering::SeqCst);
        Self(count)
    }
}

impl Drop for Sleeping<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
const MONITORED_TAIL: Duration = Duration::from_millis(1);
#[cfg(not(any(target_os = "linux", target_os = "android")))]
const MONITORED_TAIL: Duration = Duration::from_millis(2);
const COARSE_TAIL: Duration = Duration::from_micros(100);
const HIGH_RES_TAIL: Duration = Duration::from_millis(1);
const FINE_SPIN: Duration = Duration::from_micros(20);
const WAIT_TICKS: u32 = 10_000;
const POLL_TICKS: u32 = 100_000;

#[cfg(test)]
pub(crate) fn serialize_timing_test() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
pub(crate) fn median_below(
    attempts: usize,
    bound: Duration,
    mut sample: impl FnMut() -> Vec<Duration>,
) -> Result<(), Vec<Duration>> {
    let mut last = Vec::new();
    for _ in 0..attempts {
        let mut values = sample();
        values.sort_unstable();
        if values.get(values.len() / 2).is_some_and(|median| *median < bound) {
            return Ok(());
        }
        last = values;
    }
    Err(last)
}

fn effective_tail(
    backend: WaitBackend,
    requested: Duration,
    remaining: Option<Duration>,
    precise: bool,
    high_resolution: bool,
) -> Duration {
    if backend == WaitBackend::Os && !precise {
        Duration::ZERO
    } else if high_resolution && remaining.is_some_and(|time| time > requested) {
        requested.min(HIGH_RES_TAIL)
    } else {
        requested
    }
}

impl Default for Futex {
    fn default() -> Self {
        Self::new()
    }
}

impl Futex {
    pub fn new() -> Self {
        Self::with_backend(wait_backend())
    }

    fn with_backend(backend: WaitBackend) -> Self {
        Self {
            word: WaitWord(AtomicU32::new(0)),
            sleepers: AtomicU32::new(0),
            registration: Mutex::new(Registration::default()),
            wake: Condvar::new(),
            backend,
        }
    }

    pub fn generation(&self) -> u32 {
        self.word.0.load(Ordering::Acquire)
    }

    pub fn notify_all(&self) {
        self.word.0.fetch_add(1, Ordering::SeqCst);
        if self.sleepers.load(Ordering::SeqCst) == 0 {
            return;
        }
        let registration = self.registration.lock().unwrap();
        #[cfg(windows)]
        for waiter in &registration.waiters {
            waiter.signal();
        }
        self.wake.notify_all();
        drop(registration);
    }

    pub fn wait(&self, expected: u32, timeout: Duration) -> WaitOutcome {
        self.wait_inner(expected, Instant::now().checked_add(timeout), COARSE_TAIL, false)
    }

    pub fn wait_until(&self, expected: u32, deadline: Instant) -> WaitOutcome {
        self.wait_inner(expected, Some(deadline), MONITORED_TAIL, true)
    }

    fn wait_inner(
        &self,
        expected: u32,
        deadline: Option<Instant>,
        monitored_tail: Duration,
        precise: bool,
    ) -> WaitOutcome {
        #[cfg(windows)]
        return self.wait_registered(
            expected,
            deadline,
            monitored_tail,
            precise,
            windows_wait::current(),
        );
        #[cfg(not(windows))]
        self.wait_registered(expected, deadline, monitored_tail, precise)
    }

    fn wait_registered(
        &self,
        expected: u32,
        deadline: Option<Instant>,
        monitored_tail: Duration,
        precise: bool,
        #[cfg(windows)] mut windows_waiter: Option<std::sync::Arc<windows_wait::Waiter>>,
    ) -> WaitOutcome {
        loop {
            if self.generation() != expected {
                return WaitOutcome::Changed;
            }
            let remaining = deadline.map(|end| end.saturating_duration_since(Instant::now()));
            if remaining == Some(Duration::ZERO) {
                return WaitOutcome::TimedOut;
            }
            #[cfg(windows)]
            let high_resolution = windows_waiter.is_some();
            #[cfg(not(windows))]
            let high_resolution = false;
            let tail =
                effective_tail(self.backend, monitored_tail, remaining, precise, high_resolution);
            if let Some(time) = remaining.filter(|time| *time <= tail) {
                if precise && time <= FINE_SPIN {
                    std::hint::spin_loop();
                } else {
                    self.pause(expected);
                }
                continue;
            }
            let registration = self.registration.lock().unwrap();
            #[cfg(windows)]
            let mut registration = registration;
            let _sleeping = Sleeping::enter(&self.sleepers);
            if self.word.0.load(Ordering::SeqCst) != expected {
                return WaitOutcome::Changed;
            }
            let coarse = match deadline {
                Some(end) => {
                    let remaining = end.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return WaitOutcome::TimedOut;
                    }
                    let tail = effective_tail(
                        self.backend,
                        monitored_tail,
                        Some(remaining),
                        precise,
                        high_resolution,
                    );
                    let coarse = remaining.saturating_sub(tail);
                    if coarse.is_zero() {
                        continue;
                    }
                    Some(coarse)
                }
                None => None,
            };
            #[cfg(windows)]
            if let Some(waiter) = windows_waiter.as_ref() {
                if waiter.prepare(coarse) {
                    registration.waiters.push(std::sync::Arc::clone(waiter));
                    drop(registration);
                    let success = waiter.wait(coarse.is_some());
                    let mut registration = self.registration.lock().unwrap();
                    registration
                        .waiters
                        .retain(|registered| !std::sync::Arc::ptr_eq(registered, waiter));
                    waiter.cancel();
                    drop(registration);
                    if !success {
                        windows_waiter = None;
                    }
                    continue;
                }
                windows_waiter = None;
                continue;
            }
            match coarse {
                Some(timeout) => drop(self.wake.wait_timeout(registration, timeout).unwrap()),
                None => drop(self.wake.wait(registration).unwrap()),
            }
        }
    }

    pub fn pause(&self, expected: u32) {
        self.pause_ticks(expected, WAIT_TICKS);
    }

    pub fn poll_pause(&self, expected: u32) {
        let ticks = if self.backend == WaitBackend::Mwaitx { POLL_TICKS } else { WAIT_TICKS };
        self.pause_ticks(expected, ticks);
    }

    fn pause_ticks(&self, expected: u32, ticks: u32) {
        if self.generation() != expected {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            match self.backend {
                WaitBackend::Umwait => umwait(&self.word.0, expected, ticks),
                WaitBackend::Mwaitx => mwaitx(&self.word.0, expected, ticks),
                WaitBackend::Os => spin_pause(),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = ticks;
            spin_pause();
        }
    }
}

#[cfg(windows)]
mod windows_wait {
    use std::ffi::{c_char, c_void};
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle, RawHandle};
    use std::sync::{Arc, OnceLock};
    use std::time::Duration;

    type CreateTimer = unsafe extern "system" fn(*const c_void, *const u16, u32, u32) -> RawHandle;
    type TimerApc = unsafe extern "system" fn(*mut c_void, u32, u32);

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleA(name: *const c_char) -> RawHandle;
        fn GetProcAddress(module: RawHandle, name: *const c_char) -> *mut c_void;
        fn CreateEventW(
            attributes: *const c_void,
            manual_reset: i32,
            initial_state: i32,
            name: *const u16,
        ) -> RawHandle;
        fn SetEvent(event: RawHandle) -> i32;
        fn ResetEvent(event: RawHandle) -> i32;
        fn SetWaitableTimer(
            timer: RawHandle,
            due: *const i64,
            period: i32,
            callback: Option<TimerApc>,
            argument: *mut c_void,
            resume: i32,
        ) -> i32;
        fn CancelWaitableTimer(timer: RawHandle) -> i32;
        fn WaitForMultipleObjects(
            count: u32,
            handles: *const RawHandle,
            all: i32,
            timeout: u32,
        ) -> u32;
    }

    pub(super) struct Waiter {
        timer: OwnedHandle,
        event: OwnedHandle,
    }

    thread_local! {
        static WAITER: Option<Arc<Waiter>> = Waiter::new().map(Arc::new);
    }

    pub(super) fn current() -> Option<Arc<Waiter>> {
        WAITER.try_with(Clone::clone).ok().flatten()
    }

    impl Waiter {
        fn new() -> Option<Self> {
            static CREATE_TIMER: OnceLock<Option<CreateTimer>> = OnceLock::new();
            let create_timer = CREATE_TIMER.get_or_init(|| unsafe {
                let module = GetModuleHandleA(c"kernel32.dll".as_ptr());
                if module.is_null() {
                    return None;
                }
                let address = GetProcAddress(module, c"CreateWaitableTimerExW".as_ptr());
                (!address.is_null())
                    .then(|| std::mem::transmute::<*mut c_void, CreateTimer>(address))
            });
            let create_timer = (*create_timer)?;
            unsafe {
                let timer = create_timer(std::ptr::null(), std::ptr::null(), 2, 0x0010_0002);
                if timer.is_null() {
                    return None;
                }
                let timer = OwnedHandle::from_raw_handle(timer);
                let event = CreateEventW(std::ptr::null(), 1, 0, std::ptr::null());
                if event.is_null() {
                    return None;
                }
                Some(Self {
                    timer,
                    event: OwnedHandle::from_raw_handle(event),
                })
            }
        }

        pub(super) fn prepare(&self, timeout: Option<Duration>) -> bool {
            unsafe {
                if ResetEvent(self.event.as_raw_handle()) == 0 {
                    return false;
                }
                match timeout {
                    Some(timeout) => {
                        let due = relative_due_time(timeout);
                        SetWaitableTimer(
                            self.timer.as_raw_handle(),
                            &due,
                            0,
                            None,
                            std::ptr::null_mut(),
                            0,
                        ) != 0
                    }
                    None => CancelWaitableTimer(self.timer.as_raw_handle()) != 0,
                }
            }
        }

        pub(super) fn wait(&self, timed: bool) -> bool {
            let handles = [self.event.as_raw_handle(), self.timer.as_raw_handle()];
            let count = if timed { 2 } else { 1 };
            let outcome = unsafe { WaitForMultipleObjects(count, handles.as_ptr(), 0, u32::MAX) };
            outcome < count
        }

        pub(super) fn signal(&self) {
            unsafe {
                SetEvent(self.event.as_raw_handle());
            }
        }

        pub(super) fn cancel(&self) {
            unsafe {
                CancelWaitableTimer(self.timer.as_raw_handle());
            }
        }
    }

    fn relative_due_time(timeout: Duration) -> i64 {
        -(timeout.as_nanos().div_ceil(100).clamp(1, i64::MAX as u128) as i64)
    }

    #[cfg(test)]
    pub(super) fn timer_units(timeout: Duration) -> i64 {
        relative_due_time(timeout)
    }
}

fn spin_pause() {
    for _ in 0..64 {
        std::hint::spin_loop();
    }
}

pub fn wait_backend() -> WaitBackend {
    static BACKEND: OnceLock<WaitBackend> = OnceLock::new();
    *BACKEND.get_or_init(detect_backend)
}

pub fn mwaitx_supported() -> bool {
    capabilities().1 && std::env::var_os("NEXIUM_NO_MWAITX").is_none()
}

pub fn umwait_supported() -> bool {
    capabilities().0 && std::env::var_os("NEXIUM_NO_UMWAIT").is_none()
}

fn detect_backend() -> WaitBackend {
    if std::env::var_os("NEXIUM_NO_MONITORED_WAIT").is_some() {
        return WaitBackend::Os;
    }
    let (umwait, mwaitx) = capabilities();
    select_backend(
        umwait && std::env::var_os("NEXIUM_NO_UMWAIT").is_none(),
        mwaitx && std::env::var_os("NEXIUM_NO_MWAITX").is_none(),
    )
}

fn select_backend(umwait: bool, mwaitx: bool) -> WaitBackend {
    if umwait {
        WaitBackend::Umwait
    } else if mwaitx {
        WaitBackend::Mwaitx
    } else {
        WaitBackend::Os
    }
}

fn capabilities() -> (bool, bool) {
    static CAPS: OnceLock<(bool, bool)> = OnceLock::new();
    *CAPS.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{__cpuid, __cpuid_count};
            let basic = __cpuid(0);
            let extended = __cpuid(0x8000_0000).eax;
            let leaf1 = __cpuid(1);
            let tsc = leaf1.edx & (1 << 4) != 0;
            let invariant = extended >= 0x8000_0007 && __cpuid(0x8000_0007).edx & (1 << 8) != 0;
            let leaf7 = (basic.eax >= 7).then(|| __cpuid_count(7, 0));
            let unaffected =
                leaf7.is_some_and(|leaf| leaf.eax >= 2 && __cpuid_count(7, 2).edx & (1 << 7) != 0);
            let vendor_intel =
                basic.ebx == 0x756e_6547 && basic.edx == 0x4965_6e69 && basic.ecx == 0x6c65_746e;
            let family = (leaf1.eax >> 8) & 15;
            let model = ((leaf1.eax >> 4) & 15) | ((leaf1.eax >> 12) & 0xf0);
            let affected = vendor_intel
                && family == 6
                && matches!(model, 0x8f | 0xcf | 0xaf | 0xad)
                && !unaffected;
            let umwait =
                tsc && invariant && !affected && leaf7.is_some_and(|leaf| leaf.ecx & (1 << 5) != 0);
            let mwaitx = extended >= 0x8000_0001 && __cpuid(0x8000_0001).ecx & (1 << 29) != 0;
            (umwait, mwaitx)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            (false, false)
        }
    })
}

#[cfg(target_arch = "x86_64")]
unsafe fn umwait(word: &AtomicU32, expected: u32, ticks: u32) {
    let deadline = unsafe { std::arch::x86_64::_rdtsc() }.wrapping_add(ticks as u64);
    unsafe {
        std::arch::asm!(
        "umonitor {word}",
        "cmp dword ptr [{word}], {expected:e}",
        "jne 2f",
        "umwait ecx",
        "2:",
        word = in(reg) word.as_ptr(),
        expected = in(reg) expected,
        in("eax") deadline as u32,
        in("edx") (deadline >> 32) as u32,
        in("ecx") 1u32,
        options(nostack)
        );
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn mwaitx(word: &AtomicU32, expected: u32, ticks: u32) {
    unsafe {
        std::arch::asm!(
        "monitorx",
        "cmp dword ptr [rax], {expected:e}",
        "jne 2f",
        "mov {saved_rbx}, rbx",
        "mov ebx, {ticks:e}",
        "xor eax, eax",
        "mov ecx, 2",
        "mwaitx",
        "mov rbx, {saved_rbx}",
        "2:",
        expected = in(reg) expected,
        ticks = in(reg) ticks,
        saved_rbx = out(reg) _,
        inout("rax") word.as_ptr() => _,
        inout("ecx") 0u32 => _,
        in("edx") 0u32,
        options(nostack)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier, mpsc};

    #[test]
    fn independent_capabilities_select_only_legal_backend() {
        assert_eq!(select_backend(false, false), WaitBackend::Os);
        assert_eq!(select_backend(true, false), WaitBackend::Umwait);
        assert_eq!(select_backend(false, true), WaitBackend::Mwaitx);
        assert_eq!(select_backend(true, true), WaitBackend::Umwait);
    }

    #[test]
    fn wait_word_is_cache_line_separated() {
        let event = Futex::with_backend(WaitBackend::Os);
        let word = &event.word as *const _ as usize;
        let registration = &event.registration as *const _ as usize;
        let sleepers = &event.sleepers as *const _ as usize;
        assert_eq!(word % 128, 0);
        assert!(word.abs_diff(registration) >= 128);
        assert!(word.abs_diff(sleepers) >= 128);
    }

    #[test]
    fn tails_cover_relative_absolute_and_os_waits() {
        let far = Some(Duration::from_millis(17));
        let near = Some(MONITORED_TAIL.saturating_sub(Duration::from_micros(500)));
        for backend in [WaitBackend::Mwaitx, WaitBackend::Umwait] {
            assert_eq!(effective_tail(backend, MONITORED_TAIL, far, true, true), HIGH_RES_TAIL);
            assert_eq!(effective_tail(backend, MONITORED_TAIL, near, true, true), MONITORED_TAIL);
            assert_eq!(
                effective_tail(backend, MONITORED_TAIL, Some(MONITORED_TAIL), true, true),
                MONITORED_TAIL
            );
            assert_eq!(effective_tail(backend, MONITORED_TAIL, far, true, false), MONITORED_TAIL);
            assert_eq!(effective_tail(backend, MONITORED_TAIL, None, true, true), MONITORED_TAIL);
            assert_eq!(effective_tail(backend, COARSE_TAIL, far, false, true), COARSE_TAIL);
            assert_eq!(effective_tail(backend, COARSE_TAIL, far, false, false), COARSE_TAIL);
        }
        let os = WaitBackend::Os;
        assert_eq!(effective_tail(os, MONITORED_TAIL, far, true, true), HIGH_RES_TAIL);
        assert_eq!(effective_tail(os, MONITORED_TAIL, near, true, true), MONITORED_TAIL);
        assert_eq!(effective_tail(os, MONITORED_TAIL, far, true, false), MONITORED_TAIL);
        assert_eq!(effective_tail(os, COARSE_TAIL, far, false, true), Duration::ZERO);
        assert_eq!(effective_tail(os, COARSE_TAIL, None, false, false), Duration::ZERO);
        assert!(FINE_SPIN < HIGH_RES_TAIL && FINE_SPIN < COARSE_TAIL);
    }

    #[test]
    fn notification_before_wait_is_not_lost() {
        let event = Futex::with_backend(WaitBackend::Os);
        let expected = event.generation();
        event.notify_all();
        assert_eq!(event.wait(expected, Duration::ZERO), WaitOutcome::Changed);
    }

    #[test]
    fn unchanged_word_times_out_without_an_early_return() {
        for backend in [WaitBackend::Os, wait_backend()] {
            let event = Futex::with_backend(backend);
            let deadline = Instant::now() + Duration::from_micros(200);
            assert_eq!(
                event.wait_until(event.generation(), deadline),
                WaitOutcome::TimedOut
            );
            assert!(Instant::now() >= deadline);
        }
    }

    #[test]
    fn concurrent_registration_and_signal_do_not_lose_wakes() {
        for backend in [WaitBackend::Os, wait_backend()] {
            let event = Arc::new(Futex::with_backend(backend));
            let payload = Arc::new(AtomicU32::new(0));
            let (ready_tx, ready_rx) = mpsc::sync_channel(1);
            let (done_tx, done_rx) = mpsc::sync_channel(1);
            let waiter = {
                let event = Arc::clone(&event);
                let payload = Arc::clone(&payload);
                std::thread::spawn(move || {
                    for value in 1..=1024 {
                        let expected = event.generation();
                        ready_tx.send(()).unwrap();
                        assert_eq!(
                            event.wait(expected, Duration::from_secs(2)),
                            WaitOutcome::Changed
                        );
                        assert_eq!(payload.load(Ordering::Relaxed), value);
                        done_tx.send(()).unwrap();
                    }
                })
            };
            for value in 1..=1024 {
                ready_rx.recv_timeout(Duration::from_secs(3)).unwrap();
                payload.store(value, Ordering::Relaxed);
                event.notify_all();
                done_rx.recv_timeout(Duration::from_secs(3)).unwrap();
            }
            waiter.join().unwrap();
            assert_eq!(event.sleepers.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn notify_without_sleepers_skips_the_registration_lock() {
        let event = Arc::new(Futex::with_backend(WaitBackend::Os));
        let expected = event.generation();
        let held = event.registration.lock().unwrap();
        let (done_tx, done_rx) = mpsc::sync_channel(1);
        let notifier = {
            let event = Arc::clone(&event);
            std::thread::spawn(move || {
                event.notify_all();
                done_tx.send(()).unwrap();
            })
        };
        let completed = done_rx.recv_timeout(Duration::from_secs(2));
        drop(held);
        notifier.join().unwrap();
        assert!(completed.is_ok());
        assert_ne!(event.generation(), expected);
    }

    #[test]
    fn notification_during_the_monitored_tail_ends_the_wait() {
        if std::thread::available_parallelism().map_or(1, |count| count.get()) < 2 {
            return;
        }
        let _timing = serialize_timing_test();
        for backend in [WaitBackend::Os, wait_backend()] {
            let mut confirmed = false;
            for _ in 0..50 {
                let event = Arc::new(Futex::with_backend(backend));
                let expected = event.generation();
                let base = Instant::now();
                let notified_at = Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX));
                let state = Arc::new(AtomicU32::new(0));
                let notifier = {
                    let event = Arc::clone(&event);
                    let state = Arc::clone(&state);
                    let notified_at = Arc::clone(&notified_at);
                    std::thread::spawn(move || {
                        state.store(1, Ordering::Release);
                        while state.load(Ordering::Acquire) != 2 {
                            std::hint::spin_loop();
                        }
                        let start = Instant::now();
                        while start.elapsed() < Duration::from_micros(200) {
                            std::hint::spin_loop();
                        }
                        notified_at.store(base.elapsed().as_nanos() as u64, Ordering::Release);
                        event.notify_all();
                    })
                };
                while state.load(Ordering::Acquire) != 1 {
                    std::thread::yield_now();
                }
                let deadline =
                    Instant::now() + MONITORED_TAIL.saturating_sub(Duration::from_micros(100));
                state.store(2, Ordering::Release);
                let outcome = event.wait_until(expected, deadline);
                let woke = Instant::now();
                notifier.join().unwrap();
                let notified = base + Duration::from_nanos(notified_at.load(Ordering::Acquire));
                if notified + Duration::from_micros(100) < deadline
                    && outcome == WaitOutcome::Changed
                    && woke < deadline
                {
                    confirmed = true;
                    break;
                }
            }
            assert!(confirmed, "{backend:?}");
        }
    }

    #[test]
    fn monitored_pause_wakes_on_notification() {
        let backend = wait_backend();
        if backend == WaitBackend::Os {
            return;
        }
        let _timing = serialize_timing_test();
        let event = Arc::new(Futex::with_backend(backend));
        let result = median_below(5, Duration::from_micros(20), || {
            (0..11)
                .map(|_| {
                    let expected = event.generation();
                    let base = Instant::now();
                    let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let waiter = {
                        let event = Arc::clone(&event);
                        let ready = Arc::clone(&ready);
                        std::thread::spawn(move || {
                            ready.store(true, Ordering::Release);
                            while event.generation() == expected {
                                event.pause_ticks(expected, u32::MAX);
                            }
                            base.elapsed()
                        })
                    };
                    while !ready.load(Ordering::Acquire) {
                        std::thread::yield_now();
                    }
                    std::thread::sleep(Duration::from_millis(3));
                    let notified = base.elapsed();
                    event.notify_all();
                    waiter.join().unwrap().saturating_sub(notified)
                })
                .collect()
        });
        assert!(result.is_ok(), "{backend:?} {result:?}");
    }

    #[test]
    fn absolute_deadlines_do_not_inherit_os_timer_lateness() {
        let _timing = serialize_timing_test();
        for backend in [WaitBackend::Os, wait_backend()] {
            let event = Futex::with_backend(backend);
            for delay in [Duration::from_micros(300), Duration::from_micros(1500)] {
                let result = median_below(5, Duration::from_micros(10), || {
                    (0..15)
                        .map(|_| {
                            let deadline = Instant::now() + delay;
                            assert_eq!(event.wait_until(event.generation(), deadline), WaitOutcome::TimedOut);
                            Instant::now().saturating_duration_since(deadline)
                        })
                        .collect()
                });
                assert!(result.is_ok(), "{backend:?} {delay:?} {result:?}");
            }
        }
    }

    #[test]
    fn absolute_deadlines_finish_with_a_fine_spin() {
        let _timing = serialize_timing_test();
        for backend in [WaitBackend::Os, wait_backend()] {
            let event = Futex::with_backend(backend);
            let result = median_below(5, Duration::from_micros(1), || {
                (0..21)
                    .map(|_| {
                        let deadline = Instant::now() + Duration::from_micros(300);
                        assert_eq!(event.wait_until(event.generation(), deadline), WaitOutcome::TimedOut);
                        Instant::now().saturating_duration_since(deadline)
                    })
                    .collect()
            });
            assert!(result.is_ok(), "{backend:?} {result:?}");
        }
    }

    #[test]
    fn one_event_wakes_every_registered_waiter() {
        let event = Arc::new(Futex::with_backend(WaitBackend::Os));
        let barrier = Arc::new(Barrier::new(5));
        let waiters: Vec<_> = (0..4)
            .map(|_| {
                let event = Arc::clone(&event);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let expected = event.generation();
                    barrier.wait();
                    event.wait(expected, Duration::from_secs(2))
                })
            })
            .collect();
        barrier.wait();
        event.notify_all();
        for waiter in waiters {
            assert_eq!(waiter.join().unwrap(), WaitOutcome::Changed);
        }
    }

    fn wait_for_sleepers<T>(event: &Futex, expected_count: u32, waiter: &std::thread::JoinHandle<T>) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while event.sleepers.load(Ordering::SeqCst) != expected_count && !waiter.is_finished() {
            assert!(Instant::now() < deadline, "waiters did not enter the OS wait");
            std::thread::yield_now();
        }
    }

    #[test]
    fn spurious_os_notifications_do_not_complete_the_wait() {
        let event = Arc::new(Futex::with_backend(WaitBackend::Os));
        let expected = event.generation();
        let waiter = {
            let event = Arc::clone(&event);
            std::thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_millis(50);
                #[cfg(windows)]
                let outcome =
                    event.wait_registered(expected, Some(deadline), COARSE_TAIL, false, None);
                #[cfg(not(windows))]
                let outcome = event.wait_registered(expected, Some(deadline), COARSE_TAIL, false);
                (outcome, Instant::now(), deadline)
            })
        };
        wait_for_sleepers(&event, 1, &waiter);
        for _ in 0..32 {
            drop(event.registration.lock().unwrap());
            event.wake.notify_all();
            std::thread::yield_now();
        }
        let (outcome, completed, deadline) = waiter.join().unwrap();
        assert_eq!(outcome, WaitOutcome::TimedOut);
        assert!(completed >= deadline);
        assert_eq!(event.generation(), expected);
        assert_eq!(event.sleepers.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn short_pause_is_bounded_and_observes_old_generations() {
        let event = Futex::new();
        let expected = event.generation();
        let start = Instant::now();
        for _ in 0..32 {
            event.pause(expected);
        }
        assert!(start.elapsed() < Duration::from_secs(1));
        let start = Instant::now();
        for _ in 0..4 {
            event.poll_pause(expected);
        }
        assert!(start.elapsed() < Duration::from_secs(1));
        event.notify_all();
        let start = Instant::now();
        for _ in 0..10_000 {
            event.pause(expected);
            event.poll_pause(expected);
        }
        assert!(start.elapsed() < Duration::from_millis(10));
    }

    #[test]
    fn generation_wrap_still_changes_the_observed_value() {
        let event = Futex::with_backend(WaitBackend::Os);
        event.word.0.store(u32::MAX, Ordering::Relaxed);
        event.notify_all();
        assert_eq!(event.generation(), 0);
        assert_eq!(event.wait(u32::MAX, Duration::ZERO), WaitOutcome::Changed);
    }

    #[cfg(windows)]
    fn wait_for_windows_registration(
        event: &Futex,
        expected_count: usize,
        finished: impl Fn() -> bool,
    ) {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if event.registration.lock().unwrap().waiters.len() == expected_count || finished() {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "Windows waiters did not register"
            );
            std::thread::yield_now();
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_relative_timer_intervals_round_up_without_overflow() {
        assert_eq!(windows_wait::timer_units(Duration::ZERO), -1);
        assert_eq!(windows_wait::timer_units(Duration::from_nanos(1)), -1);
        assert_eq!(windows_wait::timer_units(Duration::from_nanos(100)), -1);
        assert_eq!(windows_wait::timer_units(Duration::from_nanos(101)), -2);
        assert_eq!(
            windows_wait::timer_units(Duration::from_micros(16_667)),
            -166_670
        );
        assert_eq!(windows_wait::timer_units(Duration::MAX), -i64::MAX);
    }

    #[cfg(windows)]
    #[test]
    fn windows_event_signal_is_retained_before_entering_the_os_wait() {
        let Some(waiter) = windows_wait::current() else {
            return;
        };
        let event = Futex::with_backend(WaitBackend::Os);
        let expected = event.generation();
        let sleeping = {
            let mut registration = event.registration.lock().unwrap();
            let sleeping = Sleeping::enter(&event.sleepers);
            assert_eq!(event.generation(), expected);
            assert!(waiter.prepare(Some(Duration::from_secs(3))));
            registration.waiters.push(Arc::clone(&waiter));
            sleeping
        };
        event.notify_all();
        let start = Instant::now();
        assert!(waiter.wait(true));
        assert!(start.elapsed() < Duration::from_secs(1));
        let mut registration = event.registration.lock().unwrap();
        registration.waiters.clear();
        waiter.cancel();
        drop(registration);
        drop(sleeping);
        assert_ne!(event.generation(), expected);
        assert_eq!(event.sleepers.load(Ordering::SeqCst), 0);
    }

    #[cfg(windows)]
    #[test]
    fn windows_notification_interrupts_all_registered_coarse_waits() {
        if windows_wait::current().is_none() {
            return;
        }
        let event = Arc::new(Futex::with_backend(WaitBackend::Os));
        let expected = event.generation();
        let (completed_tx, completed_rx) = mpsc::channel();
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let event = Arc::clone(&event);
                let completed_tx = completed_tx.clone();
                std::thread::spawn(move || {
                    completed_tx
                        .send(event.wait(expected, Duration::from_secs(5)))
                        .unwrap();
                })
            })
            .collect();
        wait_for_windows_registration(&event, 4, || threads.iter().all(|thread| thread.is_finished()));
        event.notify_all();
        for _ in 0..4 {
            assert_eq!(
                completed_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
                WaitOutcome::Changed
            );
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert!(event.registration.lock().unwrap().waiters.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn windows_private_event_spurious_signals_keep_the_original_deadline() {
        if windows_wait::current().is_none() {
            return;
        }
        let event = Arc::new(Futex::with_backend(WaitBackend::Os));
        let expected = event.generation();
        let waiter = {
            let event = Arc::clone(&event);
            std::thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_millis(50);
                let result = event.wait_until(expected, deadline);
                (result, Instant::now(), deadline)
            })
        };
        wait_for_windows_registration(&event, 1, || waiter.is_finished());
        for _ in 0..32 {
            let registration = event.registration.lock().unwrap();
            for waiter in &registration.waiters {
                waiter.signal();
            }
            drop(registration);
            std::thread::yield_now();
        }
        let (result, completed, deadline) = waiter.join().unwrap();
        assert_eq!(result, WaitOutcome::TimedOut);
        assert!(completed >= deadline);
        assert!(completed.saturating_duration_since(deadline) < Duration::from_secs(1));
        assert_eq!(event.generation(), expected);
        assert!(event.registration.lock().unwrap().waiters.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn windows_timer_and_event_resources_are_reused_and_released_at_thread_exit() {
        let Some(first) = windows_wait::current() else {
            return;
        };
        let events = [
            Futex::with_backend(WaitBackend::Os),
            Futex::with_backend(WaitBackend::Os),
        ];
        for index in 0..32 {
            let cached = windows_wait::current().unwrap();
            assert!(Arc::ptr_eq(&first, &cached));
            drop(cached);
            let event = &events[index % events.len()];
            let deadline = Instant::now() + Duration::from_micros(200);
            assert_eq!(
                event.wait(event.generation(), Duration::from_micros(200)),
                WaitOutcome::TimedOut
            );
            assert!(Instant::now() >= deadline);
            assert!(event.registration.lock().unwrap().waiters.is_empty());
            assert_eq!(Arc::strong_count(&first), 2);
        }
        let weak = std::thread::spawn(|| {
            let waiter = windows_wait::current().unwrap();
            let weak = Arc::downgrade(&waiter);
            let event = Futex::with_backend(WaitBackend::Os);
            assert_eq!(
                event.wait(event.generation(), Duration::from_micros(200)),
                WaitOutcome::TimedOut
            );
            assert!(event.registration.lock().unwrap().waiters.is_empty());
            weak
        })
        .join()
        .unwrap();
        assert!(weak.upgrade().is_none());
    }

    #[cfg(windows)]
    #[test]
    fn unavailable_windows_waiter_retains_the_registered_condvar_fallback() {
        let event = Arc::new(Futex::with_backend(WaitBackend::Os));
        let expected = event.generation();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let (done_tx, done_rx) = mpsc::sync_channel(1);
        let waiter = {
            let event = Arc::clone(&event);
            std::thread::spawn(move || {
                ready_tx.send(()).unwrap();
                let result = event.wait_registered(
                    expected,
                    Some(Instant::now() + Duration::from_secs(3)),
                    COARSE_TAIL,
                    false,
                    None,
                );
                done_tx.send(result).unwrap();
            })
        };
        ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        event.notify_all();
        assert_eq!(
            done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            WaitOutcome::Changed
        );
        waiter.join().unwrap();
        let deadline = Instant::now() + Duration::from_micros(200);
        assert_eq!(
            event.wait_registered(event.generation(), Some(deadline), COARSE_TAIL, false, None),
            WaitOutcome::TimedOut
        );
        assert!(Instant::now() >= deadline);
    }
}
