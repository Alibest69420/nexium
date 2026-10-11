use std::sync::OnceLock;
use std::sync::RwLock;
use std::time::Instant;

pub use crate::monitored_futex::{
    mwaitx_supported, umwait_supported, wait_backend, Futex, WaitBackend, WaitOutcome,
};

type WakeHook = std::sync::Arc<dyn Fn() + Send + Sync>;

static HOOK: RwLock<Option<WakeHook>> = RwLock::new(None);

fn event() -> &'static Futex {
    static EVENT: OnceLock<Futex> = OnceLock::new();
    EVENT.get_or_init(Futex::new)
}

pub fn install(hook: WakeHook) {
    *HOOK.write().unwrap() = Some(hook);
}

pub fn signal() {
    event().notify_all();
    let hook = HOOK.read().unwrap().clone();
    if let Some(hook) = hook {
        hook();
    }
}

pub fn micro_pause() {
    let event = event();
    event.poll_pause(event.generation());
}

pub fn wait_until(deadline: Instant) {
    let event = event();
    while Instant::now() < deadline {
        let _ = event.wait_until(event.generation(), deadline);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn signal_without_hook_is_noop_and_hook_fires_after_install() {
        signal();
        static FIRED: AtomicUsize = AtomicUsize::new(0);
        install(std::sync::Arc::new(|| {
            FIRED.fetch_add(1, Ordering::SeqCst);
        }));
        signal();
        signal();
        assert!(FIRED.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn micro_pause_returns_promptly() {
        let start = Instant::now();
        for _ in 0..8 {
            micro_pause();
        }
        assert!(start.elapsed() < Duration::from_millis(250));
    }

    #[test]
    fn wait_until_reaches_deadline() {
        let deadline = Instant::now() + Duration::from_micros(300);
        wait_until(deadline);
        assert!(Instant::now() >= deadline);
    }

    #[test]
    #[ignore = "precision timing; run on an idle host with --ignored"]
    fn micro_pause_wakes_on_the_host_wake_event() {
        if wait_backend() != WaitBackend::Mwaitx {
            return;
        }
        let _timing = crate::monitored_futex::serialize_timing_test();
        let result = crate::monitored_futex::median_below(5, Duration::from_micros(5), || {
            (0..11)
                .map(|_| {
                    let expected = event().generation();
                    let base = Instant::now();
                    let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let waiter = {
                        let ready = std::sync::Arc::clone(&ready);
                        std::thread::spawn(move || {
                            ready.store(true, Ordering::Release);
                            while event().generation() == expected {
                                micro_pause();
                            }
                            base.elapsed()
                        })
                    };
                    while !ready.load(Ordering::Acquire) {
                        std::thread::yield_now();
                    }
                    std::thread::sleep(Duration::from_millis(3));
                    let notified = base.elapsed();
                    event().notify_all();
                    waiter.join().unwrap().saturating_sub(notified)
                })
                .collect()
        });
        assert!(result.is_ok(), "{result:?}");
    }
}
