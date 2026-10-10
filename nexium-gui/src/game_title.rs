use nexium_cpu::CpuBackendKind;

#[derive(Debug, PartialEq, Eq)]
pub struct GameWindowTitle {
    pub text: String,
    pub cpu_backend: Option<CpuBackendKind>,
}

pub fn with_active_cpu_backend(title: String, backend: Option<CpuBackendKind>) -> GameWindowTitle {
    let label = match backend {
        Some(CpuBackendKind::Dynarmic) => "Dynarmic",
        Some(CpuBackendKind::Rustarmic) => "Rustarmic",
        Some(CpuBackendKind::Nce) => "NCE",
        None => return GameWindowTitle { text: title, cpu_backend: None },
    };
    GameWindowTitle {
        text: format!("{title} | {label}"),
        cpu_backend: backend,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_title_uses_the_active_cpu_backend() {
        let title = "NeXium 0.1.0 | Mario Kart 8 Deluxe 4.0.0 | Vulkan | GPU | Unlocked";
        for (backend, label) in [
            (CpuBackendKind::Dynarmic, "Dynarmic"),
            (CpuBackendKind::Rustarmic, "Rustarmic"),
            (CpuBackendKind::Nce, "NCE"),
        ] {
            assert_eq!(
                with_active_cpu_backend(title.into(), Some(backend)),
                GameWindowTitle {
                    text: format!("{title} | {label}"),
                    cpu_backend: Some(backend),
                },
            );
        }
    }

    #[test]
    fn game_title_does_not_guess_a_backend_before_cpu_initialization() {
        for title in ["NeXium", "NeXium 0.1.0 | Loading game"] {
            let prepared = with_active_cpu_backend(title.into(), None);
            assert_eq!(prepared.text, title);
            assert_eq!(prepared.cpu_backend, None);
        }
    }

    #[test]
    fn prepared_game_title_keeps_one_backend_snapshot_for_its_cache() {
        let title = "NeXium 0.1.0 | Game";
        let before_boot = with_active_cpu_backend(title.into(), None);
        let after_boot = with_active_cpu_backend(title.into(), Some(CpuBackendKind::Rustarmic));
        let after_fallback = with_active_cpu_backend(title.into(), Some(CpuBackendKind::Dynarmic));
        assert_eq!(before_boot.cpu_backend, None);
        assert_eq!(before_boot.text, title);
        assert_eq!(after_boot.cpu_backend, Some(CpuBackendKind::Rustarmic));
        assert_eq!(after_boot.text, format!("{title} | Rustarmic"));
        assert_eq!(after_fallback.cpu_backend, Some(CpuBackendKind::Dynarmic));
        assert_eq!(after_fallback.text, format!("{title} | Dynarmic"));
    }
}
