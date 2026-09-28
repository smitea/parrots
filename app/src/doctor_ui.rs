//! Health check UI (mirrors `parrots doctor`): devices / models / voice
//! profile / hotwords, each rendered as OK / WARN / FAIL with a fix hint.

use eframe::egui;
use parrots_platform_macos::PARROTS_DEVICE_NAMES;

use crate::AppPaths;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok,
    Warn,
    Fail,
}

pub struct Check {
    pub status: Status,
    pub name: String,
    pub detail: String,
}

impl Check {
    fn label(&self) -> (&'static str, [u8; 3]) {
        match self.status {
            Status::Ok => ("OK ", [80, 200, 120]),
            Status::Warn => ("WARN", [230, 190, 80]),
            Status::Fail => ("FAIL", [230, 100, 100]),
        }
    }
}

/// Run all checks. `models_root`/`profiles_dir` follow the same conventions
/// as the CLI (`PARROTS_MODELS` / `PARROTS_PROFILES` overrides).
pub fn run_checks(paths: &AppPaths) -> Vec<Check> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let mut checks = Vec::new();

    let host = cpal::default_host();
    for (role, dev) in [
        ("default input device", host.default_input_device()),
        ("default output device", host.default_output_device()),
    ] {
        checks.push(match dev {
            Some(d) => Check {
                status: Status::Ok,
                name: role.into(),
                detail: d.name().unwrap_or_else(|_| "(unnamed)".into()),
            },
            None => Check {
                status: Status::Fail,
                name: role.into(),
                detail: "no device".into(),
            },
        });
    }

    for (name, role) in [
        (
            PARROTS_DEVICE_NAMES[0],
            "direction A exit (meeting app microphone)",
        ),
        (
            PARROTS_DEVICE_NAMES[1],
            "direction B entry (meeting app speaker)",
        ),
    ] {
        if parrots_platform_macos::find_device(name).is_some() {
            checks.push(Check {
                status: Status::Ok,
                name: format!("virtual device {name}"),
                detail: format!("installed ({role})"),
            });
        } else {
            checks.push(Check {
                status: Status::Warn,
                name: format!("virtual device {name}"),
                detail: format!(
                    "not installed ({role}); fix: cd driver/macos && ./make-pkg.sh, then install the pkg"
                ),
            });
        }
    }

    for rel in [
        "asr/sensevoice/model.int8.onnx",
        "vad/silero_vad.onnx",
        "mt/en-zh/encoder_model_quantized.onnx",
        "mt/zh-en/encoder_model_quantized.onnx",
        "tts/zipvoice/encoder.int8.onnx",
    ] {
        let path = paths.models_root.join(rel);
        checks.push(if path.is_file() {
            Check {
                status: Status::Ok,
                name: format!("model {rel}"),
                detail: path.display().to_string(),
            }
        } else {
            Check {
                status: Status::Fail,
                name: format!("model {rel}"),
                detail: format!("missing under {}", paths.models_root.display()),
            }
        });
    }

    let voice_wav = paths.profiles_dir.join(format!("{}.wav", paths.voice_name));
    let voice_txt = paths.profiles_dir.join(format!("{}.txt", paths.voice_name));
    checks.push(if voice_wav.is_file() && voice_txt.is_file() {
        Check {
            status: Status::Ok,
            name: format!("voice profile {}", paths.voice_name),
            detail: voice_wav.display().to_string(),
        }
    } else {
        Check {
            status: Status::Warn,
            name: format!("voice profile {}", paths.voice_name),
            detail: "missing; record it on the Settings tab".into(),
        }
    });

    checks
}

/// Render the health tab; returns true when everything critical passes.
pub fn show(ui: &mut egui::Ui, checks: &[Check]) -> bool {
    let mut all_ok = true;
    egui::ScrollArea::vertical().show(ui, |ui| {
        for c in checks {
            let (tag, color) = c.label();
            if c.status == Status::Fail {
                all_ok = false;
            }
            ui.horizontal(|ui| {
                ui.colored_label(egui::Color32::from_rgb(color[0], color[1], color[2]), tag);
                ui.label(format!("{} — {}", c.name, c.detail));
            });
        }
    });
    all_ok
}
