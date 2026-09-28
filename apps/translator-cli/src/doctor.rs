//! `parrots doctor`: environment health check — default devices / virtual driver / models / voiceprint readiness

use crate::wire;
use cpal::traits::{DeviceTrait, HostTrait};
use parrots_platform_macos::find_device;

enum Status {
    Ok,
    Warn,
    Fail,
}

struct Check {
    status: Status,
    name: String,
    detail: String,
}

impl Check {
    fn print(&self) {
        let tag = match self.status {
            Status::Ok => "OK ",
            Status::Warn => "WARN",
            Status::Fail => "FAIL",
        };
        println!("[{tag}] {} — {}", self.name, self.detail);
    }
}

fn default_device_check(role: &str, device: Option<cpal::Device>) -> Check {
    match device {
        Some(d) => Check {
            status: Status::Ok,
            name: role.into(),
            detail: d.name().unwrap_or_else(|_| "(unable to read name)".into()),
        },
        None => Check {
            status: Status::Fail,
            name: role.into(),
            detail: "no default device".into(),
        },
    }
}

/// A missing virtual device is only a WARN: features fall back to the default device; installing it enables driver mode
fn virtual_device_check(name: &str, role: &str) -> Check {
    match find_device(name) {
        Some(_) => Check {
            status: Status::Ok,
            name: format!("virtual device {name}"),
            detail: format!("installed ({role})"),
        },
        None => Check {
            status: Status::Warn,
            name: format!("virtual device {name}"),
            detail: format!("not installed ({role}); install: {}", wire::INSTALL_HINT),
        },
    }
}

fn model_check(root: &std::path::Path, rel: &str) -> Check {
    let path = root.join(rel);
    if path.is_file() {
        Check {
            status: Status::Ok,
            name: format!("model {rel}"),
            detail: path.display().to_string(),
        }
    } else {
        Check {
            status: Status::Fail,
            name: format!("model {rel}"),
            detail: format!("missing (not found under {})", root.display()),
        }
    }
}

/// List voice profile names under profiles/ (stems of *.wav, sorted and comma-joined)
fn voiceprints_check() -> Check {
    let dir = wire::profiles_dir();
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "wav"))
        .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
        .collect();
    names.sort();
    if names.is_empty() {
        Check {
            status: Status::Warn,
            name: format!("voice profiles {}", dir.display()),
            detail: "no profiles (talk requires parrots enroll --name my first)".into(),
        }
    } else {
        Check {
            status: Status::Ok,
            name: format!("voice profiles {}", dir.display()),
            detail: names.join(", "),
        }
    }
}

/// Hotword list (plan 6): count + how to add more
fn hotwords_check() -> Check {
    let hw = wire::load_hotwords();
    if hw.is_empty() {
        Check {
            status: Status::Warn,
            name: "hotwords".into(),
            detail: "list is empty (proper-noun correction: parrots hotword <word>)".into(),
        }
    } else {
        Check {
            status: Status::Ok,
            name: "hotwords".into(),
            detail: format!("{} entries", hw.len()),
        }
    }
}

/// Run all checks and print the report; returns whether all core checks pass (missing models fail → non-zero exit code)
pub fn run() -> anyhow::Result<bool> {
    println!("parrots doctor health check report");
    let mut checks = Vec::new();

    let host = cpal::default_host();
    checks.push(default_device_check(
        "default input device",
        host.default_input_device(),
    ));
    checks.push(default_device_check(
        "default output device",
        host.default_output_device(),
    ));

    checks.push(virtual_device_check(
        wire::VIRTUAL_MIC,
        "direction A exit (meeting app microphone)",
    ));
    checks.push(virtual_device_check(
        wire::VIRTUAL_SPEAKERS,
        "direction B entry (meeting app speakers)",
    ));

    let root = wire::models_root();
    for rel in [
        "asr/sensevoice/model.int8.onnx",
        "whisper/ggml-small.bin",
        "vad/silero_vad.onnx",
        "mt/en-zh/encoder_model_quantized.onnx",
        "mt/zh-en/encoder_model_quantized.onnx",
        "tts/zipvoice/encoder.int8.onnx",
    ] {
        checks.push(model_check(&root, rel));
    }

    checks.push(voiceprints_check());
    checks.push(hotwords_check());

    let (mut ok, mut warn, mut fail) = (0, 0, 0);
    for c in &checks {
        match c.status {
            Status::Ok => ok += 1,
            Status::Warn => warn += 1,
            Status::Fail => fail += 1,
        }
        c.print();
    }
    println!("Summary: {ok} OK / {warn} WARN / {fail} FAIL");
    Ok(fail == 0)
}
