//! Driver mode live end-to-end check (direction B: Parrots Speakers entry; requires the driver installed locally)
//!
//! Flow: a child process runs `parrots live --device "Parrots Speakers" --max-seconds 12 --out/--report`;
//! the parent waits until the child signals capture-ready (stderr marker), plays
//! `fixtures/en-meeting.wav` to Parrots Speakers via CpalSink (resampled to the device rate),
//! then waits for the child to exit on its own and validates the translated output.
//!
//! Run: `cargo test --release -p parrots-cli -- --ignored driver`

use parrots_core::{AudioPlatform, DeviceId};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

#[test]
#[ignore] // requires the Parrots virtual driver: cargo test --release -p parrots-cli -- --ignored driver
fn driver_live_parrots_speakers_produces_translation() {
    if parrots_platform_macos::find_device("Parrots Speakers").is_none() {
        eprintln!("skipping: Parrots Speakers not installed (cd driver/macos && ./make-pkg.sh, then double-click the ParrotsAudio pkg to install)");
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = std::env::temp_dir().join("parrots-driver-e2e.wav");
    let report = std::env::temp_dir().join("parrots-driver-e2e-report.json");
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&report);

    let mut child = Command::new(env!("CARGO_BIN_EXE_parrots"))
        .current_dir(&root)
        .args([
            "live",
            "--from",
            "en",
            "--to",
            "zh",
            "--device",
            "Parrots Speakers",
            "--max-seconds",
            "12",
            "--out",
            out.to_str().unwrap(),
            "--report",
            report.to_str().unwrap(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn parrots live subprocess");

    // Wait for the "capture ready" marker (printed before the live loop starts): tracing
    // writes to stdout by default, while whisper/ort C library logs go to stderr; both
    // streams must be drained continuously, or the child blocks on a full pipe
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let streams: [Option<Box<dyn std::io::Read + Send>>; 2] = [
        child.stdout.take().map(|s| Box::new(s) as _),
        child.stderr.take().map(|s| Box::new(s) as _),
    ];

    for stream in streams {
        let tx = tx.clone();
        std::thread::spawn(move || {
            if let Some(stream) = stream {
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            }
        });
    }
    drop(tx);
    let mut ready = false;
    let wait_start = Instant::now();
    while wait_start.elapsed() < Duration::from_secs(300) {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(line) => {
                // readiness markers for the incremental/whole-segment modes
                if line.contains("Incremental mode") || line.contains("Live mode") {
                    ready = true;
                    eprintln!(
                        "child capture ready ({:.1}s)",
                        wait_start.elapsed().as_secs_f32()
                    );
                    break;
                }
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break, // child exited early
        }
    }
    if !ready {
        let _ = child.kill();
        let _ = child.wait();
        panic!("child did not enter capture loop within 300s (startup failure?)");
    }

    // Play the fixture through the Parrots Speakers output (resampled to its native rate, raw samples written)
    let platform = parrots_platform_macos::MacAudioPlatform::new();
    let mut sink = platform
        .open_playback(&DeviceId(Some("Parrots Speakers".to_string())))
        .expect("failed to open Parrots Speakers playback");
    let mut reader = hound::WavReader::open(root.join("fixtures/en-meeting.wav"))
        .expect("failed to open fixture");
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| f32::from(s.unwrap()) / 32768.0)
            .collect(),
    };
    let mono: Vec<f32> = if channels > 1 {
        raw.chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        raw
    };
    let samples = parrots_platform_macos::resample::linear_resample(
        &mono,
        spec.sample_rate,
        sink.sample_rate(),
    );
    sink.write(&samples)
        .expect("failed to write fixture to Parrots Speakers");
    eprintln!(
        "played fixture: {:.2}s @ {}Hz (device rate {}Hz)",
        mono.len() as f64 / f64::from(spec.sample_rate),
        spec.sample_rate,
        sink.sample_rate()
    );

    // Wait for the child to exit on its own after max-seconds (fallback timeout guards against hangs)
    let deadline = Instant::now() + Duration::from_secs(120);
    let status = loop {
        if let Some(status) = child.try_wait().expect("try_wait failed") {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("child did not exit within 120s");
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    assert!(status.success(), "child exited abnormally: {status:?}");

    // Translated wav is non-empty and non-silent
    let mut reader = hound::WavReader::open(&out).expect("translated wav missing");
    let out_spec = reader.spec();
    let merged: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect();
    assert!(
        !merged.is_empty(),
        "translated wav is empty (no synthesized audio captured)"
    );
    let rms = (merged.iter().map(|s| s * s).sum::<f32>() / merged.len() as f32).sqrt();
    assert!(rms > 0.005, "translated wav looks silent rms={rms}");
    eprintln!(
        "translated wav: duration={:.2}s sample_rate={}Hz rms={rms:.4}",
        merged.len() as f64 / f64::from(out_spec.sample_rate),
        out_spec.sample_rate
    );

    // Latency report: live capture timing varies a lot, so relaxed to 8000ms (strict gate is in the fixture test)
    let rep: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    let first_audio = rep["first_audio_ms_mean"]
        .as_f64()
        .expect("first_audio_ms_mean should exist");
    let e2e = rep["e2e_ms_mean"]
        .as_f64()
        .expect("e2e_ms_mean should exist");
    eprintln!(
        "first audio mean: {first_audio:.0}ms | E2E mean: {e2e:.0}ms, utterances: {}",
        rep["utterances"]
    );
    assert!(
        first_audio <= 8000.0,
        "first audio mean {first_audio:.0}ms exceeds the relaxed 8000ms gate (live capture timing jitter)"
    );
    assert!(
        rep["utterances"].as_u64().unwrap() >= 1,
        "no complete utterance was translated"
    );
}
