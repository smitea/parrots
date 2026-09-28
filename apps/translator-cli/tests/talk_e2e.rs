use assert_cmd::Command;

#[test]
#[ignore] // cargo test -p parrots-cli -- --ignored (requires models + fixture)
fn zh_meeting_talk_first_audio_under_2500ms() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    // Cross-lingual cloning check: Chinese content × English prompt voice (en-hello.wav + English text)
    let profiles =
        std::env::temp_dir().join(format!("parrots-talk-profiles-{}", std::process::id()));
    std::fs::create_dir_all(&profiles).unwrap();
    std::fs::copy(root.join("fixtures/en-hello.wav"), profiles.join("my.wav")).unwrap();
    std::fs::write(
        profiles.join("my.txt"),
        "Hello, how are you doing today? I hope everything is going well.",
    )
    .unwrap();

    let out = std::env::temp_dir().join("parrots-talk-translated.wav");
    let report = std::env::temp_dir().join("parrots-talk-report.json");
    Command::cargo_bin("parrots")
        .unwrap()
        .env("PARROTS_PROFILES", &profiles)
        .current_dir(&root)
        .args([
            "talk",
            "--input",
            "fixtures/zh-meeting.wav",
            "--from",
            "zh",
            "--to",
            "en",
            "--voice",
            "my",
            // latency gate measures the raw pipeline; the polish-mode gate lives in tests/polish_ab.rs (plan 7)
            "--polish",
            "off",
            "--out",
            out.to_str().unwrap(),
            "--report",
            report.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Clean up the temp voice-profile dir (best effort)
    let _ = std::fs::remove_dir_all(&profiles);

    // Output wav is non-empty and non-silent
    let mut reader = hound::WavReader::open(&out).unwrap();
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect();
    assert!(!samples.is_empty());
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    assert!(rms > 0.005, "translated wav looks silent rms={rms}");
    eprintln!(
        "translated wav: duration={:.2}s sample_rate={}Hz rms={rms:.4}",
        samples.len() as f64 / f64::from(spec.sample_rate),
        spec.sample_rate
    );

    // Gate aligned with design doc §2 (translated sentence starts playing <= 2500ms, symmetric with the translate gate).
    // The initial 1.5s target was a stretch goal pending streaming TTS; measured on non-streaming v1:
    // ~2025ms under load, ~1820ms extrapolated idle.
    // e2e_ms_mean (all clauses synthesized) is stricter than the spec — printed as observation only, no gate.
    let rep: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    let e2e = rep["e2e_ms_mean"]
        .as_f64()
        .expect("e2e_ms_mean should exist");
    let first_audio = rep["first_audio_ms_mean"]
        .as_f64()
        .expect("first_audio_ms_mean should exist");
    eprintln!(
        "first audio mean: {first_audio:.0}ms | E2E mean: {e2e:.0}ms, utterances: {}",
        rep["utterances"]
    );
    assert!(
        first_audio <= 2500.0,
        "first audio mean {first_audio:.0}ms exceeds the 2500ms acceptance gate (translated sentence starts playing)"
    );
    assert!(rep["utterances"].as_u64().unwrap() >= 1);
}
