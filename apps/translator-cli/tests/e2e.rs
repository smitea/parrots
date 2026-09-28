use assert_cmd::Command;

#[test]
#[ignore] // cargo test -p parrots-cli -- --ignored (requires models + fixture)
          // Acceptance gate semantics = design doc §2 M1 goal "translated full sentence starts playing within 2.5s":
          // the gate sits on first_audio_ms_mean (first clause synthesized); e2e_ms_mean (all clauses
          // synthesized) is stricter than the spec — printed as observation only, no gate.
          // Note: --polish off explicitly — this gate measures raw pipeline latency; the polish-mode gate
          // lives in tests/polish_ab.rs (polish-layer budget accounted separately, plan 7).
fn fixture_meeting_first_audio_under_2500ms() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let input = root.join("fixtures/en-meeting.wav");
    let out = std::env::temp_dir().join("parrots-e2e-translated.wav");
    let report = std::env::temp_dir().join("parrots-e2e-report.json");
    Command::cargo_bin("parrots")
        .unwrap()
        .current_dir(&root)
        .args([
            "translate",
            "--input",
            input.to_str().unwrap(),
            "--from",
            "en",
            "--to",
            "zh",
            "--polish",
            "off",
            "--out",
            out.to_str().unwrap(),
            "--report",
            report.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Output wav is non-empty and non-silent
    let mut reader = hound::WavReader::open(&out).unwrap();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect();
    assert!(!samples.is_empty());
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    assert!(rms > 0.005, "translated wav looks silent rms={rms}");

    // Latency acceptance gate: first audio <=2500ms (design doc §2); e2e observed only
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
