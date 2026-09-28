//! ASR A/B benchmark: whisper-small vs SenseVoice-small (#[ignore]; run once models are ready).
//!
//! Run: cargo test -p parrots-cli --test asr_ab --release -- --ignored --nocapture
//!
//! Transcribes the zh/en fixtures with both engines: prints text and timing, and computes
//! a rough character error rate (CER, character-level Levenshtein) against the fixture .txt
//! ground truth. Acceptance gate (plan 5): on the zh fixture, SenseVoice CER <= whisper-small
//! and latency <= 1/2.

use std::path::PathBuf;

use parrots_asr_sensevoice::SenseVoiceAsr;
use parrots_asr_whisper::WhisperAsr;
use parrots_core::{AsrEngine, AudioSegment, Lang};

fn models_root() -> PathBuf {
    // cargo test's cwd = package dir → repo root is two levels up; PARROTS_MODELS overrides
    std::env::var_os("PARROTS_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models"))
}

fn fixture(name: &str) -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../fixtures/{name}"))
}

fn load_fixture(name: &str) -> AudioSegment {
    let mut reader = hound::WavReader::open(fixture(&format!("{name}.wav"))).expect("fixture wav");
    assert_eq!(reader.spec().sample_rate, 16000);
    assert_eq!(reader.spec().channels, 1);
    let samples: Vec<f32> = match reader.spec().sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| f32::from(s.unwrap()) / 32768.0)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    };
    AudioSegment::new(samples, 16000)
}

fn truth(name: &str) -> String {
    std::fs::read_to_string(fixture(&format!("{name}.txt"))).expect("fixture ground truth txt")
}

/// Character-level CER = Levenshtein(hypothesis, reference) / reference char count (rough: ignores case/punctuation).
fn cer(hypo: &str, reference: &str) -> f64 {
    let norm = |s: &str| -> Vec<char> {
        s.to_lowercase()
            .chars()
            .filter(|c| !c.is_whitespace() && !".,!?;:、。!?;:,".contains(*c))
            .collect()
    };
    let (h, r) = (norm(hypo), norm(reference));
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rc) in r.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, hc) in h.iter().enumerate() {
            let cost = if rc == hc { 0 } else { 1 };
            cur.push((prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[h.len()] as f64 / r.len() as f64
}

async fn bench(name: &str, label: &str, asr: &dyn AsrEngine, lang: Lang) -> (String, f64) {
    let audio = load_fixture(name).with_lang(lang);
    // Two passes: the first includes residual graph optimization; time the second (warm) pass
    let _ = asr.transcribe(&audio).await.expect("warm-up transcribe");
    let t0 = std::time::Instant::now();
    let out = asr.transcribe(&audio).await.expect("transcribe");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("[{name}] {label}: {:.0}ms | {}", ms, out.text);
    (out.text, ms)
}

async fn compare(name: &str, lang: Lang, assert_gate: bool) {
    let whisper = WhisperAsr::load(
        &models_root().join("whisper/ggml-small.bin"),
        &[Lang::En, Lang::Zh],
        None,
    )
    .expect("whisper model not ready");
    let sensevoice =
        SenseVoiceAsr::load(&models_root().join("asr/sensevoice"), &[Lang::En, Lang::Zh])
            .expect("SenseVoice model not ready (run download-models.sh)");

    let (wh_text, wh_ms) = bench(name, "whisper-small", &whisper, lang).await;
    let (sv_text, sv_ms) = bench(name, "sensevoice    ", &sensevoice, lang).await;

    let reference = truth(name);
    let wh_cer = cer(&wh_text, &reference);
    let sv_cer = cer(&sv_text, &reference);
    println!(
        "[{name}] CER: whisper={wh_cer:.3} sensevoice={sv_cer:.3} | time: whisper={wh_ms:.0}ms sensevoice={sv_ms:.0}ms (ratio {:.2})",
        sv_ms / wh_ms
    );
    println!("[{name}] ground truth: {reference}");

    if assert_gate {
        assert!(
            sv_cer <= wh_cer,
            "gate: zh fixture SenseVoice CER ({sv_cer:.3}) must be <= whisper ({wh_cer:.3})"
        );
        assert!(
            sv_ms * 2.0 <= wh_ms,
            "gate: zh fixture SenseVoice latency ({sv_ms:.0}ms) must be <= half of whisper ({wh_ms:.0}ms)"
        );
    }
}

#[tokio::test]
#[ignore]
async fn ab_zh_meeting_sensevoice_vs_whisper() {
    compare("zh-meeting", Lang::Zh, true).await;
}

#[tokio::test]
#[ignore]
async fn ab_en_meeting_sensevoice_vs_whisper() {
    compare("en-meeting", Lang::En, false).await;
}
