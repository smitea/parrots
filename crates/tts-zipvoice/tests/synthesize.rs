//! 需要真实模型与声纹 fixture 的端到端合成测试(`cargo test -- --ignored`)。

use std::path::Path;
use std::time::Instant;

use parrots_core::{Synthesizer, VoiceProfile};
use parrots_tts_zipvoice::{write_wav, ZipvoiceTts};

fn repo_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn english_voice() -> VoiceProfile {
    VoiceProfile::from_prompt(
        repo_root().join("fixtures/en-hello.wav"),
        "Hello, how are you doing today? I hope everything is going well.".into(),
    )
}

#[tokio::test]
#[ignore]
async fn synthesizes_zh_with_english_prompt_voice() {
    let tts = ZipvoiceTts::load(&repo_root().join("models/tts/zipvoice")).unwrap();
    let voice = english_voice();

    let start = Instant::now();
    let seg = tts
        .synthesize("你好,今天会议开得怎么样?", &voice)
        .await
        .unwrap();
    let elapsed = start.elapsed();

    assert!(!seg.samples.is_empty());
    assert!(seg.rms() > 0.001, "RMS 过低: {}", seg.rms());

    let out = Path::new("/tmp/parrots-tts-out.wav");
    write_wav(out, &seg.samples, seg.sample_rate).unwrap();

    println!(
        "duration={}ms sample_rate={} samples={} rms={:.4} synth={elapsed:?} wav={out:?}",
        seg.duration_ms(),
        seg.sample_rate,
        seg.samples.len(),
        seg.rms()
    );
}

#[tokio::test]
#[ignore]
async fn same_voice_reuses_session() {
    let tts = ZipvoiceTts::load(&repo_root().join("models/tts/zipvoice")).unwrap();
    let voice = english_voice();
    let text = "好的,我马上把会议纪要发给大家。";

    let start = Instant::now();
    let first = tts.synthesize(text, &voice).await.unwrap();
    let first_elapsed = start.elapsed();

    let start = Instant::now();
    let second = tts.synthesize(text, &voice).await.unwrap();
    let second_elapsed = start.elapsed();

    assert!(!first.samples.is_empty() && !second.samples.is_empty());
    assert_eq!(first.sample_rate, second.sample_rate);
    assert!(
        second_elapsed <= first_elapsed,
        "第二次调用({second_elapsed:?})应不慢于首次({first_elapsed:?})"
    );
    println!(
        "first(含会话创建)={first_elapsed:?} duration={}ms, second(热会话)={second_elapsed:?} duration={}ms",
        first.duration_ms(),
        second.duration_ms()
    );
}
