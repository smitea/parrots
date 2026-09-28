//! 集成测试:真实模型转写 fixture(#[ignore],模型就绪后手动运行)。
//!
//! 运行:cargo test -p parrots-asr-sensevoice --release -- --ignored

use parrots_asr_sensevoice::SenseVoiceAsr;
use parrots_core::{AsrEngine, AudioSegment, Lang};

fn models_root() -> std::path::PathBuf {
    // cargo test 的 cwd = 包目录 → 仓库根为上两级;PARROTS_MODELS 可覆盖
    std::env::var_os("PARROTS_MODELS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models"))
}

fn load_fixture_wav(path: &std::path::Path) -> AudioSegment {
    let mut reader = hound::WavReader::open(path).expect("打开 fixture wav");
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

fn contains_any(text: &str, keys: &[&str]) -> bool {
    keys.iter().any(|k| text.contains(k))
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../fixtures/{name}"))
}

#[tokio::test]
#[ignore]
async fn zh_meeting_fixture_contains_keywords() {
    let asr = SenseVoiceAsr::load(&models_root().join("asr/sensevoice"), &[Lang::Zh])
        .expect("SenseVoice 模型就绪(download-models.sh)");
    let audio = load_fixture_wav(&fixture_path("zh-meeting.wav")).with_lang(Lang::Zh);
    let t = asr.transcribe(&audio).await.expect("转写成功");
    println!("zh-meeting 转写: {}", t.text);
    assert!(
        contains_any(&t.text, &["大家好", "会议", "路线图"]),
        "zh 转写应含关键词,实际: {}",
        t.text
    );
}

#[tokio::test]
#[ignore]
async fn en_meeting_fixture_contains_keywords() {
    let asr = SenseVoiceAsr::load(&models_root().join("asr/sensevoice"), &[Lang::En])
        .expect("SenseVoice 模型就绪(download-models.sh)");
    let audio = load_fixture_wav(&fixture_path("en-meeting.wav")).with_lang(Lang::En);
    let t = asr.transcribe(&audio).await.expect("转写成功");
    println!("en-meeting 转写: {}", t.text);
    assert!(
        contains_any(&t.text.to_lowercase(), &["thank", "roadmap"]),
        "en 转写应含关键词,实际: {}",
        t.text
    );
}
