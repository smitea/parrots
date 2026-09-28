use parrots_asr_whisper::WhisperAsr;
use parrots_core::{AsrEngine, AudioSegment, Lang};
use std::path::Path;

#[tokio::test]
#[ignore] // cargo test -p parrots-asr-whisper -- --ignored
async fn transcribes_fixture_hello() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/whisper/ggml-small.bin");
    let asr = WhisperAsr::load(&model, &[Lang::En, Lang::Zh], None).unwrap();
    let mut reader = hound::WavReader::open("../../fixtures/en-hello.wav").unwrap();
    assert_eq!(reader.spec().sample_rate, 16000);
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    let seg = AudioSegment::new(samples, 16000).with_lang(Lang::En);
    let t = asr.transcribe(&seg).await.unwrap();
    eprintln!("转写: {}", t.text);
    assert!(t.text.to_lowercase().contains("hello"), "实际: {}", t.text);
    assert_eq!(t.lang, Lang::En);
}
