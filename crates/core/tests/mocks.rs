use parrots_core::engine_traits::{AsrEngine, Synthesizer, Translator};
use parrots_core::{AudioSegment, Lang, Transcript, VoiceProfile};
use std::sync::Arc;

struct MockAsr;
#[async_trait::async_trait]
impl AsrEngine for MockAsr {
    fn supported_langs(&self) -> &[Lang] {
        &[Lang::En]
    }
    async fn transcribe(&self, audio: &AudioSegment) -> parrots_core::Result<Transcript> {
        assert!(!audio.samples.is_empty());
        Ok(Transcript {
            text: "mock text".into(),
            lang: audio.lang_hint.unwrap_or(Lang::En),
            duration_ms: audio.duration_ms(),
        })
    }
}

struct MockTts;
#[async_trait::async_trait]
impl Synthesizer for MockTts {
    fn supported_langs(&self) -> &[Lang] {
        &[Lang::Zh]
    }
    async fn synthesize(
        &self,
        _text: &str,
        _voice: &VoiceProfile,
    ) -> parrots_core::Result<AudioSegment> {
        Ok(AudioSegment::new(vec![0.0; 100], 24000))
    }
}

struct MockTranslator;
#[async_trait::async_trait]
impl Translator for MockTranslator {
    fn pair(&self) -> (Lang, Lang) {
        (Lang::En, Lang::Zh)
    }
    async fn translate(&self, text: &str, _ctx: &[String]) -> parrots_core::Result<String> {
        Ok(format!("译:{text}"))
    }
}

#[tokio::test]
async fn trait_objects_usable() {
    let asr: Arc<dyn AsrEngine> = Arc::new(MockAsr);
    let seg = AudioSegment::new(vec![0.1, 0.2], 16000).with_lang(Lang::En);
    let t: Transcript = asr.transcribe(&seg).await.unwrap();
    assert_eq!(t.text, "mock text");

    let tts: Arc<dyn Synthesizer> = Arc::new(MockTts);
    let vp = VoiceProfile::from_prompt("/tmp/p.wav".into(), "文本".into());
    let out = tts.synthesize("你好", &vp).await.unwrap();
    assert_eq!(out.sample_rate, 24000);
    assert_eq!(out.duration_ms(), 4);

    let mt: Arc<dyn Translator> = Arc::new(MockTranslator);
    assert_eq!(mt.pair(), (Lang::En, Lang::Zh));
    assert_eq!(mt.translate_clause("hello").await.unwrap(), "译:hello");
}

#[test]
fn rms_zero_for_silence() {
    let seg = AudioSegment::new(vec![0.0; 1600], 16000);
    assert_eq!(seg.rms(), 0.0);
}
