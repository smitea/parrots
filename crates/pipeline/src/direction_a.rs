//! Direction A (I speak → they listen): fixed voiceprint = the user's
//! pre-enrolled profile.
pub use crate::direction_b::PipelineEvent;

use crate::clause::split_clauses;
use crate::timing::StageTimings;
use parrots_core::{
    AsrEngine, AudioSegment, Lang, Synthesizer, TextPolisher, Translator, VoiceProfile,
};
use std::sync::Arc;
use std::time::Instant;

pub struct DirectionAPipeline {
    pub asr: Arc<dyn AsrEngine>,
    pub translator: Arc<dyn Translator>,
    pub tts: Arc<dyn Synthesizer>,
    pub source_lang: Lang,
    pub target_lang: Lang,
    /// Hotword correction (Plan 6): after ASR, before MT; None = disabled
    pub hotwords: Option<Arc<crate::hotwords::Hotwords>>,
    /// Text polish layer (Plan 7): after hotword correction, before MT; None = disabled
    pub polisher: Option<Arc<dyn TextPolisher>>,
}

impl DirectionAPipeline {
    /// Processes one complete utterance (VAD segmentation is done by the
    /// caller), returning events in order
    pub async fn process_utterance(
        &self,
        audio: &AudioSegment,
        voice: &VoiceProfile,
        timings: &mut StageTimings,
    ) -> parrots_core::Result<Vec<PipelineEvent>> {
        let t0 = Instant::now();
        let transcript = self.asr.transcribe(audio).await?;
        timings.record("asr", t0.elapsed().as_secs_f64() * 1000.0);
        if transcript.text.trim().is_empty() {
            return Ok(vec![]);
        }
        let mut text = match &self.hotwords {
            Some(h) => h.correct(&transcript.text),
            None => transcript.text.clone(),
        };
        // Text polish layer (Plan 7): after hotword correction, before MT;
        // keep the original text on failure/empty output
        let hw_terms: Vec<String> = self
            .hotwords
            .as_ref()
            .map(|h| h.terms().to_vec())
            .unwrap_or_default();
        if let Some(p) = &self.polisher {
            if p.ready() {
                let t_polish = Instant::now();
                match p.polish(&text, &hw_terms).await {
                    Ok(fixed) if !fixed.trim().is_empty() => text = fixed,
                    _ => tracing::warn!("TextPolisher produced no output, keeping original text"),
                }
                timings.record("polish", t_polish.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let mut events = vec![PipelineEvent::Transcribed(text.clone())];

        let t1 = Instant::now();
        let translated = self.translator.translate(&text, &[]).await?;
        timings.record("mt", t1.elapsed().as_secs_f64() * 1000.0);
        events.push(PipelineEvent::Translated(translated.clone()));

        // First audio = the moment the first clause finishes synthesis
        // (design doc §2: the full translated sentence starts playing);
        // not recorded if the translation has no clauses
        let mut first_audio_recorded = false;
        for clause in split_clauses(&translated) {
            let t2 = Instant::now();
            let out = self.tts.synthesize(&clause, voice).await?;
            timings.record("tts", t2.elapsed().as_secs_f64() * 1000.0);
            if !first_audio_recorded {
                timings.record("first_audio", t0.elapsed().as_secs_f64() * 1000.0);
                first_audio_recorded = true;
            }
            events.push(PipelineEvent::SynthAudio(out));
        }
        timings.record("e2e", t0.elapsed().as_secs_f64() * 1000.0);
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parrots_core::{Result, Transcript};
    use std::sync::Mutex;

    struct MockAsr;
    #[async_trait::async_trait]
    impl AsrEngine for MockAsr {
        fn supported_langs(&self) -> &[Lang] {
            &[Lang::Zh]
        }
        async fn transcribe(&self, _a: &AudioSegment) -> Result<Transcript> {
            Ok(Transcript {
                text: "大家好,很高兴认识大家。今天过得怎么样?".into(),
                lang: Lang::Zh,
                duration_ms: 100,
            })
        }
    }

    struct MockMt;
    #[async_trait::async_trait]
    impl Translator for MockMt {
        fn pair(&self) -> (Lang, Lang) {
            (Lang::Zh, Lang::En)
        }
        async fn translate(&self, _text: &str, _ctx: &[String]) -> Result<String> {
            Ok("Hello everyone, nice to meet you. How was your day?".into())
        }
    }

    struct MockTts(Arc<Mutex<Vec<(String, String)>>>);
    #[async_trait::async_trait]
    impl Synthesizer for MockTts {
        fn supported_langs(&self) -> &[Lang] {
            &[Lang::En]
        }
        async fn synthesize(&self, text: &str, v: &VoiceProfile) -> Result<AudioSegment> {
            self.0
                .lock()
                .unwrap()
                .push((text.to_string(), v.prompt_text.clone()));
            Ok(AudioSegment::new(vec![0.5; 240], 24000))
        }
    }

    #[tokio::test]
    async fn utterance_flow_ordered_events() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let pipe = DirectionAPipeline {
            asr: Arc::new(MockAsr),
            translator: Arc::new(MockMt),
            tts: Arc::new(MockTts(log.clone())),
            source_lang: Lang::Zh,
            target_lang: Lang::En,
            hotwords: None,
            polisher: None,
        };
        let voice = VoiceProfile::from_prompt("/tmp/x.wav".into(), "我的声纹".into());
        let mut timings = StageTimings::default();
        let audio = AudioSegment::new(vec![0.1; 16000], 16000).with_lang(Lang::Zh);

        let events = pipe
            .process_utterance(&audio, &voice, &mut timings)
            .await
            .unwrap();

        let texts: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                PipelineEvent::Transcribed(t) | PipelineEvent::Translated(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            texts,
            vec![
                "大家好,很高兴认识大家。今天过得怎么样?",
                "Hello everyone, nice to meet you. How was your day?"
            ]
        );

        // ASCII ',' also triggers a split (Task 9 semantics): the translation
        // splits into exactly 3 segments on ','/'.'/'?', and the voice
        // parameter is passed through — each synthesis log entry carries the
        // fixed profile's prompt_text.
        assert_eq!(
            *log.lock().unwrap(),
            vec![
                ("Hello everyone,".into(), "我的声纹".into()),
                ("nice to meet you.".into(), "我的声纹".into()),
                ("How was your day?".into(), "我的声纹".into()),
            ]
        );
        let audio_events: Vec<&AudioSegment> = events
            .iter()
            .filter_map(|e| match e {
                PipelineEvent::SynthAudio(a) => Some(a),
                _ => None,
            })
            .collect();
        assert_eq!(audio_events.len(), 3);
        assert!(audio_events.iter().all(|a| a.sample_rate == 24000));

        assert!(!timings.e2e_ms.is_empty() && timings.e2e_ms[0] > 0.0);
        // First-audio metric: ready as soon as the first clause finishes
        // synthesis, always earlier than the full e2e
        assert!(
            !timings.first_audio_ms.is_empty() && timings.first_audio_ms[0] > 0.0,
            "first-audio latency should be recorded"
        );
        assert!(timings.first_audio_ms[0] <= timings.e2e_ms[0]);
    }

    #[tokio::test]
    async fn empty_transcript_short_circuits() {
        struct EmptyAsr;
        #[async_trait::async_trait]
        impl AsrEngine for EmptyAsr {
            fn supported_langs(&self) -> &[Lang] {
                &[Lang::Zh]
            }
            async fn transcribe(&self, _a: &AudioSegment) -> Result<Transcript> {
                Ok(Transcript {
                    text: "  ".into(),
                    lang: Lang::Zh,
                    duration_ms: 0,
                })
            }
        }
        struct PanickingTts;
        #[async_trait::async_trait]
        impl Synthesizer for PanickingTts {
            fn supported_langs(&self) -> &[Lang] {
                &[Lang::En]
            }
            async fn synthesize(&self, _: &str, _: &VoiceProfile) -> Result<AudioSegment> {
                panic!("synthesis should not be triggered");
            }
        }
        let pipe = DirectionAPipeline {
            asr: Arc::new(EmptyAsr),
            translator: Arc::new(MockMt),
            tts: Arc::new(PanickingTts),
            source_lang: Lang::Zh,
            target_lang: Lang::En,
            hotwords: None,
            polisher: None,
        };
        let voice = VoiceProfile::from_prompt("/tmp/x.wav".into(), "我的声纹".into());
        let mut timings = StageTimings::default();
        let events = pipe
            .process_utterance(
                &AudioSegment::new(vec![0.0; 1600], 16000),
                &voice,
                &mut timings,
            )
            .await
            .unwrap();
        assert!(events.is_empty());
    }
}
