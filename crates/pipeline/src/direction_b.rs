use crate::clause::split_clauses;
use crate::timing::StageTimings;
use crate::voiceprint::RollingVoiceprint;
use parrots_core::{AsrEngine, AudioSegment, Lang, Synthesizer, TextPolisher, Translator};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone)]
pub enum PipelineEvent {
    Transcribed(String),
    Translated(String),
    SynthAudio(AudioSegment),
}

pub struct DirectionBPipeline {
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

impl DirectionBPipeline {
    /// Processes one complete utterance (VAD segmentation is done by the
    /// caller), returning events in order
    pub async fn process_utterance(
        &self,
        audio: &AudioSegment,
        voiceprint: &mut RollingVoiceprint,
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
        // keep the original text on failure/empty output.
        // The term list is passed in as context, for fixing homophone typos
        // of words outside the term list
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

        // The other party's voiceprint = (this segment's audio, this
        // segment's transcript); update before synthesizing
        let voice = voiceprint.update(audio, &text)?;

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
            let out = self.tts.synthesize(&clause, &voice).await?;
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
    use parrots_core::{Result, Transcript, VoiceProfile};
    use std::sync::Mutex;

    struct MockAsr;
    #[async_trait::async_trait]
    impl AsrEngine for MockAsr {
        fn supported_langs(&self) -> &[Lang] {
            &[Lang::En]
        }
        async fn transcribe(&self, _a: &AudioSegment) -> Result<Transcript> {
            Ok(Transcript {
                text: "Hello! How are you?".into(),
                lang: Lang::En,
                duration_ms: 100,
            })
        }
    }

    struct MockMt;
    #[async_trait::async_trait]
    impl Translator for MockMt {
        fn pair(&self) -> (Lang, Lang) {
            (Lang::En, Lang::Zh)
        }
        async fn translate(&self, text: &str, _ctx: &[String]) -> Result<String> {
            Ok(format!("你好!你好吗?({text})"))
        }
    }

    struct MockTts(Arc<Mutex<Vec<String>>>);
    #[async_trait::async_trait]
    impl Synthesizer for MockTts {
        fn supported_langs(&self) -> &[Lang] {
            &[Lang::Zh]
        }
        async fn synthesize(&self, text: &str, _v: &VoiceProfile) -> Result<AudioSegment> {
            self.0.lock().unwrap().push(text.to_string());
            Ok(AudioSegment::new(vec![0.5; 240], 24000))
        }
    }

    #[tokio::test]
    async fn utterance_flow_ordered_events() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let pipe = DirectionBPipeline {
            asr: Arc::new(MockAsr),
            translator: Arc::new(MockMt),
            tts: Arc::new(MockTts(log.clone())),
            source_lang: Lang::En,
            target_lang: Lang::Zh,
            hotwords: None,
            polisher: None,
        };
        let tmp = tempfile::tempdir().unwrap();
        let mut vp = RollingVoiceprint::new(tmp.path(), 2).unwrap();
        let mut timings = StageTimings::default();
        let audio = AudioSegment::new(vec![0.1; 16000], 16000).with_lang(Lang::En);

        let events = pipe
            .process_utterance(&audio, &mut vp, &mut timings)
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
            vec!["Hello! How are you?", "你好!你好吗?(Hello! How are you?)"]
        );

        // Note: the plan originally expected 2 clauses; after Task 9, ASCII
        // '!'/'?' also trigger splits, so 4 segments are produced:
        // "你好!" / "你好吗?" / "(Hello!" / "How are you?"; the trailing ')'
        // belongs to the attach set — attach-only segments are dropped and
        // never reach TTS.
        let audio_count = events
            .iter()
            .filter(|e| matches!(e, PipelineEvent::SynthAudio(_)))
            .count();
        assert_eq!(
            audio_count, 4,
            "4 clauses, one synth segment each (see comment above)"
        );
        assert_eq!(
            *log.lock().unwrap(),
            vec!["你好!", "你好吗?", "(Hello!", "How are you?"]
        );
        assert!(!timings.e2e_ms.is_empty() && timings.e2e_ms[0] > 0.0);
        // First-audio metric: ready as soon as the first clause finishes
        // synthesis, always earlier than the full e2e
        assert!(
            !timings.first_audio_ms.is_empty() && timings.first_audio_ms[0] > 0.0,
            "first-audio latency should be recorded"
        );
        assert!(timings.first_audio_ms[0] <= timings.e2e_ms[0]);
        // Voiceprint persisted to disk
        assert!(vp.dir().read_dir().unwrap().count() >= 1);
    }

    #[tokio::test]
    async fn empty_transcript_short_circuits() {
        struct EmptyAsr;
        #[async_trait::async_trait]
        impl AsrEngine for EmptyAsr {
            fn supported_langs(&self) -> &[Lang] {
                &[Lang::En]
            }
            async fn transcribe(&self, _a: &AudioSegment) -> Result<Transcript> {
                Ok(Transcript {
                    text: "  ".into(),
                    lang: Lang::En,
                    duration_ms: 0,
                })
            }
        }
        struct PanickingTts;
        #[async_trait::async_trait]
        impl Synthesizer for PanickingTts {
            fn supported_langs(&self) -> &[Lang] {
                &[Lang::Zh]
            }
            async fn synthesize(&self, _: &str, _: &VoiceProfile) -> Result<AudioSegment> {
                panic!("synthesis should not be triggered");
            }
        }
        let pipe = DirectionBPipeline {
            asr: Arc::new(EmptyAsr),
            translator: Arc::new(MockMt),
            tts: Arc::new(PanickingTts),
            source_lang: Lang::En,
            target_lang: Lang::Zh,
            hotwords: None,
            polisher: None,
        };
        let tmp = tempfile::tempdir().unwrap();
        let mut vp = RollingVoiceprint::new(tmp.path(), 2).unwrap();
        let mut timings = StageTimings::default();
        let events = pipe
            .process_utterance(
                &AudioSegment::new(vec![0.0; 1600], 16000),
                &mut vp,
                &mut timings,
            )
            .await
            .unwrap();
        assert!(events.is_empty());
    }

    /// Plan 6: ASR text goes through hotword correction before MT (the
    /// Transcribed event, the translator input, and the voiceprint transcript
    /// all carry the corrected text).
    #[tokio::test]
    async fn hotwords_corrected_before_mt() {
        struct HomophoneAsr;
        #[async_trait::async_trait]
        impl AsrEngine for HomophoneAsr {
            fn supported_langs(&self) -> &[Lang] {
                &[Lang::Zh]
            }
            async fn transcribe(&self, _a: &AudioSegment) -> Result<Transcript> {
                Ok(Transcript {
                    text: "我再看稀有記".into(),
                    lang: Lang::Zh,
                    duration_ms: 100,
                })
            }
        }
        struct CapturingMt(Arc<Mutex<Vec<String>>>);
        #[async_trait::async_trait]
        impl Translator for CapturingMt {
            fn pair(&self) -> (Lang, Lang) {
                (Lang::Zh, Lang::En)
            }
            async fn translate(&self, text: &str, _ctx: &[String]) -> Result<String> {
                self.0.lock().unwrap().push(text.to_string());
                Ok("Journey to the West".into())
            }
        }
        struct SilentTts;
        #[async_trait::async_trait]
        impl Synthesizer for SilentTts {
            fn supported_langs(&self) -> &[Lang] {
                &[Lang::En]
            }
            async fn synthesize(&self, _: &str, _: &VoiceProfile) -> Result<AudioSegment> {
                Ok(AudioSegment::new(vec![0.5; 240], 24000))
            }
        }
        let mt_inputs = Arc::new(Mutex::new(Vec::new()));
        let pipe = DirectionBPipeline {
            asr: Arc::new(HomophoneAsr),
            translator: Arc::new(CapturingMt(mt_inputs.clone())),
            tts: Arc::new(SilentTts),
            source_lang: Lang::Zh,
            target_lang: Lang::En,
            hotwords: Some(Arc::new(crate::hotwords::Hotwords::from_terms(["西游记"]))),
            polisher: None,
        };
        let tmp = tempfile::tempdir().unwrap();
        let mut vp = RollingVoiceprint::new(tmp.path(), 2).unwrap();
        let mut timings = StageTimings::default();
        let events = pipe
            .process_utterance(
                &AudioSegment::new(vec![0.1; 16000], 16000),
                &mut vp,
                &mut timings,
            )
            .await
            .unwrap();
        // The Transcribed event carries the corrected text
        assert_eq!(
            events
                .iter()
                .filter_map(|e| match e {
                    PipelineEvent::Transcribed(t) => Some(t.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec!["我再看西游记"]
        );
        // MT receives the corrected text
        assert_eq!(*mt_inputs.lock().unwrap(), vec!["我再看西游记"]);
    }
}
