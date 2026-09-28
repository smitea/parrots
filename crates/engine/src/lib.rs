use parrots_core::{AsrEngine, Lang, Synthesizer, Translator};
use std::collections::HashMap;
use std::sync::Arc;

/// Language pack: the minimal unit for adding a new language (zero changes to the pipeline/engine core)
pub struct LanguagePack {
    pub lang: Lang,
    pub asr: Arc<dyn AsrEngine>,
    pub translators: Vec<Arc<dyn Translator>>,
    pub tts: Arc<dyn Synthesizer>,
}

pub struct Engine {
    packs: HashMap<Lang, LanguagePack>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            packs: HashMap::new(),
        }
    }

    pub fn register(&mut self, pack: LanguagePack) {
        self.packs.insert(pack.lang, pack);
    }

    pub fn pack(&self, lang: Lang) -> Option<&LanguagePack> {
        self.packs.get(&lang)
    }

    pub fn translator(&self, from: Lang, to: Lang) -> Option<Arc<dyn Translator>> {
        self.packs
            .get(&from)?
            .translators
            .iter()
            .find(|t| t.pair() == (from, to))
            .cloned()
    }

    pub fn langs(&self) -> Vec<Lang> {
        let mut v: Vec<Lang> = self.packs.keys().copied().collect();
        v.sort_by_key(|l| l.code());
        v
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parrots_core::{AudioSegment, Result, Transcript, VoiceProfile};

    struct MockAsr;
    #[async_trait::async_trait]
    impl AsrEngine for MockAsr {
        fn supported_langs(&self) -> &[Lang] {
            &[Lang::En]
        }
        async fn transcribe(&self, _a: &AudioSegment) -> Result<Transcript> {
            Ok(Transcript {
                text: String::new(),
                lang: Lang::En,
                duration_ms: 0,
            })
        }
    }

    struct MockTts;
    #[async_trait::async_trait]
    impl Synthesizer for MockTts {
        fn supported_langs(&self) -> &[Lang] {
            &[Lang::En]
        }
        async fn synthesize(&self, _t: &str, _v: &VoiceProfile) -> Result<AudioSegment> {
            Ok(AudioSegment::new(vec![], 24000))
        }
    }

    struct MockMt(Lang, Lang);
    #[async_trait::async_trait]
    impl Translator for MockMt {
        fn pair(&self) -> (Lang, Lang) {
            (self.0, self.1)
        }
        async fn translate(&self, t: &str, _ctx: &[String]) -> Result<String> {
            Ok(t.to_string())
        }
    }

    fn en_pack() -> LanguagePack {
        LanguagePack {
            lang: Lang::En,
            asr: Arc::new(MockAsr),
            translators: vec![
                Arc::new(MockMt(Lang::En, Lang::Zh)),
                Arc::new(MockMt(Lang::En, Lang::En)),
            ],
            tts: Arc::new(MockTts),
        }
    }

    #[test]
    fn registry_lookup() {
        let mut e = Engine::new();
        e.register(en_pack());
        assert!(e.pack(Lang::En).is_some());
        assert!(e.pack(Lang::Zh).is_none());
        assert!(e.translator(Lang::En, Lang::Zh).is_some());
        assert!(e.translator(Lang::Zh, Lang::En).is_none());
        assert_eq!(e.langs(), vec![Lang::En]);
    }
}

pub mod service;
