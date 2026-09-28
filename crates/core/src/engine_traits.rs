use crate::{AudioSegment, Lang, Result, Transcript, TranscriptUpdate, VoiceProfile};
use tokio::sync::broadcast;

#[async_trait::async_trait]
pub trait AsrEngine: Send + Sync {
    fn supported_langs(&self) -> &[Lang];
    /// Utterance-level final transcript
    async fn transcribe(&self, audio: &AudioSegment) -> Result<Transcript>;
    /// Rolling-window partial transcript (for UI), not provided by default
    fn partial_stream(
        &self,
        _stream: AudioSegment,
    ) -> Option<broadcast::Receiver<TranscriptUpdate>> {
        None
    }
}

#[async_trait::async_trait]
pub trait Translator: Send + Sync {
    fn pair(&self) -> (Lang, Lang);
    /// Whole-sentence translation with conversation context
    async fn translate(&self, text: &str, ctx: &[String]) -> Result<String>;
    /// Incremental clause translation (enables clause-level pipelined parallelism)
    async fn translate_clause(&self, clause: &str) -> Result<String> {
        self.translate(clause, &[]).await
    }
}

#[async_trait::async_trait]
pub trait Synthesizer: Send + Sync {
    fn supported_langs(&self) -> &[Lang];
    /// Synthesize speech with the given voiceprint
    async fn synthesize(&self, text: &str, voice: &VoiceProfile) -> Result<AudioSegment>;
}

/// Text polish layer (plan 7): sits between ASR and MT, polishing transcripts with context
/// (homophone typos / filler words / segmentation smoothing).
#[async_trait::async_trait]
pub trait TextPolisher: Send + Sync {
    /// Polish transcript text; hotwords provide context; timeout is up to the implementation (return the original text on timeout)
    async fn polish(&self, text: &str, hotwords: &[String]) -> Result<String>;
    /// Whether the model is ready (false if loading failed; the pipeline skips polishing)
    fn ready(&self) -> bool {
        true
    }
}
