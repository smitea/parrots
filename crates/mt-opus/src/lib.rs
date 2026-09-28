pub mod generate;

use generate::GreedyDecoder;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use parrots_core::{Error, Lang, Result, Translator};
use std::path::Path;
use std::sync::Arc;

pub struct MarianTranslator {
    inner: Arc<MarianInner>,
    pair: (Lang, Lang),
}

struct MarianInner {
    tokenizer: tokenizers::Tokenizer,
    /// encoder_model_quantized.onnx, read fully into memory
    encoder_bytes: Vec<u8>,
    /// decoder_model_quantized.onnx
    decoder_bytes: Vec<u8>,
    /// Cached ort sessions; built on the first translation, reused afterwards
    /// (rebuilding sessions from memory takes hundreds of ms).
    /// Same serialization trade-off as TTS: one lock serializes all
    /// translations, mutual exclusion in exchange for no rebuilds.
    sessions: std::sync::Mutex<Option<(Session, Session)>>,
}

impl MarianInner {
    fn build_sessions(&self) -> Result<(Session, Session)> {
        Ok((
            session_from_memory(&self.encoder_bytes)?,
            session_from_memory(&self.decoder_bytes)?,
        ))
    }
}

impl MarianTranslator {
    pub fn load(dir: &Path, from: Lang, to: Lang) -> Result<Self> {
        let tokenizer = load_tokenizer(&dir.join("tokenizer.json"))?;
        let encoder_bytes = std::fs::read(dir.join("encoder_model_quantized.onnx"))?;
        let decoder_bytes = std::fs::read(dir.join("decoder_model_quantized.onnx"))?;
        Ok(Self {
            inner: Arc::new(MarianInner {
                tokenizer,
                encoder_bytes,
                decoder_bytes,
                sessions: std::sync::Mutex::new(None),
            }),
            pair: (from, to),
        })
    }

    /// Startup warmup: builds the ort sessions ahead of time, removing the
    /// hidden session-creation cost from the first translation
    /// (same pattern as TTS warmup; slower startup for a faster first utterance).
    pub fn warmup(&self) -> Result<()> {
        let sessions = self.inner.build_sessions()?;
        let mut guard = self
            .inner
            .sessions
            .lock()
            .map_err(|e| Error::inference(format!("mt session lock poisoned: {e}")))?;
        *guard = Some(sessions);
        Ok(())
    }
}

/// tokenizers 0.21 panics outright on a Precompiled normalizer with
/// `precompiled_charsmap: null` (common in Xenova exports; semantically
/// equivalent to no normalizer), so strip it before loading.
fn load_tokenizer(path: &Path) -> Result<tokenizers::Tokenizer> {
    let raw = std::fs::read_to_string(path)?;
    let mut json: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| Error::inference(format!("{} failed to parse: {e}", path.display())))?;
    if json["normalizer"]["type"] == "Precompiled"
        && json["normalizer"]["precompiled_charsmap"].is_null()
    {
        json["normalizer"] = serde_json::Value::Null;
    }
    tokenizers::Tokenizer::from_bytes(json.to_string())
        .map_err(|e| Error::ModelMissing(format!("{}: {e}", path.display())))
}

#[async_trait::async_trait]
impl Translator for MarianTranslator {
    fn pair(&self) -> (Lang, Lang) {
        self.pair
    }

    async fn translate(&self, text: &str, _ctx: &[String]) -> Result<String> {
        let inner = Arc::clone(&self.inner);
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            // Hold the lock for the whole translation; sessions are built only
            // on the first call, then reused (avoids rebuilding both ort
            // sessions from memory on every translation)
            let mut guard = inner
                .sessions
                .lock()
                .map_err(|e| Error::inference(format!("mt session lock poisoned: {e}")))?;
            if guard.is_none() {
                *guard = Some(inner.build_sessions()?);
            }
            let (encoder, decoder) = guard.as_mut().expect("sessions guaranteed to exist above");
            let out = GreedyDecoder::new(encoder, decoder).translate(&inner.tokenizer, &text)?;
            tracing::debug!(chars = out.chars().count(), "mt translation finished");
            Ok(out)
        })
        .await
        .map_err(|e| Error::inference(format!("join: {e}")))?
    }
}

fn session_from_memory(bytes: &[u8]) -> Result<Session> {
    Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .with_intra_threads(4)
        .map_err(ort_err)?
        .commit_from_memory(bytes)
        .map_err(ort_err)
}

fn ort_err(e: impl std::fmt::Display) -> Error {
    Error::inference(e.to_string())
}
