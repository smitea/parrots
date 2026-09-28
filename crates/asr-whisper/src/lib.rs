use parrots_core::{AsrEngine, AudioSegment, Lang, Result, Transcript};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Minimum decodable duration for whisper (seconds); shorter input is padded with silence on both ends
const MIN_AUDIO_SAMPLES: usize = 16000;

/// Common hallucination texts whisper produces for short/silent segments (discarded when matched)
const HALLUCINATIONS: [&str; 8] = [
    "谢谢观看",
    "谢谢收看",
    "请订阅",
    "字幕",
    "字幕由",
    "明镜",
    "Thanks for watching",
    "Subscribe to",
];

pub struct WhisperAsr {
    state: Arc<Mutex<whisper_rs::WhisperState>>,
    langs: Vec<Lang>,
    /// Decoding bias prompt (plan 6, source layer): a concatenated hotword list that biases the decoder toward those words
    initial_prompt: Option<String>,
}

// whisper-rs upstream declares Send/Sync for Context; State is likewise a
// single-thread-use object, made thread-safe here via Mutex serialization
unsafe impl Send for WhisperAsr {}
unsafe impl Sync for WhisperAsr {}

impl WhisperAsr {
    pub fn load(model_path: &Path, langs: &[Lang], initial_prompt: Option<String>) -> Result<Self> {
        let ctx = whisper_rs::WhisperContext::new_with_params(
            model_path.to_str().ok_or_else(|| {
                parrots_core::Error::ModelMissing(model_path.display().to_string())
            })?,
            whisper_rs::WhisperContextParameters::default(),
        )
        .map_err(|e| parrots_core::Error::ModelMissing(format!("whisper load failed: {e}")))?;
        // The Metal backend initializes on create_state (hundreds of ms of kernel
        // loading); do it once and reuse across utterances instead of
        // re-initializing per utterance
        let state = ctx.create_state().map_err(|e| {
            parrots_core::Error::ModelMissing(format!("whisper state creation failed: {e}"))
        })?;
        let asr = Self {
            state: Arc::new(Mutex::new(state)),
            langs: langs.to_vec(),
            initial_prompt,
        };
        // Startup warmup: decode silence once, moving Metal's first-call compile
        // cost into the startup phase
        let warm = vec![0.0f32; 16000];
        let state_arc = asr.state.clone();
        let mut guard = state_arc
            .lock()
            .map_err(|_| parrots_core::Error::inference("whisper lock poisoned"))?;
        let _ = decode_state(&mut guard, &warm, "en", None);
        drop(guard);
        Ok(asr)
    }
}

#[async_trait::async_trait]
impl AsrEngine for WhisperAsr {
    fn supported_langs(&self) -> &[Lang] {
        &self.langs
    }

    async fn transcribe(&self, audio: &AudioSegment) -> Result<Transcript> {
        assert_eq!(
            audio.sample_rate, 16000,
            "WhisperAsr only accepts 16kHz audio"
        );
        let lang = audio.lang_hint.unwrap_or(Lang::En);
        let state = self.state.clone();
        let mut samples = audio.samples.clone();
        // whisper refuses to decode clips under 1s: pad silence on both ends to the
        // minimum duration, preserving "hello"-level short utterances; half lead /
        // half tail silence is less hallucination-prone than pure tail padding
        if samples.len() < MIN_AUDIO_SAMPLES {
            let lead = (MIN_AUDIO_SAMPLES - samples.len()) / 2;
            let mut padded = vec![0.0f32; lead];
            padded.extend_from_slice(&samples);
            padded.resize(MIN_AUDIO_SAMPLES, 0.0);
            samples = padded;
        }
        let duration_ms = audio.duration_ms();
        let prompt = self.initial_prompt.clone();
        let text = tokio::task::spawn_blocking(move || {
            decode_state(
                &mut *state
                    .lock()
                    .map_err(|_| parrots_core::Error::inference("whisper lock poisoned"))?,
                &samples,
                lang.code(),
                prompt.as_deref(),
            )
        })
        .await
        .map_err(|e| parrots_core::Error::inference(format!("join: {e}")))??;
        // Hallucination guard 1: speech under 600ms cannot possibly contain more than 8 characters
        let text = if duration_ms < 600 && text.chars().count() > 8 {
            String::new()
        } else {
            text
        };
        // Hallucination guard 2: known stock text
        let text = if HALLUCINATIONS.iter().any(|h| text.contains(h)) {
            String::new()
        } else {
            text
        };
        Ok(Transcript {
            text,
            lang,
            duration_ms,
        })
    }
}

fn decode_state(
    state: &mut whisper_rs::WhisperState,
    samples: &[f32],
    lang: &str,
    initial_prompt: Option<&str>,
) -> Result<String> {
    let mut params =
        whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    // Metal is the primary backend; CPU threads assist with scheduling
    params.set_n_threads(4);
    params.set_language(Some(lang));
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Prevent previous-utterance context from bleeding in when state is reused across utterances
    params.set_no_context(true);
    if let Some(prompt) = initial_prompt {
        params.set_initial_prompt(prompt);
    }
    state
        .full(params, samples)
        .map_err(|e| parrots_core::Error::inference(e.to_string()))?;
    let n = state
        .full_n_segments()
        .map_err(|e| parrots_core::Error::inference(e.to_string()))?;
    let mut text = String::new();
    for i in 0..n {
        let seg = state
            .full_get_segment_text(i)
            .map_err(|e| parrots_core::Error::inference(e.to_string()))?;
        text.push_str(&seg);
    }
    Ok(text.trim().to_string())
}
