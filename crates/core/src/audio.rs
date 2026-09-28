/// Mono audio segment
#[derive(Debug, Clone)]
pub struct AudioSegment {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Language hint (None = unknown)
    pub lang_hint: Option<crate::Lang>,
}

impl AudioSegment {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            lang_hint: None,
        }
    }

    pub fn with_lang(mut self, lang: crate::Lang) -> Self {
        self.lang_hint = Some(lang);
        self
    }

    pub fn duration_ms(&self) -> u64 {
        (self.samples.len() as f64 / self.sample_rate as f64 * 1000.0).round() as u64
    }

    pub fn rms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.samples.iter().map(|s| s * s).sum();
        (sum / self.samples.len() as f32).sqrt()
    }
}

/// Final ASR transcript
#[derive(Debug, Clone, serde::Serialize)]
pub struct Transcript {
    pub text: String,
    pub lang: crate::Lang,
    pub duration_ms: u64,
}

/// Transcript update (UI partial transcript / final), reserved for streaming ASR and the subtitle overlay
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum TranscriptUpdate {
    Partial(String),
    Final(String),
}

/// Voiceprint profile: (reference audio path + its transcript) required for zero-shot cloning
#[derive(Debug, Clone)]
pub struct VoiceProfile {
    pub prompt_wav_path: std::path::PathBuf,
    pub prompt_text: String,
}

impl VoiceProfile {
    pub fn from_prompt(path: std::path::PathBuf, text: String) -> Self {
        Self {
            prompt_wav_path: path,
            prompt_text: text,
        }
    }
}
