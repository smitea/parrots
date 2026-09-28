//! SenseVoice-small ASR: a thin wrapper over the sherpa-onnx C API.
//!
//! Each utterance gets its own `SherpaOnnxOfflineStream` (one AcceptWaveform →
//! Decode → GetResult); the recognizer is reused across utterances via a
//! Mutex to avoid reloading the model.

mod config;
mod ffi;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use config::ConfigBundle;
use parrots_core::{AsrEngine, AudioSegment, Error, Lang, Result, Transcript};

/// Same as asr-whisper: minimum decodable duration (seconds); shorter input is padded with silence on both ends
const MIN_AUDIO_SAMPLES: usize = 16000;

/// Same as asr-whisper: common hallucination texts for short/silent segments (discarded when matched)
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

/// Required model files (relative to model_dir).
const REQUIRED_FILES: [&str; 2] = ["model.int8.onnx", "tokens.txt"];

/// Infer the SenseVoice language hint from the target language set: a single
/// language is passed explicitly; mixed/auto uses auto.
fn sensevoice_language(langs: &[Lang]) -> &'static str {
    match langs {
        [Lang::Zh] => "zh",
        [Lang::En] => "en",
        _ => config::SENSEVOICE_LANG_AUTO,
    }
}

/// Owns a sherpa-onnx recognizer handle; destroyed on Drop.
#[derive(Debug)]
struct OwnedRecognizer {
    ptr: *const ffi::SherpaOnnxOfflineRecognizer,
}

// SAFETY: ptr points to an upstream opaque session object, used only via the C
// API; the upstream ONNX Runtime session's Run call is thread-safe, and all
// Decode calls in this crate are serialized through a Mutex.
unsafe impl Send for OwnedRecognizer {}
unsafe impl Sync for OwnedRecognizer {}

impl Drop for OwnedRecognizer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr was created by SherpaOnnxCreateOfflineRecognizer and is not destroyed elsewhere.
            unsafe { ffi::SherpaOnnxDestroyOfflineRecognizer(self.ptr) };
        }
    }
}

/// SenseVoice-small recognition engine.
#[derive(Debug)]
pub struct SenseVoiceAsr {
    /// Recognizer reused across utterances; the Mutex serializes Decode (upstream gives no concurrent-decode guarantee)
    recognizer: Arc<Mutex<OwnedRecognizer>>,
    langs: Vec<Lang>,
}

impl SenseVoiceAsr {
    /// Verifies the model files are present and creates the recognizer (model
    /// load is slow; done once at startup).
    ///
    /// `langs` determines the language hint: a single zh/en is passed
    /// explicitly, otherwise auto (the model auto-detects).
    pub fn load(model_dir: &Path, langs: &[Lang]) -> Result<Self> {
        for file in REQUIRED_FILES {
            if !model_dir.join(file).is_file() {
                return Err(Error::ModelMissing(format!(
                    "{}/{file}",
                    model_dir.display()
                )));
            }
        }
        let bundle = ConfigBundle::sensevoice(model_dir, sensevoice_language(langs));
        // SAFETY: bundle owns every string referenced by the struct, alive for the duration of the call.
        let ptr = unsafe { ffi::SherpaOnnxCreateOfflineRecognizer(bundle.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::ModelMissing(format!(
                "sherpa-onnx failed to create the SenseVoice recognizer (model files missing or corrupt): {}",
                model_dir.display()
            )));
        }
        let asr = Self {
            recognizer: Arc::new(Mutex::new(OwnedRecognizer { ptr })),
            langs: langs.to_vec(),
        };
        // Startup warmup: decode silence once, moving the first-decode graph
        // optimization cost into the loading phase
        let warm = vec![0.0f32; MIN_AUDIO_SAMPLES];
        let guard = asr.recognizer.lock().unwrap_or_else(|e| e.into_inner());
        // Warmup failure is not fatal: the first real decode still completes
        // graph optimization, just slightly slower
        let _ = decode(&guard, &warm);
        drop(guard);
        Ok(asr)
    }
}

#[async_trait::async_trait]
impl AsrEngine for SenseVoiceAsr {
    fn supported_langs(&self) -> &[Lang] {
        &self.langs
    }

    async fn transcribe(&self, audio: &AudioSegment) -> Result<Transcript> {
        assert_eq!(
            audio.sample_rate, 16000,
            "SenseVoiceAsr only accepts 16kHz audio"
        );
        let lang = audio.lang_hint.unwrap_or(Lang::En);
        let mut samples = audio.samples.clone();
        // Pad clips under 1s with silence to the minimum duration, preserving
        // "hello"-level short utterances; half lead / half tail silence is less
        // hallucination-prone than pure tail padding (same as asr-whisper)
        if samples.len() < MIN_AUDIO_SAMPLES {
            let lead = (MIN_AUDIO_SAMPLES - samples.len()) / 2;
            let mut padded = vec![0.0f32; lead];
            padded.extend_from_slice(&samples);
            padded.resize(MIN_AUDIO_SAMPLES, 0.0);
            samples = padded;
        }
        let duration_ms = audio.duration_ms();
        let recognizer = self.recognizer.clone();
        let text = tokio::task::spawn_blocking(move || {
            let guard = recognizer.lock().unwrap_or_else(|e| e.into_inner());
            decode(&guard, &samples)
        })
        .await
        .map_err(|e| Error::inference(format!("join: {e}")))??;
        // Hallucination guard 1: speech under 600ms cannot possibly contain more
        // than 8 characters (same as asr-whisper)
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

/// Fully decode one audio clip: create stream → feed audio → decode → get text → free.
fn decode(recognizer: &OwnedRecognizer, samples: &[f32]) -> Result<String> {
    // SAFETY: the recognizer handle is valid; the stream lives only within this function.
    let stream = unsafe { ffi::SherpaOnnxCreateOfflineStream(recognizer.ptr) };
    if stream.is_null() {
        return Err(Error::inference(
            "sherpa-onnx failed to create the decode stream",
        ));
    }
    let result = decode_stream(recognizer, stream, samples);
    // SAFETY: the stream was returned by CreateOfflineStream and destroyed exactly once.
    unsafe { ffi::SherpaOnnxDestroyOfflineStream(stream) };
    result
}

/// Decodes an already-created stream and returns text; the caller destroys the
/// stream whether decoding succeeds or fails.
fn decode_stream(
    recognizer: &OwnedRecognizer,
    stream: *const ffi::SherpaOnnxOfflineStream,
    samples: &[f32],
) -> Result<String> {
    // SAFETY: the stream is valid; samples outlives the call.
    unsafe {
        ffi::SherpaOnnxAcceptWaveformOffline(stream, 16000, samples.as_ptr(), samples.len() as i32);
        ffi::SherpaOnnxDecodeOfflineStream(recognizer.ptr, stream);
    }
    // SAFETY: the stream is valid; the result is allocated upstream, read and
    // destroyed within this function.
    let raw = unsafe { ffi::SherpaOnnxGetOfflineStreamResult(stream) };
    if raw.is_null() {
        return Err(Error::inference("sherpa-onnx decode result is null"));
    }
    let text = unsafe {
        let r = &*raw;
        if r.text.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(r.text)
                .to_string_lossy()
                .into_owned()
        }
    };
    // SAFETY: raw was returned by GetOfflineStreamResult and destroyed exactly once.
    unsafe { ffi::SherpaOnnxDestroyOfflineRecognizerResult(raw) };
    Ok(text.trim().to_string())
}

/// Conventional model directory (models/asr/sensevoice).
pub fn default_model_dir() -> PathBuf {
    PathBuf::from("models/asr/sensevoice")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_hint_follows_langs() {
        assert_eq!(sensevoice_language(&[Lang::Zh]), "zh");
        assert_eq!(sensevoice_language(&[Lang::En]), "en");
        assert_eq!(sensevoice_language(&[Lang::En, Lang::Zh]), "auto");
        assert_eq!(sensevoice_language(&[]), "auto");
    }

    #[test]
    fn load_reports_missing_model() {
        let err =
            SenseVoiceAsr::load(Path::new("/nonexistent-sensevoice"), &[Lang::Zh]).unwrap_err();
        assert!(matches!(err, Error::ModelMissing(_)));
    }
}
