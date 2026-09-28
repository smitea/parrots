//! ZipVoice zero-shot TTS: a thin wrapper over the sherpa-onnx C API.
//!
//! Uses `SherpaOnnxOfflineTtsGenerateWithConfig` with (prompt_wav, prompt_text)
//! as a rolling voiceprint for zero-shot voice cloning; sessions are lazily
//! created and reused, and prompt audio is cached by (path, prompt text).

mod config;
mod ffi;

use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use parrots_core::{AudioSegment, Error, Lang, Result, Synthesizer, VoiceProfile};

/// Flow-matching sampling steps: zipvoice-distill is a distilled model designed
/// for few-step sampling (the upstream non-distilled default is 5 steps; 4
/// balances audio quality against synthesis latency).
///
/// Overridable via the `PARROTS_TTS_NUM_STEPS` env var (2~5, for the
/// incremental speedup A/B, see tests/steps_latency_ab); invalid values fall
/// back to the default.
fn generation_steps() -> i32 {
    std::env::var("PARROTS_TTS_NUM_STEPS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|&n| (2..=5).contains(&n))
        .unwrap_or(4)
}

use config::ConfigBundle;

/// Required model files (relative to model_dir).
const REQUIRED_FILES: [&str; 5] = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "tokens.txt",
    "lexicon.txt",
    "vocos_24khz.onnx",
];

/// Required files inside espeak-ng-data (mirrors upstream Validate()).
const ESPEAK_REQUIRED_FILES: [&str; 4] = ["phontab", "phonindex", "phondata", "intonations"];

/// Owns a sherpa-onnx TTS session handle; destroyed on Drop.
#[derive(Debug)]
struct OwnedTts {
    ptr: *const ffi::SherpaOnnxOfflineTts,
}

// SAFETY: ptr points to an upstream opaque session object, used only via the C
// API; the upstream ONNX Runtime session's Run call is thread-safe, and every
// call in this crate is serialized through the session Mutex.
unsafe impl Send for OwnedTts {}
// SAFETY: same as above; shared access only triggers read-only C calls such as
// Generate/SampleRate.
unsafe impl Sync for OwnedTts {}

impl Drop for OwnedTts {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr was created by SherpaOnnxCreateOfflineTts and is not destroyed elsewhere.
            unsafe { ffi::SherpaOnnxDestroyOfflineTts(self.ptr) };
        }
    }
}

/// Session handle carried into `spawn_blocking` closures (raw pointers are not Send).
struct TtsHandle(*const ffi::SherpaOnnxOfflineTts);

// SAFETY: the handle is used only in C calls within this closure; calls are
// already serialized by the outer session Mutex.
unsafe impl Send for TtsHandle {}

/// Cached prompt audio (the zero-shot cloning reference).
#[derive(Debug)]
struct PromptAudio {
    samples: Vec<f32>,
    sample_rate: i32,
}

/// Prompt audio cache capacity: in long live sessions the rolling voiceprint
/// produces a new key (path, transcript) per clause; unbounded it would grow
/// linearly, so keep the 32 most recently used (zero-shot cloning references;
/// a session in practice switches between only a few prompt sources, so 32
/// is plenty).
const PROMPT_CACHE_CAPACITY: usize = 32;

/// Bounded least-recently-used (LRU) cache: small capacity, so a linear
/// timestamp scan on eviction is fine.
#[derive(Debug)]
struct PromptCache {
    map: HashMap<(PathBuf, String), (Arc<PromptAudio>, u64)>,
    clock: u64,
}

impl PromptCache {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            clock: 0,
        }
    }

    fn get(&mut self, key: &(PathBuf, String)) -> Option<Arc<PromptAudio>> {
        let stamp = self.clock + 1;
        self.map.get_mut(key).map(|(audio, s)| {
            *s = stamp;
            Arc::clone(audio)
        })
    }

    fn insert(&mut self, key: (PathBuf, String), audio: Arc<PromptAudio>) {
        self.clock += 1;
        let stamp = self.clock;
        if self.map.len() >= PROMPT_CACHE_CAPACITY && !self.map.contains_key(&key) {
            if let Some(oldest) = self
                .map
                .iter()
                .min_by_key(|(_, (_, s))| *s)
                .map(|(k, _)| k.clone())
            {
                self.map.remove(&oldest);
            }
        }
        self.map.insert(key, (audio, stamp));
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.map.len()
    }
}

/// ZipVoice voiceprint synthesizer.
#[derive(Debug)]
pub struct ZipvoiceTts {
    model_dir: PathBuf,
    /// Lazily created session; the lock is held across spawn_blocking awaits, serializing all generation calls.
    session: tokio::sync::Mutex<Option<OwnedTts>>,
    prompt_cache: Mutex<PromptCache>,
}

impl ZipvoiceTts {
    /// Verifies the model files are present ([`Error::ModelMissing`] otherwise).
    ///
    /// Does not create the ONNX session here — it is lazily loaded on the
    /// first `synthesize`, avoiding load cost when unused.
    pub fn load(model_dir: &Path) -> Result<Self> {
        for file in REQUIRED_FILES {
            if !model_dir.join(file).is_file() {
                return Err(Error::ModelMissing(format!(
                    "{}/{file}",
                    model_dir.display()
                )));
            }
        }
        for file in ESPEAK_REQUIRED_FILES {
            if !model_dir.join("espeak-ng-data").join(file).is_file() {
                return Err(Error::ModelMissing(format!(
                    "{}/espeak-ng-data/{file}",
                    model_dir.display()
                )));
            }
        }
        let model_dir = model_dir
            .canonicalize()
            .unwrap_or_else(|_| model_dir.to_path_buf());
        Ok(Self {
            model_dir,
            session: tokio::sync::Mutex::new(None),
            prompt_cache: Mutex::new(PromptCache::new()),
        })
    }

    /// Reads and caches prompt audio; the cache key is (wav path, prompt text),
    /// bounded LRU.
    async fn prompt_audio(&self, voice: &VoiceProfile) -> Result<Arc<PromptAudio>> {
        let key = (voice.prompt_wav_path.clone(), voice.prompt_text.clone());
        if let Some(hit) = self.prompt_cache.lock().unwrap().get(&key) {
            return Ok(hit);
        }
        let path = voice.prompt_wav_path.clone();
        let audio = Arc::new(
            tokio::task::spawn_blocking(move || read_prompt_wav(&path))
                .await
                .map_err(|e| Error::inference(format!("prompt audio read task failed: {e}")))??,
        );
        self.prompt_cache
            .lock()
            .unwrap()
            .insert(key, Arc::clone(&audio));
        Ok(audio)
    }

    /// Warmup: creates the ONNX session immediately (model load is slow) so the
    /// first `synthesize` no longer includes load cost. Behavior is unchanged
    /// if not called (the first synthesis still lazy-loads).
    ///
    /// Synchronous API; internally takes the lock on a separate thread so it
    /// can be called safely from an async runtime context (tokio Mutex's
    /// `blocking_lock` must not run on an executor thread).
    pub fn warmup(&self) -> Result<()> {
        let model_dir = self.model_dir.clone();
        let session = &self.session;
        std::thread::scope(|s| {
            s.spawn(move || {
                let mut guard = session.blocking_lock();
                if guard.is_none() {
                    *guard = Some(create_session(&model_dir)?);
                }
                Ok::<(), Error>(())
            })
            .join()
            .map_err(|_| Error::inference("TTS warmup thread failed (panic)"))?
        })
    }

    /// Creates the ONNX session on first call (model load is slow).
    async fn ensure_session(&self) -> Result<()> {
        let mut guard = self.session.lock().await;
        if guard.is_some() {
            return Ok(());
        }
        let model_dir = self.model_dir.clone();
        let owned = tokio::task::spawn_blocking(move || create_session(&model_dir))
            .await
            .map_err(|e| Error::inference(format!("session creation task failed: {e}")))??;
        *guard = Some(owned);
        Ok(())
    }
}

/// Creates the sherpa-onnx TTS session (the same path shared by lazy loading
/// and warmup).
fn create_session(model_dir: &Path) -> Result<OwnedTts> {
    let bundle = ConfigBundle::zipvoice(model_dir);
    // SAFETY: bundle owns every string referenced by the struct, alive for the duration of the call.
    let ptr = unsafe { ffi::SherpaOnnxCreateOfflineTts(bundle.as_ptr()) };
    if ptr.is_null() {
        return Err(Error::ModelMissing(format!(
            "sherpa-onnx failed to create the TTS session (model files missing or corrupt): {}",
            model_dir.display()
        )));
    }
    Ok(OwnedTts { ptr })
}

#[async_trait::async_trait]
impl Synthesizer for ZipvoiceTts {
    fn supported_langs(&self) -> &[Lang] {
        &[Lang::Zh, Lang::En]
    }

    async fn synthesize(&self, text: &str, voice: &VoiceProfile) -> Result<AudioSegment> {
        let prompt = self.prompt_audio(voice).await?;
        self.ensure_session().await?;

        let text_c = CString::new(text)
            .map_err(|_| Error::inference("synthesis text contains NUL bytes"))?;
        let ref_text = CString::new(voice.prompt_text.as_str())
            .map_err(|_| Error::inference("prompt text contains NUL bytes"))?;

        let guard = self.session.lock().await;
        let session = guard.as_ref().expect("ensure_session created the session");
        let handle = TtsHandle(session.ptr);

        let (samples, sample_rate) = tokio::task::spawn_blocking(move || {
            // SAFETY: the handle is valid; prompt/text_c/ref_text all outlive the
            // call, and the generated audio is copied and destroyed within this
            // closure, never escaping.
            unsafe { handle.generate(text_c, ref_text, prompt) }
        })
        .await
        .map_err(|e| Error::inference(format!("synthesis task failed: {e}")))??;

        Ok(AudioSegment::new(samples, sample_rate as u32))
    }
}

impl TtsHandle {
    /// Calls `SherpaOnnxOfflineTtsGenerateWithConfig` for one zero-shot
    /// synthesis, copying samples and freeing the C-side audio on the blocking
    /// thread.
    ///
    /// # Safety
    /// `self.0` must be a handle returned by `SherpaOnnxCreateOfflineTts` that
    /// has not been destroyed.
    unsafe fn generate(
        self,
        text: CString,
        reference_text: CString,
        prompt: Arc<PromptAudio>,
    ) -> Result<(Vec<f32>, i32)> {
        // silence_scale=0 falls back to the upstream default (0.2) via the C
        // API's SHERPA_ONNX_OR; sampling steps are explicit: zipvoice-distill
        // is a distilled flow-matching model designed for few-step sampling
        let gen = ffi::SherpaOnnxGenerationConfig {
            silence_scale: 0.0,
            speed: 1.0,
            sid: 0,
            reference_audio: prompt.samples.as_ptr(),
            reference_audio_len: prompt.samples.len() as i32,
            reference_sample_rate: prompt.sample_rate,
            reference_text: reference_text.as_ptr(),
            num_steps: generation_steps(),
            extra: std::ptr::null(),
        };
        // SAFETY: all pointer arguments are valid for the duration of the call; callback is None.
        let audio = unsafe {
            ffi::SherpaOnnxOfflineTtsGenerateWithConfig(
                self.0,
                text.as_ptr(),
                &gen,
                None,
                std::ptr::null_mut(),
            )
        };
        if audio.is_null() {
            return Err(Error::inference(
                "sherpa-onnx synthesis failed (empty result; check the prompt audio and text)",
            ));
        }
        // SAFETY: audio is non-null with fields filled by upstream; n/sample_rate are non-negative.
        let (n, sample_rate) = unsafe { ((*audio).n, (*audio).sample_rate) };
        let mut samples = vec![0.0f32; n.max(0) as usize];
        if !samples.is_empty() {
            // SAFETY: upstream guarantees samples points to n f32 values.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (*audio).samples,
                    samples.as_mut_ptr(),
                    samples.len(),
                );
            }
        }
        // SAFETY: audio was returned by upstream and destroyed exactly once.
        unsafe { ffi::SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio) };
        Ok((samples, sample_rate))
    }
}

/// Reads a wav via hound → mono f32 ([-1, 1]).
fn read_prompt_wav(path: &Path) -> Result<PromptAudio> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| Error::audio(format!("failed to open prompt audio {path:?}: {e}")))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(Error::audio(format!(
            "prompt audio {path:?} must be mono, got {} channels",
            spec.channels
        )));
    }
    let sample_rate = spec.sample_rate;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .into_samples::<i16>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::audio(format!("failed to read prompt audio {path:?}: {e}")))?
            .into_iter()
            .map(|s: i16| s as f32 / 32768.0)
            .collect(),
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::audio(format!("failed to read prompt audio {path:?}: {e}")))?,
    };
    Ok(PromptAudio {
        samples,
        sample_rate: sample_rate as i32,
    })
}

/// Writes mono f32 audio as a 16-bit PCM wav.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| Error::audio(format!("failed to create {path:?}: {e}")))?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer
            .write_sample(v)
            .map_err(|e| Error::audio(format!("failed to write {path:?}: {e}")))?;
    }
    writer
        .finalize()
        .map_err(|e| Error::audio(format!("failed to finalize {path:?}: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_wav_roundtrip() {
        let dir = std::env::temp_dir().join("parrots-tts-zipvoice-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.wav");
        let samples = vec![0.0f32, 0.5, -0.5, 1.0];
        write_wav(&path, &samples, 24000).unwrap();

        let mut reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().sample_rate, 24000);
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().bits_per_sample, 16);
        let back: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32767.0)
            .collect();
        assert_eq!(back.len(), samples.len());
        assert!((back[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn load_reports_missing_model() {
        let err = ZipvoiceTts::load(Path::new("/nonexistent-zipvoice")).unwrap_err();
        assert!(matches!(err, Error::ModelMissing(_)));
    }

    #[test]
    fn generation_steps_env_override() {
        std::env::set_var("PARROTS_TTS_NUM_STEPS", "2");
        assert_eq!(generation_steps(), 2);
        std::env::set_var("PARROTS_TTS_NUM_STEPS", "9");
        assert_eq!(
            generation_steps(),
            4,
            "invalid step count should fall back to the default"
        );
        std::env::set_var("PARROTS_TTS_NUM_STEPS", "abc");
        assert_eq!(generation_steps(), 4);
        std::env::remove_var("PARROTS_TTS_NUM_STEPS");
        assert_eq!(generation_steps(), 4);
    }

    /// Steps-vs-latency A/B benchmark (B1 speedup research): synthesis latency
    /// of the same batch of incremental clause texts under 2/3/4-step sampling;
    /// the resulting wavs are written to disk for manual quality listening.
    /// If 2~3 steps sound acceptable, the default step count can be lowered and
    /// the incremental benchmark gate tightened (4000ms → 1500~2000ms).
    ///
    /// Run: cargo test -p parrots-tts-zipvoice --release -- --ignored --nocapture steps_latency_ab
    /// Output: target/tts-steps-ab/steps{N}_{short,mid,long}.wav
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn steps_latency_ab() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let tts = ZipvoiceTts::load(&root.join("models/tts/zipvoice")).expect("TTS model ready");
        tts.warmup().expect("TTS warmup");
        let voice = VoiceProfile::from_prompt(
            root.join("fixtures/zh-hello.wav"),
            std::fs::read_to_string(root.join("fixtures/zh-hello.txt"))
                .expect("zh-hello.txt")
                .trim()
                .to_string(),
        );
        // Typical incremental clause lengths: short (~0.5s) / mid (~1.5s) /
        // long (~3s) at natural speaking rate
        let cases = [
            ("short", "大家好,今天很高兴"),
            ("mid", "首先我们来看整体的业务数据,用户的增长速度"),
            (
                "long",
                "包括实时翻译、离线模式和多人协作,最后我们再讨论一下接下来的产品路线图",
            ),
        ];
        let out_dir = root.join("target/tts-steps-ab");
        std::fs::create_dir_all(&out_dir).unwrap();
        for steps in [2, 3, 4] {
            std::env::set_var("PARROTS_TTS_NUM_STEPS", steps.to_string());
            let mut total = std::time::Duration::ZERO;
            for (label, text) in cases {
                let t = std::time::Instant::now();
                let audio = tts
                    .synthesize(text, &voice)
                    .await
                    .expect("synthesis succeeded");
                let dt = t.elapsed();
                total += dt;
                let path = out_dir.join(format!("steps{steps}_{label}.wav"));
                write_wav(&path, &audio.samples, audio.sample_rate).unwrap();
                println!(
                    "steps={steps} {label} ({} chars): {dt:.1?}",
                    text.chars().count()
                );
            }
            println!("steps={steps} total: {total:.1?}");
        }
        std::env::remove_var("PARROTS_TTS_NUM_STEPS");
    }

    /// Simulates a 500-clause long session: the rolling voiceprint changes the
    /// transcript per clause (distinct keys), the cache must stay within
    /// capacity, and the most recently used key survives while the oldest is
    /// evicted.
    #[test]
    fn prompt_cache_bounded_over_500_clauses() {
        let mut cache = PromptCache::new();
        let audio = Arc::new(PromptAudio {
            samples: vec![0.0; 8],
            sample_rate: 24000,
        });
        for i in 0..500 {
            cache.insert(
                (
                    PathBuf::from("/voice.wav"),
                    format!("clause transcript {i}"),
                ),
                Arc::clone(&audio),
            );
            assert!(
                cache.len() <= PROMPT_CACHE_CAPACITY,
                "exceeded capacity at clause {i}"
            );
        }
        assert_eq!(cache.len(), PROMPT_CACHE_CAPACITY);
        assert!(cache
            .get(&(PathBuf::from("/voice.wav"), "clause transcript 499".into()))
            .is_some());
        assert!(cache
            .get(&(PathBuf::from("/voice.wav"), "clause transcript 0".into()))
            .is_none());
    }

    /// A hit refreshes recency: the earliest-inserted but most recently hit key
    /// must not be evicted.
    #[test]
    fn prompt_cache_hit_refreshes_recency() {
        let mut cache = PromptCache::new();
        let audio = Arc::new(PromptAudio {
            samples: vec![0.0; 8],
            sample_rate: 24000,
        });
        for i in 0..PROMPT_CACHE_CAPACITY {
            cache.insert(
                (PathBuf::from("/v.wav"), format!("k{i}")),
                Arc::clone(&audio),
            );
        }
        let hot = (PathBuf::from("/v.wav"), "k0".to_string());
        assert!(cache.get(&hot).is_some());
        cache.insert(
            (PathBuf::from("/v.wav"), "new-key".into()),
            Arc::clone(&audio),
        );
        assert!(
            cache.get(&hot).is_some(),
            "recently hit key was wrongly evicted"
        );
        assert!(cache.get(&(PathBuf::from("/v.wav"), "k1".into())).is_none());
    }
}
