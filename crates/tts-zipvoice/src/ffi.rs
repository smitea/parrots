//! `#[repr(C)]` mirror declarations of the sherpa-onnx C API (v1.13.8),
//! TTS part.
//!
//! Every struct's field names and order match upstream
//! `sherpa-onnx/c-api/c-api.h` (v1.13.8, in-commit file copy at `/tmp/c-api.h`)
//! one to one; the C side reads the zipvoice config by offset, so field order
//! must not change.
//!
//! Semantic notes (source `sherpa-onnx/c-api/c-api.cc`):
//! - A numeric field of 0 is replaced by the upstream default via
//!   `SHERPA_ONNX_OR(x, y)` (`silence_scale=0 → 0.2`, `speed=0 → 1.0`,
//!   `num_steps=0 → 5`);
//! - The 4 float parameters of the zipvoice model config must be strictly
//!   positive per `offline-tts-zipvoice-model-config.cc::Validate()`
//!   (`feat_scale/target_rms/guidance_scale > 0`, `t_shift >= 0`), so 0 does
//!   not fall back to the default; the upstream documented defaults must be
//!   filled in explicitly.

use std::ffi::c_char;
use std::ffi::c_void;

/** @brief Configuration for a VITS TTS model. (c-api.h L2257) */
#[repr(C)]
pub struct SherpaOnnxOfflineTtsVitsModelConfig {
    pub model: *const c_char,
    pub lexicon: *const c_char,
    pub tokens: *const c_char,
    pub data_dir: *const c_char,
    pub noise_scale: f32,
    pub noise_scale_w: f32,
    pub length_scale: f32,
    pub dict_dir: *const c_char,
}

/** @brief Configuration for a Matcha TTS model. (c-api.h L2277) */
#[repr(C)]
pub struct SherpaOnnxOfflineTtsMatchaModelConfig {
    pub acoustic_model: *const c_char,
    pub vocoder: *const c_char,
    pub lexicon: *const c_char,
    pub tokens: *const c_char,
    pub data_dir: *const c_char,
    pub noise_scale: f32,
    pub length_scale: f32,
    pub dict_dir: *const c_char,
}

/** @brief Configuration for a Kokoro TTS model. (c-api.h L2297) */
#[repr(C)]
pub struct SherpaOnnxOfflineTtsKokoroModelConfig {
    pub model: *const c_char,
    pub voices: *const c_char,
    pub tokens: *const c_char,
    pub data_dir: *const c_char,
    pub length_scale: f32,
    pub dict_dir: *const c_char,
    pub lexicon: *const c_char,
    pub lang: *const c_char,
}

/** @brief Configuration for a Kitten TTS model. (c-api.h L2317) */
#[repr(C)]
pub struct SherpaOnnxOfflineTtsKittenModelConfig {
    pub model: *const c_char,
    pub voices: *const c_char,
    pub tokens: *const c_char,
    pub data_dir: *const c_char,
    pub length_scale: f32,
}

/** @brief Configuration for a ZipVoice TTS model. (c-api.h L2331) */
#[repr(C)]
pub struct SherpaOnnxOfflineTtsZipvoiceModelConfig {
    pub tokens: *const c_char,
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub vocoder: *const c_char,
    pub data_dir: *const c_char,
    pub lexicon: *const c_char,
    pub feat_scale: f32,
    pub t_shift: f32,
    pub target_rms: f32,
    pub guidance_scale: f32,
}

/** @brief Configuration for a Pocket TTS model. (c-api.h L2355) */
#[repr(C)]
pub struct SherpaOnnxOfflineTtsPocketModelConfig {
    pub lm_flow: *const c_char,
    pub lm_main: *const c_char,
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub text_conditioner: *const c_char,
    pub vocab_json: *const c_char,
    pub token_scores_json: *const c_char,
    pub voice_embedding_cache_capacity: i32,
}

/** @brief Configuration for a Supertonic TTS model. (c-api.h L2375) */
#[repr(C)]
pub struct SherpaOnnxOfflineTtsSupertonicModelConfig {
    pub duration_predictor: *const c_char,
    pub text_encoder: *const c_char,
    pub vector_estimator: *const c_char,
    pub vocoder: *const c_char,
    pub tts_json: *const c_char,
    pub unicode_indexer: *const c_char,
    pub voice_style: *const c_char,
}

/**
 * @brief Configuration shared by offline TTS models. (c-api.h L2409)
 *
 * Note: num_threads/debug/provider sit between vits and matcha — that is the
 * upstream layout.
 */
#[repr(C)]
pub struct SherpaOnnxOfflineTtsModelConfig {
    pub vits: SherpaOnnxOfflineTtsVitsModelConfig,
    pub num_threads: i32,
    pub debug: i32,
    pub provider: *const c_char,
    pub matcha: SherpaOnnxOfflineTtsMatchaModelConfig,
    pub kokoro: SherpaOnnxOfflineTtsKokoroModelConfig,
    pub kitten: SherpaOnnxOfflineTtsKittenModelConfig,
    pub zipvoice: SherpaOnnxOfflineTtsZipvoiceModelConfig,
    pub pocket: SherpaOnnxOfflineTtsPocketModelConfig,
    pub supertonic: SherpaOnnxOfflineTtsSupertonicModelConfig,
}

/** @brief Configuration for offline text-to-speech. (c-api.h L2450) */
#[repr(C)]
pub struct SherpaOnnxOfflineTtsConfig {
    pub model: SherpaOnnxOfflineTtsModelConfig,
    pub rule_fsts: *const c_char,
    pub max_num_sentences: i32,
    pub rule_fars: *const c_char,
    pub silence_scale: f32,
}

/** @brief Generation-time parameters. (c-api.h L2727) */
#[repr(C)]
pub struct SherpaOnnxGenerationConfig {
    pub silence_scale: f32,
    pub speed: f32,
    pub sid: i32,
    pub reference_audio: *const f32,
    pub reference_audio_len: i32,
    pub reference_sample_rate: i32,
    pub reference_text: *const c_char,
    pub num_steps: i32,
    pub extra: *const c_char,
}

/** @brief Generated waveform returned by TTS APIs. (c-api.h L2470) */
#[repr(C)]
pub struct SherpaOnnxGeneratedAudio {
    pub samples: *const f32,
    pub n: i32,
    pub sample_rate: i32,
}

/** @brief Opaque offline TTS handle. (c-api.h L2517) */
#[repr(C)]
pub struct SherpaOnnxOfflineTts {
    _opaque: [u8; 0],
}

/** (c-api.h L2513) Return 0 to stop early, 1 to continue generating. */
pub type SherpaOnnxGeneratedAudioProgressCallbackWithArg =
    unsafe extern "C" fn(samples: *const f32, n: i32, progress: f32, arg: *mut c_void) -> i32;

unsafe extern "C" {
    /** (c-api.h L2539) Returns NULL if the config is invalid. */
    pub fn SherpaOnnxCreateOfflineTts(
        config: *const SherpaOnnxOfflineTtsConfig,
    ) -> *const SherpaOnnxOfflineTts;

    /** (c-api.h L2548) The parameter is a const pointer; upstream const_casts it internally. */
    pub fn SherpaOnnxDestroyOfflineTts(tts: *const SherpaOnnxOfflineTts);

    /// Generated results carry their own sample_rate; the current
    /// implementation never queries it; kept as a complete mirror of the
    /// v1.13.8 API surface.
    #[allow(dead_code)]
    pub fn SherpaOnnxOfflineTtsSampleRate(tts: *const SherpaOnnxOfflineTts) -> i32;

    /** (c-api.h L2776) Returns NULL on generation failure or empty result. */
    pub fn SherpaOnnxOfflineTtsGenerateWithConfig(
        tts: *const SherpaOnnxOfflineTts,
        text: *const c_char,
        config: *const SherpaOnnxGenerationConfig,
        callback: Option<SherpaOnnxGeneratedAudioProgressCallbackWithArg>,
        arg: *mut c_void,
    ) -> *const SherpaOnnxGeneratedAudio;

    /** (c-api.h L2789) Frees the audio returned by Generate*. */
    pub fn SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio: *const SherpaOnnxGeneratedAudio);
}
