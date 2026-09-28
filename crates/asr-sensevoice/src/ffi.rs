//! `#[repr(C)]` mirror declarations of the sherpa-onnx C API (v1.13.8),
//! offline ASR part.
//!
//! Every struct's field names and order match upstream
//! `sherpa-onnx/c-api/c-api.h` (v1.13.8, file copy at `/tmp/c-api.h`)
//! one to one; the C side reads the config by offset, so field order must
//! not change.
//!
//! Semantic notes: a numeric field of 0 is replaced by the upstream default
//! via `SHERPA_ONNX_OR(x, y)` (e.g. `decoding_method=NULL → "greedy_search"`);
//! SenseVoice's `use_itn` is "enabled if non-zero" and must be passed as 1
//! explicitly — 0 does not fall back to the default.

use std::ffi::c_char;

/** @brief Feature extraction settings for ASR. (c-api.h L287) */
#[repr(C)]
pub struct SherpaOnnxFeatureConfig {
    pub sample_rate: i32,
    pub feature_dim: i32,
}

/** @brief Configuration for a non-streaming transducer model. (c-api.h L841) */
#[repr(C)]
pub struct SherpaOnnxOfflineTransducerModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub joiner: *const c_char,
}

/** @brief Configuration for a non-streaming Paraformer model. (c-api.h L851) */
#[repr(C)]
pub struct SherpaOnnxOfflineParaformerModelConfig {
    pub model: *const c_char,
}

/** @brief Configuration for a non-streaming NeMo CTC model. (c-api.h L857) */
#[repr(C)]
pub struct SherpaOnnxOfflineNemoEncDecCtcModelConfig {
    pub model: *const c_char,
}

/** @brief Configuration for a non-streaming Whisper model. (c-api.h L865) */
#[repr(C)]
pub struct SherpaOnnxOfflineWhisperModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub language: *const c_char,
    pub task: *const c_char,
    pub tail_paddings: i32,
    pub enable_token_timestamps: i32,
    pub enable_segment_timestamps: i32,
}

/** @brief Configuration for a TDNN model. (c-api.h L941) */
#[repr(C)]
pub struct SherpaOnnxOfflineTdnnModelConfig {
    pub model: *const c_char,
}

/** @brief Configuration for a SenseVoice model. (c-api.h L955) */
#[repr(C)]
pub struct SherpaOnnxOfflineSenseVoiceModelConfig {
    pub model: *const c_char,
    pub language: *const c_char,
    pub use_itn: i32,
}

/** @brief Configuration for a Moonshine model. (c-api.h L927) */
#[repr(C)]
pub struct SherpaOnnxOfflineMoonshineModelConfig {
    pub preprocessor: *const c_char,
    pub encoder: *const c_char,
    pub uncached_decoder: *const c_char,
    pub cached_decoder: *const c_char,
    pub merged_decoder: *const c_char,
}

/** @brief Configuration for a FireRedAsr encoder/decoder model. (c-api.h L913) */
#[repr(C)]
pub struct SherpaOnnxOfflineFireRedAsrModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
}

/** @brief Configuration for a Dolphin model. (c-api.h L965) */
#[repr(C)]
pub struct SherpaOnnxOfflineDolphinModelConfig {
    pub model: *const c_char,
}

/** @brief Configuration for an offline Zipformer CTC model. (c-api.h L971) */
#[repr(C)]
pub struct SherpaOnnxOfflineZipformerCtcModelConfig {
    pub model: *const c_char,
}

/** @brief Configuration for a Canary model. (c-api.h L885) */
#[repr(C)]
pub struct SherpaOnnxOfflineCanaryModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub src_lang: *const c_char,
    pub tgt_lang: *const c_char,
    pub use_pnc: i32,
}

/** @brief Configuration for an offline WeNet CTC model. (c-api.h L977) */
#[repr(C)]
pub struct SherpaOnnxOfflineWenetCtcModelConfig {
    pub model: *const c_char,
}

/** @brief Configuration for an omnilingual offline CTC model. (c-api.h L983) */
#[repr(C)]
pub struct SherpaOnnxOfflineOmnilingualAsrCtcModelConfig {
    pub model: *const c_char,
}

/** @brief Configuration for a MedASR CTC model. (c-api.h L1044) */
#[repr(C)]
pub struct SherpaOnnxOfflineMedAsrCtcModelConfig {
    pub model: *const c_char,
}

/** @brief Configuration for an offline FunASR Nano model. (c-api.h L989) */
#[repr(C)]
pub struct SherpaOnnxOfflineFunASRNanoModelConfig {
    pub encoder_adaptor: *const c_char,
    pub llm: *const c_char,
    pub embedding: *const c_char,
    pub tokenizer: *const c_char,
    pub system_prompt: *const c_char,
    pub user_prompt: *const c_char,
    pub max_new_tokens: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: i32,
    pub language: *const c_char,
    pub itn: i32,
    pub hotwords: *const c_char,
}

/** @brief Configuration for an offline Qwen3-ASR model. (c-api.h L1019) */
#[repr(C)]
pub struct SherpaOnnxOfflineQwen3ASRModelConfig {
    pub conv_frontend: *const c_char,
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub tokenizer: *const c_char,
    pub max_total_len: i32,
    pub max_new_tokens: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: i32,
    pub hotwords: *const c_char,
}

/** @brief Configuration for a FireRedAsr CTC model. (c-api.h L921) */
#[repr(C)]
pub struct SherpaOnnxOfflineFireRedAsrCtcModelConfig {
    pub model: *const c_char,
}

/** @brief Configuration for a Cohere Transcribe model. (c-api.h L899) */
#[repr(C)]
pub struct SherpaOnnxOfflineCohereTranscribeModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub language: *const c_char,
    pub use_punct: i32,
    pub use_itn: i32,
}

/**
 * @brief Model configuration shared by offline ASR recognizers. (c-api.h L1066)
 *
 * Note: tokens/num_threads/debug/provider/model_type/modeling_unit/bpe_vocab/
 * telespeech_ctc sit between tdnn and sense_voice — that is the upstream layout.
 */
#[repr(C)]
pub struct SherpaOnnxOfflineModelConfig {
    pub transducer: SherpaOnnxOfflineTransducerModelConfig,
    pub paraformer: SherpaOnnxOfflineParaformerModelConfig,
    pub nemo_ctc: SherpaOnnxOfflineNemoEncDecCtcModelConfig,
    pub whisper: SherpaOnnxOfflineWhisperModelConfig,
    pub tdnn: SherpaOnnxOfflineTdnnModelConfig,
    pub tokens: *const c_char,
    pub num_threads: i32,
    pub debug: i32,
    pub provider: *const c_char,
    pub model_type: *const c_char,
    pub modeling_unit: *const c_char,
    pub bpe_vocab: *const c_char,
    pub telespeech_ctc: *const c_char,
    pub sense_voice: SherpaOnnxOfflineSenseVoiceModelConfig,
    pub moonshine: SherpaOnnxOfflineMoonshineModelConfig,
    pub fire_red_asr: SherpaOnnxOfflineFireRedAsrModelConfig,
    pub dolphin: SherpaOnnxOfflineDolphinModelConfig,
    pub zipformer_ctc: SherpaOnnxOfflineZipformerCtcModelConfig,
    pub canary: SherpaOnnxOfflineCanaryModelConfig,
    pub wenet_ctc: SherpaOnnxOfflineWenetCtcModelConfig,
    pub omnilingual: SherpaOnnxOfflineOmnilingualAsrCtcModelConfig,
    pub medasr: SherpaOnnxOfflineMedAsrCtcModelConfig,
    pub funasr_nano: SherpaOnnxOfflineFunASRNanoModelConfig,
    pub fire_red_asr_ctc: SherpaOnnxOfflineFireRedAsrCtcModelConfig,
    pub qwen3_asr: SherpaOnnxOfflineQwen3ASRModelConfig,
    pub cohere_transcribe: SherpaOnnxOfflineCohereTranscribeModelConfig,
}

/** @brief Configuration for an offline language model. (c-api.h L947) */
#[repr(C)]
pub struct SherpaOnnxOfflineLMConfig {
    pub model: *const c_char,
    pub scale: f32,
}

/** @brief Configuration for homophone replacement. (c-api.h L304) */
#[repr(C)]
pub struct SherpaOnnxHomophoneReplacerConfig {
    pub dict_dir: *const c_char,
    pub lexicon: *const c_char,
    pub rule_fsts: *const c_char,
}

/** @brief Configuration for a non-streaming ASR recognizer. (c-api.h L1176) */
#[repr(C)]
pub struct SherpaOnnxOfflineRecognizerConfig {
    pub feat_config: SherpaOnnxFeatureConfig,
    pub model_config: SherpaOnnxOfflineModelConfig,
    pub lm_config: SherpaOnnxOfflineLMConfig,
    pub decoding_method: *const c_char,
    pub max_active_paths: i32,
    pub hotwords_file: *const c_char,
    pub hotwords_score: f32,
    pub rule_fsts: *const c_char,
    pub rule_fars: *const c_char,
    pub blank_penalty: f32,
    pub hr: SherpaOnnxHomophoneReplacerConfig,
}

/** @brief Non-streaming recognizer handle. (c-api.h L1206) */
#[repr(C)]
pub struct SherpaOnnxOfflineRecognizer {
    _opaque: [u8; 0],
}

/** @brief Non-streaming decoding state for one utterance. (c-api.h L1209) */
#[repr(C)]
pub struct SherpaOnnxOfflineStream {
    _opaque: [u8; 0],
}

/**
 * @brief Recognition result for a non-streaming ASR stream. (c-api.h L1470)
 *
 * Pointer fields are owned by the object returned by GetResult and become
 * invalid after Destroy; this crate only reads text.
 */
#[repr(C)]
pub struct SherpaOnnxOfflineRecognizerResult {
    pub text: *const c_char,
    pub timestamps: *mut f32,
    pub count: i32,
    pub tokens: *const c_char,
    pub tokens_arr: *const *const c_char,
    pub json: *const c_char,
    pub lang: *const c_char,
    pub emotion: *const c_char,
    pub event: *const c_char,
    pub durations: *mut f32,
    pub ys_log_probs: *mut f32,
    pub segment_timestamps: *const f32,
    pub segment_durations: *const f32,
    pub segment_texts: *const c_char,
    pub segment_texts_arr: *const *const c_char,
    pub segment_count: i32,
}

unsafe extern "C" {
    /** (c-api.h L1270) Returns NULL if the config is invalid. */
    pub fn SherpaOnnxCreateOfflineRecognizer(
        config: *const SherpaOnnxOfflineRecognizerConfig,
    ) -> *const SherpaOnnxOfflineRecognizer;

    /** (c-api.h L1284) Frees the handle returned by CreateOfflineRecognizer. */
    pub fn SherpaOnnxDestroyOfflineRecognizer(recognizer: *const SherpaOnnxOfflineRecognizer);

    /** (c-api.h L1315) One stream per utterance; the caller owns and destroys it. */
    pub fn SherpaOnnxCreateOfflineStream(
        recognizer: *const SherpaOnnxOfflineRecognizer,
    ) -> *const SherpaOnnxOfflineStream;

    /** (c-api.h L1348) Frees the handle returned by CreateOfflineStream. */
    pub fn SherpaOnnxDestroyOfflineStream(stream: *const SherpaOnnxOfflineStream);

    /**
     * (c-api.h L1377) Feeds an entire utterance at once (mono f32 [-1,1]);
     * upstream allows at most one call per stream and resamples internally
     * to the feature sample rate.
     */
    pub fn SherpaOnnxAcceptWaveformOffline(
        stream: *const SherpaOnnxOfflineStream,
        sample_rate: i32,
        samples: *const f32,
        n: i32,
    );

    /** (c-api.h L1437) Decodes a stream once, after its audio has been fed. */
    pub fn SherpaOnnxDecodeOfflineStream(
        recognizer: *const SherpaOnnxOfflineRecognizer,
        stream: *const SherpaOnnxOfflineStream,
    );

    /** (c-api.h L1541) Returns a result snapshot; must be freed with DestroyOfflineRecognizerResult. */
    pub fn SherpaOnnxGetOfflineStreamResult(
        stream: *const SherpaOnnxOfflineStream,
    ) -> *const SherpaOnnxOfflineRecognizerResult;

    /** (c-api.h L1567) Frees the result returned by GetOfflineStreamResult. */
    pub fn SherpaOnnxDestroyOfflineRecognizerResult(r: *const SherpaOnnxOfflineRecognizerResult);
}
