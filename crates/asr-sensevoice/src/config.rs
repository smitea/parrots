//! Assembles and owns `SherpaOnnxOfflineRecognizerConfig`.
//!
//! The C side reads the string pointers inside the struct during the
//! `SherpaOnnxCreateOfflineRecognizer` call, so [`ConfigBundle`] holds both
//! all `CString`s (heap buffer addresses stay stable across `Vec` growth /
//! struct moves) and the C struct, keeping every pointer valid for the whole
//! call.

use std::ffi::{c_char, CString};
use std::path::Path;

use crate::ffi;

/// Language hint values supported by SenseVoice; use auto for mixed zh/en
/// speech to let the model auto-detect.
pub const SENSEVOICE_LANG_AUTO: &str = "auto";

/// Upstream `offline-recognizer-impl` default decoding method (also the
/// fallback for NULL), passed explicitly for auditability.
pub const DECODING_METHOD: &str = "greedy_search";

/// Model feature parameters (silero fbank; upstream docs default 16000/80).
pub const FEAT_SAMPLE_RATE: i32 = 16000;
pub const FEAT_FEATURE_DIM: i32 = 80;

/// Assembled ASR config owning every string; `as_ptr()` is passed to
/// `SherpaOnnxCreateOfflineRecognizer`.
pub(crate) struct ConfigBundle {
    /// Every string referenced by the C struct; must live as long as `raw`.
    _strings: Vec<CString>,
    raw: ffi::SherpaOnnxOfflineRecognizerConfig,
}

/// Interns a string and returns a pointer usable by the C struct.
///
/// A `CString`'s heap buffer address is independent of where its ownership is
/// stored, so later growth of `strings` will not invalidate the returned
/// pointer.
fn intern(strings: &mut Vec<CString>, s: &str) -> *const c_char {
    let c = CString::new(s).expect("path/text must not contain NUL bytes");
    let ptr = c.as_ptr();
    strings.push(c);
    ptr
}

impl ConfigBundle {
    /// Builds a config with only the sense_voice model family enabled;
    /// `model_dir` should be an absolute path.
    ///
    /// `language` is the SenseVoice language hint ("zh"/"en"/"auto", etc.).
    pub(crate) fn sensevoice(model_dir: &Path, language: &str) -> Self {
        let dir = model_dir.to_string_lossy().into_owned();
        let mut strings = Vec::new();

        let model = intern(&mut strings, &format!("{dir}/model.int8.onnx"));
        let tokens = intern(&mut strings, &format!("{dir}/tokens.txt"));
        let language = intern(&mut strings, language);
        let provider = intern(&mut strings, "cpu");
        let decoding_method = intern(&mut strings, DECODING_METHOD);

        let raw = ffi::SherpaOnnxOfflineRecognizerConfig {
            feat_config: ffi::SherpaOnnxFeatureConfig {
                sample_rate: FEAT_SAMPLE_RATE,
                feature_dim: FEAT_FEATURE_DIM,
            },
            model_config: ffi::SherpaOnnxOfflineModelConfig {
                transducer: ffi::SherpaOnnxOfflineTransducerModelConfig {
                    encoder: std::ptr::null(),
                    decoder: std::ptr::null(),
                    joiner: std::ptr::null(),
                },
                paraformer: ffi::SherpaOnnxOfflineParaformerModelConfig {
                    model: std::ptr::null(),
                },
                nemo_ctc: ffi::SherpaOnnxOfflineNemoEncDecCtcModelConfig {
                    model: std::ptr::null(),
                },
                whisper: ffi::SherpaOnnxOfflineWhisperModelConfig {
                    encoder: std::ptr::null(),
                    decoder: std::ptr::null(),
                    language: std::ptr::null(),
                    task: std::ptr::null(),
                    tail_paddings: 0,
                    enable_token_timestamps: 0,
                    enable_segment_timestamps: 0,
                },
                tdnn: ffi::SherpaOnnxOfflineTdnnModelConfig {
                    model: std::ptr::null(),
                },
                tokens,
                num_threads: 2,
                debug: 0,
                provider,
                model_type: std::ptr::null(),
                modeling_unit: std::ptr::null(),
                bpe_vocab: std::ptr::null(),
                telespeech_ctc: std::ptr::null(),
                sense_voice: ffi::SherpaOnnxOfflineSenseVoiceModelConfig {
                    model,
                    language,
                    use_itn: 1,
                },
                moonshine: ffi::SherpaOnnxOfflineMoonshineModelConfig {
                    preprocessor: std::ptr::null(),
                    encoder: std::ptr::null(),
                    uncached_decoder: std::ptr::null(),
                    cached_decoder: std::ptr::null(),
                    merged_decoder: std::ptr::null(),
                },
                fire_red_asr: ffi::SherpaOnnxOfflineFireRedAsrModelConfig {
                    encoder: std::ptr::null(),
                    decoder: std::ptr::null(),
                },
                dolphin: ffi::SherpaOnnxOfflineDolphinModelConfig {
                    model: std::ptr::null(),
                },
                zipformer_ctc: ffi::SherpaOnnxOfflineZipformerCtcModelConfig {
                    model: std::ptr::null(),
                },
                canary: ffi::SherpaOnnxOfflineCanaryModelConfig {
                    encoder: std::ptr::null(),
                    decoder: std::ptr::null(),
                    src_lang: std::ptr::null(),
                    tgt_lang: std::ptr::null(),
                    use_pnc: 0,
                },
                wenet_ctc: ffi::SherpaOnnxOfflineWenetCtcModelConfig {
                    model: std::ptr::null(),
                },
                omnilingual: ffi::SherpaOnnxOfflineOmnilingualAsrCtcModelConfig {
                    model: std::ptr::null(),
                },
                medasr: ffi::SherpaOnnxOfflineMedAsrCtcModelConfig {
                    model: std::ptr::null(),
                },
                funasr_nano: ffi::SherpaOnnxOfflineFunASRNanoModelConfig {
                    encoder_adaptor: std::ptr::null(),
                    llm: std::ptr::null(),
                    embedding: std::ptr::null(),
                    tokenizer: std::ptr::null(),
                    system_prompt: std::ptr::null(),
                    user_prompt: std::ptr::null(),
                    max_new_tokens: 0,
                    temperature: 0.0,
                    top_p: 0.0,
                    seed: 0,
                    language: std::ptr::null(),
                    itn: 0,
                    hotwords: std::ptr::null(),
                },
                fire_red_asr_ctc: ffi::SherpaOnnxOfflineFireRedAsrCtcModelConfig {
                    model: std::ptr::null(),
                },
                qwen3_asr: ffi::SherpaOnnxOfflineQwen3ASRModelConfig {
                    conv_frontend: std::ptr::null(),
                    encoder: std::ptr::null(),
                    decoder: std::ptr::null(),
                    tokenizer: std::ptr::null(),
                    max_total_len: 0,
                    max_new_tokens: 0,
                    temperature: 0.0,
                    top_p: 0.0,
                    seed: 0,
                    hotwords: std::ptr::null(),
                },
                cohere_transcribe: ffi::SherpaOnnxOfflineCohereTranscribeModelConfig {
                    encoder: std::ptr::null(),
                    decoder: std::ptr::null(),
                    language: std::ptr::null(),
                    use_punct: 0,
                    use_itn: 0,
                },
            },
            lm_config: ffi::SherpaOnnxOfflineLMConfig {
                model: std::ptr::null(),
                scale: 0.0,
            },
            decoding_method,
            max_active_paths: 0,
            hotwords_file: std::ptr::null(),
            hotwords_score: 0.0,
            rule_fsts: std::ptr::null(),
            rule_fars: std::ptr::null(),
            blank_penalty: 0.0,
            hr: ffi::SherpaOnnxHomophoneReplacerConfig {
                dict_dir: std::ptr::null(),
                lexicon: std::ptr::null(),
                rule_fsts: std::ptr::null(),
            },
        };

        Self {
            _strings: strings,
            raw,
        }
    }

    pub(crate) fn as_ptr(&self) -> *const ffi::SherpaOnnxOfflineRecognizerConfig {
        &self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    /// Reads back a string referenced by the C struct (valid only while the bundle lives).
    fn str_of(ptr: *const c_char) -> String {
        assert!(!ptr.is_null());
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn sensevoice_section_populated_with_expected_paths() {
        let bundle = ConfigBundle::sensevoice(Path::new("/models/sensevoice"), "zh");
        let sv = &bundle.raw.model_config.sense_voice;

        assert_eq!(str_of(sv.model), "/models/sensevoice/model.int8.onnx");
        assert_eq!(
            str_of(bundle.raw.model_config.tokens),
            "/models/sensevoice/tokens.txt"
        );
        assert_eq!(str_of(sv.language), "zh");
        assert_eq!(sv.use_itn, 1);

        assert_eq!(bundle.raw.feat_config.sample_rate, FEAT_SAMPLE_RATE);
        assert_eq!(bundle.raw.feat_config.feature_dim, FEAT_FEATURE_DIM);
        assert_eq!(bundle.raw.model_config.num_threads, 2);
        assert_eq!(bundle.raw.model_config.debug, 0);
        assert_eq!(str_of(bundle.raw.model_config.provider), "cpu");
        assert_eq!(str_of(bundle.raw.decoding_method), DECODING_METHOD);
    }

    #[test]
    fn auto_language_is_interned() {
        let bundle =
            ConfigBundle::sensevoice(Path::new("/models/sensevoice"), SENSEVOICE_LANG_AUTO);
        assert_eq!(str_of(bundle.raw.model_config.sense_voice.language), "auto");
    }

    #[test]
    fn unused_model_sections_are_zeroed() {
        let bundle = ConfigBundle::sensevoice(Path::new("/models/sensevoice"), "auto");

        let m = &bundle.raw.model_config;
        let t = &m.transducer;
        assert!(t.encoder.is_null() && t.decoder.is_null() && t.joiner.is_null());
        assert!(m.paraformer.model.is_null());
        assert!(m.nemo_ctc.model.is_null());
        let w = &m.whisper;
        assert!(w.encoder.is_null() && w.decoder.is_null());
        assert_eq!(w.tail_paddings, 0);
        assert!(m.tdnn.model.is_null());
        assert!(m.model_type.is_null() && m.modeling_unit.is_null() && m.bpe_vocab.is_null());
        assert!(m.telespeech_ctc.is_null());

        let ms = &m.moonshine;
        assert!(ms.preprocessor.is_null() && ms.encoder.is_null());
        let fr = &m.fire_red_asr;
        assert!(fr.encoder.is_null() && fr.decoder.is_null());
        assert!(m.dolphin.model.is_null() && m.zipformer_ctc.model.is_null());
        let c = &m.canary;
        assert!(c.encoder.is_null() && c.src_lang.is_null() && c.use_pnc == 0);
        assert!(m.wenet_ctc.model.is_null() && m.omnilingual.model.is_null());
        assert!(m.medasr.model.is_null());

        let fnano = &m.funasr_nano;
        assert!(fnano.llm.is_null() && fnano.embedding.is_null());
        assert_eq!(fnano.max_new_tokens, 0);
        let q = &m.qwen3_asr;
        assert!(q.conv_frontend.is_null() && q.encoder.is_null());
        assert_eq!(q.max_total_len, 0);
        let co = &m.cohere_transcribe;
        assert!(co.encoder.is_null() && co.use_punct == 0 && co.use_itn == 0);

        assert!(bundle.raw.lm_config.model.is_null());
        assert!(bundle.raw.hotwords_file.is_null() && bundle.raw.rule_fsts.is_null());
        assert!(bundle.raw.hr.lexicon.is_null() && bundle.raw.hr.rule_fsts.is_null());
    }
}
