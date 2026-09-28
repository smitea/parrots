//! Assembles and owns `SherpaOnnxOfflineTtsConfig`.
//!
//! The C side reads the string pointers inside the struct during the
//! `SherpaOnnxCreateOfflineTts` call, so [`ConfigBundle`] holds both all
//! `CString`s (heap buffer addresses stay stable across `Vec` growth /
//! struct moves) and the C struct, keeping every pointer valid for the whole
//! call.

use std::ffi::{c_char, CString};
use std::path::Path;

use crate::ffi;

/// Documented defaults from upstream `offline-tts-zipvoice-model-config.cc`.
/// Validate() requires feat_scale/target_rms/guidance_scale > 0 and
/// t_shift >= 0; 0 does not fall back to the default, so they must be passed
/// explicitly.
pub const ZIPVOICE_FEAT_SCALE: f32 = 0.1;
pub const ZIPVOICE_T_SHIFT: f32 = 0.5;
pub const ZIPVOICE_TARGET_RMS: f32 = 0.1;
pub const ZIPVOICE_GUIDANCE_SCALE: f32 = 1.0;

/// Assembled TTS config owning every string; `as_ptr()` is passed to
/// `SherpaOnnxCreateOfflineTts`.
pub(crate) struct ConfigBundle {
    /// Every string referenced by the C struct; must live as long as `raw`.
    _strings: Vec<CString>,
    raw: ffi::SherpaOnnxOfflineTtsConfig,
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
    /// Builds a config with only the zipvoice model family enabled;
    /// `model_dir` should be an absolute path.
    pub(crate) fn zipvoice(model_dir: &Path) -> Self {
        let dir = model_dir.to_string_lossy().into_owned();
        let mut strings = Vec::new();

        let tokens = intern(&mut strings, &format!("{dir}/tokens.txt"));
        let encoder = intern(&mut strings, &format!("{dir}/encoder.int8.onnx"));
        let decoder = intern(&mut strings, &format!("{dir}/decoder.int8.onnx"));
        let vocoder = intern(&mut strings, &format!("{dir}/vocos_24khz.onnx"));
        let data_dir = intern(&mut strings, &format!("{dir}/espeak-ng-data"));
        let lexicon = intern(&mut strings, &format!("{dir}/lexicon.txt"));
        let provider = intern(&mut strings, "cpu");

        let raw = ffi::SherpaOnnxOfflineTtsConfig {
            model: ffi::SherpaOnnxOfflineTtsModelConfig {
                vits: ffi::SherpaOnnxOfflineTtsVitsModelConfig {
                    model: std::ptr::null(),
                    lexicon: std::ptr::null(),
                    tokens: std::ptr::null(),
                    data_dir: std::ptr::null(),
                    noise_scale: 0.0,
                    noise_scale_w: 0.0,
                    length_scale: 0.0,
                    dict_dir: std::ptr::null(),
                },
                num_threads: 2,
                debug: 0,
                provider,
                matcha: ffi::SherpaOnnxOfflineTtsMatchaModelConfig {
                    acoustic_model: std::ptr::null(),
                    vocoder: std::ptr::null(),
                    lexicon: std::ptr::null(),
                    tokens: std::ptr::null(),
                    data_dir: std::ptr::null(),
                    noise_scale: 0.0,
                    length_scale: 0.0,
                    dict_dir: std::ptr::null(),
                },
                kokoro: ffi::SherpaOnnxOfflineTtsKokoroModelConfig {
                    model: std::ptr::null(),
                    voices: std::ptr::null(),
                    tokens: std::ptr::null(),
                    data_dir: std::ptr::null(),
                    length_scale: 0.0,
                    dict_dir: std::ptr::null(),
                    lexicon: std::ptr::null(),
                    lang: std::ptr::null(),
                },
                kitten: ffi::SherpaOnnxOfflineTtsKittenModelConfig {
                    model: std::ptr::null(),
                    voices: std::ptr::null(),
                    tokens: std::ptr::null(),
                    data_dir: std::ptr::null(),
                    length_scale: 0.0,
                },
                zipvoice: ffi::SherpaOnnxOfflineTtsZipvoiceModelConfig {
                    tokens,
                    encoder,
                    decoder,
                    vocoder,
                    data_dir,
                    lexicon,
                    feat_scale: ZIPVOICE_FEAT_SCALE,
                    t_shift: ZIPVOICE_T_SHIFT,
                    target_rms: ZIPVOICE_TARGET_RMS,
                    guidance_scale: ZIPVOICE_GUIDANCE_SCALE,
                },
                pocket: ffi::SherpaOnnxOfflineTtsPocketModelConfig {
                    lm_flow: std::ptr::null(),
                    lm_main: std::ptr::null(),
                    encoder: std::ptr::null(),
                    decoder: std::ptr::null(),
                    text_conditioner: std::ptr::null(),
                    vocab_json: std::ptr::null(),
                    token_scores_json: std::ptr::null(),
                    voice_embedding_cache_capacity: 0,
                },
                supertonic: ffi::SherpaOnnxOfflineTtsSupertonicModelConfig {
                    duration_predictor: std::ptr::null(),
                    text_encoder: std::ptr::null(),
                    vector_estimator: std::ptr::null(),
                    vocoder: std::ptr::null(),
                    tts_json: std::ptr::null(),
                    unicode_indexer: std::ptr::null(),
                    voice_style: std::ptr::null(),
                },
            },
            rule_fsts: std::ptr::null(),
            max_num_sentences: 0,
            rule_fars: std::ptr::null(),
            silence_scale: 0.0,
        };

        Self {
            _strings: strings,
            raw,
        }
    }

    pub(crate) fn as_ptr(&self) -> *const ffi::SherpaOnnxOfflineTtsConfig {
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
    fn zipvoice_section_populated_with_expected_paths() {
        let bundle = ConfigBundle::zipvoice(Path::new("/models/zipvoice"));
        let zv = &bundle.raw.model.zipvoice;

        assert_eq!(str_of(zv.tokens), "/models/zipvoice/tokens.txt");
        assert_eq!(str_of(zv.encoder), "/models/zipvoice/encoder.int8.onnx");
        assert_eq!(str_of(zv.decoder), "/models/zipvoice/decoder.int8.onnx");
        assert_eq!(str_of(zv.vocoder), "/models/zipvoice/vocos_24khz.onnx");
        assert_eq!(str_of(zv.data_dir), "/models/zipvoice/espeak-ng-data");
        assert_eq!(str_of(zv.lexicon), "/models/zipvoice/lexicon.txt");

        assert_eq!(bundle.raw.model.num_threads, 2);
        assert_eq!(bundle.raw.model.debug, 0);
        assert_eq!(str_of(bundle.raw.model.provider), "cpu");

        assert!((zv.feat_scale - ZIPVOICE_FEAT_SCALE).abs() < f32::EPSILON);
        assert!((zv.t_shift - ZIPVOICE_T_SHIFT).abs() < f32::EPSILON);
        assert!((zv.target_rms - ZIPVOICE_TARGET_RMS).abs() < f32::EPSILON);
        assert!((zv.guidance_scale - ZIPVOICE_GUIDANCE_SCALE).abs() < f32::EPSILON);
    }

    #[test]
    fn unused_model_sections_are_zeroed() {
        let bundle = ConfigBundle::zipvoice(Path::new("/models/zipvoice"));

        let vits = &bundle.raw.model.vits;
        assert!(vits.model.is_null() && vits.lexicon.is_null() && vits.tokens.is_null());
        assert!(vits.data_dir.is_null() && vits.dict_dir.is_null());
        assert_eq!(vits.noise_scale, 0.0);

        let matcha = &bundle.raw.model.matcha;
        assert!(matcha.acoustic_model.is_null() && matcha.vocoder.is_null());
        let kokoro = &bundle.raw.model.kokoro;
        assert!(kokoro.model.is_null() && kokoro.voices.is_null());
        let kitten = &bundle.raw.model.kitten;
        assert!(kitten.model.is_null() && kitten.voices.is_null());
        let pocket = &bundle.raw.model.pocket;
        assert!(pocket.lm_flow.is_null() && pocket.lm_main.is_null());
        assert_eq!(pocket.voice_embedding_cache_capacity, 0);
        let supertonic = &bundle.raw.model.supertonic;
        assert!(supertonic.duration_predictor.is_null() && supertonic.tts_json.is_null());

        assert!(bundle.raw.rule_fsts.is_null() && bundle.raw.rule_fars.is_null());
        assert_eq!(bundle.raw.max_num_sentences, 0);
        assert_eq!(bundle.raw.silence_scale, 0.0);
    }
}
