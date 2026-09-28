use anyhow::Context;
use parrots_asr_sensevoice::SenseVoiceAsr;
use parrots_asr_whisper::WhisperAsr;
use parrots_core::{AudioSegment, Lang, TextPolisher};
use parrots_engine::{Engine, LanguagePack};
use parrots_mt_opus::MarianTranslator;
use parrots_tts_zipvoice::ZipvoiceTts;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Official BlackHole virtual device name (direction A exit: meeting apps read this as their microphone)
pub const VIRTUAL_MIC: &str = "Parrots Microphone";
/// Direction B entry: meeting apps play through this speaker; we capture from here
pub const VIRTUAL_SPEAKERS: &str = "Parrots Speakers";

/// Install hint shown when the virtual driver is missing (branded pkg preferred, official BlackHole as alternative)
pub const INSTALL_HINT: &str =
    "cd driver/macos && ./make-pkg.sh, then double-click to install the ParrotsAudio pkg \
(or brew install --cask blackhole-2ch blackhole-16ch for the official alternative)";

pub fn resolve_input_device(pref: Option<&str>) -> parrots_core::DeviceId {
    resolve_device(pref, VIRTUAL_SPEAKERS)
}

/// talk live specific: default = system default speaker (most intuitive for local monitoring);
/// in meeting mode pass `--device "Parrots Microphone"` explicitly (strict)
pub fn resolve_talk_output_device(pref: Option<&str>) -> parrots_core::DeviceId {
    match pref {
        Some(name) => parrots_core::DeviceId(Some(name.to_string())),
        None => parrots_core::DeviceId(None),
    }
}

fn resolve_device(pref: Option<&str>, virtual_name: &str) -> parrots_core::DeviceId {
    match pref {
        Some(name) => parrots_core::DeviceId(Some(name.to_string())),
        None => match parrots_platform_macos::find_device(virtual_name) {
            Some(found) => parrots_core::DeviceId(Some(found)),
            None => {
                tracing::warn!("{virtual_name} not found, falling back to the default device; install: {INSTALL_HINT}");
                parrots_core::DeviceId(None)
            }
        },
    }
}

/// Text polish layer switch (plan 7): auto = on for fixture / off for live (preserves latency)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum PolishChoice {
    /// On by default for fixture, off by default for live; auto-degrades when the model is missing
    #[default]
    Auto,
    /// Force on (degrades with a warn when the model is missing)
    On,
    /// Force off
    Off,
}

/// Polish timeout in live mode (plan 7 fixed value, keeps latency bounded)
const POLISH_TIMEOUT_LIVE: Duration = Duration::from_millis(800);
/// Relaxed for fixture/batch mode (quality first offline; plan deviation documented)
const POLISH_TIMEOUT_FIXTURE: Duration = Duration::from_millis(1500);

/// Build a TextPolisher per the switch policy; `fixture_mode` drives the auto decision and timeout choice.
/// Returns None = not enabled (policy off, or degraded due to missing model).
pub fn build_polisher(choice: PolishChoice, fixture_mode: bool) -> Option<Arc<dyn TextPolisher>> {
    let enabled = match choice {
        PolishChoice::Off => false,
        PolishChoice::On => true,
        PolishChoice::Auto => fixture_mode,
    };
    if !enabled {
        return None;
    }
    let timeout = if fixture_mode {
        POLISH_TIMEOUT_FIXTURE
    } else {
        POLISH_TIMEOUT_LIVE
    };
    let polisher = parrots_polish_qwen::QwenPolisher::load(&models_root().join("polish/qwen1.5b"));
    if !polisher.ready() {
        tracing::warn!(
            "polish model missing, text polish layer degraded off (model: download-models.sh)"
        );
        return None;
    }
    Some(Arc::new(polisher.with_timeout(timeout)))
}

/// ASR engine choice (--asr): default SenseVoice-small (beats whisper-small on both
/// zh/en accuracy and speed; A/B benchmark in tests/asr_ab.rs); whisper kept as fallback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum AsrChoice {
    /// Default: SenseVoice-small (sherpa-onnx int8, zh-en-ja-ko-yue)
    #[default]
    Sensevoice,
    /// Alternative: whisper-small (Metal)
    Whisper,
}

pub fn models_root() -> PathBuf {
    std::env::var_os("PARROTS_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"))
}

pub fn profiles_dir() -> PathBuf {
    std::env::var_os("PARROTS_PROFILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("profiles"))
}

/// Hotword list (plan 6): profiles/hotwords.txt; missing file = empty list.
pub fn load_hotwords() -> parrots_pipeline::Hotwords {
    parrots_pipeline::Hotwords::load(&profiles_dir().join("hotwords.txt"))
}

/// Append one hotword to the list file; returns the total line count of the list (for the confirmation message).
pub fn add_hotword(word: &str) -> anyhow::Result<usize> {
    let dir = profiles_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("hotwords.txt");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    writeln!(f, "{word}")?;
    let hotwords = load_hotwords();
    Ok(hotwords.len())
}

pub fn load_voice(name: &str) -> anyhow::Result<parrots_core::VoiceProfile> {
    let wav = profiles_dir().join(format!("{name}.wav"));
    let txt = profiles_dir().join(format!("{name}.txt"));
    anyhow::ensure!(
        wav.is_file() && txt.is_file(),
        "voice profile not found: {name} (run parrots enroll --name {name} first)"
    );
    Ok(parrots_core::VoiceProfile::from_prompt(
        wav,
        std::fs::read_to_string(txt)?,
    ))
}

pub fn build_engine(asr: AsrChoice) -> anyhow::Result<Engine> {
    let root = models_root();
    // Hotword source bias (plan 6): inject the list into the whisper initial_prompt to reduce proper-noun misspellings
    let hotwords = load_hotwords();
    let initial_prompt = if hotwords.is_empty() {
        None
    } else {
        tracing::info!(
            "{} hotwords injected into the whisper decoding bias",
            hotwords.len()
        );
        Some(hotwords.prompt_text())
    };
    // Both language packs share one Arc<dyn AsrEngine>: whisper already shares via an internal lock;
    // a single SenseVoice instance (auto language) naturally covers zh/en and saves a copy of model memory
    let asr_engine: Arc<dyn parrots_core::AsrEngine> = match asr {
        AsrChoice::Whisper => {
            let whisper_path = root.join("whisper/ggml-small.bin");
            Arc::new(WhisperAsr::load(
                &whisper_path,
                &[Lang::En, Lang::Zh],
                initial_prompt,
            )?)
        }
        AsrChoice::Sensevoice => Arc::new(SenseVoiceAsr::load(
            &root.join("asr/sensevoice"),
            &[Lang::En, Lang::Zh],
        )?),
    };
    let mt_en_zh = MarianTranslator::load(&root.join("mt/en-zh"), Lang::En, Lang::Zh)?;
    let mt_zh_en = MarianTranslator::load(&root.join("mt/zh-en"), Lang::Zh, Lang::En)?;
    let tts_en = ZipvoiceTts::load(&root.join("tts/zipvoice"))?;
    let tts_zh = ZipvoiceTts::load(&root.join("tts/zipvoice"))?;
    // Warm up MT sessions at startup: removes the hidden session-creation cost on the first translation (same pattern as TTS warmup)
    mt_en_zh.warmup()?;
    mt_zh_en.warmup()?;
    // Warm up TTS sessions at startup: the first synthesis produces first audio immediately (aligns with the design doc latency target)
    tts_en.warmup()?;
    tts_zh.warmup()?;

    let en = LanguagePack {
        lang: Lang::En,
        asr: asr_engine.clone(),
        translators: vec![Arc::new(mt_en_zh)],
        tts: Arc::new(tts_en),
    };
    let zh = LanguagePack {
        lang: Lang::Zh,
        asr: asr_engine,
        translators: vec![Arc::new(mt_zh_en)],
        tts: Arc::new(tts_zh),
    };
    let mut e = Engine::new();
    e.register(en);
    e.register(zh);
    Ok(e)
}

pub fn read_wav_mono(path: &Path) -> anyhow::Result<AudioSegment> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let ch = usize::from(spec.channels);
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| f32::from(s.unwrap()) / 32768.0)
            .collect(),
    };
    let mono: Vec<f32> = if ch > 1 {
        raw.chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        raw
    };
    if sr % 16000 == 0 && sr > 16000 {
        let r = (sr / 16000) as usize;
        let samples = mono
            .chunks(r)
            .map(|c| c.iter().sum::<f32>() / r as f32)
            .collect();
        Ok(AudioSegment::new(samples, 16000))
    } else {
        Ok(AudioSegment::new(mono, sr))
    }
}

pub fn write_wav(path: &Path, seg: &AudioSegment) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: seg.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in &seg.samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}
