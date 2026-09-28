use crate::wire;
use anyhow::Context;
use parrots_core::{
    AudioPlatform, AudioSegment, DeviceId, Lang, SpeechDetector, VadConfig, VoiceProfile,
};
use parrots_pipeline::{
    DirectionAPipeline, DirectionBPipeline, PipelineEvent, RollingVoiceprint, StageTimings,
    UtteranceAssembler,
};
use parrots_platform_macos::MacAudioPlatform;
use parrots_vad::{SileroVad, VAD_FRAME};
use std::path::PathBuf;

pub struct TranslateArgs {
    pub input: PathBuf,
    pub from: Lang,
    pub to: Lang,
    pub out: PathBuf,
    pub report: PathBuf,
    pub voice: Option<PathBuf>,
    pub voice_text: Option<String>,
    pub asr: wire::AsrChoice,
    /// Text polish layer (plan 7): fixture defaults to auto = on
    pub polish: wire::PolishChoice,
}

pub async fn run_translate(args: TranslateArgs) -> anyhow::Result<()> {
    let engine = wire::build_engine(args.asr)?;
    let src = engine
        .pack(args.from)
        .with_context(|| format!("{:?} language pack missing", args.from))?;
    let translator = engine
        .translator(args.from, args.to)
        .with_context(|| format!("{:?}→{:?} translator missing", args.from, args.to))?;
    let pipeline = DirectionBPipeline {
        asr: src.asr.clone(),
        translator,
        tts: src.tts.clone(),
        source_lang: args.from,
        target_lang: args.to,
        hotwords: hotwords_for_pipeline(),
        polisher: wire::build_polisher(args.polish, true),
    };

    let audio = wire::read_wav_mono(&args.input)?.with_lang(args.from);
    let tmp = tempfile::tempdir()?;
    let mut voiceprint = RollingVoiceprint::new(tmp.path(), 2)?;
    // Explicit voiceprint prompt takes priority
    if let (Some(v), Some(t)) = (&args.voice, &args.voice_text) {
        let seg = wire::read_wav_mono(v)?;
        voiceprint.update(&seg, t)?;
    }

    let mut vad = SileroVad::load(&wire::models_root().join("vad/silero_vad.onnx"))?;
    let mut detector = SpeechDetector::new(VadConfig::default(), VAD_FRAME as u64);
    let mut assembler = UtteranceAssembler::new();
    let mut timings = StageTimings::default();
    let mut collected: Vec<AudioSegment> = Vec::new();
    let mut fed_samples: usize = 0;

    for chunk_samples in audio.samples.chunks(VAD_FRAME) {
        if chunk_samples.len() < VAD_FRAME {
            break;
        }
        fed_samples += VAD_FRAME;
        assembler.push(chunk_samples);
        let prob = vad.score_frame(chunk_samples)?;
        for ev in detector.feed(prob) {
            match ev {
                parrots_vad::VadEvent::SpeechStarted => assembler.on_speech_start(),
                parrots_vad::VadEvent::SpeechEnded { end_sample, .. } => {
                    let Some(utt) = assembler.on_speech_end(end_sample) else {
                        continue;
                    };
                    let utt = utt.with_lang(args.from);
                    for event in pipeline
                        .process_utterance(&utt, &mut voiceprint, &mut timings)
                        .await?
                    {
                        if let PipelineEvent::SynthAudio(a) = event {
                            collected.push(a);
                        }
                    }
                }
            }
        }
    }

    // File input is a finite stream: if speech runs to the end, the detector never emits SpeechEnded, so force a flush at EOF
    if detector.in_speech() {
        if let Some(utt) = assembler.on_speech_end(fed_samples as u64) {
            let utt = utt.with_lang(args.from);
            for event in pipeline
                .process_utterance(&utt, &mut voiceprint, &mut timings)
                .await?
            {
                if let PipelineEvent::SynthAudio(a) = event {
                    collected.push(a);
                }
            }
        }
    }

    let sr = collected.first().map(|a| a.sample_rate).unwrap_or(24000);
    let merged: Vec<f32> = collected
        .iter()
        .flat_map(|a| a.samples.iter().copied())
        .collect();
    wire::write_wav(&args.out, &AudioSegment::new(merged, sr))?;

    let report = serde_json::json!({
        "utterances": timings.e2e_ms.len(),
        "asr_ms_mean": StageTimings::mean(&timings.asr_ms),
        "mt_ms_mean": StageTimings::mean(&timings.mt_ms),
        "tts_ms_mean": StageTimings::mean(&timings.tts_ms),
        "polish_ms_mean": StageTimings::mean(&timings.polish_ms),
        "polish_ms_mean": StageTimings::mean(&timings.polish_ms),
        "e2e_ms_mean": StageTimings::mean(&timings.e2e_ms),
        "first_audio_ms_mean": StageTimings::mean(&timings.first_audio_ms),
    });
    std::fs::write(&args.report, serde_json::to_string_pretty(&report)?)?;
    println!("{}", timings.report());
    println!("Translation audio → {}", args.out.display());
    Ok(())
}

pub struct TalkArgs {
    /// Input wav: Some = fixture mode; None = live microphone mode
    pub input: Option<PathBuf>,
    pub from: Lang,
    pub to: Lang,
    /// Voice profile name (profiles/<name>.{wav,txt}, enroll first with parrots enroll)
    pub voice: String,
    pub out: Option<PathBuf>,
    pub report: Option<PathBuf>,
    /// Output device preference: Some = strict; None = prefer the virtual microphone (Parrots Microphone)
    pub output_pref: Option<String>,
    /// Incremental clause pipeline (default on); false = whole-utterance mode (existing behavior, for comparison)
    pub incremental: bool,
    pub asr: wire::AsrChoice,
    /// Echo cancellation: capture goes through system VPIO; auto-falls back to plain capture + time-based gate on failure
    pub aec: bool,
    /// Text polish layer (plan 7): fixture auto = on, live auto = off
    pub polish: wire::PolishChoice,
    /// Playback gate (B4): None = auto (off when AEC is active, on otherwise); Some = manual override
    pub gate_playback: Option<bool>,
}

/// Direction A (I speak → the other side hears): voiceprint fixed to the user's pre-enrolled profile, cross-lingual cloning
pub async fn run_talk(args: TalkArgs) -> anyhow::Result<()> {
    let engine = wire::build_engine(args.asr)?;
    let src = engine
        .pack(args.from)
        .with_context(|| format!("{:?} language pack missing", args.from))?;
    let translator = engine
        .translator(args.from, args.to)
        .with_context(|| format!("{:?}→{:?} translator missing", args.from, args.to))?;
    let voice = wire::load_voice(&args.voice)?;
    let hotwords = hotwords_for_pipeline();
    match args.input.clone() {
        Some(input) => {
            let pipeline = DirectionAPipeline {
                asr: src.asr.clone(),
                translator,
                tts: src.tts.clone(),
                source_lang: args.from,
                target_lang: args.to,
                hotwords: hotwords.clone(),
                polisher: wire::build_polisher(args.polish, true),
            };
            talk_fixture(&args, &pipeline, &voice, input).await
        }
        None => {
            let sink_device = wire::resolve_talk_output_device(args.output_pref.as_deref());
            if args.incremental {
                // Device hot-switching (plan 3+ Phase B): capture on the default microphone, output rebuilt per preference
                let resolve_output = {
                    let pref = args.output_pref.clone();
                    std::sync::Arc::new(move || wire::resolve_talk_output_device(pref.as_deref()))
                };
                let recovery = crate::live_inc::DeviceRecovery {
                    initial_input: DeviceId(None),
                    // Migrate capture whenever the default input device changes (e.g. Bluetooth headphones connect)
                    resolve_input: std::sync::Arc::new(|| DeviceId(None)),
                    resolve_output,
                };
                // live: auto = off (preserves latency); with --polish on, enabled with an 800ms bound
                let polisher = wire::build_polisher(args.polish, false);
                talk_live_incremental(
                    args.from,
                    args.asr,
                    src.asr.clone(),
                    translator,
                    src.tts.clone(),
                    voice,
                    sink_device,
                    hotwords,
                    recovery,
                    args.aec,
                    polisher,
                    args.gate_playback,
                )
                .await
            } else {
                let pipeline = DirectionAPipeline {
                    asr: src.asr.clone(),
                    translator,
                    tts: src.tts.clone(),
                    source_lang: args.from,
                    target_lang: args.to,
                    hotwords,
                    // Comparison mode: whole-utterance live has no polish layer
                    polisher: None,
                };
                talk_live(args.from, &pipeline, &voice, sink_device).await
            }
        }
    }
}

/// Pipeline hotword correction layer (plan 6): enabled only when the word list is non-empty.
pub fn hotwords_for_pipeline() -> Option<std::sync::Arc<parrots_pipeline::Hotwords>> {
    let hw = wire::load_hotwords();
    if hw.is_empty() {
        None
    } else {
        Some(std::sync::Arc::new(hw))
    }
}

/// Incremental talk live: shares the two-stage pipeline with direction B live; the voiceprint is the fixed personal profile
#[allow(clippy::too_many_arguments)]
async fn talk_live_incremental(
    from: Lang,
    asr_choice: wire::AsrChoice,
    asr: std::sync::Arc<dyn parrots_core::AsrEngine>,
    translator: std::sync::Arc<dyn parrots_core::Translator>,
    tts: std::sync::Arc<dyn parrots_core::Synthesizer>,
    voice: VoiceProfile,
    sink_device: DeviceId,
    hotwords: Option<std::sync::Arc<parrots_pipeline::Hotwords>>,
    recovery: crate::live_inc::DeviceRecovery,
    aec: bool,
    polisher: Option<std::sync::Arc<dyn parrots_core::TextPolisher>>,
    gate_pref: Option<bool>,
) -> anyhow::Result<()> {
    let platform = MacAudioPlatform::new();
    // Capture source: default to system VPIO echo cancellation (so the other side's speech over
    // speakers is not mistakenly transcribed); on init failure fall back to plain capture +
    // the existing time-based gate (belt and braces)
    let (capture, aec_active): (Box<dyn parrots_core::AudioStream>, bool) = if aec {
        match parrots_platform_macos::aec::AecCapture::open() {
            Ok(c) => {
                tracing::info!("Echo-cancelled capture (VPIO) enabled");
                (Box::new(c), true)
            }
            Err(e) => {
                tracing::warn!(
                    "VPIO init failed ({e}), falling back to plain capture + time-based gate"
                );
                (platform.open_capture(&DeviceId(None))?, false)
            }
        }
    } else {
        (platform.open_capture(&DeviceId(None))?, false)
    };
    // Playback gate (B4, plan 5 leftover): auto = off when AEC is active (capture interruptions
    // disappear with it), on otherwise; --gate-playback true/false overrides manually
    // (can force it on until no echo loop is confirmed on real hardware)
    let gate_playback = gate_pref.unwrap_or(!aec_active);
    tracing::info!(
        "Playback gate: {} (AEC: {})",
        if gate_playback { "on" } else { "off" },
        if aec_active { "VPIO" } else { "none" }
    );
    let sink = platform.open_playback(&sink_device)?;
    let vad = SileroVad::load(&wire::models_root().join("vad/silero_vad.onnx"))?;
    let voice_source = crate::live_inc::VoiceSource::Fixed(voice);
    println!("Live conversation mode (incremental): speak {from:?} into the microphone, clause translations play as you speak; Ctrl-C to exit");
    let outcome = crate::live_inc::run_incremental_live(
        capture,
        sink,
        vad,
        crate::live_inc::IncrementalArgs {
            asr,
            translator,
            tts,
            voice_source,
            from_lang: from,
            hotwords,
            tick_interval_samples: wire::incremental_tick_interval(asr_choice),
            max_seconds: None,
            collect: false,
            gate_playback: true,
            recovery: Some(recovery),
            polisher,
        },
    )
    .await?;
    println!("{}", outcome.timings.report());
    Ok(())
}

/// Fixture mode: wav in → VAD segmentation → translation + fixed-voiceprint synthesis → optional wav/report out
async fn talk_fixture(
    args: &TalkArgs,
    pipeline: &DirectionAPipeline,
    voice: &VoiceProfile,
    input: PathBuf,
) -> anyhow::Result<()> {
    let audio = wire::read_wav_mono(&input)?.with_lang(args.from);
    let mut vad = SileroVad::load(&wire::models_root().join("vad/silero_vad.onnx"))?;
    let mut detector = SpeechDetector::new(VadConfig::default(), VAD_FRAME as u64);
    let mut assembler = UtteranceAssembler::new();
    let mut timings = StageTimings::default();
    let mut collected: Vec<AudioSegment> = Vec::new();
    let mut fed_samples: usize = 0;

    for chunk_samples in audio.samples.chunks(VAD_FRAME) {
        if chunk_samples.len() < VAD_FRAME {
            break;
        }
        fed_samples += VAD_FRAME;
        assembler.push(chunk_samples);
        let prob = vad.score_frame(chunk_samples)?;
        for ev in detector.feed(prob) {
            match ev {
                parrots_vad::VadEvent::SpeechStarted => assembler.on_speech_start(),
                parrots_vad::VadEvent::SpeechEnded { end_sample, .. } => {
                    let Some(utt) = assembler.on_speech_end(end_sample) else {
                        continue;
                    };
                    let utt = utt.with_lang(args.from);
                    collect_synth(pipeline, &utt, voice, &mut timings, &mut collected).await?;
                }
            }
        }
    }

    // File input is a finite stream: if speech runs to the end, the detector never emits SpeechEnded, so force a flush at EOF
    if detector.in_speech() {
        if let Some(utt) = assembler.on_speech_end(fed_samples as u64) {
            let utt = utt.with_lang(args.from);
            collect_synth(pipeline, &utt, voice, &mut timings, &mut collected).await?;
        }
    }

    if let Some(out) = &args.out {
        let sr = collected.first().map(|a| a.sample_rate).unwrap_or(24000);
        let merged: Vec<f32> = collected
            .iter()
            .flat_map(|a| a.samples.iter().copied())
            .collect();
        wire::write_wav(out, &AudioSegment::new(merged, sr))?;
    }

    if let Some(report_path) = &args.report {
        let report = serde_json::json!({
            "utterances": timings.e2e_ms.len(),
            "asr_ms_mean": StageTimings::mean(&timings.asr_ms),
            "mt_ms_mean": StageTimings::mean(&timings.mt_ms),
            "tts_ms_mean": StageTimings::mean(&timings.tts_ms),
        "polish_ms_mean": StageTimings::mean(&timings.polish_ms),
            "e2e_ms_mean": StageTimings::mean(&timings.e2e_ms),
            "first_audio_ms_mean": StageTimings::mean(&timings.first_audio_ms),
        });
        std::fs::write(report_path, serde_json::to_string_pretty(&report)?)?;
    }

    println!("{}", timings.report());
    if let Some(out) = &args.out {
        println!("Translation audio → {}", out.display());
    }
    Ok(())
}

/// Live mode: microphone capture → translation + fixed-voiceprint synthesis → playback on the given output device
async fn talk_live(
    from: Lang,
    pipeline: &DirectionAPipeline,
    voice: &VoiceProfile,
    sink_device: DeviceId,
) -> anyhow::Result<()> {
    let platform = MacAudioPlatform::new();
    let mut capture = platform.open_capture(&DeviceId(None))?;
    let mut sink = platform.open_playback(&sink_device)?;
    let mut vad = SileroVad::load(&wire::models_root().join("vad/silero_vad.onnx"))?;
    // Live tuning: trailing silence 30 frames (480ms) → 40 frames (640ms), so fast speech is less likely to be chopped at short pauses
    let mut detector = SpeechDetector::new(
        VadConfig {
            end_frames: 40,
            ..Default::default()
        },
        VAD_FRAME as u64,
    );
    let mut assembler = UtteranceAssembler::new();
    let mut timings = StageTimings::default();
    // Playback gate (echo-loop mitigation, design doc §3.3): the translation actually leaves
    // the speaker within seconds after write() (the queue drains in real time), so the gate
    // must be timed as "written sample duration + 300ms margin", not just the write() instant
    let loop_t0 = std::time::Instant::now();
    let now_ms = || loop_t0.elapsed().as_millis() as u64;
    let gate_until_ms = std::sync::atomic::AtomicU64::new(0);
    println!("Live conversation mode: speak {from:?} into the microphone, translations play from the speakers; Ctrl-C to exit");
    // Known tradeoff: serial design; during synthesis the capture ring buffer overwrites old data (speech in that window is dropped)
    loop {
        let chunk = capture.next_chunk().context("capture stream ended")?;
        if now_ms() < gate_until_ms.load(std::sync::atomic::Ordering::Relaxed) {
            continue; // playing (incl. 300ms acoustic margin): drop the whole frame (no scoring, no feeding)
        }
        assembler.push(&chunk);
        let prob = vad.score_frame(&chunk)?;
        for ev in detector.feed(prob) {
            match ev {
                parrots_vad::VadEvent::SpeechStarted => {
                    assembler.on_speech_start();
                    println!("Speech detected...")
                }
                parrots_vad::VadEvent::SpeechEnded { end_sample, .. } => {
                    let Some(utt) = assembler.on_speech_end(end_sample) else {
                        continue;
                    };
                    let utt = utt.with_lang(from);
                    let mut played_samples: usize = 0;
                    let mut sink_rate = 48000u32;
                    for event in pipeline
                        .process_utterance(&utt, voice, &mut timings)
                        .await?
                    {
                        match event {
                            PipelineEvent::Transcribed(t) => println!("Recognized: {t}"),
                            PipelineEvent::Translated(t) => println!("Translation: {t}"),
                            PipelineEvent::SynthAudio(a) => {
                                let resampled = parrots_platform_macos::resample::linear_resample(
                                    &a.samples,
                                    a.sample_rate,
                                    sink.sample_rate(),
                                );
                                sink_rate = sink.sample_rate();
                                played_samples += resampled.len();
                                sink.write(&resampled)?;
                            }
                        }
                    }
                    // Playback duration = written samples / sample rate, plus a 300ms acoustic margin (reverb/tail)
                    if played_samples > 0 {
                        let play_ms = played_samples as f64 / sink_rate as f64 * 1000.0 + 300.0;
                        gate_until_ms.store(
                            now_ms() + play_ms as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    println!("{}", timings.report());
                }
            }
        }
    }
}

/// Process one utterance, printing transcript/translation for manual review and collecting synthesized audio for merging
async fn collect_synth(
    pipeline: &DirectionAPipeline,
    utt: &AudioSegment,
    voice: &VoiceProfile,
    timings: &mut StageTimings,
    collected: &mut Vec<AudioSegment>,
) -> anyhow::Result<()> {
    for event in pipeline.process_utterance(utt, voice, timings).await? {
        match event {
            PipelineEvent::Transcribed(t) => println!("Recognized: {t}"),
            PipelineEvent::Translated(t) => println!("Translation: {t}"),
            PipelineEvent::SynthAudio(a) => collected.push(a),
        }
    }
    Ok(())
}

pub fn parse_lang(s: &str) -> anyhow::Result<Lang> {
    Lang::parse(s).with_context(|| format!("unknown language {s}"))
}
