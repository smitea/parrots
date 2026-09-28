mod doctor;
mod enroll;
mod live_inc;
mod run;
mod wire;

use anyhow::Context;
use clap::{Parser, Subcommand};
use parrots_core::{AudioPlatform, AudioSegment, DeviceId, Lang, SpeechDetector, VadConfig};
use parrots_pipeline::{
    DirectionBPipeline, PipelineEvent, RollingVoiceprint, StageTimings, UtteranceAssembler,
    UtteranceRollingVoiceprint,
};
use parrots_platform_macos::MacAudioPlatform;
use parrots_vad::{SileroVad, VAD_FRAME};
use run::TranslateArgs;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "parrots",
    about = "Parrots real-time translation CLI (plan 1: direction B)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enroll a voice profile: read the prompt sentence aloud, written into profiles/
    Enroll {
        #[arg(long, default_value = "my")]
        name: String,
        #[arg(long, default_value = "60")]
        max_seconds: u64,
    },
    /// Add a hotword (plan 6): appends to profiles/hotwords.txt,
    /// enabling ASR homophone auto-correction + whisper decoding bias
    Hotword {
        /// Hotword spelling (e.g. 西游记)
        word: String,
    },
    /// Translate a wav file, outputting a translation wav + latency report
    Translate {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "en")]
        from: String,
        #[arg(long, default_value = "zh")]
        to: String,
        #[arg(long, default_value = "translated.wav")]
        out: PathBuf,
        #[arg(long, default_value = "report.json")]
        report: PathBuf,
        /// Voiceprint prompt audio (defaults to the first utterance itself)
        #[arg(long)]
        voice: Option<PathBuf>,
        /// Voiceprint prompt transcript (paired with --voice)
        #[arg(long)]
        voice_text: Option<String>,
        /// ASR engine (default sensevoice; whisper is the fallback)
        #[arg(long, value_enum, default_value_t = wire::AsrChoice::default())]
        asr: wire::AsrChoice,
        /// Text polish layer (plan 7): auto = on (auto-degrades when the model is missing)
        #[arg(long, value_enum, default_value_t = wire::PolishChoice::default())]
        polish: wire::PolishChoice,
    },
    /// Live microphone → translated speech playback
    Live {
        #[arg(long, default_value = "en")]
        from: String,
        #[arg(long, default_value = "zh")]
        to: String,
        /// Capture device name (default = prefer Parrots Speakers, fall back to the default microphone if missing)
        #[arg(long)]
        device: Option<String>,
        /// Optional: path to write the merged translation wav on exit (for end-to-end verification)
        #[arg(long)]
        out: Option<PathBuf>,
        /// Optional: path to write the latency report json on exit (for end-to-end verification)
        #[arg(long)]
        report: Option<PathBuf>,
        /// Optional: exit cleanly after running N seconds (default = unlimited, until Ctrl-C)
        #[arg(long)]
        max_seconds: Option<u64>,
        /// Incremental clause pipeline: translate clause-by-clause while speaking (default on);
        /// `--incremental false` falls back to whole-utterance mode (existing behavior, for comparison)
        #[arg(long, default_value = "true", num_args = 0..=1, default_missing_value = "true")]
        incremental: bool,
        /// Voiceprint update granularity (incremental mode only): default updates per clause (adapts fast);
        /// `--voice-utterance true` updates only when a whole utterance ends (more stable timbre,
        /// clauses reuse the previous profile; for voice-clone quality A/B comparison)
        #[arg(long, default_value = "false", num_args = 0..=1, default_missing_value = "true")]
        voice_utterance: bool,
        /// ASR engine (default sensevoice; whisper is the fallback)
        #[arg(long, value_enum, default_value_t = wire::AsrChoice::default())]
        asr: wire::AsrChoice,
        /// Text polish layer (plan 7): in live mode auto = off (preserves latency)
        #[arg(long, value_enum, default_value_t = wire::PolishChoice::default())]
        polish: wire::PolishChoice,
    },
    /// Live conversation (direction A): fixed personal voiceprint, speak {from}, play {to}
    Talk {
        /// Input wav (default = live microphone mode)
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long, default_value = "zh")]
        from: String,
        #[arg(long, default_value = "en")]
        to: String,
        /// Voice profile name (enroll first with parrots enroll)
        #[arg(long, default_value = "my")]
        voice: String,
        /// Fixture mode: translation wav output path
        #[arg(long)]
        out: Option<PathBuf>,
        /// Fixture mode: latency report output path
        #[arg(long)]
        report: Option<PathBuf>,
        /// Output device name (default = prefer Parrots Microphone, fall back to the default speaker if missing; live mode only)
        #[arg(long)]
        device: Option<String>,
        /// Incremental clause pipeline: translate clause-by-clause while speaking (default on);
        /// `--incremental false` falls back to whole-utterance mode (existing behavior, for comparison)
        #[arg(long, default_value = "true", num_args = 0..=1, default_missing_value = "true")]
        incremental: bool,
        /// ASR engine (default sensevoice; whisper is the fallback)
        #[arg(long, value_enum, default_value_t = wire::AsrChoice::default())]
        asr: wire::AsrChoice,
        /// Echo cancellation (live capture goes through system VPIO; default on,
        /// auto-falls back to plain capture + time-based gate if initialization fails)
        #[arg(long, default_value = "true", num_args = 0..=1, default_missing_value = "true")]
        aec: bool,
        /// Text polish layer (plan 7): in live mode auto = off (preserves latency)
        #[arg(long, value_enum, default_value_t = wire::PolishChoice::default())]
        polish: wire::PolishChoice,
        /// Playback gate / echo-loop mitigation (B4): auto = off when AEC is active, on otherwise;
        /// true/false overrides manually (defaults to off once no echo loop is confirmed on real hardware)
        #[arg(long, num_args = 0..=1, default_missing_value = "true")]
        gate_playback: Option<bool>,
    },
    /// Benchmark: same path as translate, with a detailed latency report
    Bench {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "en")]
        from: String,
        #[arg(long, default_value = "zh")]
        to: String,
        /// ASR engine (default sensevoice; whisper is the fallback)
        #[arg(long, value_enum, default_value_t = wire::AsrChoice::default())]
        asr: wire::AsrChoice,
    },
    /// Environment health check: default devices / virtual driver / models / voiceprint readiness
    Doctor,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    match Cli::parse().cmd {
        Cmd::Translate {
            input,
            from,
            to,
            out,
            report,
            voice,
            voice_text,
            asr,
            polish,
        } => {
            let args = TranslateArgs {
                input,
                from: run::parse_lang(&from)?,
                to: run::parse_lang(&to)?,
                out,
                report,
                voice,
                voice_text,
                asr,
                polish,
            };
            run::run_translate(args).await
        }
        Cmd::Enroll { name, max_seconds } => enroll::run_enroll(&name, max_seconds),
        Cmd::Hotword { word } => {
            let total = wire::add_hotword(word.trim())?;
            println!("Hotword added: {word} ({total} total in the list)");
            Ok(())
        }
        Cmd::Bench {
            input,
            from,
            to,
            asr,
        } => {
            let args = TranslateArgs {
                input,
                from: run::parse_lang(&from)?,
                to: run::parse_lang(&to)?,
                out: PathBuf::from("/tmp/parrots-bench.wav"),
                report: PathBuf::from("/tmp/parrots-bench-report.json"),
                voice: None,
                voice_text: None,
                asr,
                // Benchmark = raw pipeline, no polish layer attached
                polish: wire::PolishChoice::Off,
            };
            run::run_translate(args).await
        }
        Cmd::Talk {
            input,
            from,
            to,
            voice,
            out,
            report,
            device,
            incremental,
            asr,
            aec,
            polish,
            gate_playback,
        } => {
            let args = run::TalkArgs {
                input,
                from: run::parse_lang(&from)?,
                to: run::parse_lang(&to)?,
                voice,
                out,
                report,
                output_pref: device,
                incremental,
                asr,
                aec,
                polish,
                gate_playback,
            };
            run::run_talk(args).await
        }
        Cmd::Live {
            from,
            to,
            device,
            out,
            report,
            max_seconds,
            incremental,
            voice_utterance,
            asr,
            polish,
        } => {
            let from = run::parse_lang(&from)?;
            let to = run::parse_lang(&to)?;
            if incremental {
                live_incremental(
                    from,
                    to,
                    device,
                    out,
                    report,
                    max_seconds,
                    asr,
                    voice_utterance,
                    polish,
                )
                .await
            } else {
                let capture = wire::resolve_input_device(device.as_deref());
                live(from, to, capture, out, report, max_seconds, asr).await
            }
        }
        Cmd::Doctor => {
            if doctor::run()? {
                Ok(())
            } else {
                std::process::exit(1);
            }
        }
    }
}

async fn live(
    from: Lang,
    to: Lang,
    capture_device: DeviceId,
    out: Option<PathBuf>,
    report: Option<PathBuf>,
    max_seconds: Option<u64>,
    asr: wire::AsrChoice,
) -> anyhow::Result<()> {
    let engine = wire::build_engine(asr)?;
    let src = engine.pack(from).context("language pack missing")?;
    let translator = engine.translator(from, to).context("translator missing")?;
    let pipeline = DirectionBPipeline {
        asr: src.asr.clone(),
        translator,
        tts: src.tts.clone(),
        source_lang: from,
        target_lang: to,
        hotwords: run::hotwords_for_pipeline(),
        // Comparison mode: whole-utterance live has no polish layer
        polisher: None,
    };
    let platform = MacAudioPlatform::new();
    let mut capture = platform.open_capture(&capture_device)?;
    let mut sink = platform.open_playback(&DeviceId(None))?;
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
    let tmp = tempfile::tempdir()?;
    let mut voiceprint = RollingVoiceprint::new(tmp.path(), 2)?;
    let mut timings = StageTimings::default();
    let mut collected: Vec<AudioSegment> = Vec::new();
    tracing::info!("Live mode: speak {from:?} into the microphone, translations play from the speakers; Ctrl-C to exit");
    // The auto-exit timer starts at the capture loop: model loading does not consume the max_seconds window (aligns the e2e playback window)
    let start = std::time::Instant::now();
    // Known tradeoff: during synthesis the capture ring buffer overwrites old data (speech in that window is dropped); v1 serial design, task pipelining later
    loop {
        if max_seconds.is_some_and(|n| start.elapsed() >= std::time::Duration::from_secs(n)) {
            break;
        }
        let chunk = capture.next_chunk().context("capture stream ended")?;
        assembler.push(&chunk);
        let prob = vad.score_frame(&chunk)?;
        for ev in detector.feed(prob) {
            match ev {
                parrots_vad::VadEvent::SpeechStarted => {
                    assembler.on_speech_start();
                    tracing::info!("Speech detected...")
                }
                parrots_vad::VadEvent::SpeechEnded { end_sample, .. } => {
                    let Some(utt) = assembler.on_speech_end(end_sample) else {
                        continue;
                    };
                    let utt = utt.with_lang(from);
                    let events = pipeline
                        .process_utterance(&utt, &mut voiceprint, &mut timings)
                        .await?;
                    for event in events {
                        if let PipelineEvent::SynthAudio(a) = event {
                            let resampled = parrots_platform_macos::resample::linear_resample(
                                &a.samples,
                                a.sample_rate,
                                sink.sample_rate(),
                            );
                            sink.write(&resampled)?;
                            // Long-running mode without --out does not collect, avoiding unbounded memory growth
                            if out.is_some() {
                                collected.push(a);
                            }
                        }
                    }
                    tracing::info!("{}", timings.report());
                }
            }
        }
    }

    if let Some(out) = &out {
        let sr = collected.first().map(|a| a.sample_rate).unwrap_or(24000);
        let merged: Vec<f32> = collected
            .iter()
            .flat_map(|a| a.samples.iter().copied())
            .collect();
        wire::write_wav(out, &AudioSegment::new(merged, sr))?;
        println!("Translation audio → {}", out.display());
    }
    if let Some(report) = &report {
        let rep = serde_json::json!({
            "utterances": timings.e2e_ms.len(),
            "asr_ms_mean": StageTimings::mean(&timings.asr_ms),
            "mt_ms_mean": StageTimings::mean(&timings.mt_ms),
            "tts_ms_mean": StageTimings::mean(&timings.tts_ms),
            "e2e_ms_mean": StageTimings::mean(&timings.e2e_ms),
            "first_audio_ms_mean": StageTimings::mean(&timings.first_audio_ms),
        });
        std::fs::write(report, serde_json::to_string_pretty(&rep)?)?;
    }
    Ok(())
}

/// Incremental mode: two-stage pipeline of capture/segmentation + clause worker pool (shared by direction B live and talk_live)
/// Device hot-switching (plan 3+ Phase B): automatically rebuilds the stream on plug/unplug or Bluetooth changes; the current utterance is dropped.
#[allow(clippy::too_many_arguments)]
async fn live_incremental(
    from: Lang,
    to: Lang,
    device_pref: Option<String>,
    out: Option<PathBuf>,
    report: Option<PathBuf>,
    max_seconds: Option<u64>,
    asr: wire::AsrChoice,
    voice_utterance: bool,
    polish: wire::PolishChoice,
) -> anyhow::Result<()> {
    let engine = wire::build_engine(asr)?;
    let src = engine.pack(from).context("language pack missing")?;
    let translator = engine.translator(from, to).context("translator missing")?;
    let platform = MacAudioPlatform::new();
    let capture_device = wire::resolve_input_device(device_pref.as_deref());
    let capture = platform.open_capture(&capture_device)?;
    let vad = SileroVad::load(&wire::models_root().join("vad/silero_vad.onnx"))?;
    let sink = platform.open_playback(&DeviceId(None))?;
    let tmp = tempfile::tempdir()?;
    let voice_source = if voice_utterance {
        live_inc::VoiceSource::RollingUtterance(Arc::new(std::sync::Mutex::new(
            UtteranceRollingVoiceprint::new(tmp.path(), 2)?,
        )))
    } else {
        live_inc::VoiceSource::Rolling(Arc::new(std::sync::Mutex::new(RollingVoiceprint::new(
            tmp.path(),
            2,
        )?)))
    };
    let hotwords = run::hotwords_for_pipeline();
    let resolve_input = {
        let pref = device_pref.clone();
        Arc::new(move || wire::resolve_input_device(pref.as_deref()))
    };
    let recovery = live_inc::DeviceRecovery {
        initial_input: capture_device,
        resolve_input,
        // Live translation playback uses the system default speaker: on write failure rebuild with the default output (headphone plug/unplug migration)
        resolve_output: Arc::new(|| DeviceId(None)),
    };
    tracing::info!(
        "Incremental mode: speak {from:?}, translations play clause-by-clause; Ctrl-C to exit"
    );
    let outcome = live_inc::run_incremental_live(
        capture,
        sink,
        vad,
        live_inc::IncrementalArgs {
            asr: src.asr.clone(),
            translator,
            tts: src.tts.clone(),
            voice_source,
            from_lang: from,
            hotwords,
            tick_interval_samples: wire::incremental_tick_interval(asr),
            max_seconds,
            collect: out.is_some(),
            gate_playback: true,
            recovery: Some(recovery),
            // live: auto = off (preserves latency); with --polish on, enabled with an 800ms bound
            polisher: wire::build_polisher(polish, false),
        },
    )
    .await?;
    tracing::info!("{}", outcome.timings.report());
    if let Some(out) = &out {
        let sr = outcome
            .collected
            .first()
            .map(|a| a.sample_rate)
            .unwrap_or(24000);
        let merged: Vec<f32> = outcome
            .collected
            .iter()
            .flat_map(|a| a.samples.iter().copied())
            .collect();
        wire::write_wav(out, &AudioSegment::new(merged, sr))?;
        println!("Translation audio → {}", out.display());
    }
    if let Some(report) = &report {
        let t = &outcome.timings;
        let rep = serde_json::json!({
            "utterances": t.e2e_ms.len(),
            "clauses": t.e2e_ms.len(),
            "asr_ms_mean": StageTimings::mean(&t.asr_ms),
            "mt_ms_mean": StageTimings::mean(&t.mt_ms),
            "tts_ms_mean": StageTimings::mean(&t.tts_ms),
            "polish_ms_mean": StageTimings::mean(&t.polish_ms),
            "e2e_ms_mean": StageTimings::mean(&t.e2e_ms),
            "first_audio_ms_mean": StageTimings::mean(&t.first_audio_ms),
        });
        std::fs::write(report, serde_json::to_string_pretty(&rep)?)?;
    }
    Ok(())
}
