//! Incremental clause live pipeline (two stages: capture/segmentation + clause worker pool).
//!
//! - Capture thread (thread A): blocking-pull 16kHz chunks → VAD scoring → events, producing
//!   (chunk, events); whole frames are dropped during playback-gate windows (same echo-loop
//!   mitigation as the whole-utterance mode).
//! - Segmentation task: rolling buffer + rolling ASR once per tick ([`IncrementalSegmenter`]),
//!   submits clauses → bounded channel (depth 2, natural backpressure; submission order =
//!   playback order).
//! - Clause worker pool: a single-consumer task runs MT→TTS→sink in submission order
//!   (single-consumer queues are naturally ordered, playback never interleaves); after writing
//!   it updates the gate as "total remaining duration of unplayed clauses + 300ms margin".

use cpal::traits::{DeviceTrait, HostTrait};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use parrots_core::{
    AsrEngine, AudioPlatform, AudioSegment, AudioSink, AudioStream, DeviceId, Result,
    SpeechDetector, Synthesizer, Translator, VadConfig, VoiceProfile,
};
use parrots_pipeline::{
    ClauseDecision, IncrementalSegmenter, RollingVoiceprint, StageTimings,
    UtteranceRollingVoiceprint,
};
use parrots_vad::{SileroVad, VadEvent, VAD_FRAME};
use tokio::sync::mpsc;

/// Acoustic margin for the playback gate (reverb/tail), same as the whole-utterance mode
const GATE_TAIL_MS: u64 = 300;
/// Capture→segmentation channel depth (512 samples/chunk ≈ 32ms, 32 chunks ≈ 1s buffer)
const CAPTURE_CHANNEL_DEPTH: usize = 32;
/// Clause channel depth = max clauses in flight in the pipeline: single-consumer serial, ≤2 in flight
const CLAUSE_CHANNEL_DEPTH: usize = 2;
/// Live trailing-silence threshold 40 frames (640ms), same as the whole-utterance mode
const END_FRAMES: u32 = 40;
/// Device self-healing: max stream-rebuild retries (1s apart; consecutive failures exit with an error)
const RECOVERY_MAX_ATTEMPTS: u32 = 5;
/// Device self-healing: rebuild interval
const RECOVERY_BACKOFF: Duration = Duration::from_secs(1);

/// TTS voiceprint source: direction A = fixed profile; direction B = the other side's rolling voiceprint.
/// Rolling updates per clause (1~3s short prompt); RollingUtterance updates only when a whole
/// utterance ends (clauses reuse the latest utterance profile, more stable timbre; see plan 5 leftover B3).
pub enum VoiceSource {
    Fixed(VoiceProfile),
    Rolling(Arc<Mutex<RollingVoiceprint>>),
    RollingUtterance(Arc<Mutex<UtteranceRollingVoiceprint>>),
}

impl VoiceSource {
    async fn voice_for(
        &self,
        audio: &AudioSegment,
        text: &str,
        is_final: bool,
    ) -> Result<VoiceProfile> {
        match self {
            VoiceSource::Fixed(v) => Ok(v.clone()),
            VoiceSource::Rolling(vp) => {
                let mut vp = vp.lock().unwrap_or_else(|e| e.into_inner());
                vp.update(audio, text)
            }
            VoiceSource::RollingUtterance(vp) => {
                let mut vp = vp.lock().unwrap_or_else(|e| e.into_inner());
                vp.update(audio, text, is_final)
            }
        }
    }
}

/// Message from the capture thread to the segmentation task.
enum CaptureMsg {
    Chunk {
        samples: Vec<f32>,
        /// Capture time (ms relative to pipeline start)
        at_ms: u64,
        events: Vec<VadEvent>,
    },
    /// Capture device was rebuilt (hot-switch): the current utterance buffer is not comparable across devices, drop it
    Reset,
    Fatal(anyhow::Error),
    Eof,
}

/// Message from the segmentation task to the worker pool (submission order = playback order).
struct ClauseMsg {
    clause: ClauseDecision,
    /// Capture time of the clause's last sample (ms relative to pipeline start), for e2e/first-audio measurement
    spoken_at_ms: u64,
}

/// Device self-healing spec (plan 3+ Phase B): passed by real-device live; None for fixture/tests.
#[derive(Clone)]
pub struct DeviceRecovery {
    /// Initial capture device (already resolved by the caller)
    pub initial_input: DeviceId,
    /// Re-resolve the capture device (called on hot-switch; auto/explicit semantics are encapsulated in the closure)
    pub resolve_input: Arc<dyn Fn() -> DeviceId + Send + Sync>,
    /// Re-resolve the playback device
    pub resolve_output: Arc<dyn Fn() -> DeviceId + Send + Sync>,
}

/// Device logical identity: explicit name or system default device name (for hot-switch comparison)
fn device_identity(id: &DeviceId, is_input: bool) -> Option<String> {
    match id {
        DeviceId(Some(n)) => Some(n.clone()),
        DeviceId(None) => {
            let host = cpal::default_host();
            let d = if is_input {
                host.default_input_device()
            } else {
                host.default_output_device()
            };
            d.and_then(|d| d.name().ok())
        }
    }
}

fn identity_label(ident: &Option<String>) -> String {
    ident.clone().unwrap_or_else(|| "(system default)".into())
}

/// Incremental pipeline arguments.
pub struct IncrementalArgs {
    pub asr: Arc<dyn AsrEngine>,
    pub translator: Arc<dyn Translator>,
    pub tts: Arc<dyn Synthesizer>,
    pub voice_source: VoiceSource,
    /// Source language: lang_hint for the rolling ASR (whisper follows it strictly; when missing
    /// it decodes as En by default, so Chinese gets "translated" into English)
    pub from_lang: parrots_core::Lang,
    /// Hotword correction (plan 6): fixes homophone typos before clauses enter MT; None = off
    pub hotwords: Option<Arc<parrots_pipeline::Hotwords>>,
    /// Rolling ASR period (samples): SenseVoice 8000 (500ms) / whisper 16000 (1s)
    pub tick_interval_samples: usize,
    /// Exit cleanly after running N seconds (None = unlimited)
    pub max_seconds: Option<u64>,
    /// Collect synthesized audio (for --out persistence; long runs don't collect to avoid unbounded memory)
    pub collect: bool,
    /// Playback gate (echo-loop mitigation): no gate needed when fixture/benchmark injects a NullSink, set false
    pub gate_playback: bool,
    /// Device hot-switch self-healing (plan 3+ Phase B): None = no self-healing (fixture/tests)
    pub recovery: Option<DeviceRecovery>,
    /// Text polish layer (plan 7): after hotword correction, before MT; None = off (live defaults off)
    pub polisher: Option<Arc<dyn parrots_core::TextPolisher>>,
}

/// Incremental pipeline output (handed back to the caller on exit for persistence/reporting).
pub struct IncrementalOutcome {
    pub collected: Vec<AudioSegment>,
    pub timings: StageTimings,
}

/// Run the incremental clause pipeline until the stream ends / max_seconds elapses.
///
/// Voiceprint and printing are injected via [`VoiceSource`]; directions A/B share the same loop.
pub async fn run_incremental_live(
    capture: Box<dyn AudioStream>,
    sink: Box<dyn AudioSink>,
    mut vad: SileroVad,
    args: IncrementalArgs,
) -> anyhow::Result<IncrementalOutcome> {
    let t0 = Instant::now();
    let gate_until_ms = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let recovery = args.recovery;
    let out_recovery = recovery.clone();

    // ---- Device watcher (plan 3+ Phase B): topology changes set a flag, checked per chunk by the capture thread ----
    let device_changed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _watcher = recovery.as_ref().map(|_| {
        let flag = device_changed.clone();
        parrots_platform_macos::watch::DeviceWatcher::new(Arc::new(move || {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }))
    });

    // ---- Thread A: capture + VAD scoring + event production (blocking API, dedicated OS thread) ----
    let (ctx, mut crx) = mpsc::channel::<CaptureMsg>(CAPTURE_CHANNEL_DEPTH);
    let gate = gate_until_ms.clone();
    let max_seconds = args.max_seconds;
    let gate_playback = args.gate_playback;
    let capture_thread = std::thread::spawn(move || {
        let mut capture = capture;
        let platform = parrots_platform_macos::MacAudioPlatform::new();
        // Self-healing state: current capture device identity (explicit or default device name) and consecutive rebuild failures
        let mut cur_ident = recovery
            .as_ref()
            .map(|r| device_identity(&r.initial_input, true));
        let new_detector = || {
            SpeechDetector::new(
                VadConfig {
                    end_frames: END_FRAMES,
                    ..Default::default()
                },
                VAD_FRAME as u64,
            )
        };
        let mut detector = new_detector();

        // Hot-switch rebuild: re-resolve → reopen the stream; returns the new device identity on success, None after 5 consecutive failures
        let reopen_input = |capture: &mut Box<dyn AudioStream>| -> Option<Option<String>> {
            let rec = recovery.as_ref()?;
            for attempt in 1..=RECOVERY_MAX_ATTEMPTS {
                let dev = (rec.resolve_input)();
                match platform.open_capture(&dev) {
                    Ok(s) => {
                        *capture = s;
                        let ident = device_identity(&dev, true);
                        println!("[device] switched to {}", identity_label(&ident));
                        return Some(ident);
                    }
                    Err(e) => {
                        tracing::warn!("failed to rebuild capture stream ({attempt}/{RECOVERY_MAX_ATTEMPTS}): {e}");
                        std::thread::sleep(RECOVERY_BACKOFF);
                    }
                }
            }
            None
        };

        loop {
            if let Some(n) = max_seconds {
                if t0.elapsed() >= Duration::from_secs(n) {
                    let _ = ctx.blocking_send(CaptureMsg::Eof);
                    break;
                }
            }
            // Conditions 2/3: watcher topology change → current device gone / re-resolve points elsewhere → migrate
            if device_changed.swap(false, std::sync::atomic::Ordering::Relaxed) {
                if let (Some(rec), Some(ident)) = (&recovery, &cur_ident) {
                    let gone = ident
                        .as_deref()
                        .is_none_or(|n| parrots_platform_macos::find_device(n).is_none());
                    let fresh = device_identity(&(rec.resolve_input)(), true);
                    if gone || fresh != *ident {
                        tracing::info!(
                            "Device topology changed (gone={gone}): {} → {}",
                            identity_label(ident),
                            identity_label(&fresh)
                        );
                        if let Some(new_ident) = reopen_input(&mut capture) {
                            cur_ident = Some(new_ident);
                            // Audio across devices is not comparable: drop the current utterance (VAD state reset too)
                            detector = new_detector();
                            let _ = ctx.blocking_send(CaptureMsg::Reset);
                        }
                    }
                }
            }
            let Some(chunk) = capture.next_chunk() else {
                // Condition 1: stream ended (device unplugged/underlying stream died) → try rebuild; without self-healing this is a normal EOF
                if recovery.is_none() {
                    let _ = ctx.blocking_send(CaptureMsg::Eof);
                    break;
                }
                match reopen_input(&mut capture) {
                    Some(new_ident) => {
                        cur_ident = Some(new_ident);
                        detector = new_detector();
                        let _ = ctx.blocking_send(CaptureMsg::Reset);
                    }
                    None => {
                        let _ = ctx.blocking_send(CaptureMsg::Fatal(anyhow::anyhow!(
                            "capture device failed to rebuild {RECOVERY_MAX_ATTEMPTS} consecutive times, exiting"
                        )));
                        break;
                    }
                }
                continue;
            };
            if chunk.len() < VAD_FRAME {
                continue;
            }
            // Playing (incl. acoustic margin): drop the whole frame (no scoring, no feeding); same echo-loop mitigation as the whole-utterance mode
            if gate_playback
                && (t0.elapsed().as_millis() as u64)
                    < gate.load(std::sync::atomic::Ordering::Relaxed)
            {
                continue;
            }
            let prob = match vad.score_frame(&chunk) {
                Ok(p) => p,
                Err(e) => {
                    let _ = ctx.blocking_send(CaptureMsg::Fatal(anyhow::anyhow!(e)));
                    break;
                }
            };
            let events = detector.feed(prob);
            if ctx
                .blocking_send(CaptureMsg::Chunk {
                    samples: chunk,
                    at_ms: t0.elapsed().as_millis() as u64,
                    events,
                })
                .is_err()
            {
                break;
            }
        }
    });

    // ---- Segmentation task: rolling buffer + rolling ASR → clause submission ----
    let (wtx, mut wrx) = mpsc::channel::<ClauseMsg>(CLAUSE_CHANNEL_DEPTH);
    let asr_for_seg = args.asr.clone();
    let from_lang = args.from_lang;
    let seg_task = tokio::spawn(async move {
        // Rolling ASR bridge: the pure-logic segmenter requires a sync closure; SenseVoice/whisper's
        // transcribe is async → bridged with block_in_place + block_on (the segmentation task is
        // already the pipeline's slow stage; blocking it does not affect the capture thread or worker pool)
        let call_asr = move |samples: &[f32]| -> Result<String> {
            let seg = AudioSegment::new(samples.to_vec(), 16000).with_lang(from_lang);
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(asr_for_seg.transcribe(&seg))
                    .map(|t| t.text)
            })
        };

        let mut segmenter: Option<IncrementalSegmenter> = None;
        let mut since_tick = 0usize;
        let mut last_at_ms;
        while let Some(msg) = crx.recv().await {
            match msg {
                CaptureMsg::Fatal(e) => return Err(e),
                CaptureMsg::Eof => break,
                CaptureMsg::Reset => {
                    // Device hot-switch: the current utterance is not comparable across devices, drop it entirely
                    segmenter = None;
                    since_tick = 0;
                }
                CaptureMsg::Chunk {
                    samples,
                    at_ms,
                    events,
                } => {
                    last_at_ms = at_ms;
                    let mut seg = segmenter.take();
                    let mut ended: Option<(u64, u64)> = None;
                    for ev in events {
                        match ev {
                            VadEvent::SpeechStarted => {
                                seg = Some(IncrementalSegmenter::new(0));
                                since_tick = 0;
                            }
                            VadEvent::SpeechEnded {
                                start_sample,
                                end_sample,
                            } => {
                                ended = Some((start_sample, end_sample));
                            }
                        }
                    }
                    let Some(s) = seg.as_mut() else { continue };
                    s.push(&samples);
                    since_tick += samples.len();
                    let mut utterance_done = false;
                    let decision = if let Some((start, end)) = ended {
                        // VAD ended: submit all remaining tail audio (is_final)
                        utterance_done = true;
                        s.finish(&call_asr, end.saturating_sub(start))?
                    } else if since_tick >= args.tick_interval_samples {
                        since_tick = 0;
                        s.tick(&call_asr)?
                    } else {
                        None
                    };
                    if let Some(d) = decision {
                        if !d.text.trim().is_empty()
                            && wtx
                                .send(ClauseMsg {
                                    clause: d,
                                    spoken_at_ms: last_at_ms,
                                })
                                .await
                                .is_err()
                        {
                            return Ok(());
                        }
                    }
                    segmenter = if utterance_done { None } else { seg };
                }
            }
        }
        Ok(())
    });

    // ---- Clause worker pool: single consumer runs MT→TTS→sink in submission order ----
    let gate = gate_until_ms.clone();
    let collect = args.collect;
    let translator = args.translator.clone();
    let tts = args.tts.clone();
    let hotwords = args.hotwords.clone();
    let polisher = args.polisher.clone();
    let voice_source = Arc::new(args.voice_source);

    let worker_task = tokio::spawn(async move {
        let mut sink = sink;
        let out_platform = parrots_platform_macos::MacAudioPlatform::new();
        let mut timings = StageTimings::default();
        let mut collected: Vec<AudioSegment> = Vec::new();
        let mut first_audio_recorded = false;
        while let Some(msg) = wrx.recv().await {
            let ClauseDecision {
                audio: clause_audio,
                text,
                is_final,
            } = msg.clause;
            // Hotword correction (plan 6): rewrite homophone typos per the word list before MT/voiceprint
            let corrected = match &hotwords {
                Some(h) => h.correct(&text),
                None => text,
            };
            println!("Recognized: {corrected}");
            if corrected.trim().is_empty() {
                continue;
            }
            // Text polish layer (plan 7): after hotword correction, before MT; keeps the original text on failure/empty
            let corrected = match polisher.as_ref() {
                Some(p) if p.ready() => {
                    let t_polish = Instant::now();
                    let out = p
                        .polish(&corrected, &[])
                        .await
                        .unwrap_or_else(|_| corrected.clone());
                    if !out.trim().is_empty() {
                        timings.record("polish", t_polish.elapsed().as_secs_f64() * 1000.0);
                        out
                    } else {
                        corrected
                    }
                }
                _ => corrected,
            };
            let t1 = Instant::now();
            let translated = match translator.translate_clause(&corrected).await {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("clause translation failed, skipping: {e}");
                    continue;
                }
            };
            timings.record("mt", t1.elapsed().as_secs_f64() * 1000.0);
            println!("Translation: {translated}");
            if translated.trim().is_empty() {
                continue;
            }
            let voice = match voice_source
                .voice_for(&clause_audio, &corrected, is_final)
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("clause voiceprint update failed, skipping: {e}");
                    continue;
                }
            };
            let t2 = Instant::now();
            let audio = match tts.synthesize(&translated, &voice).await {
                Ok(a) => a,
                Err(e) => {
                    tracing::warn!("clause synthesis failed, skipping: {e}");
                    continue;
                }
            };
            timings.record("tts", t2.elapsed().as_secs_f64() * 1000.0);

            let resampled = parrots_platform_macos::resample::linear_resample(
                &audio.samples,
                audio.sample_rate,
                sink.sample_rate(),
            );
            let sink_rate = sink.sample_rate();
            if let Err(e) = sink.write(&resampled) {
                // Condition 1 (playback side): write failure → rebuild the playback stream (plan 3+ Phase B)
                let Some(rec) = &out_recovery else {
                    return Err(e.into());
                };
                let mut rebuilt = None;
                let mut new_ident = None;
                for attempt in 1..=RECOVERY_MAX_ATTEMPTS {
                    tokio::time::sleep(RECOVERY_BACKOFF).await;
                    let dev = (rec.resolve_output)();
                    match out_platform.open_playback(&dev) {
                        Ok(s) => {
                            new_ident = device_identity(&dev, false);
                            rebuilt = Some(s);
                            break;
                        }
                        Err(e2) => tracing::warn!(
                            "failed to rebuild playback stream ({attempt}/{RECOVERY_MAX_ATTEMPTS}): {e2}"
                        ),
                    }
                }
                let Some(mut new_sink) = rebuilt else {
                    return Err(anyhow::anyhow!(
                        "playback device failed to rebuild {RECOVERY_MAX_ATTEMPTS} consecutive times, exiting"
                    ));
                };
                let new_rate = new_sink.sample_rate();
                let pending = if new_rate != sink_rate {
                    // New device has a different sample rate: resample the pending samples before writing
                    parrots_platform_macos::resample::linear_resample(
                        &resampled, sink_rate, new_rate,
                    )
                } else {
                    resampled
                };
                new_sink.write(&pending)?;
                sink = new_sink;
                println!("[device] switched to {}", identity_label(&new_ident));
            } else {
                // Gate = total remaining duration of unplayed clauses (in a serial queue, just the current clause) + 300ms margin
                let played_ms =
                    resampled.len() as f64 / sink_rate as f64 * 1000.0 + GATE_TAIL_MS as f64;
                gate.store(
                    t0.elapsed().as_millis() as u64 + played_ms as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }

            if !first_audio_recorded {
                // First audio = time the first clause is ready to play - time its last sample was spoken
                timings.record(
                    "first_audio",
                    (t0.elapsed().as_millis() as u64).saturating_sub(msg.spoken_at_ms) as f64,
                );
                first_audio_recorded = true;
            }
            timings.record(
                "e2e",
                (t0.elapsed().as_millis() as u64).saturating_sub(msg.spoken_at_ms) as f64,
            );
            if collect {
                collected.push(audio);
            }
        }
        Ok::<_, anyhow::Error>((timings, collected))
    });

    // ---- Join: capture thread ends naturally (Eof/time limit) → drain segmentation → drain worker pool ----
    capture_thread
        .join()
        .map_err(|_| anyhow::anyhow!("capture thread panicked"))?;
    seg_task
        .await
        .map_err(|e| anyhow::anyhow!("segmentation task panicked: {e}"))??;
    let (timings, collected) = worker_task
        .await
        .map_err(|e| anyhow::anyhow!("worker pool panicked: {e}"))??;
    Ok(IncrementalOutcome { collected, timings })
}

#[cfg(test)]
mod tests {
    use super::*;
    use parrots_asr_sensevoice::SenseVoiceAsr;
    use parrots_core::Lang;
    use parrots_mt_opus::MarianTranslator;
    use parrots_tts_zipvoice::ZipvoiceTts;
    use std::path::{Path, PathBuf};

    /// Real-time-paced fixture stream: plays a 16kHz mono wav at 32ms/chunk (mimics real-device capture pacing)
    struct PacedWavStream {
        samples: Vec<f32>,
        pos: usize,
    }

    impl PacedWavStream {
        fn load(path: &Path) -> anyhow::Result<Self> {
            let mut reader = hound::WavReader::open(path)?;
            assert_eq!(reader.spec().sample_rate, 16000);
            assert_eq!(reader.spec().channels, 1);
            let samples: Vec<f32> = match reader.spec().sample_format {
                hound::SampleFormat::Int => reader
                    .samples::<i16>()
                    .map(|s| f32::from(s.unwrap()) / 32768.0)
                    .collect(),
                hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
            };
            Ok(Self { samples, pos: 0 })
        }
    }

    impl AudioStream for PacedWavStream {
        fn sample_rate(&self) -> u32 {
            16000
        }
        fn next_chunk(&mut self) -> Option<Vec<f32>> {
            if self.pos >= self.samples.len() {
                return None;
            }
            let end = (self.pos + VAD_FRAME).min(self.samples.len());
            let chunk = self.samples[self.pos..end].to_vec();
            self.pos = end;
            // Real-time pacing: 512 samples = 32ms (no wait on the last chunk, EOF immediately)
            if self.pos < self.samples.len() {
                std::thread::sleep(Duration::from_millis(32));
            }
            Some(chunk)
        }
    }

    /// Null sink: for benchmarks/tests, discards on write
    struct NullSink;

    impl AudioSink for NullSink {
        fn sample_rate(&self) -> u32 {
            24000
        }
        fn write(&mut self, _samples: &[f32]) -> Result<()> {
            Ok(())
        }
    }

    fn models_root() -> PathBuf {
        // cargo test cwd = package dir → repo root is two levels up; PARROTS_MODELS overrides
        std::env::var_os("PARROTS_MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
            })
    }

    fn fixture_root() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
    }

    /// Incremental-mode latency benchmark: zh-long fixture (~27s of continuous Chinese) streamed at real-time pace.
    ///
    /// Gate ≤4000ms: measured 3.3~3.5s; the pipeline's own overhead (Tick ≤0.5s + rolling ASR
    /// ~0.3s + MT ~0.35s) is ~1.1s; the rest is ZipVoice synthesis time (RTF≈1.2,
    /// longer clauses are slower). 1500ms is the design target, to be tightened once TTS speeds up.
    ///
    /// Run: cargo test -p parrots-cli -- --ignored --nocapture incremental
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn incremental_first_audio_within_1500ms() {
        let asr = Arc::new(
            SenseVoiceAsr::load(&models_root().join("asr/sensevoice"), &[Lang::Zh])
                .expect("SenseVoice model ready (download-models.sh)"),
        );
        let mt = Arc::new(
            MarianTranslator::load(&models_root().join("mt/zh-en"), Lang::Zh, Lang::En)
                .expect("MT model ready"),
        );
        let tts = Arc::new(
            ZipvoiceTts::load(&models_root().join("tts/zipvoice")).expect("TTS model ready"),
        );
        tts.warmup().expect("TTS warmup");
        mt.warmup().expect("MT warmup");
        let vad =
            SileroVad::load(&models_root().join("vad/silero_vad.onnx")).expect("VAD model ready");
        let tmp = tempfile::tempdir().unwrap();
        let voice_source = VoiceSource::Rolling(Arc::new(Mutex::new(
            RollingVoiceprint::new(tmp.path(), 2).unwrap(),
        )));

        let stream = PacedWavStream::load(&fixture_root().join("zh-long.wav"))
            .expect("zh-long fixture (generated by make-fixture.sh)");
        let outcome = run_incremental_live(
            Box::new(stream),
            Box::new(NullSink),
            vad,
            IncrementalArgs {
                asr,
                translator: mt,
                tts,
                voice_source,
                from_lang: Lang::Zh,
                hotwords: None,
                tick_interval_samples: 8000,
                max_seconds: None,
                collect: false,
                gate_playback: false,
                recovery: None,
                polisher: None,
            },
        )
        .await
        .expect("incremental pipeline run succeeded");

        let t = &outcome.timings;
        println!("{}", t.report());
        println!("Clauses: {}", t.e2e_ms.len());
        assert!(
            !t.first_audio_ms.is_empty(),
            "first audio latency should be recorded"
        );
        let first = t.first_audio_ms[0];
        println!("First clause translation ready (incremental): {first:.0}ms (gate 4000ms, design target 1500ms)");
        assert!(
            first <= 4000.0,
            "incremental first-clause translation gate is 4000ms (pipeline ~1.1s + TTS dominated), got {first:.0}ms"
        );
    }

    /// Whisper incremental path smoke test (plan 5 leftover B6): whisper-small on rolling short
    /// windows was never verified; this benchmark drives the real model with the zh-long fixture
    /// (tick 1s) to observe whether the "Recognized/Translation" output fragments. Manual
    /// judgment, no hard gate (assertions only verify the pipeline runs and produces clauses).
    ///
    /// Run: cargo test -p parrots-cli --release -- --ignored --nocapture incremental_whisper
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn incremental_whisper_smoke() {
        use parrots_asr_whisper::WhisperAsr;

        let asr = Arc::new(
            WhisperAsr::load(
                &models_root().join("whisper/ggml-small.bin"),
                &[Lang::Zh],
                None,
            )
            .expect("whisper model ready (download-models.sh)"),
        );
        let mt = Arc::new(
            MarianTranslator::load(&models_root().join("mt/zh-en"), Lang::Zh, Lang::En)
                .expect("MT model ready"),
        );
        let tts = Arc::new(
            ZipvoiceTts::load(&models_root().join("tts/zipvoice")).expect("TTS model ready"),
        );
        tts.warmup().expect("TTS warmup");
        mt.warmup().expect("MT warmup");
        let vad =
            SileroVad::load(&models_root().join("vad/silero_vad.onnx")).expect("VAD model ready");
        let tmp = tempfile::tempdir().unwrap();
        let voice_source = VoiceSource::Rolling(Arc::new(Mutex::new(
            RollingVoiceprint::new(tmp.path(), 2).unwrap(),
        )));

        let stream = PacedWavStream::load(&fixture_root().join("zh-long.wav"))
            .expect("zh-long fixture (generated by make-fixture.sh)");
        let outcome = run_incremental_live(
            Box::new(stream),
            Box::new(NullSink),
            vad,
            IncrementalArgs {
                asr,
                translator: mt,
                tts,
                voice_source,
                from_lang: Lang::Zh,
                hotwords: None,
                // Same as wire.rs: whisper is slower, rolling period throttled to 1s
                tick_interval_samples: 16000,
                max_seconds: None,
                collect: false,
                gate_playback: false,
                recovery: None,
                polisher: None,
            },
        )
        .await
        .expect("whisper incremental pipeline run succeeded");

        let t = &outcome.timings;
        println!("{}", t.report());
        println!("Clauses: {}", t.e2e_ms.len());
        assert!(
            !t.e2e_ms.is_empty(),
            "whisper incremental should produce at least one clause"
        );
        assert!(
            !t.first_audio_ms.is_empty(),
            "whisper incremental should record first audio latency"
        );
    }
}
