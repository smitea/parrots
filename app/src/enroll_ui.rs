//! Voice enrollment (direction A setup) run on a background thread; progress
//! and the outcome are reported through the app event channel.

use std::path::PathBuf;
use std::sync::mpsc::Sender;

use parrots_core::{AudioPlatform, AudioStream, DeviceId};
use parrots_vad::SileroVad;

use crate::{AppEvent, AppPaths};

/// The sentence the user reads aloud; stored verbatim as the cloning prompt.
pub const ENROLL_SENTENCE: &str = "大家好,很高兴认识大家。今天我想和大家聊聊我们产品的最新进展,以及接下来的计划。希望这些内容对大家有所帮助,也欢迎随时提出问题。";

/// Spawn the enrollment flow (3s countdown -> record -> VAD trim -> save).
pub fn spawn_enroll(
    tx: Sender<AppEvent>,
    paths: AppPaths,
    name: String,
    max_seconds: u64,
) -> anyhow::Result<()> {
    std::thread::Builder::new()
        .name("enroll".into())
        .spawn(move || {
            let report = |m: String| {
                let _ = tx.send(AppEvent::Enroll(m));
            };
            for left in (1..=3).rev() {
                report(format!("Recording starts in {left}..."));
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            report(format!(
                "Recording... read the prompt sentence aloud (max {max_seconds}s)"
            ));

            let outcome = (|| -> anyhow::Result<()> {
                let platform = parrots_platform_macos::MacAudioPlatform::new();
                let capture = platform.open_capture(&DeviceId(None))?;
                let mut chunks = CaptureChunks(capture);
                let mut vad = SileroVad::load(&paths.models_root.join("vad/silero_vad.onnx"))?;
                let mut scorer = |c: &[f32]| vad.score_frame(c);
                let seg = parrots_pipeline::enroll::record_utterance(
                    &mut chunks,
                    &mut scorer,
                    max_seconds as usize * 31,
                )?;

                let duration_ms = seg.duration_ms();
                anyhow::ensure!(
                    duration_ms >= 6000,
                    "recording too short ({duration_ms}ms); read the full prompt sentence and try again"
                );

                std::fs::create_dir_all(&paths.profiles_dir)?;
                let wav_path = paths.profiles_dir.join(format!("{name}.wav"));
                let txt_path = paths.profiles_dir.join(format!("{name}.txt"));
                parrots_tts_zipvoice::write_wav(&wav_path, &seg.samples, seg.sample_rate)?;
                std::fs::write(&txt_path, ENROLL_SENTENCE)?;
                report(format!("voice profile saved: {}", wav_path.display()));
                Ok(())
            })();

            let _ = tx.send(AppEvent::EnrollDone(outcome));
        })
        .map_err(|e| anyhow::anyhow!("failed to spawn enroll thread: {e}"))?;
    Ok(())
}

struct CaptureChunks(Box<dyn AudioStream>);

impl Iterator for CaptureChunks {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next_chunk()
    }
}

// Paths are passed by value so the worker owns what it needs.
#[allow(dead_code)]
fn _paths_witness(p: &AppPaths) -> (PathBuf, PathBuf, String) {
    (
        p.models_root.clone(),
        p.profiles_dir.clone(),
        p.voice_name.clone(),
    )
}
