use crate::wire;
use parrots_core::{AudioPlatform, AudioStream, DeviceId};
use parrots_vad::SileroVad;

pub const ENROLL_SENTENCE: &str = "大家好,很高兴认识大家。今天我想和大家聊聊我们产品的最新进展,以及接下来的计划。希望这些内容对大家有所帮助,也欢迎随时提出问题。";

pub fn run_enroll(name: &str, max_seconds: u64) -> anyhow::Result<()> {
    println!("Recording starts in 3 seconds; please read the following sentence aloud:");
    println!("{ENROLL_SENTENCE}");
    std::thread::sleep(std::time::Duration::from_secs(3));
    println!(
        "Recording... (stops automatically once a sentence finishes, max {max_seconds} seconds)"
    );

    let platform = parrots_platform_macos::MacAudioPlatform::new();
    let capture = platform.open_capture(&DeviceId(None))?;
    let mut chunks = CaptureChunks(capture);
    let mut vad = SileroVad::load(&wire::models_root().join("vad/silero_vad.onnx"))?;
    let mut scorer = |c: &[f32]| vad.score_frame(c);
    let seg = parrots_pipeline::enroll::record_utterance(
        &mut chunks,
        &mut scorer,
        max_seconds as usize * 31,
    )?;

    let duration_ms = seg.duration_ms();
    if duration_ms < 6000 {
        anyhow::bail!("recording too short ({duration_ms}ms); please read the full prompt sentence and try again");
    }

    let dir = wire::profiles_dir();
    std::fs::create_dir_all(&dir)?;
    let wav_path = dir.join(format!("{name}.wav"));
    let txt_path = dir.join(format!("{name}.txt"));
    parrots_tts_zipvoice::write_wav(&wav_path, &seg.samples, seg.sample_rate)?;
    std::fs::write(&txt_path, ENROLL_SENTENCE)?;

    println!("Voice enrollment complete: {name} (duration {duration_ms}ms)");
    println!("Audio → {}", wav_path.display());
    println!("Transcript → {}", txt_path.display());
    Ok(())
}

struct CaptureChunks(Box<dyn AudioStream>);

impl Iterator for CaptureChunks {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next_chunk()
    }
}
