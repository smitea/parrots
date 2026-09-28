use crate::utterance::UtteranceAssembler;
use parrots_core::{AudioSegment, Error, Result};
use parrots_core::{SpeechDetector, VadConfig, VadEvent};

pub const ENROLL_FRAME: usize = 512;

/// Records one complete utterance from a 16k capture stream: returns as soon
/// as the first VAD-completed utterance is observed.
/// scorer: per frame (512 samples) → speech probability (production =
/// SileroVad wrapper, tests = closures).
pub fn record_utterance(
    stream: &mut dyn Iterator<Item = Vec<f32>>,
    mut scorer: impl FnMut(&[f32]) -> Result<f32>,
    max_frames: usize,
) -> Result<AudioSegment> {
    let mut vad = SpeechDetector::new(VadConfig::default(), ENROLL_FRAME as u64);
    let mut asm = UtteranceAssembler::new();
    for _ in 0..max_frames {
        let Some(chunk) = stream.next() else {
            break;
        };
        if chunk.len() < ENROLL_FRAME {
            break;
        }
        asm.push(&chunk);
        let prob = scorer(&chunk)?;
        for ev in vad.feed(prob) {
            match ev {
                VadEvent::SpeechStarted => asm.on_speech_start(),
                VadEvent::SpeechEnded { end_sample, .. } => {
                    if let Some(seg) = asm.on_speech_end(end_sample) {
                        return Ok(seg);
                    }
                }
            }
        }
    }
    Err(Error::audio(
        "enrollment timed out: no complete utterance detected",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Probability script: 0..10 silence, 10..50 voiced, 50..100 silence
    fn voiced(i: usize) -> bool {
        (10..50).contains(&i)
    }

    fn make_stream(n: usize) -> impl Iterator<Item = Vec<f32>> {
        (0..n)
            .map(|i| vec![if voiced(i) { 1.0 } else { 0.0 }; ENROLL_FRAME])
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn returns_first_completed_utterance() {
        let mut stream = make_stream(100);
        let mut scorer = |c: &[f32]| -> Result<f32> { Ok(if c[0] == 1.0 { 0.95 } else { 0.01 }) };
        let seg = record_utterance(&mut stream, &mut scorer, 100)
            .expect("should return the first complete utterance");

        // State machine walkthrough (VadConfig::default: start_frames=3, end_frames=30, preroll_frames=2):
        // - Frame 12 (0-based) reaches 3 consecutive voiced frames →
        //   SpeechStarted; the assembler receives the event only after
        //   push(frame 12) → start cur_start = 13*512 = 6656.
        // - Frames 49+ go silent; frame 79 has low_run=30 → SpeechEnded,
        //   end_sample = 80*512 - 30*512 = 50*512 = 25600.
        // - Segment = [13*512, 50*512) = 37*512 = 18944 samples, exactly the
        //   37 voiced frames 13..=49.
        assert_eq!(seg.sample_rate, 16000);
        assert_eq!(seg.samples.len(), 37 * ENROLL_FRAME);
        assert_eq!(seg.duration_ms(), 1184);
        assert!(
            seg.samples.iter().all(|&s| s == 1.0),
            "segment should contain exactly the voiced frames (silence trimmed at both ends)"
        );
    }

    #[test]
    fn times_out_without_speech() {
        let mut stream = make_stream(100);
        let mut scorer = |_c: &[f32]| -> Result<f32> { Ok(0.01) };
        let err = record_utterance(&mut stream, &mut scorer, 100).unwrap_err();
        assert!(
            err.to_string().contains("enrollment timed out"),
            "actual error: {err}"
        );
    }
}
