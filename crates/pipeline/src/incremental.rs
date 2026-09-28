//! Incremental clause segmenter: a pure-logic state machine over rolling ASR
//! plus clause commits.
//!
//! In the live loop the "segmentation task" calls [`IncrementalSegmenter::tick`]
//! every tick (about 500ms): it runs one rolling ASR pass over the uncommitted
//! audio, and commits a clause when the recognized text shows sentence
//! punctuation (。!?;, etc.) or a 400ms micro-pause (an audio proxy for a
//! short-term dip in VAD probability: an RMS-silent run) with enough
//! characters accumulated. The punctuation path's cut point first snaps to a
//! micro-pause silence boundary within the window (falling back to a
//! character-proportion estimate when no silence is available); the cut audio
//! plus the clause text are handed to the upper layer to feed the work pool
//! (MT→TTS). After VAD SpeechEnded, [`IncrementalSegmenter::finish`] commits
//! the residual tail segment (`is_final = true`).
//!
//! ASR results are injected as a closure (`&dyn Fn(&[f32]) -> Result<String>`);
//! unit tests drive it with canned closures, no real model required.

use crate::clause::{is_attach, is_clause_punct};
use parrots_core::{AudioSegment, Result};

/// 16kHz sample rate (agreed with capture/ASR).
const SAMPLE_RATE: u32 = 16000;

/// Minimum clause duration: 1s of audio (wait for the next tick if shorter).
const MIN_CLAUSE_SAMPLES: usize = 16000;

/// Micro-pause frame length: 100ms.
const PAUSE_FRAME_SAMPLES: usize = 1600;

/// Minimum micro-pause duration: 4 frames = 400ms.
const PAUSE_MIN_FRAMES: usize = 4;

/// Silence RMS threshold (synthesized speech/normal pickup is far above this).
const PAUSE_SILENCE_RMS: f32 = 0.01;

/// Minimum new characters between micro-pause commits (keeps stray
/// interjections from being chopped up).
const MIN_CHARS_BETWEEN_COMMITS: usize = 6;

/// Minimum body-content characters to commit a clause (body = non-punctuation,
/// non-attach; blocks "。"-style punctuation-only fragments, avoiding
/// downstream MT hallucinations).
const MIN_CLAUSE_CONTENT_CHARS: usize = 2;

/// Minimum body-content characters for the tail segment (the tail is the last
/// piece, so the threshold is relaxed to 1).
const MIN_TAIL_CONTENT_CHARS: usize = 1;

/// Maximum deviation for snapping a punctuation cut point to a silence
/// boundary: 0.5s (about 2 characters; beyond that the silence is unrelated
/// to the cut point and the proportion estimate is kept).
const SNAP_TOLERANCE_SAMPLES: usize = 8000;

/// One clause commit: audio slice + clause text + whether it is the tail.
pub struct ClauseDecision {
    /// Audio slice of the committed clause (16kHz)
    pub audio: AudioSegment,
    /// Clause text (from rolling ASR)
    pub text: String,
    /// Tail commit triggered by VAD end
    pub is_final: bool,
}

/// Incremental clause state machine: maintains the rolling buffer and
/// committed cursors, reused across ticks.
pub struct IncrementalSegmenter {
    /// Uncommitted audio (window head is drained on commit; always starts at
    /// the committed cursor)
    window: Vec<f32>,
    /// Absolute sample cursor of committed clauses (counted from SpeechStarted)
    committed_samples: u64,
    /// Cumulative character count of committed clauses (aligned with the
    /// rolling ASR full text, taking incremental text)
    committed_chars: usize,
}

impl IncrementalSegmenter {
    /// Starts a speech round (`start_sample` = absolute sample position of
    /// SpeechStarted).
    pub fn new(start_sample: u64) -> Self {
        Self {
            window: Vec::new(),
            committed_samples: start_sample,
            committed_chars: 0,
        }
    }

    /// Appends accumulated audio (called continuously by the segmentation
    /// task while in_speech).
    pub fn push(&mut self, chunk: &[f32]) {
        self.window.extend_from_slice(chunk);
    }

    /// Called every tick: runs rolling ASR over uncommitted audio and decides
    /// whether to commit a clause based on punctuation/micro-pause.
    ///
    /// Returns `None` when this tick commits nothing (not enough audio / no
    /// punctuation and no qualifying micro-pause).
    /// Known tradeoff: if rolling ASR rewrites the already-committed prefix,
    /// the increment is taken blindly by character cursor; punctuation cut
    /// points are snapped to silence (tolerance 0.5s), but severe misalignment
    /// between ASR segmentation and audio pauses can still drift.
    pub fn tick(
        &mut self,
        asr: &dyn Fn(&[f32]) -> Result<String>,
    ) -> Result<Option<ClauseDecision>> {
        let total = self.window.len();
        if total < MIN_CLAUSE_SAMPLES {
            return Ok(None);
        }
        let text = asr(&self.window)?;
        let new_text: String = text.chars().skip(self.committed_chars).collect();
        let total_chars = new_text.chars().count();
        if total_chars == 0 {
            return Ok(None);
        }
        // Trigger 1: the text shows punctuation (and ASR has written past it;
        // trailing punctuation does not count)
        if let Some(cut_chars) = last_committable_punct(&new_text) {
            // The cut point first snaps to a real silence boundary in the
            // window (when ASR segmentation and audio pauses are misaligned,
            // this avoids pulling the next clause's beginning into the slice);
            // falls back to a character-proportion estimate when no silence
            // is available
            let estimate = proportion(cut_chars, total_chars, total);
            let cut_samples = snap_to_silence(&self.window, estimate).unwrap_or(estimate);
            if cut_samples >= MIN_CLAUSE_SAMPLES
                && content_chars(&new_text.chars().take(cut_chars).collect::<String>())
                    >= MIN_CLAUSE_CONTENT_CHARS
            {
                return Ok(Some(self.commit(&new_text, cut_chars, cut_samples)));
            }
        }
        // Trigger 2: >= N characters since the last commit and the audio
        // shows a 400ms micro-pause (followed by speech)
        if total_chars >= MIN_CHARS_BETWEEN_COMMITS {
            if let Some(pause_start) = last_pause_start(&self.window) {
                if pause_start >= MIN_CLAUSE_SAMPLES {
                    let cut_chars =
                        proportion(pause_start, total, total_chars).clamp(1, total_chars - 1);
                    if content_chars(&new_text.chars().take(cut_chars).collect::<String>())
                        >= MIN_CLAUSE_CONTENT_CHARS
                    {
                        return Ok(Some(self.commit(&new_text, cut_chars, pause_start)));
                    }
                }
            }
        }
        Ok(None)
    }

    /// VAD SpeechEnded: commits all remaining audio as the tail segment
    /// (`is_final = true`).
    ///
    /// The tail text comes from one ASR pass over the residual audio (short
    /// utterance fallback: when nothing was committed during the round, it is
    /// committed here, so sub-1s "hello"-level short utterances are not lost).
    pub fn finish(
        &mut self,
        asr: &dyn Fn(&[f32]) -> Result<String>,
        end_sample: u64,
    ) -> Result<Option<ClauseDecision>> {
        let end_rel =
            (end_sample.saturating_sub(self.committed_samples) as usize).min(self.window.len());
        if end_rel == 0 {
            return Ok(None);
        }
        let residual: Vec<f32> = self.window[..end_rel].to_vec();
        let text = asr(&residual)?;
        if content_chars(&text) < MIN_TAIL_CONTENT_CHARS {
            return Ok(None);
        }
        self.window.drain(..end_rel);
        self.committed_samples += end_rel as u64;
        self.committed_chars += text.chars().count();
        Ok(Some(ClauseDecision {
            audio: AudioSegment::new(residual, SAMPLE_RATE),
            text,
            is_final: true,
        }))
    }

    /// Commits `[0, cut_samples)` as a clause: slices the audio, advances the
    /// cursors, drains the buffer.
    fn commit(&mut self, new_text: &str, cut_chars: usize, cut_samples: usize) -> ClauseDecision {
        let audio: Vec<f32> = self.window[..cut_samples].to_vec();
        self.window.drain(..cut_samples);
        self.committed_samples += cut_samples as u64;
        self.committed_chars += cut_chars;
        let text: String = new_text.chars().take(cut_chars).collect();
        ClauseDecision {
            audio: AudioSegment::new(audio, SAMPLE_RATE),
            text,
            is_final: false,
        }
    }
}

/// Position of the last punctuation in the incremental text that is followed
/// by content (character count including that punctuation mark).
///
/// Trailing punctuation does not count: rolling ASR often appends a sentence
/// mark to a partial window's end; splitting on it would commit an
/// unfinished clause together with a speculative period (fragmentation +
/// MT hallucinations).
fn last_committable_punct(text: &str) -> Option<usize> {
    let chars: Vec<char> = text.chars().collect();
    (0..chars.len().saturating_sub(1))
        .filter(|&i| is_clause_punct(chars[i]))
        .map(|i| i + 1)
        .next_back()
}

/// Body-content character count (non-punctuation, non-attach).
fn content_chars(text: &str) -> usize {
    text.chars()
        .filter(|&c| !is_clause_punct(c) && !is_attach(c))
        .count()
}

/// Finds the last micro-pause start in the audio that is >= 400ms of silence
/// with speech after it (relative sample offset).
///
/// This is the audio proxy for a "short-term dip in VAD probability": the
/// silence run's start is a natural cut point.
/// Pure trailing silence does not trigger (that is handled by VAD
/// SpeechEnded → finish).
fn last_pause_start(samples: &[f32]) -> Option<usize> {
    pause_starts(samples).into_iter().next_back()
}

/// All micro-pause starts in the window (relative sample offsets, ascending).
fn pause_starts(samples: &[f32]) -> Vec<usize> {
    let frame_rms = |f: usize| -> f32 {
        let seg = &samples[f * PAUSE_FRAME_SAMPLES..(f + 1) * PAUSE_FRAME_SAMPLES];
        let sum: f32 = seg.iter().map(|s| s * s).sum();
        (sum / seg.len() as f32).sqrt()
    };
    let n_frames = samples.len() / PAUSE_FRAME_SAMPLES;
    if n_frames < PAUSE_MIN_FRAMES + 1 {
        return Vec::new();
    }
    let quiet: Vec<bool> = (0..n_frames)
        .map(|f| frame_rms(f) < PAUSE_SILENCE_RMS)
        .collect();
    // Find the start of each silence run (length >= PAUSE_MIN_FRAMES, not at
    // the end, followed by speech frames)
    let mut f = 0usize;
    let mut starts = Vec::new();
    while f < n_frames {
        if quiet[f] {
            let start = f;
            while f < n_frames && quiet[f] {
                f += 1;
            }
            if f - start >= PAUSE_MIN_FRAMES && f < n_frames {
                starts.push(start * PAUSE_FRAME_SAMPLES);
            }
        } else {
            f += 1;
        }
    }
    starts
}

/// Snaps a punctuation-proportion cut point to the nearest micro-pause
/// boundary (deviation <= 0.5s).
///
/// The clause audio ends at a real silence start, so a poor proportion
/// estimate cannot pull the next clause's beginning into the slice (audio
/// bleeding between translated clauses); returns `None` when no silence can
/// absorb the snap.
fn snap_to_silence(samples: &[f32], estimate: usize) -> Option<usize> {
    pause_starts(samples)
        .into_iter()
        .min_by_key(|start| start.abs_diff(estimate))
        .filter(|start| start.abs_diff(estimate) <= SNAP_TOLERANCE_SAMPLES)
}

/// Proportional conversion: `part / whole * total` (rounded).
fn proportion(part: usize, whole: usize, total: usize) -> usize {
    ((part as f64 / whole as f64) * total as f64).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 0.5-amplitude "speech" block (RMS 0.5, far above the silence threshold).
    fn speech(ms: u64) -> Vec<f32> {
        vec![0.5; (SAMPLE_RATE as u64 * ms / 1000) as usize]
    }

    /// A digital silence block.
    fn silence(ms: u64) -> Vec<f32> {
        vec![0.0; (SAMPLE_RATE as u64 * ms / 1000) as usize]
    }

    /// Canned ASR mapping audio duration to text: steps = (min duration ms, text).
    fn canned_asr<'a>(steps: &'a [(u64, &'a str)]) -> impl Fn(&[f32]) -> Result<String> + 'a {
        move |samples: &[f32]| {
            let ms = samples.len() as u64 * 1000 / SAMPLE_RATE as u64;
            let mut out = "";
            for (at, text) in steps {
                if ms >= *at {
                    out = text;
                }
            }
            Ok(out.to_string())
        }
    }

    #[test]
    fn punctuation_triggers_commit() {
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(2000));
        let asr = canned_asr(&[(1500, "大家好,今天")]);
        let d = seg
            .tick(&asr)
            .unwrap()
            .expect("punctuation should trigger a commit");
        assert_eq!(d.text, "大家好,");
        assert!(!d.is_final);
        // 4/6 characters → proportionally cut audio
        assert_eq!(d.audio.samples.len(), 21333);
        assert_eq!(d.audio.sample_rate, 16000);
        // Buffer drained after commit: below minimum clause duration, no more commits
        assert!(seg.tick(&asr).unwrap().is_none());
    }

    #[test]
    fn trailing_punct_is_not_committable() {
        // Rolling ASR often appends a period to a partial window's end: not a
        // clause end until ASR writes past it
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(2000));
        let asr = canned_asr(&[(1500, "大家好,今天。")]);
        // The last committable punctuation is "," (followed by 今天); the
        // trailing 。 does not count
        let d = seg
            .tick(&asr)
            .unwrap()
            .expect("punctuation should trigger a commit");
        assert_eq!(d.text, "大家好,");
        // Pure trailing period (no earlier punctuation): no commit
        let mut seg2 = IncrementalSegmenter::new(0);
        seg2.push(&speech(2000));
        let asr2 = canned_asr(&[(1500, "大家好。")]);
        assert!(seg2.tick(&asr2).unwrap().is_none());
    }

    #[test]
    fn pure_punct_cut_is_rejected() {
        // Fragments with no body content inside the cut (punctuation only) are
        // not committed, avoiding downstream MT hallucinations
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(2000));
        let asr = canned_asr(&[(1500, "。今天")]);
        assert!(seg.tick(&asr).unwrap().is_none());
    }

    #[test]
    fn punctuation_cut_snaps_to_silence_boundary() {
        // 1s speech + 0.5s silence + 1.5s speech: punctuation in "大家好啊,"
        // at 5/11 characters → proportion estimate 21818, which is 5818 from
        // the real pause start 16000, <= tolerance 8000 → the cut point snaps
        // to the silence boundary; the slice contains no next-clause speech
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(1000));
        seg.push(&silence(500));
        seg.push(&speech(1500));
        let asr = canned_asr(&[(1500, "大家好啊,今天你们好吗")]);
        let d = seg
            .tick(&asr)
            .unwrap()
            .expect("punctuation should trigger a commit");
        assert_eq!(d.text, "大家好啊,");
        assert_eq!(d.audio.samples.len(), 16000);
    }

    #[test]
    fn punctuation_cut_falls_back_when_silence_too_far() {
        // 1.8s speech + 0.5s silence + 0.7s speech: the pause start 28800 is
        // 8800 from the estimate 20000, > tolerance 8000 → that silence is
        // unrelated to the cut point; keep the proportion estimate
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(1800));
        seg.push(&silence(500));
        seg.push(&speech(700));
        let asr = canned_asr(&[(1500, "十二个字,符切五处吧这样")]);
        let d = seg
            .tick(&asr)
            .unwrap()
            .expect("punctuation should trigger a commit");
        assert_eq!(d.text, "十二个字,");
        assert_eq!(d.audio.samples.len(), 20000);
    }

    #[test]
    fn snap_prefers_nearest_pause() {
        // Two pauses (16000 and 40000), estimate 16000 → snaps to the nearest
        // 16000, not the farther one
        let window = &[
            speech(1000),
            silence(500),
            speech(1000),
            silence(500),
            speech(1000),
        ]
        .concat();
        assert_eq!(pause_starts(window), vec![16000, 40000]);
        assert_eq!(snap_to_silence(window, 16000), Some(16000));
    }

    #[test]
    fn proportional_cut_point() {
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(2000));
        // 6 characters; clause "你好吗," is 4 characters with punctuation →
        // cut = 4/6 * 2s
        let asr = canned_asr(&[(1500, "你好吗,朋友")]);
        let d = seg
            .tick(&asr)
            .unwrap()
            .expect("punctuation should trigger a commit");
        assert_eq!(d.text, "你好吗,");
        assert_eq!(d.audio.samples.len(), 21333);
        // No new punctuation, no pause: the remaining 朋友 is not committed
        assert!(seg.tick(&asr).unwrap().is_none());
    }

    #[test]
    fn no_punctuation_no_commit() {
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(2000));
        // No punctuation and below MIN_CHARS_BETWEEN_COMMITS (5 < 6);
        // continuous speech with no pause
        let asr = canned_asr(&[(1500, "大家好今天")]);
        assert!(seg.tick(&asr).unwrap().is_none());
    }

    #[test]
    fn micro_pause_triggers_commit() {
        // 1s speech + 0.5s silence + 0.5s speech: pause start 16000, followed
        // by speech → triggers
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(1000));
        seg.push(&silence(500));
        seg.push(&speech(500));
        // 7 characters, no punctuation, >= 6 → micro-pause path; cut =
        // 16000/32000*7 ≈ 4 characters
        let asr = canned_asr(&[(1500, "大家好啊你们好")]);
        let d = seg
            .tick(&asr)
            .unwrap()
            .expect("micro-pause should trigger a commit");
        assert!(!d.is_final);
        assert_eq!(d.audio.samples.len(), 16000);
        assert_eq!(d.text.chars().count(), 4);
    }

    #[test]
    fn trailing_silence_does_not_trigger() {
        // Trailing silence is not a micro-pause "followed by speech": left to
        // VAD SpeechEnded → finish
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(1000));
        seg.push(&silence(600));
        let asr = canned_asr(&[(1500, "大家好啊你们好")]);
        assert!(seg.tick(&asr).unwrap().is_none());
    }

    #[test]
    fn min_clause_duration_guards() {
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(500));
        // 0.5s < 1s: ASR must not even be called, let alone commit
        let asr = |_s: &[f32]| -> Result<String> {
            panic!("ASR must not be called below the minimum duration")
        };
        assert!(seg.tick(&asr).unwrap().is_none());
    }

    #[test]
    fn finish_commits_tail_as_final() {
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(2000));
        let asr = canned_asr(&[(1500, "你好,大家")]);
        let d = seg
            .tick(&asr)
            .unwrap()
            .expect("punctuation should trigger a commit");
        assert_eq!(d.text, "你好,");
        // Commits 3/5 * 32000 = 19200, leaving 12800; after appending a 1s
        // tail, SpeechEnded
        seg.push(&speech(1000));
        let asr_tail = canned_asr(&[(500, "再见")]);
        let tail = seg
            .finish(&asr_tail, 35200)
            .unwrap()
            .expect("tail should be committed");
        assert_eq!(tail.text, "再见");
        assert!(tail.is_final);
        assert_eq!(tail.audio.samples.len(), 16000);
        // Drained: a second finish has nothing left
        assert!(seg.finish(&asr_tail, 35200).unwrap().is_none());
    }

    #[test]
    fn finish_covers_short_utterance_without_tick() {
        // Nothing committed during the whole round (e.g. a 0.5s 你好): finish
        // acts as the fallback, preventing lost utterances
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(500));
        let asr = canned_asr(&[(0, "你好")]);
        let d = seg
            .finish(&asr, 8000)
            .unwrap()
            .expect("short utterance should be committed by the finish fallback");
        assert_eq!(d.text, "你好");
        assert!(d.is_final);
    }

    #[test]
    fn reset_starts_new_window() {
        let mut seg = IncrementalSegmenter::new(0);
        seg.push(&speech(1000));
        seg.push(&speech(1000));
        let asr = canned_asr(&[(0, "全提交,好")]);
        let d = seg.tick(&asr).unwrap().expect("should commit");
        assert_eq!(d.audio.samples.len(), 25600);
        // New speech round: cursors reset to the new start
        let mut seg2 = IncrementalSegmenter::new(96000);
        seg2.push(&speech(500));
        let asr2 = canned_asr(&[(0, "你好")]);
        let d2 = seg2
            .finish(&asr2, 112000)
            .unwrap()
            .expect("new round's short utterance should be committed");
        assert_eq!(d2.text, "你好");
        assert!(d2.is_final);
    }
}
