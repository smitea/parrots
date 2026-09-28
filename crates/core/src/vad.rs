/// VAD event (absolute sample index, 16kHz)
#[derive(Debug, Clone, PartialEq)]
pub enum VadEvent {
    SpeechStarted,
    SpeechEnded { start_sample: u64, end_sample: u64 },
}

#[derive(Debug, Clone)]
pub struct VadConfig {
    pub threshold: f32,
    pub start_frames: u32,
    pub end_frames: u32,
    /// Force segmentation of overly long speech (in samples)
    pub max_speech_samples: u64,
    /// Preroll frames at utterance start (prevents clipping the onset)
    pub preroll_frames: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            start_frames: 3,
            end_frames: 30,
            max_speech_samples: 16000 * 12,
            preroll_frames: 2,
        }
    }
}

/// Pure-logic state machine: feed per-frame speech probabilities, emit events
pub struct SpeechDetector {
    cfg: VadConfig,
    frame_samples: u64,
    in_speech: bool,
    high_run: u32,
    low_run: u32,
    speech_start: u64,
    total_samples: u64,
}

impl SpeechDetector {
    pub fn new(cfg: VadConfig, frame_samples: u64) -> Self {
        Self {
            cfg,
            frame_samples,
            in_speech: false,
            high_run: 0,
            low_run: 0,
            speech_start: 0,
            total_samples: 0,
        }
    }

    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    pub fn feed(&mut self, prob: f32) -> Vec<VadEvent> {
        let mut events = Vec::new();
        let frame_start = self.total_samples;
        self.total_samples += self.frame_samples;
        let voiced = prob >= self.cfg.threshold;

        if !self.in_speech {
            self.high_run = if voiced { self.high_run + 1 } else { 0 };
            if self.high_run >= self.cfg.start_frames {
                self.in_speech = true;
                self.high_run = 0;
                self.low_run = 0;
                self.speech_start = frame_start
                    .saturating_sub(u64::from(self.cfg.preroll_frames) * self.frame_samples);
                events.push(VadEvent::SpeechStarted);
            }
        } else {
            self.low_run = if voiced { 0 } else { self.low_run + 1 };
            let duration = self.total_samples - self.speech_start;
            let ended = self.low_run >= self.cfg.end_frames;
            let overflow = duration >= self.cfg.max_speech_samples;
            if ended || overflow {
                let end = if ended {
                    self.total_samples - u64::from(self.low_run) * self.frame_samples
                } else {
                    self.total_samples
                };
                self.in_speech = false;
                self.low_run = 0;
                events.push(VadEvent::SpeechEnded {
                    start_sample: self.speech_start,
                    end_sample: end,
                });
            }
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det() -> SpeechDetector {
        SpeechDetector::new(VadConfig::default(), 512)
    }

    #[test]
    fn silence_produces_nothing() {
        let mut d = det();
        for _ in 0..100 {
            assert!(d.feed(0.01).is_empty());
        }
    }

    #[test]
    fn short_burst_ignored() {
        let mut d = det();
        d.feed(0.9);
        d.feed(0.9);
        assert!(
            d.feed(0.01).is_empty(),
            "does not trigger below start_frames"
        );
        assert!(!d.in_speech());
    }

    #[test]
    fn start_then_end_with_bounds() {
        let mut d = det();
        let mut started = false;
        let mut ended = None;
        for i in 0..150 {
            let p = if (20..80).contains(&i) { 0.95 } else { 0.01 };
            for e in d.feed(p) {
                match e {
                    VadEvent::SpeechStarted => started = true,
                    VadEvent::SpeechEnded {
                        start_sample,
                        end_sample,
                    } => ended = Some((start_sample, end_sample)),
                }
            }
        }
        assert!(started);
        let (s, e) = ended.expect("expected a speech-end event");
        // Trigger frame = frame 22 (frames 20-22, 3 consecutive voiced), preroll 2 frames -> start = 20*512
        assert_eq!(s, 20 * 512);
        // The 30th consecutive low frame (i=109) ends speech: end = 110*512 - 30*512 = 80*512
        assert_eq!(e, 80 * 512);
    }

    #[test]
    fn max_duration_forces_split() {
        let mut d = SpeechDetector::new(
            VadConfig {
                max_speech_samples: 10 * 512,
                end_frames: 1_000_000,
                ..Default::default()
            },
            512,
        );
        let mut splits = 0;
        for _ in 0..40 {
            for e in d.feed(0.95) {
                if matches!(e, VadEvent::SpeechEnded { .. }) {
                    splits += 1;
                }
            }
        }
        assert!(
            splits >= 3,
            "continuous speech should force segmentation, got {splits}"
        );
    }
}
