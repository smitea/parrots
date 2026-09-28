use parrots_core::AudioSegment;

/// Carves complete utterances out of a continuous 16k sample stream with VAD.
/// Call order: push(chunk) → handle detector.feed(prob) events → on_started/on_ended
#[derive(Default)]
pub struct UtteranceAssembler {
    total_pushed: u64,
    collecting: bool,
    cur_start: u64,
    buf: Vec<f32>,
}

impl UtteranceAssembler {
    pub fn new() -> Self {
        Self {
            total_pushed: 0,
            collecting: false,
            cur_start: 0,
            buf: Vec::new(),
        }
    }

    /// Pushes a 512-sample chunk (must precede VAD event handling for the
    /// same frame)
    pub fn push(&mut self, chunk: &[f32]) {
        if self.collecting {
            self.buf.extend_from_slice(chunk);
        }
        self.total_pushed += chunk.len() as u64;
    }

    pub fn on_speech_start(&mut self) {
        self.collecting = true;
        self.cur_start = self.total_pushed;
        self.buf.clear();
    }

    /// Called on SpeechEnded; returns the carved segment (None and discarded
    /// if too short)
    pub fn on_speech_end(&mut self, end_sample: u64) -> Option<AudioSegment> {
        self.collecting = false;
        let len = (end_sample.saturating_sub(self.cur_start)) as usize;
        if len == 0 || self.buf.is_empty() {
            self.buf.clear();
            return None;
        }
        let len = len.min(self.buf.len());
        Some(AudioSegment::new(self.buf.drain(..len).collect(), 16000))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_between_start_and_end() {
        let mut a = UtteranceAssembler::new();
        a.push(&[0.0; 512]); // frame 0 (silence)
        a.on_speech_start();
        let c1: Vec<f32> = (0..512).map(|i| i as f32).collect();
        let c2: Vec<f32> = (512..1024).map(|i| i as f32).collect();
        a.push(&c1);
        a.push(&c2);
        // Absolute sample indices: silence block 0..512, speech c1+c2 = 512..1536
        let seg = a.on_speech_end(1536).expect("assembly should complete");
        assert_eq!(seg.sample_rate, 16000);
        assert_eq!(seg.samples.len(), 1024);
        assert_eq!(seg.samples[0], 0.0);
        assert_eq!(seg.samples[1023], 1023.0);
    }

    #[test]
    fn end_before_any_sample_returns_none() {
        let mut a = UtteranceAssembler::new();
        a.on_speech_start();
        assert!(a.on_speech_end(0).is_none());
    }
}
