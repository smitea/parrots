use serde::Serialize;

#[derive(Debug, Default, Clone, Serialize)]
pub struct StageTimings {
    pub asr_ms: Vec<f64>,
    pub mt_ms: Vec<f64>,
    pub tts_ms: Vec<f64>,
    pub e2e_ms: Vec<f64>,
    /// Time until the first synthesized audio chunk is ready (matches the
    /// design doc §2 "full translated sentence starts playing" latency
    /// definition)
    pub first_audio_ms: Vec<f64>,
    /// Text polish layer time (Plan 7; empty when disabled)
    pub polish_ms: Vec<f64>,
}

impl StageTimings {
    pub fn record(&mut self, stage: &str, ms: f64) {
        match stage {
            "asr" => self.asr_ms.push(ms),
            "mt" => self.mt_ms.push(ms),
            "tts" => self.tts_ms.push(ms),
            "e2e" => self.e2e_ms.push(ms),
            "first_audio" => self.first_audio_ms.push(ms),
            "polish" => self.polish_ms.push(ms),
            _ => {}
        }
    }

    pub fn mean(v: &[f64]) -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }

    pub fn report(&self) -> String {
        format!(
            "asr mean {:.0}ms | mt mean {:.0}ms | tts mean {:.0}ms | polish mean {:.0}ms | end-to-end mean {:.0}ms | first audio {:.0}ms",
            Self::mean(&self.asr_ms),
            Self::mean(&self.mt_ms),
            Self::mean(&self.tts_ms),
            Self::mean(&self.polish_ms),
            Self::mean(&self.e2e_ms),
            Self::mean(&self.first_audio_ms)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_report() {
        let mut t = StageTimings::default();
        t.record("asr", 100.0);
        t.record("asr", 300.0);
        assert_eq!(StageTimings::mean(&t.asr_ms), 200.0);
        assert!(t.report().contains("200ms"));
        assert!(t.report().contains("end-to-end mean 0ms"));
        t.record("first_audio", 1500.0);
        assert_eq!(StageTimings::mean(&t.first_audio_ms), 1500.0);
        assert!(t.report().contains("first audio 1500ms"));
    }
}
