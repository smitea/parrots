use parrots_core::{AudioSegment, Result, VoiceProfile};
use std::path::PathBuf;

/// The other party's rolling voiceprint: each utterance writes a wav +
/// transcript to disk, rotating through the most recent `keep` slots
pub struct RollingVoiceprint {
    dir: PathBuf,
    keep: usize,
    counter: usize,
}

impl RollingVoiceprint {
    pub fn new(dir: impl Into<PathBuf>, keep: usize) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            keep,
            counter: 0,
        })
    }

    pub fn update(&mut self, audio: &AudioSegment, transcript: &str) -> Result<VoiceProfile> {
        self.counter += 1;
        let slot = self.counter % self.keep.max(1);
        let path = self.dir.join(format!("voice-{slot}.wav"));
        write_wav16(&path, &audio.samples, audio.sample_rate)?;
        Ok(VoiceProfile::from_prompt(path, transcript.to_string()))
    }

    pub fn dir(&self) -> &std::path::Path {
        &self.dir
    }
}

/// Utterance-level rolling voiceprint (B3): clause audio accumulates into a
/// full utterance, and the profile is updated as a whole only when the
/// utterance ends (`is_final`) — clause synthesis keeps using the most recent
/// utterance-level profile, avoiding timbre drift from 1-3s short prompts.
/// Before the first utterance ends there is no profile, so it falls back to
/// clause-level updates, guaranteeing every clause has a voiceprint available.
pub struct UtteranceRollingVoiceprint {
    inner: RollingVoiceprint,
    acc_audio: Vec<f32>,
    acc_text: String,
    acc_rate: u32,
    last: Option<VoiceProfile>,
}

impl UtteranceRollingVoiceprint {
    pub fn new(dir: impl Into<PathBuf>, keep: usize) -> Result<Self> {
        Ok(Self {
            inner: RollingVoiceprint::new(dir, keep)?,
            acc_audio: Vec::new(),
            acc_text: String::new(),
            acc_rate: 16000,
            last: None,
        })
    }

    /// Advances per clause, returning the voiceprint this clause's synthesis
    /// should use.
    pub fn update(
        &mut self,
        audio: &AudioSegment,
        transcript: &str,
        is_final: bool,
    ) -> Result<VoiceProfile> {
        if self.acc_audio.is_empty() {
            self.acc_rate = audio.sample_rate;
        }
        self.acc_audio.extend_from_slice(&audio.samples);
        self.acc_text.push_str(transcript);
        if !is_final {
            if let Some(p) = &self.last {
                return Ok(p.clone());
            }
            // First utterance: no profile to reuse, fall back to clause level
            return self.inner.update(audio, transcript);
        }
        let full = AudioSegment::new(std::mem::take(&mut self.acc_audio), self.acc_rate);
        let profile = self.inner.update(&full, &self.acc_text.clone())?;
        self.acc_text.clear();
        self.last = Some(profile.clone());
        Ok(profile)
    }
}

fn write_wav16(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .map_err(|e| parrots_core::Error::audio(e.to_string()))?;
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .map_err(|e| parrots_core::Error::audio(e.to_string()))?;
    }
    w.finalize()
        .map_err(|e| parrots_core::Error::audio(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_writes_rotating_files() {
        let tmp = tempfile::tempdir().unwrap();
        let mut vp = RollingVoiceprint::new(tmp.path(), 2).unwrap();
        let seg = AudioSegment::new(vec![0.1; 1600], 16000);
        let p1 = vp.update(&seg, "hello").unwrap();
        assert!(p1.prompt_wav_path.exists());
        assert_eq!(p1.prompt_text, "hello");
        let p2 = vp.update(&seg, "world").unwrap();
        assert_ne!(p1.prompt_wav_path, p2.prompt_wav_path);
        let p3 = vp.update(&seg, "again").unwrap();
        assert_eq!(p1.prompt_wav_path, p3.prompt_wav_path, "slot rotation");
    }

    /// B3: the first utterance falls back to clause level; the whole profile
    /// updates only when the utterance ends, and later clauses keep the
    /// previous utterance profile (path/transcript unchanged).
    #[test]
    fn utterance_level_freezes_between_utterances() {
        let tmp = tempfile::tempdir().unwrap();
        let mut vp = UtteranceRollingVoiceprint::new(tmp.path(), 2).unwrap();
        let clause = AudioSegment::new(vec![0.1; 16000], 16000);
        // First utterance: clause 1/2 has no profile → clause-level fallback
        // (transcript = clause text)
        let p1 = vp.update(&clause, "第一句,", false).unwrap();
        assert_eq!(p1.prompt_text, "第一句,");
        let p2 = vp.update(&clause, "第二句", true).unwrap();
        assert_eq!(
            p2.prompt_text, "第一句,第二句",
            "tail should update the whole profile"
        );
        // Second utterance: non-tail clauses keep the previous profile
        let p3 = vp.update(&clause, "第三句,", false).unwrap();
        assert_eq!(p3.prompt_text, p2.prompt_text);
        assert_eq!(p3.prompt_wav_path, p2.prompt_wav_path);
        // Utterance end: profile refreshed with the accumulated content
        let p4 = vp.update(&clause, "第四句", true).unwrap();
        assert_eq!(p4.prompt_text, "第三句,第四句");
    }

    /// Accumulated audio duration = sum of clauses (becomes the profile when
    /// the utterance ends).
    #[test]
    fn utterance_level_accumulates_audio() {
        let tmp = tempfile::tempdir().unwrap();
        let mut vp = UtteranceRollingVoiceprint::new(tmp.path(), 2).unwrap();
        let clause = AudioSegment::new(vec![0.1; 8000], 16000);
        vp.update(&clause, "a", false).unwrap();
        let final_profile = vp.update(&clause, "b", true).unwrap();
        let mut reader = hound::WavReader::open(&final_profile.prompt_wav_path).unwrap();
        let n: u32 = reader.samples::<i16>().count() as u32;
        assert_eq!(n, 16000, "two accumulated 0.5s clauses = 1s");
    }
}
