use cpal::traits::{DeviceTrait, HostTrait};
use parrots_core::{AudioStream, DeviceId, Error, Result};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// cpal real-time callback thread → lock-free ring buffer → blocking
/// next_chunk() pulls (downmix + resample to 16k)
pub struct CpalCapture {
    ring: Arc<OverwriteRing>,
    leftover: Vec<f32>,
    /// Raw input samples needed per 512-sample output chunk (integral:
    /// 512*in_rate is divisible by 16k)
    need: usize,
    in_rate: u32,
    _stream: crate::StreamGuard, // dropped = stopped
}

const TARGET_RATE: u32 = 16000;
const CHUNK_OUT: usize = 512;

impl CpalCapture {
    pub fn open(device: &DeviceId) -> Result<Self> {
        let host = cpal::default_host();
        let dev = match &device.0 {
            Some(name) => host
                .input_devices()
                .map_err(audio_err)?
                .find(|d| d.name().map(|n| n.to_string()).ok().as_deref() == Some(name.as_str()))
                .ok_or_else(|| Error::audio(format!("input device not found: {name}")))?,
            None => host
                .default_input_device()
                .ok_or_else(|| Error::audio("no default input device"))?,
        };
        let cfg = dev.default_input_config().map_err(audio_err)?;
        let in_rate = cfg.sample_rate().0;
        let channels = usize::from(cfg.channels());
        // Input samples per 512-sample output chunk must be integral (true
        // for 16k/24k/32k/48k/96k)
        let raw_per_chunk = u64::from(CHUNK_OUT as u32) * u64::from(in_rate);
        if raw_per_chunk % u64::from(TARGET_RATE) != 0 {
            return Err(Error::audio(format!(
                "{in_rate}Hz → {TARGET_RATE}Hz not yet supported"
            )));
        }
        let need = (raw_per_chunk / u64::from(TARGET_RATE)) as usize;
        let stream_cfg = cpal::StreamConfig {
            channels: channels as u16,
            sample_rate: cpal::SampleRate(in_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let ring = Arc::new(OverwriteRing::new(1 << 16));
        let r = ring.clone();
        let err_fn = |e| tracing::error!("cpal input stream error: {e}");
        let _stream = crate::StreamGuard::start(move || {
            let stream = dev
                .build_input_stream(
                    &stream_cfg,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        // Real-time thread: downmix + write the ring buffer only
                        // (oldest dropped when full)
                        for frame in data.chunks(channels) {
                            let mono = frame.iter().sum::<f32>() / channels as f32;
                            r.push(mono);
                        }
                    },
                    err_fn,
                    None,
                )
                .map_err(audio_err)?;
            Ok(stream)
        })?;
        Ok(Self {
            ring,
            leftover: Vec::new(),
            need,
            in_rate,
            _stream,
        })
    }
}

impl AudioStream for CpalCapture {
    fn sample_rate(&self) -> u32 {
        TARGET_RATE
    }

    fn next_chunk(&mut self) -> Option<Vec<f32>> {
        let need = self.need;
        let mut raw = std::mem::take(&mut self.leftover);
        while raw.len() < need {
            match self.ring.pop() {
                Some(s) => raw.push(s),
                None => std::thread::sleep(Duration::from_millis(2)),
            }
        }
        let window = &raw[..need];
        let out = if self.in_rate == TARGET_RATE {
            window.to_vec()
        } else if self.in_rate % TARGET_RATE == 0 {
            crate::resample::decimate(window, (self.in_rate / TARGET_RATE) as usize)
        } else {
            crate::resample::linear_resample(window, self.in_rate, TARGET_RATE)
        };
        self.leftover = raw[need..].to_vec();
        Some(out)
    }
}

/// Lock-free single-writer single-reader ring buffer that drops the oldest
/// data when full (written by the real-time callback thread, read by the
/// reader thread). Deliberately not rtrb's pop-then-push: those three steps
/// transiently empty the ring, so the reader could observe a gap; here a
/// single fetch_max atomic RMW drops the oldest sample, strictly better for
/// the real-time thread.
pub(crate) struct OverwriteRing {
    slots: Box<[AtomicU32]>,
    cap: u64,
    head: AtomicU64, // next write sequence (advanced only by the writer thread)
    tail: AtomicU64, // next read sequence (advanced by the reader thread; moved forward by the writer on overflow to drop oldest)
}

impl OverwriteRing {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            slots: (0..cap).map(|_| AtomicU32::new(0)).collect(),
            cap: cap as u64,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
        }
    }

    /// Writer thread: writes one sample; when full, drops the oldest first,
    /// then writes (keeps the newest audio).
    pub(crate) fn push(&self, value: f32) {
        let h = self.head.load(Ordering::Relaxed); // writer-only advance
        let t = self.tail.load(Ordering::Acquire);
        if h - t == self.cap {
            // Full: advancing tail by one = dropping the oldest sample; the
            // new sample then overwrites its slot
            self.tail.fetch_max(h - self.cap + 1, Ordering::AcqRel);
        }
        self.slots[(h % self.cap) as usize].store(value.to_bits(), Ordering::Relaxed);
        self.head.store(h + 1, Ordering::Release);
    }

    /// Reader thread: pops one sample; None when empty.
    pub(crate) fn pop(&self) -> Option<f32> {
        let h = self.head.load(Ordering::Acquire);
        let t = self.tail.load(Ordering::Relaxed);
        if t == h {
            return None;
        }
        // Under an extreme race (the writer concurrently moves tail forward on
        // overflow and overwrites this slot), reading an old or new value via
        // AtomicU32 is still a single atomic read with no data race; the
        // sample is already inside the "overflow drop" window, so taking
        // either value is acceptable.
        let v = f32::from_bits(self.slots[(t % self.cap) as usize].load(Ordering::Relaxed));
        self.tail.fetch_max(t + 1, Ordering::AcqRel);
        Some(v)
    }
}

fn audio_err(e: impl std::fmt::Display) -> Error {
    Error::audio(e.to_string())
}
