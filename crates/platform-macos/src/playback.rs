use cpal::traits::{DeviceTrait, HostTrait};
use parrots_core::{AudioSink, DeviceId, Error, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub struct CpalSink {
    queue: Arc<Mutex<VecDeque<f32>>>,
    out_rate: u32,
    _stream: crate::StreamGuard, // dropped = stopped
}

impl CpalSink {
    pub fn open(device: &DeviceId) -> Result<Self> {
        let host = cpal::default_host();
        let dev = match &device.0 {
            Some(name) => host
                .output_devices()
                .map_err(audio_err)?
                .find(|d| d.name().map(|n| n.to_string()).ok().as_deref() == Some(name.as_str()))
                .ok_or_else(|| Error::audio(format!("output device not found: {name}")))?,
            None => host
                .default_output_device()
                .ok_or_else(|| Error::audio("no default output device"))?,
        };
        let cfg = dev.default_output_config().map_err(audio_err)?;
        let out_rate = cfg.sample_rate().0;
        let channels = usize::from(cfg.channels());
        let stream_cfg = cpal::StreamConfig {
            channels: channels as u16,
            sample_rate: cpal::SampleRate(out_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let queue = Arc::new(Mutex::new(VecDeque::<f32>::new()));
        let q = queue.clone();
        let err_fn = |e| tracing::error!("cpal output stream error: {e}");
        let _stream = crate::StreamGuard::start(move || {
            let stream = dev
                .build_output_stream(
                    &stream_cfg,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        // Real-time thread: output silence on try_lock contention
                        // (writer briefly holding the lock); never block/panic
                        match q.try_lock() {
                            Ok(mut q) => {
                                for frame in data.chunks_mut(channels) {
                                    let s = q.pop_front().unwrap_or(0.0);
                                    for c in frame.iter_mut() {
                                        *c = s;
                                    }
                                }
                            }
                            Err(_) => {
                                for c in data.iter_mut() {
                                    *c = 0.0;
                                }
                            }
                        }
                    },
                    err_fn,
                    None,
                )
                .map_err(audio_err)?;
            Ok(stream)
        })?;
        Ok(Self {
            queue,
            out_rate,
            _stream,
        })
    }
}

impl AudioSink for CpalSink {
    fn sample_rate(&self) -> u32 {
        self.out_rate
    }

    fn write(&mut self, samples: &[f32]) -> Result<()> {
        // The queue carries a mono stream at the source sample rate: each
        // source sample occupies one frame, and the callback expands it to
        // the output channels. Never enqueue copies per channel — the queue
        // would drain at 1/channels real time, stretching audio N times and
        // dropping pitch N times (on 16ch, speech turns straight into
        // infrasonic rumble).
        let mut q = self
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for s in samples {
            q.push_back(*s);
        }
        Ok(())
    }
}

fn audio_err(e: impl std::fmt::Display) -> Error {
    Error::audio(e.to_string())
}
