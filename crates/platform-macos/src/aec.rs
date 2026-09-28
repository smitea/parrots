//! VoiceProcessingIO echo-cancelled capture path (Plan 3+ Phase C).
//!
//! Captures the default microphone via the system-level VPIO audio unit
//! (kAudioUnitSubType_VoiceProcessingIO, the same AEC + AGC + noise
//! suppression as FaceTime/Zoom), outputting 16k mono f32 in 512-sample
//! chunks, matching CpalCapture's AudioStream contract.
//!
//! Threading model: the AudioUnit is created/held/disposed on a dedicated OS
//! thread (CoreAudio callbacks run on system real-time threads, unrelated to
//! the creating thread); the real-time callback only does AudioUnitRender +
//! writes to [`OverwriteRing`](crate::capture::OverwriteRing), and
//! `next_chunk()` pulls blocking.
//!
//! Compatibility note: since the macOS 26 SDK, VPIO no longer accepts an
//! app-preset 16k stream format (AudioUnitInitialize fails with -10875), so
//! rendering happens at the hardware's native rate and this module resamples
//! to 16k. Known boundary: VPIO always uses the system default input device;
//! if VPIO initialization fails, the caller (talk --aec) automatically falls
//! back to normal capture.

use std::sync::{mpsc, Arc};
use std::time::Duration;

use parrots_core::{AudioStream, Error, Result};

use crate::capture::OverwriteRing;

const TARGET_RATE: u32 = 16000;
const CHUNK_OUT: usize = 512;
/// Upper bound on frames rendered per real-time callback (scratch reserve;
/// actual VPIO frame counts are far smaller)
const MAX_FRAMES_PER_CB: usize = 4096;

/// Callback context (accessed by the real-time thread)
struct CallbackCtx {
    unit: coreaudio_sys::AudioUnit,
    ring: Arc<OverwriteRing>,
    /// Render scratch (allocated at thread start, written exclusively by the
    /// real-time thread)
    scratch: std::cell::RefCell<Vec<f32>>,
}

unsafe extern "C" fn input_callback(
    in_ref_con: *mut std::ffi::c_void,
    io_action_flags: *mut coreaudio_sys::AudioUnitRenderActionFlags,
    in_time_stamp: *const coreaudio_sys::AudioTimeStamp,
    _in_bus_number: coreaudio_sys::UInt32,
    in_number_frames: coreaudio_sys::UInt32,
    _io_data: *mut coreaudio_sys::AudioBufferList,
) -> coreaudio_sys::OSStatus {
    unsafe {
        let ctx = &*(in_ref_con as *const CallbackCtx);
        let frames = in_number_frames as usize;
        let mut scratch = ctx.scratch.borrow_mut();
        scratch.resize(frames, 0.0);
        let buffer = coreaudio_sys::AudioBuffer {
            mNumberChannels: 1,
            mDataByteSize: (frames * std::mem::size_of::<f32>()) as u32,
            mData: scratch.as_mut_ptr() as *mut std::ffi::c_void,
        };
        let mut buffer_list = coreaudio_sys::AudioBufferList {
            mNumberBuffers: 1,
            mBuffers: [buffer],
        };
        // Real-time thread: pull (AEC/AGC/NS-processed) audio from input bus 1
        let status = coreaudio_sys::AudioUnitRender(
            ctx.unit,
            io_action_flags,
            in_time_stamp,
            1,
            in_number_frames,
            &mut buffer_list,
        );
        if status != 0 {
            return status;
        }
        for &s in scratch.iter() {
            ctx.ring.push(s);
        }
        0
    }
}

/// VPIO echo-cancelled capture (16k mono).
pub struct AecCapture {
    ring: Arc<OverwriteRing>,
    /// Post-resample 16k buffer awaiting pop (topped up by taking more
    /// windows when short of a 512 chunk)
    out16: Vec<f32>,
    /// Native-rate input samples needed per 512-sample output window
    need: usize,
    in_rate: u32,
    stop_tx: Option<mpsc::Sender<()>>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl AecCapture {
    pub fn open() -> Result<Self> {
        let ring = Arc::new(OverwriteRing::new(1 << 16));
        let (ready_tx, ready_rx) = mpsc::channel::<Result<u32>>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let r = ring.clone();
        let join = std::thread::Builder::new()
            .name("vpio-unit-owner".into())
            .spawn(move || unsafe {
                match build_and_start_unit(r) {
                    Ok((unit, in_rate)) => {
                        let _ = ready_tx.send(Ok(in_rate));
                        // Block until the stop signal; then dispose the unit on this thread
                        let _ = stop_rx.recv();
                        coreaudio_sys::AudioOutputUnitStop(unit);
                        coreaudio_sys::AudioUnitUninitialize(unit);
                        coreaudio_sys::AudioComponentInstanceDispose(unit);
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                    }
                }
            })
            .map_err(|e| Error::audio(format!("failed to spawn VPIO thread: {e}")))?;
        match ready_rx.recv() {
            Ok(Ok(in_rate)) => {
                // Integer ratio (48k→16k) uses exact decimation; otherwise take
                // ceil windows, linearly interpolate, and let the surplus flow
                // into the 16k buffer
                let need = if in_rate % TARGET_RATE == 0 {
                    CHUNK_OUT * (in_rate / TARGET_RATE) as usize
                } else {
                    (CHUNK_OUT as f64 * f64::from(in_rate) / f64::from(TARGET_RATE)).ceil() as usize
                };
                Ok(Self {
                    ring,
                    out16: Vec::new(),
                    need,
                    in_rate,
                    stop_tx: Some(stop_tx),
                    join: Some(join),
                })
            }
            Ok(Err(e)) => {
                let _ = join.join();
                Err(e)
            }
            Err(_) => {
                let _ = join.join();
                Err(Error::audio("VPIO thread exited unexpectedly"))
            }
        }
    }
}

impl Drop for AecCapture {
    fn drop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl AudioStream for AecCapture {
    fn sample_rate(&self) -> u32 {
        TARGET_RATE
    }

    fn next_chunk(&mut self) -> Option<Vec<f32>> {
        // Top up the 16k buffer: take native samples in need-sized windows,
        // resample, and append (surplus <= 1 sample/window)
        while self.out16.len() < CHUNK_OUT {
            let mut raw = Vec::with_capacity(self.need);
            while raw.len() < self.need {
                match self.ring.pop() {
                    Some(s) => raw.push(s),
                    None => std::thread::sleep(Duration::from_millis(2)),
                }
            }
            let resampled = if self.in_rate == TARGET_RATE {
                raw
            } else if self.in_rate % TARGET_RATE == 0 {
                crate::resample::decimate(&raw, (self.in_rate / TARGET_RATE) as usize)
            } else {
                crate::resample::linear_resample(&raw, self.in_rate, TARGET_RATE)
            };
            self.out16.extend(resampled);
        }
        let out = self.out16[..CHUNK_OUT].to_vec();
        self.out16.drain(..CHUNK_OUT);
        Some(out)
    }
}

/// Builds the VPIO unit on the current (dedicated) thread: enable the input
/// bus, attach the input callback, initialize, query the native sample rate,
/// start. Returns (unit, native sample rate).
unsafe fn build_and_start_unit(
    ring: Arc<OverwriteRing>,
) -> Result<(coreaudio_sys::AudioUnit, u32)> {
    unsafe {
        let desc = coreaudio_sys::AudioComponentDescription {
            componentType: coreaudio_sys::kAudioUnitType_Output,
            componentSubType: coreaudio_sys::kAudioUnitSubType_VoiceProcessingIO,
            componentManufacturer: coreaudio_sys::kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0,
        };
        let component = coreaudio_sys::AudioComponentFindNext(std::ptr::null_mut(), &desc);
        if component.is_null() {
            return Err(Error::audio("VoiceProcessingIO audio unit not found"));
        }
        let mut unit: coreaudio_sys::AudioUnit = std::ptr::null_mut();
        let status = coreaudio_sys::AudioComponentInstanceNew(component, &mut unit);
        if status != 0 {
            return Err(Error::audio(format!(
                "AudioComponentInstanceNew failed: {status}"
            )));
        }
        // Failure-cleanup macro: stop → uninitialize → dispose instance
        macro_rules! bail {
            ($msg:expr, $uninit:expr) => {{
                if $uninit {
                    coreaudio_sys::AudioUnitUninitialize(unit);
                }
                coreaudio_sys::AudioComponentInstanceDispose(unit);
                return Err(Error::audio($msg));
            }};
        }

        // Enable input (element 1 = input bus). Output element 0 stays at its
        // default: explicitly disabling it fails initialization on newer
        // macOS; with no render callback attached, the unit only feeds silence
        // to the speakers while providing the AEC reference path.
        let enable: coreaudio_sys::UInt32 = 1;
        let status = coreaudio_sys::AudioUnitSetProperty(
            unit,
            coreaudio_sys::kAudioOutputUnitProperty_EnableIO,
            coreaudio_sys::kAudioUnitScope_Input,
            1,
            &enable as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<coreaudio_sys::UInt32>() as u32,
        );
        if status != 0 {
            bail!(format!("failed to enable VPIO input: {status}"), false);
        }

        // Attach the input callback (SetInputCallback; the callback pulls data
        // from bus 1 via AudioUnitRender)
        let ctx = Box::into_raw(Box::new(CallbackCtx {
            unit,
            ring,
            scratch: std::cell::RefCell::new(Vec::with_capacity(MAX_FRAMES_PER_CB)),
        }));
        let callback = coreaudio_sys::AURenderCallbackStruct {
            inputProc: Some(input_callback),
            inputProcRefCon: ctx as *mut std::ffi::c_void,
        };
        let status = coreaudio_sys::AudioUnitSetProperty(
            unit,
            coreaudio_sys::kAudioOutputUnitProperty_SetInputCallback,
            coreaudio_sys::kAudioUnitScope_Global,
            0,
            &callback as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<coreaudio_sys::AURenderCallbackStruct>() as u32,
        );
        if status != 0 {
            drop(Box::from_raw(ctx));
            bail!(format!("failed to set input callback: {status}"), false);
        }

        let status = coreaudio_sys::AudioUnitInitialize(unit);
        if status != 0 {
            drop(Box::from_raw(ctx));
            bail!(format!("AudioUnitInitialize failed: {status}"), false);
        }

        // Query the input bus's actual (native) sample rate after initialization
        let mut asbd: coreaudio_sys::AudioStreamBasicDescription = std::mem::zeroed();
        let mut prop_size =
            std::mem::size_of::<coreaudio_sys::AudioStreamBasicDescription>() as u32;
        let status = coreaudio_sys::AudioUnitGetProperty(
            unit,
            coreaudio_sys::kAudioUnitProperty_StreamFormat,
            coreaudio_sys::kAudioUnitScope_Output,
            1,
            &mut asbd as *mut _ as *mut std::ffi::c_void,
            &mut prop_size,
        );
        if status != 0 {
            drop(Box::from_raw(ctx));
            bail!(format!("failed to query stream format: {status}"), true);
        }
        if asbd.mSampleRate <= 0.0 {
            drop(Box::from_raw(ctx));
            bail!("invalid VPIO sample rate", true);
        }

        let status = coreaudio_sys::AudioOutputUnitStart(unit);
        if status != 0 {
            drop(Box::from_raw(ctx));
            bail!(format!("AudioOutputUnitStart failed: {status}"), true);
        }
        Ok((unit, asbd.mSampleRate as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Requires microphone permission + the default input device:
    /// cargo test -p parrots-platform-macos -- --ignored aec
    #[test]
    #[ignore]
    fn aec_capture_delivers_512_sample_chunks() {
        let mut cap = AecCapture::open().expect("VPIO initialization failed");
        assert_eq!(cap.sample_rate(), 16000);
        // Collect ~0.5s of data: 16 chunks × 512
        let mut all = Vec::new();
        for _ in 0..16 {
            let chunk = cap
                .next_chunk()
                .expect("VPIO stream interrupted unexpectedly");
            assert_eq!(chunk.len(), 512);
            all.extend(chunk);
        }
        let rms = (all.iter().map(|s| s * s).sum::<f32>() / all.len() as f32).sqrt();
        eprintln!("AecCapture rms={rms:.4} (can be low in a quiet room; path check only)");
        assert!(rms.is_finite() && rms >= 0.0);
    }
}
