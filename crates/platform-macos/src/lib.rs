pub mod aec;
pub mod capture;
pub mod playback;
pub mod resample;
pub mod watch;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parrots_core::{
    AudioPlatform, AudioSink, AudioStream, DeviceId, Error, Result, VirtualDeviceInfo,
};

/// cpal::Stream is !Send on macOS (CoreAudio) while AudioStream/AudioSink
/// require Send: the Stream is built, played, and held on a dedicated thread;
/// on Drop a signal makes that thread drop the Stream (i.e. stop the stream).
pub(crate) struct StreamGuard {
    stop: Option<std::sync::mpsc::Sender<()>>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl StreamGuard {
    pub(crate) fn start<F>(build: F) -> Result<Self>
    where
        F: FnOnce() -> Result<cpal::Stream> + Send + 'static,
    {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let join = std::thread::Builder::new()
            .name("cpal-stream-owner".into())
            .spawn(move || {
                let mut stream = match build() {
                    Ok(s) => Some(s),
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                if let Err(e) = stream.as_ref().unwrap().play() {
                    let _ = ready_tx.send(Err(Error::audio(e.to_string())));
                    return;
                }
                let _ = ready_tx.send(Ok(()));
                // Block until the stop signal; then drop the Stream on this
                // thread (stopping capture/playback)
                let _ = stop_rx.recv();
                drop(stream.take());
            })
            .map_err(|e| Error::audio(format!("failed to spawn audio stream thread: {e}")))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                stop: Some(stop_tx),
                join: Some(join),
            }),
            Ok(Err(e)) => {
                let _ = join.join();
                Err(e)
            }
            Err(_) => {
                let _ = join.join();
                Err(Error::audio("audio stream thread exited unexpectedly"))
            }
        }
    }
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// Parrots-branded virtual device names (dual-sided: input+output; Plan 3+,
/// branded fork of the official driver)
const PARROTS_DEVICE_NAMES: [&str; 2] = ["Parrots Microphone", "Parrots Speakers"];

/// Enumerates all audio device names (input + output; order-preserving dedup)
pub fn device_names() -> Vec<String> {
    let host = cpal::default_host();
    let mut names: Vec<String> = Vec::new();
    let inputs = host.input_devices().into_iter().flatten();
    let outputs = host.output_devices().into_iter().flatten();
    for dev in inputs.chain(outputs) {
        // Some devices may refuse to report their name; just skip them
        let name = match dev.name() {
            Ok(n) => n,
            Err(_) => continue,
        };
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// Finds a device by exact name match (returns the device name on hit)
pub fn find_device(name: &str) -> Option<String> {
    device_names().into_iter().find(|n| n == name)
}

pub struct MacAudioPlatform;

impl MacAudioPlatform {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MacAudioPlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioPlatform for MacAudioPlatform {
    fn virtual_devices(&self) -> Vec<VirtualDeviceInfo> {
        // Plan 3+: branded Parrots dual-variant detection (the official
        // BlackHole can still be used explicitly via --device).
        // Virtual devices are dual-sided (input+output); registered on detection.
        let mut devices = Vec::new();
        for name in PARROTS_DEVICE_NAMES {
            if let Some(found) = find_device(name) {
                devices.push(VirtualDeviceInfo {
                    uid: found.clone(),
                    name: found,
                    is_input: true,
                    is_output: true,
                });
            }
        }
        devices
    }

    fn open_capture(&self, device: &DeviceId) -> Result<Box<dyn AudioStream>> {
        Ok(Box::new(capture::CpalCapture::open(device)?))
    }

    fn open_playback(&self, device: &DeviceId) -> Result<Box<dyn AudioSink>> {
        Ok(Box::new(playback::CpalSink::open(device)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_names_no_panic() {
        // Extreme VMs may have no devices at all: only require no panic and a
        // returned Vec; do not assert non-empty
        let names: Vec<String> = device_names();
        let _ = names;
    }

    #[test]
    fn find_device_missing_is_none() {
        assert_eq!(find_device("Definitely Not A Device 12345"), None);
    }

    #[test]
    fn virtual_devices_reflect_reality() {
        let found = PARROTS_DEVICE_NAMES
            .iter()
            .filter(|n| find_device(n).is_some())
            .count();
        let platform = MacAudioPlatform::new();
        assert_eq!(platform.virtual_devices().len(), found);
    }
}
