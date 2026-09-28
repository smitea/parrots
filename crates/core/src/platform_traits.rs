use crate::Result;

/// Abstract input stream: pull-based producer of 16kHz mono f32 chunks (resampling/downmixing done inside the implementation)
pub trait AudioStream: Send {
    fn sample_rate(&self) -> u32;
    /// Blockingly pull the next 512-sample chunk; None = end of stream
    fn next_chunk(&mut self) -> Option<Vec<f32>>;
}

/// Abstract output sink
pub trait AudioSink: Send {
    fn sample_rate(&self) -> u32;
    fn write(&mut self, samples: &[f32]) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct VirtualDeviceInfo {
    pub name: String,
    pub uid: String,
    pub is_input: bool,
    pub is_output: bool,
}

/// None = system default device
#[derive(Debug, Clone, Default)]
pub struct DeviceId(pub Option<String>);

/// Platform abstraction: engines are OS-agnostic; a new platform = a new platform-* implementation
pub trait AudioPlatform: Send + Sync {
    /// Virtual device install detection (plan 1 returns empty; plan 3 wires in the driver)
    fn virtual_devices(&self) -> Vec<VirtualDeviceInfo>;
    fn open_capture(&self, device: &DeviceId) -> Result<Box<dyn AudioStream>>;
    fn open_playback(&self, device: &DeviceId) -> Result<Box<dyn AudioSink>>;
}
