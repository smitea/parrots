pub mod audio;
pub mod engine_traits;
pub mod error;
pub mod lang;
pub mod platform_traits;
pub mod vad;

pub use audio::{AudioSegment, Transcript, TranscriptUpdate, VoiceProfile};
pub use engine_traits::{AsrEngine, Synthesizer, TextPolisher, Translator};
pub use error::Error;
pub use lang::Lang;
pub use platform_traits::{AudioPlatform, AudioSink, AudioStream, DeviceId, VirtualDeviceInfo};
pub use vad::{SpeechDetector, VadConfig, VadEvent};

pub type Result<T> = std::result::Result<T, Error>;
