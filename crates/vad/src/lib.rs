pub mod detector;
pub mod silero;

pub use detector::{SpeechDetector, VadConfig, VadEvent};
pub use silero::{SileroVad, VAD_FRAME};
