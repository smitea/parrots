use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unsupported language: {0}")]
    UnsupportedLang(String),
    #[error("model file missing: {0}")]
    ModelMissing(String),
    #[error("audio error: {0}")]
    Audio(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
}

impl Error {
    pub fn inference(msg: impl Into<String>) -> Self {
        Self::Inference(msg.into())
    }
    pub fn audio(msg: impl Into<String>) -> Self {
        Self::Audio(msg.into())
    }
}
