//! App settings: JSON persistence next to the user's voice profiles.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Persistent app settings (plan 4 Task 3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    /// Direction A voice profile name (profiles/<name>.{wav,txt})
    pub voice: String,
    /// Capture device override (None = policy default; direction B prefers
    /// `Parrots Speakers`, direction A uses the system default microphone)
    pub input_device: Option<String>,
    /// Playback device override (None = system default; meeting mode picks
    /// `Parrots Microphone` explicitly)
    pub output_device: Option<String>,
    /// VoiceProcessingIO echo cancellation for direction A
    pub aec: bool,
    /// Text polisher (plan 7) enabled
    pub polish: bool,
    /// Playback gate override for direction A: None = auto (off while AEC
    /// active), Some(true/false) forces it
    pub gate_playback: Option<bool>,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            voice: "my".into(),
            input_device: None,
            output_device: None,
            aec: true,
            polish: false,
            gate_playback: None,
        }
    }
}

fn settings_path(profiles_dir: &Path) -> PathBuf {
    profiles_dir.join("app-settings.json")
}

impl AppSettings {
    pub fn load(profiles_dir: &Path) -> Self {
        let path = settings_path(profiles_dir);
        match std::fs::read_to_string(&path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_else(|e| {
                tracing::warn!(
                    "settings parse failed ({e}), using defaults: {}",
                    path.display()
                );
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, profiles_dir: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(profiles_dir)?;
        let path = settings_path(profiles_dir);
        std::fs::write(&path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}
