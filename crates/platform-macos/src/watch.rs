//! Audio device change watcher (Plan 3+ Phase B): polls the device topology
//! every 1s, triggering a callback on change.

use cpal::traits::{DeviceTrait, HostTrait};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Device topology change callback (called synchronously on the watcher
/// thread; must be lightweight and non-blocking)
pub type DeviceChangeCallback = Arc<dyn Fn() + Send + Sync>;

const POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Fingerprint of the current device topology: a hash of all device names
/// (sorted) plus the default input/output device names.
/// Only for comparing two samples within the same process (the hash algorithm
/// may change across Rust versions; do not persist it to disk).
pub fn device_fingerprint() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut names = crate::device_names();
    names.sort();
    let host = cpal::default_host();
    let default_in = host.default_input_device().and_then(|d| d.name().ok());
    let default_out = host.default_output_device().and_then(|d| d.name().ok());
    let mut h = DefaultHasher::new();
    names.hash(&mut h);
    default_in.hash(&mut h);
    default_out.hash(&mut h);
    h.finish()
}

/// A background thread samples the fingerprint once per second and fires the
/// callback on change; `Drop` stops and joins the thread (blocking at most
/// one polling period).
pub struct DeviceWatcher {
    stop: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl DeviceWatcher {
    pub fn new(on_change: DeviceChangeCallback) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = stop.clone();
        let join = std::thread::Builder::new()
            .name("device-watcher".into())
            .spawn(move || {
                let mut last = device_fingerprint();
                while !stop_flag.load(Ordering::Relaxed) {
                    std::thread::sleep(POLL_INTERVAL);
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    let cur = device_fingerprint();
                    if cur != last {
                        last = cur;
                        tracing::info!("audio device change detected");
                        on_change();
                    }
                }
            })
            .expect("failed to spawn device watcher thread");
        Self {
            stop,
            join: Some(join),
        }
    }
}

impl Drop for DeviceWatcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn fingerprint_stable_across_samples() {
        // Two samples without plug/unplug events must produce identical
        // fingerprints
        let a = device_fingerprint();
        let b = device_fingerprint();
        assert_eq!(a, b);
    }

    #[test]
    fn watcher_polls_and_stops_cleanly() {
        // Plug/unplug cannot be simulated in tests: only verify the polling
        // thread starts/stops cleanly and the callback channel works
        let fires = Arc::new(AtomicUsize::new(0));
        let fires_cb = fires.clone();
        let watcher = DeviceWatcher::new(Arc::new(move || {
            fires_cb.fetch_add(1, Ordering::Relaxed);
        }));
        std::thread::sleep(Duration::from_millis(1300)); // at least one polling period
        drop(watcher);
        // The callback must not fire when no device changes
        assert_eq!(fires.load(Ordering::Relaxed), 0);
    }
}
