use parrots_core::{AudioPlatform, DeviceId};
use parrots_platform_macos::{device_names, find_device, MacAudioPlatform};

#[test]
fn virtual_devices_reflect_installed_parrots_driver() {
    // 与本机实际设备保持一致:装了 Parrots 驱动则 2 个,没装则 0 个
    let expected = ["Parrots Microphone", "Parrots Speakers"]
        .iter()
        .filter(|n| find_device(n).is_some())
        .count();
    assert_eq!(MacAudioPlatform::new().virtual_devices().len(), expected);
    // 枚举接口可用(不断言非空:极端虚拟机可能无设备)
    let _names: Vec<String> = device_names();
}

#[test]
#[ignore] // 本机运行:cargo test -p parrots-platform-macos -- --ignored(需麦克风权限)
fn default_capture_delivers_512_sample_chunks() {
    let p = MacAudioPlatform::new();
    let mut s = p.open_capture(&DeviceId(None)).unwrap();
    assert_eq!(s.sample_rate(), 16000);
    let chunk = s.next_chunk().unwrap();
    assert_eq!(chunk.len(), 512);
}
