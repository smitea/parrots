//! 虚拟音频设备双管道回环测试(需本机已安装驱动,默认 `#[ignore]` 跳过)
//!
//! 运行:`cargo test --release -p parrots-platform-macos -- --ignored loopback`
//!
//! 主测 Parrots 品牌化驱动(计划 3+);BlackHole 官方设备保留为替代路线回归。

use parrots_core::{AudioPlatform, DeviceId};
use parrots_platform_macos::{find_device, MacAudioPlatform};

/// 同一 BlackHole 设备双面回环:先开捕获(避免丢头部),再开播放,
/// 写 1.0s 440Hz 正弦(幅度 0.5,播放端原生采样率),
/// 从捕获端收 ~1.2s 数据(16k/512 块)并验证非静音。
fn run_loopback(device_name: &str) {
    if find_device(device_name).is_none() {
        eprintln!("跳过 {device_name}:未安装驱动(cd driver/macos && ./make-pkg.sh 后双击安装)");
        return;
    }
    let platform = MacAudioPlatform::new();
    let mut capture = platform
        .open_capture(&DeviceId(Some(device_name.to_string())))
        .unwrap_or_else(|e| panic!("打开 {device_name} 捕获失败: {e}"));
    let mut sink = platform
        .open_playback(&DeviceId(Some(device_name.to_string())))
        .unwrap_or_else(|e| panic!("打开 {device_name} 播放失败: {e}"));

    // 播放端原生采样率生成正弦;捕获端由 CpalCapture 统一重采样到 16k
    let out_rate = sink.sample_rate();
    let samples: Vec<f32> = (0..out_rate as usize)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / out_rate as f32).sin())
        .collect();
    sink.write(&samples).expect("写入正弦失败");

    // 捕获固定 16k/512 块;读 ~1.2s = 38 帧 × 512 = 19456 样本
    let mut collected = Vec::with_capacity(38 * 512);
    for _ in 0..38 {
        collected.extend(capture.next_chunk().expect("捕获流意外中断"));
    }

    assert!(!collected.is_empty(), "{device_name}: 未收到任何捕获样本");
    let rms = (collected.iter().map(|s| s * s).sum::<f32>() / collected.len() as f32).sqrt();
    let max_abs = collected.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    eprintln!("{device_name}: out_rate={out_rate} rms={rms:.4} max_abs={max_abs:.4}");
    // 正弦幅度 0.5 → 理论 RMS ~0.35;容忍管道衰减,但要显著高于静音
    assert!(
        rms >= 0.05,
        "{device_name}: 捕获 RMS {rms:.4} < 0.05(近乎静音,回环未通)"
    );
    assert!(
        max_abs > 0.05,
        "{device_name}: 峰值 {max_abs:.4} ≤ 0.05(恒定静音)"
    );
}

#[test]
#[ignore] // 需虚拟驱动:cargo test --release -p parrots-platform-macos -- --ignored loopback
fn loopback_parrots_microphone() {
    run_loopback("Parrots Microphone");
}

#[test]
#[ignore]
fn loopback_parrots_speakers() {
    run_loopback("Parrots Speakers");
}

#[test]
#[ignore] // 替代路线:官方 BlackHole(装了才跑)
fn loopback_blackhole_2ch() {
    run_loopback("BlackHole 2ch");
}

#[test]
#[ignore]
fn loopback_blackhole_16ch() {
    run_loopback("BlackHole 16ch");
}
