//! 采样步数延迟 A/B(B1,TTS 提速调研;#[ignore],需模型 + fixture 声纹)。
//!
//! zipvoice-distill 为蒸馏模型,支持少步采样:`PARROTS_TTS_NUM_STEPS`
//! 2~5 可调(默认 4)。本测试固定同一文本/声纹,对比 2/3/4/5 步的
//! 合成耗时与音频时长,产物 wav 落盘供人工听测评质。
//!
//! 运行:cargo test --release -p parrots-tts-zipvoice -- --ignored --nocapture steps_latency

use std::path::Path;
use std::time::Instant;

use parrots_core::{Synthesizer, VoiceProfile};
use parrots_tts_zipvoice::{write_wav, ZipvoiceTts};

fn repo_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const TEXT: &str = "好的,我马上把会议纪要发给大家,请各位查收。";

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn steps_latency_ab() {
    let tts = ZipvoiceTts::load(&repo_root().join("models/tts/zipvoice")).unwrap();
    let voice = VoiceProfile::from_prompt(
        repo_root().join("fixtures/en-hello.wav"),
        "Hello, how are you doing today? I hope everything is going well.".into(),
    );
    let out_dir = std::env::temp_dir().join("parrots-tts-steps-ab");
    std::fs::create_dir_all(&out_dir).unwrap();

    println!("{}", "=".repeat(64));
    println!("文本:{TEXT}");
    let mut results = Vec::new();
    for steps in [2, 3, 4, 5] {
        // 通过环境变量切步数(读取发生在每次 synthesize 内)
        std::env::set_var("PARROTS_TTS_NUM_STEPS", steps.to_string());
        // 每档先跑一次热身(消除首次会话/缓存差异),再计时
        let _ = tts.synthesize(TEXT, &voice).await.unwrap();
        let t0 = Instant::now();
        let seg = tts.synthesize(TEXT, &voice).await.unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let rtf = ms / seg.duration_ms() as f64;
        let wav = out_dir.join(format!("steps{steps}.wav"));
        write_wav(&wav, &seg.samples, seg.sample_rate).unwrap();
        let rms = seg.rms();
        println!(
            "steps={steps}: 合成 {ms:.0}ms | 音频 {}ms | RTF {rtf:.2} | rms {rms:.4} | {}",
            seg.duration_ms(),
            wav.display()
        );
        results.push((steps, ms, rtf, rms));
    }
    println!("{}", "=".repeat(64));
    println!("听测文件目录:{out_dir:?}(人工评质:2~5 步音质对比)");

    // 门线:各步数均应有非静音产出
    for (steps, _, _, rms) in &results {
        assert!(*rms > 0.01, "steps={steps} 产出疑似静音 rms={rms}");
    }
}
