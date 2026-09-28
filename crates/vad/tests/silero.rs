use parrots_vad::{SileroVad, VAD_FRAME};

fn model() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/vad/silero_vad.onnx")
}

#[test]
#[ignore] // cargo test -p parrots-vad -- --ignored
fn scores_are_valid_and_silence_is_low() {
    let mut vad = SileroVad::load(&model()).unwrap();
    let silence = vec![0.0f32; VAD_FRAME];
    let p_silence = vad.score_frame(&silence).unwrap();
    assert!(p_silence < 0.3, "静音概率 {p_silence} 应低于 0.3");
    let tone: Vec<f32> = (0..VAD_FRAME)
        .map(|i| 0.5 * (i as f32 * 0.2).sin() + 0.3 * (i as f32 * 0.07).sin())
        .collect();
    let mut p_tone = 0.0f32;
    for _ in 0..10 {
        p_tone = p_tone.max(vad.score_frame(&tone).unwrap());
    }
    assert!((0.0..=1.0).contains(&p_tone), "概率输出必须有效: {p_tone}");
}

#[test]
#[ignore] // 真实语音帧必须出高概率(回归:v5 需 64 样本音频上下文)
fn real_speech_scores_high() {
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/en-meeting.wav");
    let mut reader = hound::WavReader::open(fixture).unwrap();
    assert_eq!(reader.spec().sample_rate, 16000);
    assert_eq!(reader.spec().channels, 1);
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect();
    let mut vad = SileroVad::load(&model()).unwrap();
    let mut max_prob = 0.0f32;
    for c in samples.chunks(VAD_FRAME) {
        if c.len() < VAD_FRAME {
            break;
        }
        max_prob = max_prob.max(vad.score_frame(c).unwrap());
    }
    assert!(
        max_prob > 0.5,
        "真实语音最大概率 {max_prob} 应显著高于阈值 0.5"
    );
}
