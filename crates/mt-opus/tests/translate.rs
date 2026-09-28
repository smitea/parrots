use parrots_core::{Lang, Translator};
use parrots_mt_opus::MarianTranslator;
use std::path::Path;

#[tokio::test]
#[ignore]
async fn en_to_zh_basic() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/mt/en-zh");
    let mt = MarianTranslator::load(&dir, Lang::En, Lang::Zh).unwrap();
    let out = mt
        .translate("Hello, how are you today?", &[])
        .await
        .unwrap();
    eprintln!("译文: {out}");
    assert!(out.contains("你好") || out.contains("您"), "实际: {out}");
}

#[tokio::test]
#[ignore]
async fn zh_to_en_basic() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/mt/zh-en");
    let mt = MarianTranslator::load(&dir, Lang::Zh, Lang::En).unwrap();
    let out = mt.translate("今天天气很好。", &[]).await.unwrap();
    eprintln!("译文: {out}");
    let lower = out.to_lowercase();
    assert!(
        lower.contains("weather") || lower.contains("day"),
        "实际: {out}"
    );
}
