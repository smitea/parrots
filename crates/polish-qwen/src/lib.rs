//! Qwen2.5-1.5B text polish layer (plan 7): pure-Rust candle inference (GGUF Q4), fully local.
//!
//! Context-level polishing of transcript text between ASR and MT: homophone typo
//! fixes (hotword-aware), filler-word cleanup, sentence-boundary smoothing.
//! Greedy decoding (temperature 0) with per-token deadline checks; truncates on
//! timeout: returns the partial output, or falls back to the original text if empty.
//! When the model is missing `ready()`=false and `polish()` passes text through
//! unchanged (zero-cost pipeline degradation).
//!
//! Model selection notes (measured on the plan-7 risk-table fallback path, 2026-09-28):
//! 0.5B-Instruct (Q4/Q8) cannot fix homophone typos ("稀有記"+hotword not fixed; the
//! probe matrix confirmed a capability boundary rather than quantization loss).
//! Upgrading to 1.5B-Instruct Q4_K_M makes hotword repair reliable with zero
//! integration changes; only the model file grows (1.0GB, optional download).

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use parrots_core::{Error, Result, TextPolisher};

/// Polish timeout: truncate when exceeded to keep latency bounded (within the live budget)
const POLISH_TIMEOUT: Duration = Duration::from_millis(800);
/// Max generated tokens (polish tasks are short; rarely reached in practice)
const MAX_NEW_TOKENS: usize = 256;

const SYSTEM_PROMPT: &str = "你是语音转写修正器。把同音错字改成正确汉字(参考领域词表),删除口头语,不增删信息,只输出修正后的文本。";

const FEWSHOT: &str = "转写:我最喜欢看稀有記\n修正:我最喜欢看西游记\n转写:嗯 那个 我们下周三吧然后然后见\n修正:我们下周三见\n";

const MODEL_FILE: &str = "model.gguf";
const TOKENIZER_FILE: &str = "tokenizer.json";

struct Inner {
    model: Mutex<candle_transformers::models::quantized_qwen2::ModelWeights>,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
    eos_ids: Vec<u32>,
}

pub struct QwenPolisher {
    inner: Option<Arc<Inner>>,
    timeout: Duration,
}

impl QwenPolisher {
    /// Override the default timeout (live uses the default 800ms; fixture/batch may relax it)
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Load from a model directory (GGUF + tokenizer.json).
    /// Load failure does not panic: warn + degrade to passthrough (ready()=false).
    pub fn load(model_dir: &Path) -> Self {
        let gguf = model_dir.join(
            std::env::var_os("POLISH_MODEL_FILE")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| MODEL_FILE.into()),
        );
        let tokenizer_path = model_dir.join(TOKENIZER_FILE);
        let inner = (|| -> Result<Arc<Inner>> {
            for f in [&gguf, &tokenizer_path] {
                if !f.is_file() {
                    return Err(Error::audio(format!("model file missing: {}", f.display())));
                }
            }
            // Prefer Metal; fall back to CPU if unavailable
            let device = candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu);
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Error::audio(format!("failed to load tokenizer: {e}")))?;
            let eos_ids = ["<|im_end|>", "<|endoftext|>"]
                .iter()
                .filter_map(|t| tokenizer.token_to_id(t))
                .collect();
            let mut file = std::fs::File::open(&gguf)
                .map_err(|e| Error::audio(format!("failed to open GGUF: {e}")))?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut file)
                .map_err(|e| Error::audio(format!("failed to parse GGUF: {e}")))?;
            let model = candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf(
                content, &mut file, &device,
            )
            .map_err(|e| Error::audio(format!("failed to load model: {e}")))?;
            let inner = Arc::new(Inner {
                model: Mutex::new(model),
                tokenizer,
                device,
                eos_ids,
            });
            tracing::info!("TextPolisher ready (Qwen2.5-1.5B, {})", gguf.display());
            Ok(inner)
        })();
        match inner {
            Ok(inner) => {
                // Warmup: Metal's first kernel compilation takes seconds and must not
                // count against the per-call 800ms deadline (same pattern as MT/TTS warmup)
                let _ = Self::generate(&inner, "预热", &[], None);
                Self {
                    inner: Some(inner),
                    timeout: POLISH_TIMEOUT,
                }
            }
            Err(e) => {
                tracing::warn!("TextPolisher not enabled: {e}");
                Self {
                    inner: None,
                    timeout: POLISH_TIMEOUT,
                }
            }
        }
    }

    fn build_prompt(text: &str, hotwords: &[String]) -> String {
        let hw = if hotwords.is_empty() {
            "无".to_string()
        } else {
            hotwords.join("、")
        };
        format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
             <|im_start|>user\n{FEWSHOT}领域词表:{hw}\n转写:{text}<|im_end|>\n\
             <|im_start|>assistant\n"
        )
    }

    /// Synchronous generation (called on a spawn_blocking thread; greedy decoding;
    /// when deadline = Some, checked per token and truncated on timeout; warmup passes None)
    fn generate(
        inner: &Inner,
        text: &str,
        hotwords: &[String],
        deadline: Option<Duration>,
    ) -> Result<String> {
        let start = Instant::now();
        let prompt = Self::build_prompt(text, hotwords);
        let prompt_tokens: Vec<u32> = inner
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| Error::audio(format!("failed to encode: {e}")))?
            .get_ids()
            .to_vec();
        let mut model = inner.model.lock().unwrap_or_else(|e| e.into_inner());

        let mut all_tokens: Vec<u32> = Vec::new();
        let mut pos = 0usize;
        let mut timed_out = false;
        let mut next_token = *prompt_tokens.last().unwrap_or(&0);
        for index in 0..MAX_NEW_TOKENS {
            if deadline.is_some_and(|d| start.elapsed() > d) {
                timed_out = true;
                break;
            }
            let context: Vec<u32> = if index == 0 {
                prompt_tokens.clone()
            } else {
                vec![next_token]
            };
            let input = candle_core::Tensor::new(context.as_slice(), &inner.device)
                .map_err(|e| Error::audio(format!("failed to create input tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| Error::audio(format!("unsqueeze failed: {e}")))?;
            let logits = model
                .forward(&input, pos)
                .map_err(|e| Error::audio(format!("forward failed: {e}")))?;
            pos += context.len();
            let logits = logits
                .squeeze(0)
                .map_err(|e| Error::audio(format!("squeeze failed: {e}")))?;
            // First pass processes multiple tokens: take the logits at the last position
            let last = if logits.dims().len() > 1 {
                let seq_len = logits.dims()[0];
                logits
                    .get(seq_len - 1)
                    .map_err(|e| Error::audio(format!("failed to get last-position logits: {e}")))?
            } else {
                logits
            };
            let token = last
                .argmax(candle_core::D::Minus1)
                .map_err(|e| Error::audio(format!("argmax failed: {e}")))?
                .to_scalar::<u32>()
                .map_err(|e| Error::audio(format!("failed to get token: {e}")))?;
            if inner.eos_ids.contains(&token) {
                break;
            }
            all_tokens.push(token);
            next_token = token;
        }
        if timed_out && all_tokens.is_empty() {
            // Timed out with no output: fall back to the original text
            return Ok(text.to_string());
        }
        let out = inner
            .tokenizer
            .decode(&all_tokens, true)
            .map_err(|e| Error::audio(format!("failed to decode: {e}")))?;
        // The model occasionally leaks the few-shot label into the output; strip it
        let trimmed = out.trim();
        let stripped = trimmed
            .strip_prefix("修正:")
            .or_else(|| trimmed.strip_prefix("修正："))
            .or_else(|| trimmed.strip_prefix("Output:"))
            .unwrap_or(trimmed);
        let out = stripped.trim().to_string();
        if out.is_empty() {
            Ok(text.to_string())
        } else {
            Ok(out)
        }
    }
}

#[async_trait::async_trait]
impl TextPolisher for QwenPolisher {
    async fn polish(&self, text: &str, hotwords: &[String]) -> Result<String> {
        let Some(inner) = &self.inner else {
            return Ok(text.to_string());
        };
        let inner = inner.clone();
        let raw = text.to_string();
        let raw2 = raw.clone();
        let hotwords = hotwords.to_vec();
        let timeout = self.timeout;
        let result = tokio::task::spawn_blocking(move || {
            Self::generate(&inner, &raw2, &hotwords, Some(timeout))
        })
        .await
        .map_err(|e| Error::audio(format!("polish task failed: {e}")))?;
        match result {
            Ok(out) => Ok(out),
            Err(e) => {
                tracing::warn!("TextPolisher error, falling back to original text: {e}");
                Ok(raw)
            }
        }
    }

    fn ready(&self) -> bool {
        self.inner.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models_root() -> std::path::PathBuf {
        // cargo test cwd = package dir → repo root is two levels up; PARROTS_MODELS overrides
        std::env::var_os("PARROTS_MODELS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
            })
    }

    fn load() -> QwenPolisher {
        let p = QwenPolisher::load(&models_root().join("polish/qwen1.5b"));
        assert!(
            p.ready(),
            "model not ready (run download-models.sh to download the polish model first)"
        );
        // Tests focus on polish quality: relax the deadline (the latency bound is guaranteed by construction)
        p.with_timeout(Duration::from_millis(2500))
    }

    /// Requires models: cargo test -p parrots-polish-qwen -- --ignored --nocapture
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn hotword_correction() {
        let p = load();
        let out = p
            .polish("我再看稀有記", &["西游记".to_string()])
            .await
            .unwrap();
        println!("hotword: {out}");
        assert!(
            out.contains("西游记"),
            "should be polished to the hotword '西游记', got: {out}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn filler_word_cleanup() {
        let p = load();
        let raw = "嗯,那个,我觉得这个方案还行,然后然后就这样吧";
        let out = p.polish(raw, &[]).await.unwrap();
        println!("filler: {out}");
        assert!(
            !out.contains("然后然后"),
            "filler words should be cleaned up, got: {out}"
        );
        assert!(
            out.chars().count() <= raw.chars().count(),
            "polish should not lengthen the text"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn timeout_falls_back_to_original() {
        let p = load();
        // Inject a zero timeout: generation truncates with no output → fall back to the original
        let p = QwenPolisher {
            inner: p.inner.clone(),
            timeout: Duration::ZERO,
        };
        let raw = "这是一个测试句子";
        let out = p.polish(raw, &[]).await.unwrap();
        assert_eq!(
            out, raw,
            "zero timeout should fall back to the original text"
        );
    }

    /// Missing model: no panic, ready()=false, polish passes through
    #[tokio::test]
    async fn missing_model_passthrough() {
        let p = QwenPolisher::load(Path::new("/nonexistent/polish"));
        assert!(!p.ready());
        let out = p.polish("原文", &[]).await.unwrap();
        assert_eq!(out, "原文");
    }
}
