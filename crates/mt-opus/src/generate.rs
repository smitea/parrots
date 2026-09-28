use ort::session::Session;
use ort::value::Tensor;
use parrots_core::{Error, Result};

pub const MAX_NEW_TOKENS: usize = 96;

/// Borrows two prebuilt ort sessions; greedy decoding, with the decoder
/// recomputing the full sequence at each step (fast enough for short sentences)
pub struct GreedyDecoder<'a> {
    encoder: &'a mut Session,
    decoder: &'a mut Session,
}

impl<'a> GreedyDecoder<'a> {
    pub fn new(encoder: &'a mut Session, decoder: &'a mut Session) -> Self {
        Self { encoder, decoder }
    }

    pub fn translate(&mut self, tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<String> {
        let pad = id_of(tokenizer, "<pad>")
            .ok_or_else(|| Error::inference("tokenizer missing <pad> token"))?;
        let eos = id_of(tokenizer, "</s>")
            .ok_or_else(|| Error::inference("tokenizer missing </s> token"))?;

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| Error::inference(format!("tokenize failed: {e}")))?;
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let src_len = input_ids.len();

        let encoder_outputs = self
            .encoder
            .run(ort::inputs![
                "input_ids" => Tensor::from_array(([1usize, src_len], input_ids))
                    .map_err(ort_err)?,
                "attention_mask" => Tensor::from_array(([1usize, src_len], attention_mask.clone()))
                    .map_err(ort_err)?,
            ])
            .map_err(ort_err)?;
        let (hidden_shape, hidden_data) = encoder_outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        let dims = hidden_shape.to_vec();
        let hidden_dim = dims
            .get(2)
            .copied()
            .ok_or_else(|| Error::inference("last_hidden_state has rank < 3"))?
            as usize;
        let encoder_hidden_states = hidden_data.to_vec();

        let mut decoder_ids = vec![pad];
        for _ in 0..MAX_NEW_TOKENS {
            let dec_len = decoder_ids.len();
            let outputs = self
                .decoder
                .run(ort::inputs![
                    "input_ids" => Tensor::from_array(([1usize, dec_len], decoder_ids.clone()))
                        .map_err(ort_err)?,
                    "encoder_attention_mask" => Tensor::from_array(
                        ([1usize, src_len], attention_mask.clone())
                    )
                    .map_err(ort_err)?,
                    "encoder_hidden_states" => Tensor::from_array(
                        ([1usize, src_len, hidden_dim], encoder_hidden_states.clone())
                    )
                    .map_err(ort_err)?,
                ])
                .map_err(ort_err)?;
            let (logits_shape, logits) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            let vocab = *logits_shape
                .get(2)
                .ok_or_else(|| Error::inference("logits has rank < 3"))?
                as usize;
            let last = &logits[(dec_len - 1) * vocab..dec_len * vocab];
            let next = argmax(last);
            if next == eos {
                break;
            }
            decoder_ids.push(next);
        }

        let generated: Vec<u32> = decoder_ids[1..].iter().map(|&i| i as u32).collect();
        let out = tokenizer
            .decode(&generated, true)
            .map_err(|e| Error::inference(format!("decode failed: {e}")))?;
        Ok(out.trim().to_string())
    }
}

fn argmax(v: &[f32]) -> i64 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i64)
        .unwrap_or(0)
}

fn id_of(tok: &tokenizers::Tokenizer, special: &str) -> Option<i64> {
    tok.token_to_id(special).map(|i| i as i64)
}

fn ort_err(e: impl std::fmt::Display) -> Error {
    Error::inference(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_max() {
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
    }
}
