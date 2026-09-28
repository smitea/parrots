/// Integer-ratio decimation downsampling (in-block averaging for
/// anti-aliasing). 48k→16k uses ratio=3.
pub fn decimate(samples: &[f32], ratio: usize) -> Vec<f32> {
    if ratio <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(ratio)
        .map(|c| c.iter().sum::<f32>() / ratio as f32)
        .collect()
}

/// Arbitrary-ratio linear-interpolation resampling (TTS 24k → output device
/// 48k, etc.)
pub fn linear_resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = f64::from(from) / f64::from(to);
    let n_out = ((samples.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let pos = i as f64 * ratio;
        let i0 = pos as usize;
        let i1 = (i0 + 1).min(samples.len() - 1);
        let f = (pos - i0 as f64) as f32;
        out.push(samples[i0] * (1.0 - f) + samples[i1] * f);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimate_3x_averages() {
        assert_eq!(decimate(&[0.0, 0.3, 0.6, 1.0, 1.0, 1.0], 3), vec![0.3, 1.0]);
    }

    #[test]
    fn decimate_tail_dropped() {
        assert_eq!(decimate(&[1.0, 1.0], 3).len(), 0);
    }

    #[test]
    fn linear_24k_to_48k_doubles_length() {
        let out = linear_resample(&[0.0, 1.0], 24000, 48000);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linear_same_rate_passthrough() {
        let v = vec![0.5f32; 10];
        assert_eq!(linear_resample(&v, 16000, 16000), v);
    }
}
