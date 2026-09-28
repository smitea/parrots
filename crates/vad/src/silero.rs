use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use parrots_core::{Error, Result};

/// silero-vad v5: 512 samples per frame @16k, state maintained across frames
pub const VAD_FRAME: usize = 512;
/// Total element count of the state tensor with shape (2,1,128)
const STATE_ELEMS: usize = 2 * 128;
/// v5 audio context sample count: model input = context(64) + new samples(512), rolling window maintained by the caller
const CONTEXT_SIZE: usize = 64;

pub struct SileroVad {
    session: Session,
    state: Vec<f32>,
    context: Vec<f32>,
}

impl SileroVad {
    pub fn load(model_path: &std::path::Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_intra_threads(2)
            .map_err(ort_err)?
            .commit_from_file(model_path)
            .map_err(|e| Error::ModelMissing(format!("{}: {e}", model_path.display())))?;
        Ok(Self {
            session,
            state: vec![0.0; STATE_ELEMS],
            context: vec![0.0; CONTEXT_SIZE],
        })
    }

    pub fn reset(&mut self) {
        self.state = vec![0.0; STATE_ELEMS];
        self.context = vec![0.0; CONTEXT_SIZE];
    }

    /// Feed exactly 512 16k samples; returns the speech probability for that frame
    pub fn score_frame(&mut self, samples: &[f32]) -> Result<f32> {
        assert_eq!(samples.len(), VAD_FRAME, "VAD frame size must be 512");
        let sr: i64 = 16000;
        let mut input = Vec::with_capacity(CONTEXT_SIZE + VAD_FRAME);
        input.extend_from_slice(&self.context);
        input.extend_from_slice(samples);
        let inputs = ort::inputs![
            "input" => Tensor::from_array(([1usize, CONTEXT_SIZE + VAD_FRAME], input.clone())).map_err(ort_err)?,
            "state" => Tensor::from_array(([2usize, 1, 128], self.state.clone())).map_err(ort_err)?,
            "sr" => Tensor::from_array(((), vec![sr])).map_err(ort_err)?,
        ];
        let outputs = self.session.run(inputs).map_err(ort_err)?;
        let (_, output) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        let prob = output.iter().copied().fold(f32::MIN, f32::max);
        let (_, new_state) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        self.state = new_state.to_vec();
        self.context = input[input.len() - CONTEXT_SIZE..].to_vec();
        Ok(prob)
    }
}

fn ort_err(e: impl std::fmt::Display) -> Error {
    Error::inference(e.to_string())
}
