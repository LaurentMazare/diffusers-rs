//! # Pipelines

pub mod stable_diffusion_v1_5;
pub mod stable_diffusion_v2_1;

pub mod stable_diffusion {
    use crate::transformers::clip;

    pub fn build_clip_transformer(
        clip_weights: &str,
        config: &clip::Config,
        device: tch::Device,
    ) -> anyhow::Result<clip::ClipTextTransformer> {
        let mut vs = tch::nn::VarStore::new(device);
        let text_model = clip::ClipTextTransformer::new(vs.root(), config);
        vs.load(clip_weights)?;
        Ok(text_model)
    }

    pub use super::stable_diffusion_v1_5 as v1_5;
    pub use super::stable_diffusion_v2_1 as v2_1;
}
