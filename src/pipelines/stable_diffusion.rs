use crate::models::{unet_2d, vae};
use crate::schedulers::ddim;
use crate::schedulers::PredictionType;
use crate::transformers::clip;
use safetensors::tensor::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tch::nn::Variables;
use tch::{nn, Device, TchError, Tensor};

#[derive(Clone, Debug)]
pub struct StableDiffusionConfig {
    pub width: i64,
    pub height: i64,
    pub clip: clip::Config,
    autoencoder: vae::AutoEncoderKLConfig,
    unet: unet_2d::UNet2DConditionModelConfig,
    scheduler: ddim::DDIMSchedulerConfig,
}

impl StableDiffusionConfig {
    pub fn v1_5(
        sliced_attention_size: Option<i64>,
        height: Option<i64>,
        width: Option<i64>,
    ) -> Self {
        let bc = |out_channels, use_cross_attn, attention_head_dim| unet_2d::BlockConfig {
            out_channels,
            use_cross_attn,
            attention_head_dim,
        };
        // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/unet/config.json
        let unet = unet_2d::UNet2DConditionModelConfig {
            blocks: vec![bc(320, true, 8), bc(640, true, 8), bc(1280, true, 8), bc(1280, false, 8)],
            center_input_sample: false,
            cross_attention_dim: 768,
            downsample_padding: 1,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            layers_per_block: 2,
            mid_block_scale_factor: 1.,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            sliced_attention_size,
            use_linear_projection: false,
        };
        let autoencoder = vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
        };
        let height = if let Some(height) = height {
            assert_eq!(height % 8, 0, "heigh has to be divisible by 8");
            height
        } else {
            512
        };

        let width = if let Some(width) = width {
            assert_eq!(width % 8, 0, "width has to be divisible by 8");
            width
        } else {
            512
        };

        Self {
            width,
            height,
            clip: clip::Config::v1_5(),
            autoencoder,
            scheduler: Default::default(),
            unet,
        }
    }

    pub fn v2_1(
        sliced_attention_size: Option<i64>,
        height: Option<i64>,
        width: Option<i64>,
    ) -> Self {
        let bc = |out_channels, use_cross_attn, attention_head_dim| unet_2d::BlockConfig {
            out_channels,
            use_cross_attn,
            attention_head_dim,
        };
        // https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/unet/config.json
        let unet = unet_2d::UNet2DConditionModelConfig {
            blocks: vec![
                bc(320, true, 5),
                bc(640, true, 10),
                bc(1280, true, 20),
                bc(1280, false, 20),
            ],
            center_input_sample: false,
            cross_attention_dim: 1024,
            downsample_padding: 1,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            layers_per_block: 2,
            mid_block_scale_factor: 1.,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            sliced_attention_size,
            use_linear_projection: true,
        };
        // https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/vae/config.json
        let autoencoder = vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
        };
        let scheduler = ddim::DDIMSchedulerConfig {
            prediction_type: PredictionType::VPrediction,
            ..Default::default()
        };

        let height = if let Some(height) = height {
            assert_eq!(height % 8, 0, "heigh has to be divisible by 8");
            height
        } else {
            768
        };

        let width = if let Some(width) = width {
            assert_eq!(width % 8, 0, "width has to be divisible by 8");
            width
        } else {
            768
        };

        Self { width, height, clip: clip::Config::v2_1(), autoencoder, scheduler, unet }
    }

    pub fn build_vae(
        &self,
        vae_weights: &str,
        device: Device,
    ) -> anyhow::Result<vae::AutoEncoderKL> {
        let mut vs_ae = nn::VarStore::new(device);
        // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/config.json
        let autoencoder = vae::AutoEncoderKL::new(vs_ae.root(), 3, 3, self.autoencoder.clone());
        vs_ae.load(vae_weights)?;
        Ok(autoencoder)
    }

    pub fn build_unet(
        &self,
        unet_weights: &str,
        device: Device,
        in_channels: i64,
    ) -> anyhow::Result<unet_2d::UNet2DConditionModel> {
        let mut vs_unet = nn::VarStore::new(device);
        let unet =
            unet_2d::UNet2DConditionModel::new(vs_unet.root(), in_channels, 4, self.unet.clone());
        if unet_weights.ends_with("safetensors") {
            safe_tensor_to_var_store(&mut vs_unet.variables_, unet_weights)?
        } else {
            vs_unet.load(unet_weights)?
        }
        Ok(unet)
    }

    pub fn build_scheduler(&self, n_steps: usize) -> ddim::DDIMScheduler {
        ddim::DDIMScheduler::new(n_steps, self.scheduler)
    }

    pub fn build_clip_transformer(
        &self,
        clip_weights: &str,
        device: tch::Device,
    ) -> anyhow::Result<clip::ClipTextTransformer> {
        let mut vs = tch::nn::VarStore::new(device);
        let text_model = clip::ClipTextTransformer::new(vs.root(), &self.clip);
        vs.load(clip_weights)?;
        Ok(text_model)
    }
}

fn safe_tensor_to_var_store(store: &mut Arc<Mutex<Variables>>, path: &str) -> anyhow::Result<()> {
    let buffer = std::fs::read(path)?;
    let body = SafeTensors::deserialize(&buffer)?;
    let mut named_tensors = HashMap::new();
    for (name, tensor) in body.tensors() {
        named_tensors.insert(name, tensor);
    }
    let mut variables = store.lock().unwrap();
    for (name, var) in variables.named_variables.iter_mut() {
        match named_tensors.get(name) {
            Some(src) => {
                let kind = dtype_to_tkind(&src.dtype());
                let size = shape_to_size(&src.shape());
                let tensor = Tensor::f_of_data_size(src.data(), &size, kind)?;
                tch::no_grad(|| var.f_copy_(&tensor).map_err(|e| e.path_context(name)))?
            }
            None => {
                Err(TchError::TensorNameNotFound(name.to_string(), path.to_string()))?;
            }
        }
    }
    Ok(())
}

#[inline]
fn dtype_to_tkind(dtype: &Dtype) -> tch::Kind {
    match dtype {
        Dtype::F16 => tch::Kind::Half,
        Dtype::BF16 => tch::Kind::BFloat16,
        Dtype::F32 => tch::Kind::Float,
        Dtype::F64 => tch::Kind::Double,
        _ => unimplemented!(),
    }
}

#[inline]
fn shape_to_size(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|x| *x as i64).collect()
}
