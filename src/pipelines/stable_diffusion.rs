use crate::{
    models::{unet_2d, vae},
    transformers::clip,
};
use tch::{nn, Device};

pub fn build_clip_transformer(
    clip_weights: &str,
    device: Device,
) -> anyhow::Result<clip::ClipTextTransformer> {
    let mut vs = nn::VarStore::new(device);
    let text_model = clip::ClipTextTransformer::new(vs.root());
    vs.load(clip_weights)?;
    Ok(text_model)
}

pub fn build_vae(vae_weights: &str, device: Device) -> anyhow::Result<vae::AutoEncoderKL> {
    let mut vs_ae = nn::VarStore::new(device);
    // https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/vae/config.json
    let autoencoder_cfg = vae::AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
    };
    let autoencoder = vae::AutoEncoderKL::new(vs_ae.root(), 3, 3, autoencoder_cfg);
    vs_ae.load(vae_weights)?;
    Ok(autoencoder)
}

pub fn build_unet(
    unet_weights: &str,
    device: Device,
    sliced_attention_size: Option<i64>,
) -> anyhow::Result<unet_2d::UNet2DConditionModel> {
    let mut vs_unet = nn::VarStore::new(device);
    // https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/config.json
    let unet_cfg = unet_2d::UNet2DConditionModelConfig {
        attention_head_dim: 8,
        blocks: vec![
            unet_2d::BlockConfig { out_channels: 320, use_cross_attn: true },
            unet_2d::BlockConfig { out_channels: 640, use_cross_attn: true },
            unet_2d::BlockConfig { out_channels: 1280, use_cross_attn: true },
            unet_2d::BlockConfig { out_channels: 1280, use_cross_attn: false },
        ],
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
    };
    let unet = unet_2d::UNet2DConditionModel::new(vs_unet.root(), 4, 4, unet_cfg);
    vs_unet.load(unet_weights)?;
    Ok(unet)
}
