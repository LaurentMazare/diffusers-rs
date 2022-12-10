use crate::models::{unet_2d, vae};
use tch::{nn, Device};

pub fn clip_config() -> crate::transformers::clip::Config {
    crate::transformers::clip::Config::v1_5()
}

pub fn build_vae(vae_weights: &str, device: Device) -> anyhow::Result<vae::AutoEncoderKL> {
    let mut vs_ae = nn::VarStore::new(device);
    // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/config.json
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
    in_channels: i64,
    sliced_attention_size: Option<i64>,
) -> anyhow::Result<unet_2d::UNet2DConditionModel> {
    let mut vs_unet = nn::VarStore::new(device);
    let bc = |out_channels, use_cross_attn, attention_head_dim| unet_2d::BlockConfig {
        out_channels,
        use_cross_attn,
        attention_head_dim,
    };

    // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/unet/config.json
    let unet_cfg = unet_2d::UNet2DConditionModelConfig {
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
    let unet = unet_2d::UNet2DConditionModel::new(vs_unet.root(), in_channels, 4, unet_cfg);
    vs_unet.load(unet_weights)?;
    Ok(unet)
}
