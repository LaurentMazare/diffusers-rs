//! # Variational Auto-Encoder (VAE) Models.
//!
//! Auto-encoder models compress their input to a usually smaller latent space
//! before expanding it back to its original shape. This results in the latent values
//! compressing the original information.
use crate::models::unet_2d_blocks::{
    DownEncoderBlock2D, DownEncoderBlock2DConfig, UNetMidBlock2D, UNetMidBlock2DConfig,
    UpDecoderBlock2D, UpDecoderBlock2DConfig,
};
use tch::{nn, nn::Module, Tensor};

#[derive(Debug, Clone)]
struct EncoderConfig {
    // down_block_types: DownEncoderBlock2D
    block_out_channels: Vec<i64>,
    layers_per_block: i64,
    norm_num_groups: i64,
    double_z: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 2,
            norm_num_groups: 32,
            double_z: true,
        }
    }
}

#[derive(Debug)]
struct Encoder {
    conv_in: nn::Conv2D,
    down_blocks: Vec<DownEncoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2D,
    #[allow(dead_code)]
    config: EncoderConfig,
}

impl Encoder {
    fn new(vs: nn::Path, in_channels: i64, out_channels: i64, config: EncoderConfig) -> Self {
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let conv_in =
            nn::conv2d(&vs / "conv_in", in_channels, config.block_out_channels[0], 3, conv_cfg);
        let mut down_blocks = vec![];
        let vs_down_blocks = &vs / "down_blocks";
        for index in 0..config.block_out_channels.len() {
            let out_channels = config.block_out_channels[index];
            let in_channels = if index > 0 {
                config.block_out_channels[index - 1]
            } else {
                config.block_out_channels[0]
            };
            let is_final = index + 1 == config.block_out_channels.len();
            let cfg = DownEncoderBlock2DConfig {
                num_layers: config.layers_per_block,
                resnet_eps: 1e-6,
                resnet_groups: config.norm_num_groups,
                add_downsample: !is_final,
                downsample_padding: 0,
                ..Default::default()
            };
            let down_block =
                DownEncoderBlock2D::new(&vs_down_blocks / index, in_channels, out_channels, cfg);
            down_blocks.push(down_block)
        }
        let last_block_out_channels = *config.block_out_channels.last().unwrap();
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block =
            UNetMidBlock2D::new(&vs / "mid_block", last_block_out_channels, None, mid_cfg);
        let group_cfg = nn::GroupNormConfig { eps: 1e-6, ..Default::default() };
        let conv_norm_out = nn::group_norm(
            &vs / "conv_norm_out",
            config.norm_num_groups,
            last_block_out_channels,
            group_cfg,
        );
        let conv_out_channels = if config.double_z { 2 * out_channels } else { out_channels };
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        let conv_out =
            nn::conv2d(&vs / "conv_out", last_block_out_channels, conv_out_channels, 3, conv_cfg);
        Self { conv_in, down_blocks, mid_block, conv_norm_out, conv_out, config }
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.apply(&self.conv_in);
        for down_block in self.down_blocks.iter() {
            xs = xs.apply(down_block)
        }
        self.mid_block.forward(&xs, None).apply(&self.conv_norm_out).silu().apply(&self.conv_out)
    }
}

#[derive(Debug, Clone)]
struct DecoderConfig {
    // up_block_types: UpDecoderBlock2D
    block_out_channels: Vec<i64>,
    layers_per_block: i64,
    norm_num_groups: i64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self { block_out_channels: vec![64], layers_per_block: 2, norm_num_groups: 32 }
    }
}

#[derive(Debug)]
struct Decoder {
    conv_in: nn::Conv2D,
    up_blocks: Vec<UpDecoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2D,
    #[allow(dead_code)]
    config: DecoderConfig,
}

impl Decoder {
    fn new(vs: nn::Path, in_channels: i64, out_channels: i64, config: DecoderConfig) -> Self {
        let n_block_out_channels = config.block_out_channels.len();
        let last_block_out_channels = *config.block_out_channels.last().unwrap();
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let conv_in =
            nn::conv2d(&vs / "conv_in", in_channels, last_block_out_channels, 3, conv_cfg);
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block =
            UNetMidBlock2D::new(&vs / "mid_block", last_block_out_channels, None, mid_cfg);
        let mut up_blocks = vec![];
        let vs_up_blocks = &vs / "up_blocks";
        let reversed_block_out_channels: Vec<_> =
            config.block_out_channels.iter().copied().rev().collect();
        for index in 0..n_block_out_channels {
            let out_channels = reversed_block_out_channels[index];
            let in_channels = if index > 0 {
                reversed_block_out_channels[index - 1]
            } else {
                reversed_block_out_channels[0]
            };
            let is_final = index + 1 == n_block_out_channels;
            let cfg = UpDecoderBlock2DConfig {
                num_layers: config.layers_per_block + 1,
                resnet_eps: 1e-6,
                resnet_groups: config.norm_num_groups,
                add_upsample: !is_final,
                ..Default::default()
            };
            let up_block =
                UpDecoderBlock2D::new(&vs_up_blocks / index, in_channels, out_channels, cfg);
            up_blocks.push(up_block)
        }
        let group_cfg = nn::GroupNormConfig { eps: 1e-6, ..Default::default() };
        let conv_norm_out = nn::group_norm(
            &vs / "conv_norm_out",
            config.norm_num_groups,
            config.block_out_channels[0],
            group_cfg,
        );
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        let conv_out =
            nn::conv2d(&vs / "conv_out", config.block_out_channels[0], out_channels, 3, conv_cfg);
        Self { conv_in, up_blocks, mid_block, conv_norm_out, conv_out, config }
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = self.mid_block.forward(&xs.apply(&self.conv_in), None);
        for up_block in self.up_blocks.iter() {
            xs = xs.apply(up_block)
        }
        xs.apply(&self.conv_norm_out).silu().apply(&self.conv_out)
    }
}

#[derive(Debug, Clone)]
pub struct AutoEncoderKLConfig {
    pub block_out_channels: Vec<i64>,
    pub layers_per_block: i64,
    pub latent_channels: i64,
    pub norm_num_groups: i64,
}

impl Default for AutoEncoderKLConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 1,
            latent_channels: 4,
            norm_num_groups: 32,
        }
    }
}

pub struct DiagonalGaussianDistribution {
    mean: Tensor,
    std: Tensor,
    device: tch::Device,
}

impl DiagonalGaussianDistribution {
    pub fn new(parameters: &Tensor) -> Self {
        let mut parameters = parameters.chunk(2, 1).into_iter();
        let mean = parameters.next().unwrap();
        let logvar = parameters.next().unwrap();
        let std = (logvar * 0.5).exp();
        let device = std.device();
        DiagonalGaussianDistribution { mean, std, device }
    }

    pub fn sample(&self) -> Tensor {
        let sample = Tensor::randn_like(&self.mean).to(self.device);
        &self.mean + &self.std * sample
    }
}

// https://github.com/huggingface/diffusers/blob/970e30606c2944e3286f56e8eb6d3dc6d1eb85f7/src/diffusers/models/vae.py#L485
// This implementation is specific to the config used in stable-diffusion-v1-5
// https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/config.json
#[derive(Debug)]
pub struct AutoEncoderKL {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: nn::Conv2D,
    post_quant_conv: nn::Conv2D,
    pub config: AutoEncoderKLConfig,
}

impl AutoEncoderKL {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        config: AutoEncoderKLConfig,
    ) -> Self {
        let latent_channels = config.latent_channels;
        let encoder_cfg = EncoderConfig {
            block_out_channels: config.block_out_channels.clone(),
            layers_per_block: config.layers_per_block,
            norm_num_groups: config.norm_num_groups,
            double_z: true,
        };
        let encoder = Encoder::new(&vs / "encoder", in_channels, latent_channels, encoder_cfg);
        let decoder_cfg = DecoderConfig {
            block_out_channels: config.block_out_channels.clone(),
            layers_per_block: config.layers_per_block,
            norm_num_groups: config.norm_num_groups,
        };
        let decoder = Decoder::new(&vs / "decoder", latent_channels, out_channels, decoder_cfg);
        let conv_cfg = Default::default();
        let quant_conv =
            nn::conv2d(&vs / "quant_conv", 2 * latent_channels, 2 * latent_channels, 1, conv_cfg);
        let post_quant_conv =
            nn::conv2d(&vs / "post_quant_conv", latent_channels, latent_channels, 1, conv_cfg);
        Self { encoder, decoder, quant_conv, post_quant_conv, config }
    }

    /// Returns the distribution in the latent space.
    pub fn encode(&self, xs: &Tensor) -> DiagonalGaussianDistribution {
        let parameters = xs.apply(&self.encoder).apply(&self.quant_conv);
        DiagonalGaussianDistribution::new(&parameters)
    }

    /// Takes as input some sampled values.
    pub fn decode(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.post_quant_conv).apply(&self.decoder)
    }
}
