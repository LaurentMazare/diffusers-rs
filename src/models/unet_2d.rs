//! 2D UNet Denoising Models
//!
//! The 2D Unet models take as input a noisy sample and the current diffusion
//! timestep and return a denoised version of the input.
use crate::models::embeddings::{TimestepEmbedding, Timesteps};
use crate::models::unet_2d_blocks::*;
use tch::{nn, Kind, Tensor};

#[derive(Debug, Clone, Copy)]
pub struct BlockConfig {
    pub out_channels: i64,
    pub use_cross_attn: bool,
    pub attention_head_dim: i64,
}

#[derive(Debug, Clone)]
pub struct UNet2DConditionModelConfig {
    pub center_input_sample: bool,
    pub flip_sin_to_cos: bool,
    pub freq_shift: f64,
    pub blocks: Vec<BlockConfig>,
    pub layers_per_block: i64,
    pub downsample_padding: i64,
    pub mid_block_scale_factor: f64,
    pub norm_num_groups: i64,
    pub norm_eps: f64,
    pub cross_attention_dim: i64,
    pub sliced_attention_size: Option<i64>,
    pub use_linear_projection: bool,
}

impl Default for UNet2DConditionModelConfig {
    fn default() -> Self {
        Self {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: vec![
                BlockConfig { out_channels: 320, use_cross_attn: true, attention_head_dim: 8 },
                BlockConfig { out_channels: 640, use_cross_attn: true, attention_head_dim: 8 },
                BlockConfig { out_channels: 1280, use_cross_attn: true, attention_head_dim: 8 },
                BlockConfig { out_channels: 1280, use_cross_attn: false, attention_head_dim: 8 },
            ],
            layers_per_block: 2,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 1280,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }
}

#[derive(Debug)]
enum UNetDownBlock {
    Basic(DownBlock2D),
    CrossAttn(CrossAttnDownBlock2D),
}

#[derive(Debug)]
enum UNetUpBlock {
    Basic(UpBlock2D),
    CrossAttn(CrossAttnUpBlock2D),
}

#[derive(Debug)]
pub struct UNet2DConditionModel {
    conv_in: nn::Conv2D,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    down_blocks: Vec<UNetDownBlock>,
    mid_block: UNetMidBlock2DCrossAttn,
    up_blocks: Vec<UNetUpBlock>,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2D,
    config: UNet2DConditionModelConfig,
}

impl UNet2DConditionModel {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        config: UNet2DConditionModelConfig,
    ) -> Self {
        let n_blocks = config.blocks.len();
        let b_channels = config.blocks[0].out_channels;
        let bl_channels = config.blocks.last().unwrap().out_channels;
        let bl_attention_head_dim = config.blocks.last().unwrap().attention_head_dim;
        let time_embed_dim = b_channels * 4;
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let conv_in = nn::conv2d(&vs / "conv_in", in_channels, b_channels, 3, conv_cfg);

        let time_proj =
            Timesteps::new(b_channels, config.flip_sin_to_cos, config.freq_shift, vs.device());
        let time_embedding =
            TimestepEmbedding::new(&vs / "time_embedding", b_channels, time_embed_dim);

        let vs_db = &vs / "down_blocks";
        let down_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig { out_channels, use_cross_attn, attention_head_dim } =
                    config.blocks[i];

                // Enable automatic attention slicing if the config sliced_attention_size is set to 0.
                let sliced_attention_size = match config.sliced_attention_size {
                    Some(0) => Some(attention_head_dim / 2),
                    _ => config.sliced_attention_size,
                };

                let in_channels =
                    if i > 0 { config.blocks[i - 1].out_channels } else { b_channels };
                let db_cfg = DownBlock2DConfig {
                    num_layers: config.layers_per_block,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_downsample: i < n_blocks - 1,
                    downsample_padding: config.downsample_padding,
                    ..Default::default()
                };
                if use_cross_attn {
                    let config = CrossAttnDownBlock2DConfig {
                        downblock: db_cfg,
                        attn_num_head_channels: attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                        sliced_attention_size,
                        use_linear_projection: config.use_linear_projection,
                    };
                    let block = CrossAttnDownBlock2D::new(
                        &vs_db / i,
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        config,
                    );
                    UNetDownBlock::CrossAttn(block)
                } else {
                    let block = DownBlock2D::new(
                        &vs_db / i,
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        db_cfg,
                    );
                    UNetDownBlock::Basic(block)
                }
            })
            .collect();

        let mid_cfg = UNetMidBlock2DCrossAttnConfig {
            resnet_eps: config.norm_eps,
            output_scale_factor: config.mid_block_scale_factor,
            cross_attn_dim: config.cross_attention_dim,
            attn_num_head_channels: bl_attention_head_dim,
            resnet_groups: Some(config.norm_num_groups),
            use_linear_projection: config.use_linear_projection,
            ..Default::default()
        };
        let mid_block = UNetMidBlock2DCrossAttn::new(
            &vs / "mid_block",
            bl_channels,
            Some(time_embed_dim),
            mid_cfg,
        );

        let vs_ub = &vs / "up_blocks";
        let up_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig { out_channels, use_cross_attn, attention_head_dim } =
                    config.blocks[n_blocks - 1 - i];

                // Enable automatic attention slicing if the config sliced_attention_size is set to 0.
                let sliced_attention_size = match config.sliced_attention_size {
                    Some(0) => Some(attention_head_dim / 2),
                    _ => config.sliced_attention_size,
                };

                let prev_out_channels =
                    if i > 0 { config.blocks[n_blocks - i].out_channels } else { bl_channels };
                let in_channels = {
                    let index = if i == n_blocks - 1 { 0 } else { n_blocks - i - 2 };
                    config.blocks[index].out_channels
                };
                let ub_cfg = UpBlock2DConfig {
                    num_layers: config.layers_per_block + 1,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_upsample: i < n_blocks - 1,
                    ..Default::default()
                };
                if use_cross_attn {
                    let config = CrossAttnUpBlock2DConfig {
                        upblock: ub_cfg,
                        attn_num_head_channels: attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                        sliced_attention_size,
                        use_linear_projection: config.use_linear_projection,
                    };
                    let block = CrossAttnUpBlock2D::new(
                        &vs_ub / i,
                        in_channels,
                        prev_out_channels,
                        out_channels,
                        Some(time_embed_dim),
                        config,
                    );
                    UNetUpBlock::CrossAttn(block)
                } else {
                    let block = UpBlock2D::new(
                        &vs_ub / i,
                        in_channels,
                        prev_out_channels,
                        out_channels,
                        Some(time_embed_dim),
                        ub_cfg,
                    );
                    UNetUpBlock::Basic(block)
                }
            })
            .collect();

        let group_cfg = nn::GroupNormConfig { eps: config.norm_eps, ..Default::default() };
        let conv_norm_out =
            nn::group_norm(&vs / "conv_norm_out", config.norm_num_groups, b_channels, group_cfg);
        let conv_out = nn::conv2d(&vs / "conv_out", b_channels, out_channels, 3, conv_cfg);
        Self {
            conv_in,
            time_proj,
            time_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            config,
        }
    }
}

impl UNet2DConditionModel {
    pub fn forward(&self, xs: &Tensor, timestep: f64, encoder_hidden_states: &Tensor) -> Tensor {
        let (bsize, _channels, height, width) = xs.size4().unwrap();
        let device = xs.device();
        let n_blocks = self.config.blocks.len();
        let num_upsamplers = n_blocks - 1;
        let default_overall_up_factor = 2i64.pow(num_upsamplers as u32);
        let forward_upsample_size =
            height % default_overall_up_factor != 0 || width % default_overall_up_factor != 0;
        // 0. center input if necessary
        let xs = if self.config.center_input_sample { xs * 2.0 - 1.0 } else { xs.shallow_clone() };
        // 1. time
        let emb = (Tensor::ones([bsize], (Kind::Float, device)) * timestep)
            .apply(&self.time_proj)
            .apply(&self.time_embedding);
        // 2. pre-process
        let xs = xs.apply(&self.conv_in);
        // 3. down
        let mut down_block_res_xs = vec![xs.shallow_clone()];
        let mut xs = xs;
        for down_block in self.down_blocks.iter() {
            let (_xs, res_xs) = match down_block {
                UNetDownBlock::Basic(b) => b.forward(&xs, Some(&emb)),
                UNetDownBlock::CrossAttn(b) => {
                    b.forward(&xs, Some(&emb), Some(encoder_hidden_states))
                }
            };
            down_block_res_xs.extend(res_xs);
            xs = _xs;
        }
        // 4. mid
        let xs = self.mid_block.forward(&xs, Some(&emb), Some(encoder_hidden_states));
        // 5. up
        let mut xs = xs;
        let mut upsample_size = None;
        for (i, up_block) in self.up_blocks.iter().enumerate() {
            let n_resnets = match up_block {
                UNetUpBlock::Basic(b) => b.resnets.len(),
                UNetUpBlock::CrossAttn(b) => b.upblock.resnets.len(),
            };
            let res_xs = down_block_res_xs.split_off(down_block_res_xs.len() - n_resnets);
            if i < n_blocks - 1 && forward_upsample_size {
                let (_, _, h, w) = down_block_res_xs.last().unwrap().size4().unwrap();
                upsample_size = Some((h, w))
            }
            xs = match up_block {
                UNetUpBlock::Basic(b) => b.forward(&xs, &res_xs, Some(&emb), upsample_size),
                UNetUpBlock::CrossAttn(b) => {
                    b.forward(&xs, &res_xs, Some(&emb), upsample_size, Some(encoder_hidden_states))
                }
            };
        }
        // 6. post-process
        xs.apply(&self.conv_norm_out).silu().apply(&self.conv_out)
    }
}
