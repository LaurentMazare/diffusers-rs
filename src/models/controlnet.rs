// https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnet.py
use super::unet_2d::{BlockConfig, UNetDownBlock};
use crate::models::embeddings::{TimestepEmbedding, Timesteps};
use crate::models::unet_2d_blocks::*;
use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct ControlNetConditioningEmbedding {
    conv_in: nn::Conv2D,
    conv_out: nn::Conv2D,
    blocks: Vec<(nn::Conv2D, nn::Conv2D)>,
}

impl ControlNetConditioningEmbedding {
    pub fn new(
        vs: nn::Path,
        conditioning_embedding_channels: i64,
        conditioning_channels: i64,
        blocks: &[i64],
    ) -> Self {
        let b_channels = blocks[0];
        let bl_channels = *blocks.last().unwrap();
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        let conv_cfg2 = nn::ConvConfig { stride: 2, padding: 1, ..Default::default() };
        let conv_in = nn::conv2d(&vs / "conv_in", conditioning_channels, b_channels, 3, conv_cfg);
        let conv_out =
            nn::conv2d(&vs / "conv_out", bl_channels, conditioning_embedding_channels, 3, conv_cfg);
        let vs_b = &vs / "blocks";
        let blocks = (0..(blocks.len() - 1))
            .map(|i| {
                let channel_in = blocks[i];
                let channel_out = blocks[i + 1];
                let c1 = nn::conv2d(&vs_b / (2 * i), channel_in, channel_in, 3, conv_cfg);
                let c2 = nn::conv2d(&vs_b / (2 * i + 1), channel_in, channel_out, 3, conv_cfg2);
                (c1, c2)
            })
            .collect();
        Self { conv_in, conv_out, blocks }
    }
}

impl tch::nn::Module for ControlNetConditioningEmbedding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.apply(&self.conv_in).silu();
        for (c1, c2) in self.blocks.iter() {
            xs = xs.apply(c1).silu().apply(c2).silu();
        }
        xs.apply(&self.conv_out)
    }
}

pub struct ControlNetConfig {
    pub flip_sin_to_cos: bool,
    pub freq_shift: f64,
    pub blocks: Vec<BlockConfig>,
    pub conditioning_embedding_out_channels: Vec<i64>,
    pub layers_per_block: i64,
    pub downsample_padding: i64,
    pub mid_block_scale_factor: f64,
    pub norm_num_groups: i64,
    pub norm_eps: f64,
    pub cross_attention_dim: i64,
    pub use_linear_projection: bool,
}

impl Default for ControlNetConfig {
    // https://huggingface.co/lllyasviel/sd-controlnet-canny/blob/main/config.json
    fn default() -> Self {
        Self {
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: vec![
                BlockConfig { out_channels: 320, use_cross_attn: true, attention_head_dim: 8 },
                BlockConfig { out_channels: 640, use_cross_attn: true, attention_head_dim: 8 },
                BlockConfig { out_channels: 1280, use_cross_attn: true, attention_head_dim: 8 },
                BlockConfig { out_channels: 1280, use_cross_attn: false, attention_head_dim: 8 },
            ],
            conditioning_embedding_out_channels: vec![16, 32, 96, 256],
            layers_per_block: 2,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            // The default value for the following is 1280 in diffusers/models/controlnet.py but
            // 768 in the actual config file.
            cross_attention_dim: 768,
            use_linear_projection: false,
        }
    }
}

#[allow(dead_code)]
pub struct ControlNet {
    conv_in: nn::Conv2D,
    controlnet_mid_block: nn::Conv2D,
    controlnet_cond_embedding: ControlNetConditioningEmbedding,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    down_blocks: Vec<UNetDownBlock>,
    controlnet_down_blocks: Vec<nn::Conv2D>,
    mid_block: UNetMidBlock2DCrossAttn,
    pub config: ControlNetConfig,
}

impl ControlNet {
    pub fn new(vs: nn::Path, in_channels: i64, config: ControlNetConfig) -> Self {
        let n_blocks = config.blocks.len();
        let b_channels = config.blocks[0].out_channels;
        let bl_channels = config.blocks.last().unwrap().out_channels;
        let time_embed_dim = b_channels * 4;
        let time_proj =
            Timesteps::new(b_channels, config.flip_sin_to_cos, config.freq_shift, vs.device());
        let time_embedding =
            TimestepEmbedding::new(&vs / "time_embedding", b_channels, time_embed_dim);
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let conv_in = nn::conv2d(&vs / "conv_in", in_channels, b_channels, 3, conv_cfg);
        let controlnet_mid_block = nn::conv2d(
            &vs / "controlnet_mid_block",
            bl_channels,
            bl_channels,
            1,
            Default::default(),
        );
        let controlnet_cond_embedding = ControlNetConditioningEmbedding::new(
            &vs / "controlnet_cond_embedding",
            b_channels,
            3,
            &config.conditioning_embedding_out_channels,
        );
        let vs_db = &vs / "down_blocks";
        let down_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig { out_channels, use_cross_attn, attention_head_dim } =
                    config.blocks[i];

                let in_channels =
                    if i > 0 { config.blocks[i - 1].out_channels } else { b_channels };
                let db_cfg = DownBlock2DConfig {
                    num_layers: config.layers_per_block,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_downsample: i < n_blocks - 1,
                    downsample_padding: config.downsample_padding,
                    output_scale_factor: 1.,
                };
                if use_cross_attn {
                    let config = CrossAttnDownBlock2DConfig {
                        downblock: db_cfg,
                        attn_num_head_channels: attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                        sliced_attention_size: None,
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
        let bl_channels = config.blocks.last().unwrap().out_channels;
        let bl_attention_head_dim = config.blocks.last().unwrap().attention_head_dim;
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

        let vs_c = &vs / "controlnet_down_blocks";
        let controlnet_block = nn::conv2d(&vs_c / 0, b_channels, b_channels, 1, Default::default());
        let mut controlnet_down_blocks = vec![controlnet_block];
        for (i, block) in config.blocks.iter().enumerate() {
            let out_channels = block.out_channels;
            for _ in 0..config.layers_per_block {
                let conv1 = nn::conv2d(
                    &vs_c / controlnet_down_blocks.len(),
                    out_channels,
                    out_channels,
                    1,
                    Default::default(),
                );
                controlnet_down_blocks.push(conv1);
            }
            if i + 1 != config.blocks.len() {
                let conv2 = nn::conv2d(
                    &vs_c / controlnet_down_blocks.len(),
                    out_channels,
                    out_channels,
                    1,
                    Default::default(),
                );
                controlnet_down_blocks.push(conv2);
            }
        }

        Self {
            conv_in,
            controlnet_mid_block,
            controlnet_cond_embedding,
            controlnet_down_blocks,
            time_proj,
            time_embedding,
            down_blocks,
            mid_block,
            config,
        }
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        controlnet_cond: &Tensor,
        conditioning_scale: f64,
    ) -> (Vec<Tensor>, Tensor) {
        let (bsize, _channels, _height, _width) = xs.size4().unwrap();
        let device = xs.device();
        // Only support:
        // - The default channel order (rgb).
        // - No class embedding, class_embed_type and num_class_embeds are both None.
        // - No guess mode.

        // 1. Time
        let emb = (Tensor::ones([bsize], (Kind::Float, device)) * timestep)
            .apply(&self.time_proj)
            .apply(&self.time_embedding);

        // 2. Pre-process.
        let xs = xs.apply(&self.conv_in);
        let controlnet_cond = controlnet_cond.apply(&self.controlnet_cond_embedding);
        let xs = xs + controlnet_cond;

        // 3. Down.
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

        // 4. Mid.
        let xs = self.mid_block.forward(&xs, Some(&emb), Some(encoder_hidden_states));

        // 5. ControlNet blocks.
        let controlnet_down_block_res_xs = self
            .controlnet_down_blocks
            .iter()
            .enumerate()
            .map(|(i, block)| block.forward(&down_block_res_xs[i]) * conditioning_scale)
            .collect::<Vec<_>>();

        let xs = xs.apply(&self.controlnet_mid_block);
        (controlnet_down_block_res_xs, xs * conditioning_scale)
    }
}
