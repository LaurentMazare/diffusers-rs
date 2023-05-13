//! 2D UNet Building Blocks
//!
use crate::models::attention::{
    AttentionBlock, AttentionBlockConfig, SpatialTransformer, SpatialTransformerConfig,
};
use crate::models::resnet::{ResnetBlock2D, ResnetBlock2DConfig};
use tch::{nn, nn::Module, Tensor};

#[derive(Debug)]
struct Downsample2D {
    conv: Option<nn::Conv2D>,
    padding: i64,
}

impl Downsample2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        use_conv: bool,
        out_channels: i64,
        padding: i64,
    ) -> Self {
        let conv = if use_conv {
            let config = nn::ConvConfig { stride: 2, padding, ..Default::default() };
            let conv = nn::conv2d(&vs / "conv", in_channels, out_channels, 3, config);
            Some(conv)
        } else {
            None
        };
        Downsample2D { conv, padding }
    }
}

impl Module for Downsample2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match &self.conv {
            None => xs.avg_pool2d([2, 2], [2, 2], [0, 0], false, true, None),
            Some(conv) => {
                if self.padding == 0 {
                    xs.pad([0, 1, 0, 1], "constant", Some(0.)).apply(conv)
                } else {
                    xs.apply(conv)
                }
            }
        }
    }
}

// This does not support the conv-transpose mode.
#[derive(Debug)]
struct Upsample2D {
    conv: nn::Conv2D,
}

impl Upsample2D {
    fn new(vs: nn::Path, in_channels: i64, out_channels: i64) -> Self {
        let config = nn::ConvConfig { padding: 1, ..Default::default() };
        let conv = nn::conv2d(&vs / "conv", in_channels, out_channels, 3, config);
        Self { conv }
    }
}

impl Upsample2D {
    fn forward(&self, xs: &Tensor, size: Option<(i64, i64)>) -> Tensor {
        let xs = match size {
            None => {
                // The following does not work and it's tricky to pass no fixed
                // dimensions so hack our way around this.
                // xs.upsample_nearest2d(&[], Some(2.), Some(2.)
                let (_bsize, _channels, h, w) = xs.size4().unwrap();
                xs.upsample_nearest2d([2 * h, 2 * w], Some(2.), Some(2.))
            }
            Some((h, w)) => xs.upsample_nearest2d([h, w], None, None),
        };
        xs.apply(&self.conv)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DownEncoderBlock2DConfig {
    pub num_layers: i64,
    pub resnet_eps: f64,
    pub resnet_groups: i64,
    pub output_scale_factor: f64,
    pub add_downsample: bool,
    pub downsample_padding: i64,
}

impl Default for DownEncoderBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_downsample: true,
            downsample_padding: 1,
        }
    }
}

#[derive(Debug)]
pub struct DownEncoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<Downsample2D>,
    pub config: DownEncoderBlock2DConfig,
}

impl DownEncoderBlock2D {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        config: DownEncoderBlock2DConfig,
    ) -> Self {
        let resnets: Vec<_> = {
            let vs = &vs / "resnets";
            let conv_cfg = ResnetBlock2DConfig {
                eps: config.resnet_eps,
                out_channels: Some(out_channels),
                groups: config.resnet_groups,
                output_scale_factor: config.output_scale_factor,
                temb_channels: None,
                ..Default::default()
            };
            (0..(config.num_layers))
                .map(|i| {
                    let in_channels = if i == 0 { in_channels } else { out_channels };
                    ResnetBlock2D::new(&vs / i, in_channels, conv_cfg)
                })
                .collect()
        };
        let downsampler = if config.add_downsample {
            let downsample = Downsample2D::new(
                &(&vs / "downsamplers") / 0,
                out_channels,
                true,
                out_channels,
                config.downsample_padding,
            );
            Some(downsample)
        } else {
            None
        };
        Self { resnets, downsampler, config }
    }
}

impl Module for DownEncoderBlock2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.shallow_clone();
        for resnet in self.resnets.iter() {
            xs = resnet.forward(&xs, None)
        }
        match &self.downsampler {
            Some(downsampler) => xs.apply(downsampler),
            None => xs,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UpDecoderBlock2DConfig {
    pub num_layers: i64,
    pub resnet_eps: f64,
    pub resnet_groups: i64,
    pub output_scale_factor: f64,
    pub add_upsample: bool,
}

impl Default for UpDecoderBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_upsample: true,
        }
    }
}

#[derive(Debug)]
pub struct UpDecoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Upsample2D>,
    pub config: UpDecoderBlock2DConfig,
}

impl UpDecoderBlock2D {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        config: UpDecoderBlock2DConfig,
    ) -> Self {
        let resnets: Vec<_> = {
            let vs = &vs / "resnets";
            let conv_cfg = ResnetBlock2DConfig {
                out_channels: Some(out_channels),
                eps: config.resnet_eps,
                groups: config.resnet_groups,
                output_scale_factor: config.output_scale_factor,
                temb_channels: None,
                ..Default::default()
            };
            (0..(config.num_layers))
                .map(|i| {
                    let in_channels = if i == 0 { in_channels } else { out_channels };
                    ResnetBlock2D::new(&vs / i, in_channels, conv_cfg)
                })
                .collect()
        };
        let upsampler = if config.add_upsample {
            let upsample = Upsample2D::new(&vs / "upsamplers" / 0, out_channels, out_channels);
            Some(upsample)
        } else {
            None
        };
        Self { resnets, upsampler, config }
    }
}

impl Module for UpDecoderBlock2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.shallow_clone();
        for resnet in self.resnets.iter() {
            xs = resnet.forward(&xs, None)
        }
        match &self.upsampler {
            Some(upsampler) => upsampler.forward(&xs, None),
            None => xs,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UNetMidBlock2DConfig {
    pub num_layers: i64,
    pub resnet_eps: f64,
    pub resnet_groups: Option<i64>,
    pub attn_num_head_channels: Option<i64>,
    // attention_type "default"
    pub output_scale_factor: f64,
}

impl Default for UNetMidBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: Some(32),
            attn_num_head_channels: Some(1),
            output_scale_factor: 1.,
        }
    }
}

#[derive(Debug)]
pub struct UNetMidBlock2D {
    resnet: ResnetBlock2D,
    attn_resnets: Vec<(AttentionBlock, ResnetBlock2D)>,
    pub config: UNetMidBlock2DConfig,
}

impl UNetMidBlock2D {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        temb_channels: Option<i64>,
        config: UNetMidBlock2DConfig,
    ) -> Self {
        let vs_resnets = &vs / "resnets";
        let vs_attns = &vs / "attentions";
        let resnet_groups = config.resnet_groups.unwrap_or_else(|| i64::min(in_channels / 4, 32));
        let resnet_cfg = ResnetBlock2DConfig {
            eps: config.resnet_eps,
            groups: resnet_groups,
            output_scale_factor: config.output_scale_factor,
            temb_channels,
            ..Default::default()
        };
        let resnet = ResnetBlock2D::new(&vs_resnets / "0", in_channels, resnet_cfg);
        let attn_cfg = AttentionBlockConfig {
            num_head_channels: config.attn_num_head_channels,
            num_groups: resnet_groups,
            rescale_output_factor: config.output_scale_factor,
            eps: config.resnet_eps,
        };
        let mut attn_resnets = vec![];
        for index in 0..config.num_layers {
            let attn = AttentionBlock::new(&vs_attns / index, in_channels, attn_cfg);
            let resnet = ResnetBlock2D::new(&vs_resnets / (index + 1), in_channels, resnet_cfg);
            attn_resnets.push((attn, resnet))
        }
        Self { resnet, attn_resnets, config }
    }

    pub fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> Tensor {
        let mut xs = self.resnet.forward(xs, temb);
        for (attn, resnet) in self.attn_resnets.iter() {
            xs = resnet.forward(&xs.apply(attn), temb)
        }
        xs
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UNetMidBlock2DCrossAttnConfig {
    pub num_layers: i64,
    pub resnet_eps: f64,
    pub resnet_groups: Option<i64>,
    pub attn_num_head_channels: i64,
    // attention_type "default"
    pub output_scale_factor: f64,
    pub cross_attn_dim: i64,
    pub sliced_attention_size: Option<i64>,
    pub use_linear_projection: bool,
}

impl Default for UNetMidBlock2DCrossAttnConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: Some(32),
            attn_num_head_channels: 1,
            output_scale_factor: 1.,
            cross_attn_dim: 1280,
            sliced_attention_size: None, // Sliced attention disabled
            use_linear_projection: false,
        }
    }
}

#[derive(Debug)]
pub struct UNetMidBlock2DCrossAttn {
    resnet: ResnetBlock2D,
    attn_resnets: Vec<(SpatialTransformer, ResnetBlock2D)>,
    pub config: UNetMidBlock2DCrossAttnConfig,
}

impl UNetMidBlock2DCrossAttn {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        temb_channels: Option<i64>,
        config: UNetMidBlock2DCrossAttnConfig,
    ) -> Self {
        let vs_resnets = &vs / "resnets";
        let vs_attns = &vs / "attentions";
        let resnet_groups = config.resnet_groups.unwrap_or_else(|| i64::min(in_channels / 4, 32));
        let resnet_cfg = ResnetBlock2DConfig {
            eps: config.resnet_eps,
            groups: resnet_groups,
            output_scale_factor: config.output_scale_factor,
            temb_channels,
            ..Default::default()
        };
        let resnet = ResnetBlock2D::new(&vs_resnets / "0", in_channels, resnet_cfg);
        let n_heads = config.attn_num_head_channels;
        let attn_cfg = SpatialTransformerConfig {
            depth: 1,
            num_groups: resnet_groups,
            context_dim: Some(config.cross_attn_dim),
            sliced_attention_size: config.sliced_attention_size,
            use_linear_projection: config.use_linear_projection,
        };
        let mut attn_resnets = vec![];
        for index in 0..config.num_layers {
            let attn = SpatialTransformer::new(
                &vs_attns / index,
                in_channels,
                n_heads,
                in_channels / n_heads,
                attn_cfg,
            );
            let resnet = ResnetBlock2D::new(&vs_resnets / (index + 1), in_channels, resnet_cfg);
            attn_resnets.push((attn, resnet))
        }
        Self { resnet, attn_resnets, config }
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Tensor {
        let mut xs = self.resnet.forward(xs, temb);
        for (attn, resnet) in self.attn_resnets.iter() {
            xs = resnet.forward(&attn.forward(&xs, encoder_hidden_states), temb)
        }
        xs
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DownBlock2DConfig {
    pub num_layers: i64,
    pub resnet_eps: f64,
    // resnet_time_scale_shift: "default"
    // resnet_act_fn: "swish"
    pub resnet_groups: i64,
    pub output_scale_factor: f64,
    pub add_downsample: bool,
    pub downsample_padding: i64,
}

impl Default for DownBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_downsample: true,
            downsample_padding: 1,
        }
    }
}

#[derive(Debug)]
pub struct DownBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<Downsample2D>,
    pub config: DownBlock2DConfig,
}

impl DownBlock2D {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        temb_channels: Option<i64>,
        config: DownBlock2DConfig,
    ) -> Self {
        let vs_resnets = &vs / "resnets";
        let resnet_cfg = ResnetBlock2DConfig {
            out_channels: Some(out_channels),
            eps: config.resnet_eps,
            output_scale_factor: config.output_scale_factor,
            temb_channels,
            ..Default::default()
        };
        let resnets = (0..config.num_layers)
            .map(|i| {
                let in_channels = if i == 0 { in_channels } else { out_channels };
                ResnetBlock2D::new(&vs_resnets / i, in_channels, resnet_cfg)
            })
            .collect();
        let downsampler = if config.add_downsample {
            let downsampler = Downsample2D::new(
                &vs / "downsamplers" / 0,
                out_channels,
                true,
                out_channels,
                config.downsample_padding,
            );
            Some(downsampler)
        } else {
            None
        };
        Self { resnets, downsampler, config }
    }

    pub fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> (Tensor, Vec<Tensor>) {
        let mut xs = xs.shallow_clone();
        let mut output_states = vec![];
        for resnet in self.resnets.iter() {
            xs = resnet.forward(&xs, temb);
            output_states.push(xs.shallow_clone());
        }
        let xs = match &self.downsampler {
            Some(downsampler) => {
                let xs = xs.apply(downsampler);
                output_states.push(xs.shallow_clone());
                xs
            }
            None => xs,
        };
        (xs, output_states)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CrossAttnDownBlock2DConfig {
    pub downblock: DownBlock2DConfig,
    pub attn_num_head_channels: i64,
    pub cross_attention_dim: i64,
    // attention_type: "default"
    pub sliced_attention_size: Option<i64>,
    pub use_linear_projection: bool,
}

impl Default for CrossAttnDownBlock2DConfig {
    fn default() -> Self {
        Self {
            downblock: Default::default(),
            attn_num_head_channels: 1,
            cross_attention_dim: 1280,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }
}

#[derive(Debug)]
pub struct CrossAttnDownBlock2D {
    downblock: DownBlock2D,
    attentions: Vec<SpatialTransformer>,
    pub config: CrossAttnDownBlock2DConfig,
}

impl CrossAttnDownBlock2D {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        temb_channels: Option<i64>,
        config: CrossAttnDownBlock2DConfig,
    ) -> Self {
        let downblock = DownBlock2D::new(
            vs.clone(),
            in_channels,
            out_channels,
            temb_channels,
            config.downblock,
        );
        let n_heads = config.attn_num_head_channels;
        let cfg = SpatialTransformerConfig {
            depth: 1,
            context_dim: Some(config.cross_attention_dim),
            num_groups: config.downblock.resnet_groups,
            sliced_attention_size: config.sliced_attention_size,
            use_linear_projection: config.use_linear_projection,
        };
        let vs_attn = &vs / "attentions";
        let attentions = (0..config.downblock.num_layers)
            .map(|i| {
                SpatialTransformer::new(
                    &vs_attn / i,
                    out_channels,
                    n_heads,
                    out_channels / n_heads,
                    cfg,
                )
            })
            .collect();
        Self { downblock, attentions, config }
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> (Tensor, Vec<Tensor>) {
        let mut output_states = vec![];
        let mut xs = xs.shallow_clone();
        for (resnet, attn) in self.downblock.resnets.iter().zip(self.attentions.iter()) {
            xs = resnet.forward(&xs, temb);
            xs = attn.forward(&xs, encoder_hidden_states);
            output_states.push(xs.shallow_clone());
        }
        let xs = match &self.downblock.downsampler {
            Some(downsampler) => {
                let xs = xs.apply(downsampler);
                output_states.push(xs.shallow_clone());
                xs
            }
            None => xs,
        };
        (xs, output_states)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UpBlock2DConfig {
    pub num_layers: i64,
    pub resnet_eps: f64,
    // resnet_time_scale_shift: "default"
    // resnet_act_fn: "swish"
    pub resnet_groups: i64,
    pub output_scale_factor: f64,
    pub add_upsample: bool,
}

impl Default for UpBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_upsample: true,
        }
    }
}

#[derive(Debug)]
pub struct UpBlock2D {
    pub resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Upsample2D>,
    pub config: UpBlock2DConfig,
}

impl UpBlock2D {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        prev_output_channels: i64,
        out_channels: i64,
        temb_channels: Option<i64>,
        config: UpBlock2DConfig,
    ) -> Self {
        let vs_resnets = &vs / "resnets";
        let resnet_cfg = ResnetBlock2DConfig {
            out_channels: Some(out_channels),
            temb_channels,
            eps: config.resnet_eps,
            output_scale_factor: config.output_scale_factor,
            ..Default::default()
        };
        let resnets = (0..config.num_layers)
            .map(|i| {
                let res_skip_channels =
                    if i == config.num_layers - 1 { in_channels } else { out_channels };
                let resnet_in_channels = if i == 0 { prev_output_channels } else { out_channels };
                let in_channels = resnet_in_channels + res_skip_channels;
                ResnetBlock2D::new(&vs_resnets / i, in_channels, resnet_cfg)
            })
            .collect();
        let upsampler = if config.add_upsample {
            let upsampler = Upsample2D::new(&vs / "upsamplers" / 0, out_channels, out_channels);
            Some(upsampler)
        } else {
            None
        };
        Self { resnets, upsampler, config }
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        res_xs: &[Tensor],
        temb: Option<&Tensor>,
        upsample_size: Option<(i64, i64)>,
    ) -> Tensor {
        let mut xs = xs.shallow_clone();
        for (index, resnet) in self.resnets.iter().enumerate() {
            xs = Tensor::cat(&[&xs, &res_xs[res_xs.len() - index - 1]], 1);
            xs = resnet.forward(&xs, temb);
        }
        match &self.upsampler {
            Some(upsampler) => upsampler.forward(&xs, upsample_size),
            None => xs,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CrossAttnUpBlock2DConfig {
    pub upblock: UpBlock2DConfig,
    pub attn_num_head_channels: i64,
    pub cross_attention_dim: i64,
    // attention_type: "default"
    pub sliced_attention_size: Option<i64>,
    pub use_linear_projection: bool,
}

impl Default for CrossAttnUpBlock2DConfig {
    fn default() -> Self {
        Self {
            upblock: Default::default(),
            attn_num_head_channels: 1,
            cross_attention_dim: 1280,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }
}

#[derive(Debug)]
pub struct CrossAttnUpBlock2D {
    pub upblock: UpBlock2D,
    pub attentions: Vec<SpatialTransformer>,
    pub config: CrossAttnUpBlock2DConfig,
}

impl CrossAttnUpBlock2D {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        prev_output_channels: i64,
        out_channels: i64,
        temb_channels: Option<i64>,
        config: CrossAttnUpBlock2DConfig,
    ) -> Self {
        let upblock = UpBlock2D::new(
            vs.clone(),
            in_channels,
            prev_output_channels,
            out_channels,
            temb_channels,
            config.upblock,
        );
        let n_heads = config.attn_num_head_channels;
        let cfg = SpatialTransformerConfig {
            depth: 1,
            context_dim: Some(config.cross_attention_dim),
            num_groups: config.upblock.resnet_groups,
            sliced_attention_size: config.sliced_attention_size,
            use_linear_projection: config.use_linear_projection,
        };
        let vs_attn = &vs / "attentions";
        let attentions = (0..config.upblock.num_layers)
            .map(|i| {
                SpatialTransformer::new(
                    &vs_attn / i,
                    out_channels,
                    n_heads,
                    out_channels / n_heads,
                    cfg,
                )
            })
            .collect();
        Self { upblock, attentions, config }
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        res_xs: &[Tensor],
        temb: Option<&Tensor>,
        upsample_size: Option<(i64, i64)>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Tensor {
        let mut xs = xs.shallow_clone();
        for (index, resnet) in self.upblock.resnets.iter().enumerate() {
            xs = Tensor::cat(&[&xs, &res_xs[res_xs.len() - index - 1]], 1);
            xs = resnet.forward(&xs, temb);
            xs = self.attentions[index].forward(&xs, encoder_hidden_states);
        }
        match &self.upblock.upsampler {
            Some(upsampler) => upsampler.forward(&xs, upsample_size),
            None => xs,
        }
    }
}
