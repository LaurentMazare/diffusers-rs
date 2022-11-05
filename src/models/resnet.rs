use tch::{nn, Tensor};

#[derive(Debug, Clone, Copy)]
pub struct ResnetBlock2DConfig {
    pub out_channels: Option<i64>,
    pub temb_channels: Option<i64>,
    pub groups: i64,
    pub groups_out: Option<i64>,
    pub eps: f64,
    pub use_in_shortcut: Option<bool>,
    // non_linearity: silu
    pub output_scale_factor: f64,
}

impl Default for ResnetBlock2DConfig {
    fn default() -> Self {
        Self {
            out_channels: None,
            temb_channels: Some(512),
            groups: 32,
            groups_out: None,
            eps: 1e-6,
            use_in_shortcut: None,
            output_scale_factor: 1.,
        }
    }
}

#[derive(Debug)]
pub struct ResnetBlock2D {
    norm1: nn::GroupNorm,
    conv1: nn::Conv2D,
    norm2: nn::GroupNorm,
    conv2: nn::Conv2D,
    time_emb_proj: Option<nn::Linear>,
    conv_shortcut: Option<nn::Conv2D>,
    config: ResnetBlock2DConfig,
}

impl ResnetBlock2D {
    pub fn new(vs: nn::Path, in_channels: i64, config: ResnetBlock2DConfig) -> Self {
        let out_channels = config.out_channels.unwrap_or(in_channels);
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let group_cfg = nn::GroupNormConfig { eps: config.eps, affine: true, ..Default::default() };
        let norm1 = nn::group_norm(&vs / "norm1", config.groups, in_channels, group_cfg);
        let conv1 = nn::conv2d(&vs / "conv1", in_channels, out_channels, 3, conv_cfg);
        let groups_out = config.groups_out.unwrap_or(config.groups);
        let norm2 = nn::group_norm(&vs / "norm2", groups_out, out_channels, group_cfg);
        let conv2 = nn::conv2d(&vs / "conv2", out_channels, out_channels, 3, conv_cfg);
        let use_in_shortcut = config.use_in_shortcut.unwrap_or(in_channels != out_channels);
        let conv_shortcut = if use_in_shortcut {
            let conv_cfg = nn::ConvConfig { stride: 1, padding: 0, ..Default::default() };
            Some(nn::conv2d(&vs / "conv_shortcut", in_channels, out_channels, 1, conv_cfg))
        } else {
            None
        };
        let time_emb_proj = config.temb_channels.map(|temb_channels| {
            nn::linear(&vs / "time_emb_proj", temb_channels, out_channels, Default::default())
        });
        Self { norm1, conv1, norm2, conv2, time_emb_proj, config, conv_shortcut }
    }

    pub fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> Tensor {
        let shortcut_xs = match &self.conv_shortcut {
            Some(conv_shortcut) => xs.apply(conv_shortcut),
            None => xs.shallow_clone(),
        };
        let xs = xs.apply(&self.norm1).silu().apply(&self.conv1);
        let xs = match (temb, &self.time_emb_proj) {
            (Some(temb), Some(time_emb_proj)) => {
                temb.silu().apply(time_emb_proj).unsqueeze(-1).unsqueeze(-1) + xs
            }
            _ => xs,
        };
        let xs = xs.apply(&self.norm2).silu().apply(&self.conv2);
        (shortcut_xs + xs) / self.config.output_scale_factor
    }
}
