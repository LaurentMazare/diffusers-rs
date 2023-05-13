//! Attention Based Building Blocks
use tch::{nn, nn::Module, IndexOp, Kind, Tensor};
#[derive(Debug)]
struct GeGlu {
    proj: nn::Linear,
}

impl GeGlu {
    fn new(vs: nn::Path, dim_in: i64, dim_out: i64) -> Self {
        let proj = nn::linear(&vs / "proj", dim_in, dim_out * 2, Default::default());
        Self { proj }
    }
}

impl Module for GeGlu {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let hidden_states_and_gate = xs.apply(&self.proj).chunk(2, -1);
        &hidden_states_and_gate[0] * hidden_states_and_gate[1].gelu("none")
    }
}

/// A feed-forward layer.
#[derive(Debug)]
struct FeedForward {
    project_in: GeGlu,
    linear: nn::Linear,
}

impl FeedForward {
    // The glu parameter in the python code is unused?
    // https://github.com/huggingface/diffusers/blob/d3d22ce5a894becb951eec03e663951b28d45135/src/diffusers/models/attention.py#L347
    /// Creates a new feed-forward layer based on some given input dimension, some
    /// output dimension, and a multiplier to be used for the intermediary layer.
    fn new(vs: nn::Path, dim: i64, dim_out: Option<i64>, mult: i64) -> Self {
        let inner_dim = dim * mult;
        let dim_out = dim_out.unwrap_or(dim);
        let vs = &vs / "net";
        let project_in = GeGlu::new(&vs / 0, dim, inner_dim);
        let linear = nn::linear(&vs / 2, inner_dim, dim_out, Default::default());
        Self { project_in, linear }
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.project_in).apply(&self.linear)
    }
}

#[derive(Debug)]
struct CrossAttention {
    to_q: nn::Linear,
    to_k: nn::Linear,
    to_v: nn::Linear,
    to_out: nn::Linear,
    heads: i64,
    scale: f64,
    slice_size: Option<i64>,
}

impl CrossAttention {
    // Defaults should be heads = 8, dim_head = 64, context_dim = None
    fn new(
        vs: nn::Path,
        query_dim: i64,
        context_dim: Option<i64>,
        heads: i64,
        dim_head: i64,
        slice_size: Option<i64>,
    ) -> Self {
        let no_bias = nn::LinearConfig { bias: false, ..Default::default() };
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / f64::sqrt(dim_head as f64);
        let to_q = nn::linear(&vs / "to_q", query_dim, inner_dim, no_bias);
        let to_k = nn::linear(&vs / "to_k", context_dim, inner_dim, no_bias);
        let to_v = nn::linear(&vs / "to_v", context_dim, inner_dim, no_bias);
        let to_out = nn::linear(&vs / "to_out" / 0, inner_dim, query_dim, Default::default());
        Self { to_q, to_k, to_v, to_out, heads, scale, slice_size }
    }

    fn reshape_heads_to_batch_dim(&self, xs: &Tensor) -> Tensor {
        let (batch_size, seq_len, dim) = xs.size3().unwrap();
        xs.reshape([batch_size, seq_len, self.heads, dim / self.heads])
            .permute([0, 2, 1, 3])
            .reshape([batch_size * self.heads, seq_len, dim / self.heads])
    }

    fn reshape_batch_dim_to_heads(&self, xs: &Tensor) -> Tensor {
        let (batch_size, seq_len, dim) = xs.size3().unwrap();
        xs.reshape([batch_size / self.heads, self.heads, seq_len, dim])
            .permute([0, 2, 1, 3])
            .reshape([batch_size / self.heads, seq_len, dim * self.heads])
    }

    fn sliced_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        sequence_length: i64,
        dim: i64,
        slice_size: i64,
    ) -> Tensor {
        let batch_size_attention = query.size()[0];
        let mut hidden_states = Tensor::zeros(
            [batch_size_attention, sequence_length, dim / self.heads],
            (query.kind(), query.device()),
        );

        for i in 0..batch_size_attention / slice_size {
            let start_idx = i * slice_size;
            let end_idx = (i + 1) * slice_size;

            let xs = query
                .i(start_idx..end_idx)
                .matmul(&(key.i(start_idx..end_idx).transpose(-1, -2) * self.scale))
                .softmax(-1, Kind::Float)
                .matmul(&value.i(start_idx..end_idx));

            let idx = Tensor::arange_start(start_idx, end_idx, (Kind::Int64, query.device()));
            let _ = hidden_states.index_put_(&[Some(idx), None, None], &xs, false);
        }

        self.reshape_batch_dim_to_heads(&hidden_states)
    }

    fn attention(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
        let xs = query
            .matmul(&(key.transpose(-1, -2) * self.scale))
            .softmax(-1, Kind::Float)
            .matmul(value);
        self.reshape_batch_dim_to_heads(&xs)
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Tensor {
        let sequence_length = xs.size()[1];
        let query = xs.apply(&self.to_q);
        let dim = *query.size().last().unwrap();
        let context = context.unwrap_or(xs);
        let key = context.apply(&self.to_k);
        let value = context.apply(&self.to_v);
        let query = self.reshape_heads_to_batch_dim(&query);
        let key = self.reshape_heads_to_batch_dim(&key);
        let value = self.reshape_heads_to_batch_dim(&value);
        match self.slice_size {
            None => self.attention(&query, &key, &value).apply(&self.to_out),
            Some(slice_size) => {
                if query.size()[0] / slice_size <= 1 {
                    self.attention(&query, &key, &value).apply(&self.to_out)
                } else {
                    self.sliced_attention(&query, &key, &value, sequence_length, dim, slice_size)
                        .apply(&self.to_out)
                }
            }
        }
    }
}

/// A basic Transformer block.
#[derive(Debug)]
struct BasicTransformerBlock {
    attn1: CrossAttention,
    ff: FeedForward,
    attn2: CrossAttention,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    norm3: nn::LayerNorm,
}

impl BasicTransformerBlock {
    fn new(
        vs: nn::Path,
        dim: i64,
        n_heads: i64,
        d_head: i64,
        context_dim: Option<i64>,
        sliced_attention_size: Option<i64>,
    ) -> Self {
        let attn1 =
            CrossAttention::new(&vs / "attn1", dim, None, n_heads, d_head, sliced_attention_size);
        let ff = FeedForward::new(&vs / "ff", dim, None, 4);
        let attn2 = CrossAttention::new(
            &vs / "attn2",
            dim,
            context_dim,
            n_heads,
            d_head,
            sliced_attention_size,
        );
        let norm1 = nn::layer_norm(&vs / "norm1", vec![dim], Default::default());
        let norm2 = nn::layer_norm(&vs / "norm2", vec![dim], Default::default());
        let norm3 = nn::layer_norm(&vs / "norm3", vec![dim], Default::default());
        Self { attn1, ff, attn2, norm1, norm2, norm3 }
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Tensor {
        let xs = self.attn1.forward(&xs.apply(&self.norm1), None) + xs;
        let xs = self.attn2.forward(&xs.apply(&self.norm2), context) + xs;
        xs.apply(&self.norm3).apply(&self.ff) + xs
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialTransformerConfig {
    pub depth: i64,
    pub num_groups: i64,
    pub context_dim: Option<i64>,
    pub sliced_attention_size: Option<i64>,
    pub use_linear_projection: bool,
}

impl Default for SpatialTransformerConfig {
    fn default() -> Self {
        Self {
            depth: 1,
            num_groups: 32,
            context_dim: None,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }
}

#[derive(Debug)]
enum Proj {
    Conv2D(nn::Conv2D),
    Linear(nn::Linear),
}

// Aka Transformer2DModel
#[derive(Debug)]
pub struct SpatialTransformer {
    norm: nn::GroupNorm,
    proj_in: Proj,
    transformer_blocks: Vec<BasicTransformerBlock>,
    proj_out: Proj,
    pub config: SpatialTransformerConfig,
}

impl SpatialTransformer {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        n_heads: i64,
        d_head: i64,
        config: SpatialTransformerConfig,
    ) -> Self {
        let inner_dim = n_heads * d_head;
        let group_cfg = nn::GroupNormConfig { eps: 1e-6, affine: true, ..Default::default() };
        let norm = nn::group_norm(&vs / "norm", config.num_groups, in_channels, group_cfg);
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 0, ..Default::default() };
        let proj_in = if config.use_linear_projection {
            Proj::Linear(nn::linear(&vs / "proj_in", in_channels, inner_dim, Default::default()))
        } else {
            Proj::Conv2D(nn::conv2d(&vs / "proj_in", in_channels, inner_dim, 1, conv_cfg))
        };
        let mut transformer_blocks = vec![];
        let vs_tb = &vs / "transformer_blocks";
        for index in 0..config.depth {
            let tb = BasicTransformerBlock::new(
                &vs_tb / index,
                inner_dim,
                n_heads,
                d_head,
                config.context_dim,
                config.sliced_attention_size,
            );
            transformer_blocks.push(tb)
        }
        let proj_out = if config.use_linear_projection {
            Proj::Linear(nn::linear(&vs / "proj_out", in_channels, inner_dim, Default::default()))
        } else {
            Proj::Conv2D(nn::conv2d(&vs / "proj_out", inner_dim, in_channels, 1, conv_cfg))
        };
        Self { norm, proj_in, transformer_blocks, proj_out, config }
    }

    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Tensor {
        let (batch, _channel, height, weight) = xs.size4().unwrap();
        let residual = xs;
        let xs = xs.apply(&self.norm);
        let (inner_dim, xs) = match &self.proj_in {
            Proj::Conv2D(p) => {
                let xs = xs.apply(p);
                let inner_dim = xs.size()[1];
                let xs = xs.permute([0, 2, 3, 1]).view((batch, height * weight, inner_dim));
                (inner_dim, xs)
            }
            Proj::Linear(p) => {
                let inner_dim = xs.size()[1];
                let xs = xs.permute([0, 2, 3, 1]).view((batch, height * weight, inner_dim));
                (inner_dim, xs.apply(p))
            }
        };
        let mut xs = xs;
        for block in self.transformer_blocks.iter() {
            xs = block.forward(&xs, context)
        }
        let xs = match &self.proj_out {
            Proj::Conv2D(p) => {
                xs.view((batch, height, weight, inner_dim)).permute([0, 3, 1, 2]).apply(p)
            }
            Proj::Linear(p) => {
                xs.apply(p).view((batch, height, weight, inner_dim)).permute([0, 3, 1, 2])
            }
        };
        xs + residual
    }
}

/// Configuration for an attention block.
#[derive(Debug, Clone, Copy)]
pub struct AttentionBlockConfig {
    pub num_head_channels: Option<i64>,
    pub num_groups: i64,
    pub rescale_output_factor: f64,
    pub eps: f64,
}

impl Default for AttentionBlockConfig {
    fn default() -> Self {
        Self { num_head_channels: None, num_groups: 32, rescale_output_factor: 1., eps: 1e-5 }
    }
}

#[derive(Debug)]
pub struct AttentionBlock {
    group_norm: nn::GroupNorm,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    proj_attn: nn::Linear,
    channels: i64,
    num_heads: i64,
    config: AttentionBlockConfig,
}

impl AttentionBlock {
    pub fn new(vs: nn::Path, channels: i64, config: AttentionBlockConfig) -> Self {
        let num_head_channels = config.num_head_channels.unwrap_or(channels);
        let num_heads = channels / num_head_channels;
        let group_cfg = nn::GroupNormConfig { eps: config.eps, affine: true, ..Default::default() };
        let group_norm = nn::group_norm(&vs / "group_norm", config.num_groups, channels, group_cfg);
        let query = nn::linear(&vs / "query", channels, channels, Default::default());
        let key = nn::linear(&vs / "key", channels, channels, Default::default());
        let value = nn::linear(&vs / "value", channels, channels, Default::default());
        let proj_attn = nn::linear(&vs / "proj_attn", channels, channels, Default::default());
        Self { group_norm, query, key, value, proj_attn, channels, num_heads, config }
    }

    fn transpose_for_scores(&self, xs: Tensor) -> Tensor {
        let (batch, t, _h_times_d) = xs.size3().unwrap();
        xs.view((batch, t, self.num_heads, -1)).permute([0, 2, 1, 3])
    }
}

impl Module for AttentionBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let residual = xs;
        let (batch, channel, height, width) = xs.size4().unwrap();
        let xs = xs.apply(&self.group_norm).view((batch, channel, height * width)).transpose(1, 2);

        let query_proj = xs.apply(&self.query);
        let key_proj = xs.apply(&self.key);
        let value_proj = xs.apply(&self.value);

        let query_states = self.transpose_for_scores(query_proj);
        let key_states = self.transpose_for_scores(key_proj);
        let value_states = self.transpose_for_scores(value_proj);

        let scale = f64::powf((self.channels as f64) / (self.num_heads as f64), -0.25);
        let attention_scores =
            (query_states * scale).matmul(&(key_states.transpose(-1, -2) * scale));
        let attention_probs = attention_scores.softmax(-1, Kind::Float);

        let xs = attention_probs.matmul(&value_states);
        let xs = xs.permute([0, 2, 1, 3]).contiguous();
        let mut new_xs_shape = xs.size();
        new_xs_shape.pop();
        new_xs_shape.pop();
        new_xs_shape.push(self.channels);

        let xs = xs
            .view(new_xs_shape.as_slice())
            .apply(&self.proj_attn)
            .transpose(-1, -2)
            .view((batch, channel, height, width));
        (xs + residual) / self.config.rescale_output_factor
    }
}
