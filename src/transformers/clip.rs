//! Contrastive Language-Image Pre-Training
//!
//! Contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
//! pairs of images with related texts.
//!
//! https://github.com/openai/CLIP
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use tch::{nn, nn::Module, Device, Kind, Tensor};

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    QuickGelu,
    Gelu,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self {
            Activation::QuickGelu => xs * (xs * 1.702).sigmoid(),
            Activation::Gelu => xs.gelu("none"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    vocab_size: i64,
    embed_dim: i64,         // aka config.hidden_size
    activation: Activation, // aka config.hidden_act
    intermediate_size: i64,
    max_position_embeddings: usize,
    // The character to use for padding, use EOS when not set.
    pad_with: Option<String>,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    #[allow(dead_code)]
    projection_dim: i64,
}

impl Config {
    // The config details can be found in the "text_config" section of this json file:
    // https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
    pub fn v1_5() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 768,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 768,
            activation: Activation::QuickGelu,
        }
    }

    // https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/text_encoder/config.json
    pub fn v2_1() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 1024,
            intermediate_size: 4096,
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 23,
            num_attention_heads: 16,
            projection_dim: 512,
            activation: Activation::Gelu,
        }
    }
}

const BYTES_TO_UNICODE: [(u8, char); 256] = [
    (33, '!'),
    (34, '"'),
    (35, '#'),
    (36, '$'),
    (37, '%'),
    (38, '&'),
    (39, '\''),
    (40, '('),
    (41, ')'),
    (42, '*'),
    (43, '+'),
    (44, ','),
    (45, '-'),
    (46, '.'),
    (47, '/'),
    (48, '0'),
    (49, '1'),
    (50, '2'),
    (51, '3'),
    (52, '4'),
    (53, '5'),
    (54, '6'),
    (55, '7'),
    (56, '8'),
    (57, '9'),
    (58, ':'),
    (59, ';'),
    (60, '<'),
    (61, '='),
    (62, '>'),
    (63, '?'),
    (64, '@'),
    (65, 'A'),
    (66, 'B'),
    (67, 'C'),
    (68, 'D'),
    (69, 'E'),
    (70, 'F'),
    (71, 'G'),
    (72, 'H'),
    (73, 'I'),
    (74, 'J'),
    (75, 'K'),
    (76, 'L'),
    (77, 'M'),
    (78, 'N'),
    (79, 'O'),
    (80, 'P'),
    (81, 'Q'),
    (82, 'R'),
    (83, 'S'),
    (84, 'T'),
    (85, 'U'),
    (86, 'V'),
    (87, 'W'),
    (88, 'X'),
    (89, 'Y'),
    (90, 'Z'),
    (91, '['),
    (92, '\\'),
    (93, ']'),
    (94, '^'),
    (95, '_'),
    (96, '`'),
    (97, 'a'),
    (98, 'b'),
    (99, 'c'),
    (100, 'd'),
    (101, 'e'),
    (102, 'f'),
    (103, 'g'),
    (104, 'h'),
    (105, 'i'),
    (106, 'j'),
    (107, 'k'),
    (108, 'l'),
    (109, 'm'),
    (110, 'n'),
    (111, 'o'),
    (112, 'p'),
    (113, 'q'),
    (114, 'r'),
    (115, 's'),
    (116, 't'),
    (117, 'u'),
    (118, 'v'),
    (119, 'w'),
    (120, 'x'),
    (121, 'y'),
    (122, 'z'),
    (123, '{'),
    (124, '|'),
    (125, '}'),
    (126, '~'),
    (161, '¡'),
    (162, '¢'),
    (163, '£'),
    (164, '¤'),
    (165, '¥'),
    (166, '¦'),
    (167, '§'),
    (168, '¨'),
    (169, '©'),
    (170, 'ª'),
    (171, '«'),
    (172, '¬'),
    (174, '®'),
    (175, '¯'),
    (176, '°'),
    (177, '±'),
    (178, '²'),
    (179, '³'),
    (180, '´'),
    (181, 'µ'),
    (182, '¶'),
    (183, '·'),
    (184, '¸'),
    (185, '¹'),
    (186, 'º'),
    (187, '»'),
    (188, '¼'),
    (189, '½'),
    (190, '¾'),
    (191, '¿'),
    (192, 'À'),
    (193, 'Á'),
    (194, 'Â'),
    (195, 'Ã'),
    (196, 'Ä'),
    (197, 'Å'),
    (198, 'Æ'),
    (199, 'Ç'),
    (200, 'È'),
    (201, 'É'),
    (202, 'Ê'),
    (203, 'Ë'),
    (204, 'Ì'),
    (205, 'Í'),
    (206, 'Î'),
    (207, 'Ï'),
    (208, 'Ð'),
    (209, 'Ñ'),
    (210, 'Ò'),
    (211, 'Ó'),
    (212, 'Ô'),
    (213, 'Õ'),
    (214, 'Ö'),
    (215, '×'),
    (216, 'Ø'),
    (217, 'Ù'),
    (218, 'Ú'),
    (219, 'Û'),
    (220, 'Ü'),
    (221, 'Ý'),
    (222, 'Þ'),
    (223, 'ß'),
    (224, 'à'),
    (225, 'á'),
    (226, 'â'),
    (227, 'ã'),
    (228, 'ä'),
    (229, 'å'),
    (230, 'æ'),
    (231, 'ç'),
    (232, 'è'),
    (233, 'é'),
    (234, 'ê'),
    (235, 'ë'),
    (236, 'ì'),
    (237, 'í'),
    (238, 'î'),
    (239, 'ï'),
    (240, 'ð'),
    (241, 'ñ'),
    (242, 'ò'),
    (243, 'ó'),
    (244, 'ô'),
    (245, 'õ'),
    (246, 'ö'),
    (247, '÷'),
    (248, 'ø'),
    (249, 'ù'),
    (250, 'ú'),
    (251, 'û'),
    (252, 'ü'),
    (253, 'ý'),
    (254, 'þ'),
    (255, 'ÿ'),
    (0, 'Ā'),
    (1, 'ā'),
    (2, 'Ă'),
    (3, 'ă'),
    (4, 'Ą'),
    (5, 'ą'),
    (6, 'Ć'),
    (7, 'ć'),
    (8, 'Ĉ'),
    (9, 'ĉ'),
    (10, 'Ċ'),
    (11, 'ċ'),
    (12, 'Č'),
    (13, 'č'),
    (14, 'Ď'),
    (15, 'ď'),
    (16, 'Đ'),
    (17, 'đ'),
    (18, 'Ē'),
    (19, 'ē'),
    (20, 'Ĕ'),
    (21, 'ĕ'),
    (22, 'Ė'),
    (23, 'ė'),
    (24, 'Ę'),
    (25, 'ę'),
    (26, 'Ě'),
    (27, 'ě'),
    (28, 'Ĝ'),
    (29, 'ĝ'),
    (30, 'Ğ'),
    (31, 'ğ'),
    (32, 'Ġ'),
    (127, 'ġ'),
    (128, 'Ģ'),
    (129, 'ģ'),
    (130, 'Ĥ'),
    (131, 'ĥ'),
    (132, 'Ħ'),
    (133, 'ħ'),
    (134, 'Ĩ'),
    (135, 'ĩ'),
    (136, 'Ī'),
    (137, 'ī'),
    (138, 'Ĭ'),
    (139, 'ĭ'),
    (140, 'Į'),
    (141, 'į'),
    (142, 'İ'),
    (143, 'ı'),
    (144, 'Ĳ'),
    (145, 'ĳ'),
    (146, 'Ĵ'),
    (147, 'ĵ'),
    (148, 'Ķ'),
    (149, 'ķ'),
    (150, 'ĸ'),
    (151, 'Ĺ'),
    (152, 'ĺ'),
    (153, 'Ļ'),
    (154, 'ļ'),
    (155, 'Ľ'),
    (156, 'ľ'),
    (157, 'Ŀ'),
    (158, 'ŀ'),
    (159, 'Ł'),
    (160, 'ł'),
    (173, 'Ń'),
];

const PAT: &str =
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

// This is mostly a Rust rewrite of the original Python CLIP code.
// https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
/// A tokenizer for CLIP.
pub struct Tokenizer {
    re: regex::Regex,
    encoder: HashMap<String, usize>,
    decoder: HashMap<usize, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    start_of_text_token: usize,
    end_of_text_token: usize,
    config: Config,
}

impl Tokenizer {
    /// Creates a new CLIP tokenizer, this takes as input the path for the bpe vocabulary file.
    pub fn create<T: AsRef<std::path::Path> + std::fmt::Debug>(
        bpe_path: T,
        c: &Config,
    ) -> anyhow::Result<Tokenizer> {
        let bpe_file = crate::utils::file_open(bpe_path)?;
        let bpe_lines: Result<Vec<String>, _> = std::io::BufReader::new(bpe_file).lines().collect();
        let bpe_lines = bpe_lines?;
        let bpe_lines: Result<Vec<_>, _> = bpe_lines[1..49152 - 256 - 2 + 1]
            .iter()
            .map(|line| {
                let vs: Vec<_> = line.split_whitespace().collect();
                if vs.len() != 2 {
                    anyhow::bail!("expected two items got {} '{}'", vs.len(), line)
                }
                Ok((vs[0].to_string(), vs[1].to_string()))
            })
            .collect();
        let bpe_lines = bpe_lines?;
        let mut vocab: Vec<String> = Vec::new();
        for (_index, elem) in BYTES_TO_UNICODE {
            vocab.push(elem.into())
        }
        for (_index, elem) in BYTES_TO_UNICODE {
            vocab.push(format!("{elem}</w>"));
        }
        for elem in bpe_lines.iter() {
            vocab.push(format!("{}{}", elem.0, elem.1))
        }
        let start_of_text_token = vocab.len();
        vocab.push("<|startoftext|>".to_string());
        let end_of_text_token = vocab.len();
        vocab.push("<|endoftext|>".to_string());
        let encoder: HashMap<_, _> = vocab.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        let decoder: HashMap<_, _> = encoder.iter().map(|(k, v)| (*v, k.clone())).collect();
        let bpe_ranks: HashMap<_, _> =
            bpe_lines.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        let re = regex::Regex::new(PAT)?;
        let tokenizer = Tokenizer {
            encoder,
            re,
            bpe_ranks,
            decoder,
            start_of_text_token,
            end_of_text_token,
            config: c.clone(),
        };
        Ok(tokenizer)
    }

    fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
        let mut pairs = HashSet::new();
        for (i, v) in word.iter().enumerate() {
            if i > 0 {
                pairs.insert((word[i - 1].clone(), v.clone()));
            }
        }
        pairs
    }

    fn bpe(&self, token: &str) -> Vec<usize> {
        let mut word: Vec<String> = token.chars().map(|x| x.to_string()).collect();
        if word.is_empty() {
            return Vec::new();
        }
        let last_index = word.len() - 1;
        word[last_index] = format!("{}</w>", word[last_index]);
        while word.len() > 1 {
            let mut current_min = None;
            let pairs = Self::get_pairs(&word);
            for p in pairs.iter() {
                match self.bpe_ranks.get(p) {
                    None => {}
                    Some(v) => {
                        let should_replace = match current_min {
                            None => true,
                            Some((current_min, _)) => v < current_min,
                        };
                        if should_replace {
                            current_min = Some((v, p))
                        }
                    }
                }
            }
            let (first, second) = match current_min {
                None => break,
                Some((_v, (first, second))) => (first, second),
            };
            let mut new_word = vec![];
            let mut index = 0;
            while index < word.len() {
                let w = &word[index];
                if index + 1 < word.len() && w == first && &word[index + 1] == second {
                    new_word.push(format!("{first}{second}"));
                    index += 2
                } else {
                    new_word.push(w.clone());
                    index += 1
                }
            }
            word = new_word
        }
        word.iter().filter_map(|x| self.encoder.get(x)).copied().collect()
    }

    pub fn encode_pad(&self, s: &str, pad_size_to: Option<usize>) -> anyhow::Result<Vec<usize>> {
        let s = s.to_lowercase();
        let mut bpe_tokens: Vec<usize> = vec![self.start_of_text_token];
        for token in self.re.captures_iter(&s) {
            let token = token.get(0).unwrap().as_str();
            bpe_tokens.extend(self.bpe(token))
        }
        match pad_size_to {
            None => bpe_tokens.push(self.end_of_text_token),
            Some(pad_size_to) => {
                bpe_tokens.push(self.end_of_text_token);
                bpe_tokens.resize_with(
                    std::cmp::min(bpe_tokens.len(), pad_size_to - 1),
                    Default::default,
                );
                let pad_with = match &self.config.pad_with {
                    None => self.end_of_text_token,
                    Some(pad_with) => match self.encoder.get(pad_with) {
                        None => anyhow::bail!("no encoding for padding character {}", pad_with),
                        Some(v) => *v,
                    },
                };
                while bpe_tokens.len() < pad_size_to {
                    bpe_tokens.push(pad_with)
                }
            }
        }
        Ok(bpe_tokens)
    }

    /// The main tokenization entry point, takes as input a string and returns the list of tokens.
    pub fn encode(&self, s: &str) -> anyhow::Result<Vec<usize>> {
        self.encode_pad(s, Some(self.config.max_position_embeddings))
    }

    /// The inverse of the tokenization process, takes as input a list of tokens and returns a
    /// string that produces this tokenization.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let s: String = tokens.iter().map(|token| self.decoder[token].as_str()).collect();
        s.replace("</w>", " ")
    }
}

// CLIP Text Model
// https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py
#[derive(Debug)]
struct ClipTextEmbeddings {
    token_embedding: nn::Embedding,
    position_embedding: nn::Embedding,
    position_ids: Tensor,
}

impl ClipTextEmbeddings {
    fn new(vs: nn::Path, c: &Config) -> Self {
        let token_embedding =
            nn::embedding(&vs / "token_embedding", c.vocab_size, c.embed_dim, Default::default());
        let position_embedding = nn::embedding(
            &vs / "position_embedding",
            c.max_position_embeddings as i64,
            c.embed_dim,
            Default::default(),
        );
        let position_ids =
            Tensor::arange(c.max_position_embeddings as i64, (Kind::Int64, vs.device()))
                .expand([1, -1], false);
        ClipTextEmbeddings { token_embedding, position_embedding, position_ids }
    }
}

impl Module for ClipTextEmbeddings {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let token_embedding = self.token_embedding.forward(xs);
        let position_embedding = self.position_embedding.forward(&self.position_ids);
        token_embedding + position_embedding
    }
}

#[derive(Debug)]
struct ClipAttention {
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    q_proj: nn::Linear,
    out_proj: nn::Linear,
    head_dim: i64,
    scale: f64,
    num_attention_heads: i64,
}

impl ClipAttention {
    fn new(vs: nn::Path, c: &Config) -> Self {
        let embed_dim = c.embed_dim;
        let num_attention_heads = c.num_attention_heads;
        let k_proj = nn::linear(&vs / "k_proj", embed_dim, embed_dim, Default::default());
        let v_proj = nn::linear(&vs / "v_proj", embed_dim, embed_dim, Default::default());
        let q_proj = nn::linear(&vs / "q_proj", embed_dim, embed_dim, Default::default());
        let out_proj = nn::linear(&vs / "out_proj", embed_dim, embed_dim, Default::default());
        let head_dim = embed_dim / num_attention_heads;
        let scale = (head_dim as f64).powf(-0.5);
        ClipAttention { k_proj, v_proj, q_proj, out_proj, head_dim, scale, num_attention_heads }
    }

    fn shape(&self, xs: &Tensor, seq_len: i64, bsz: i64) -> Tensor {
        xs.view((bsz, seq_len, self.num_attention_heads, self.head_dim))
            .transpose(1, 2)
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Tensor {
        let (bsz, tgt_len, embed_dim) = xs.size3().unwrap();
        let query_states = xs.apply(&self.q_proj) * self.scale;
        let proj_shape = (bsz * self.num_attention_heads, -1, self.head_dim);
        let query_states = self.shape(&query_states, tgt_len, bsz).view(proj_shape);
        let key_states = self.shape(&xs.apply(&self.k_proj), -1, bsz).view(proj_shape);
        let value_states = self.shape(&xs.apply(&self.v_proj), -1, bsz).view(proj_shape);
        let attn_weights = query_states.bmm(&key_states.transpose(1, 2));

        let src_len = key_states.size()[1];
        let attn_weights = attn_weights.view((bsz, self.num_attention_heads, tgt_len, src_len))
            + causal_attention_mask;
        let attn_weights = attn_weights.view((bsz * self.num_attention_heads, tgt_len, src_len));
        let attn_weights = attn_weights.softmax(-1, Kind::Float);

        let attn_output = attn_weights.bmm(&value_states);
        attn_output
            .view((bsz, self.num_attention_heads, tgt_len, self.head_dim))
            .transpose(1, 2)
            .reshape([bsz, tgt_len, embed_dim])
            .apply(&self.out_proj)
    }
}

#[derive(Debug)]
struct ClipMlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
    activation: Activation,
}

impl ClipMlp {
    fn new(vs: nn::Path, c: &Config) -> Self {
        let fc1 = nn::linear(&vs / "fc1", c.embed_dim, c.intermediate_size, Default::default());
        let fc2 = nn::linear(&vs / "fc2", c.intermediate_size, c.embed_dim, Default::default());
        ClipMlp { fc1, fc2, activation: c.activation }
    }
}

impl Module for ClipMlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs.apply(&self.fc1);
        self.activation.forward(&xs).apply(&self.fc2)
    }
}

#[derive(Debug)]
struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: nn::LayerNorm,
    mlp: ClipMlp,
    layer_norm2: nn::LayerNorm,
}

impl ClipEncoderLayer {
    fn new(vs: nn::Path, c: &Config) -> Self {
        let self_attn = ClipAttention::new(&vs / "self_attn", c);
        let layer_norm1 =
            nn::layer_norm(&vs / "layer_norm1", vec![c.embed_dim], Default::default());
        let mlp = ClipMlp::new(&vs / "mlp", c);
        let layer_norm2 =
            nn::layer_norm(&vs / "layer_norm2", vec![c.embed_dim], Default::default());
        ClipEncoderLayer { self_attn, layer_norm1, mlp, layer_norm2 }
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Tensor {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs);
        let xs = self.self_attn.forward(&xs, causal_attention_mask);
        let xs = xs + residual;

        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs);
        let xs = self.mlp.forward(&xs);
        xs + residual
    }
}

#[derive(Debug)]
struct ClipEncoder {
    layers: Vec<ClipEncoderLayer>,
}

impl ClipEncoder {
    fn new(vs: nn::Path, c: &Config) -> Self {
        let vs = &vs / "layers";
        let mut layers: Vec<ClipEncoderLayer> = Vec::new();
        for index in 0..c.num_hidden_layers {
            let layer = ClipEncoderLayer::new(&vs / index, c);
            layers.push(layer)
        }
        ClipEncoder { layers }
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Tensor {
        let mut xs = xs.shallow_clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)
        }
        xs
    }
}

/// A CLIP transformer based model.
#[derive(Debug)]
pub struct ClipTextTransformer {
    embeddings: ClipTextEmbeddings,
    encoder: ClipEncoder,
    final_layer_norm: nn::LayerNorm,
}

impl ClipTextTransformer {
    pub fn new(vs: nn::Path, c: &Config) -> Self {
        let vs = &vs / "text_model";
        let embeddings = ClipTextEmbeddings::new(&vs / "embeddings", c);
        let encoder = ClipEncoder::new(&vs / "encoder", c);
        let final_layer_norm =
            nn::layer_norm(&vs / "final_layer_norm", vec![c.embed_dim], Default::default());
        ClipTextTransformer { embeddings, encoder, final_layer_norm }
    }

    // https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py#L678
    fn build_causal_attention_mask(bsz: i64, seq_len: i64, device: Device) -> Tensor {
        let mut mask = Tensor::ones([bsz, seq_len, seq_len], (Kind::Float, device));
        mask.fill_(f32::MIN as f64).triu_(1).unsqueeze(1)
    }
}

impl Module for ClipTextTransformer {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (bsz, seq_len) = xs.size2().unwrap();
        let xs = self.embeddings.forward(xs);
        let causal_attention_mask = Self::build_causal_attention_mask(bsz, seq_len, xs.device());
        let xs = self.encoder.forward(&xs, &causal_attention_mask);
        xs.apply(&self.final_layer_norm)
    }
}
