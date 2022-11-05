// Stable Diffusion implementation inspired:
// - Huggingface's amazing diffuser Python api: https://huggingface.co/blog/annotated-diffusion
// - Huggingface's (also amazing) blog post: https://huggingface.co/blog/annotated-diffusion
// - The "Grokking Stable Diffusion" notebook by Jonathan Whitaker.
// https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing
//
// In order to run this, first download the following and extract the file in data/
//
// mkdir -p data && cd data
// wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
// gunzip bpe_simple_vocab_16e6.txt.gz
//
// Download and convert the weights:
//
// 1. Clip Encoding Weights
// wget https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin
// From python, extract the weights and save them as a .npz file.
//   import numpy as np
//   import torch
//   model = torch.load("./pytorch_model.bin")
//   np.savez("./pytorch_model.npz", **{k: v.numpy() for k, v in model.items() if "text_model" in k})
//
// Then use tensor-tools from the tch-rs repo to convert this to a .ot file that tch can use.
// In the tch-rs repo (https://github.com/LaurentMazare/tch-rs):
//   cargo run --release --example tensor-tools cp ./data/pytorch_model.npz ./data/pytorch_model.ot
//
// 2. VAE and Unet Weights
// https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/vae/diffusion_pytorch_model.bin
// https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/diffusion_pytorch_model.bin
//
//   import numpy as np
//   import torch
//   model = torch.load("./vae.bin")
//   np.savez("./vae.npz", **{k: v.numpy() for k, v in model.items()})
//   model = torch.load("./unet.bin")
//   np.savez("./unet.npz", **{k: v.numpy() for k, v in model.items()})
//
//   cargo run --release --example tensor-tools cp ./data/vae.npz ./data/vae.ot
//   cargo run --release --example tensor-tools cp ./data/unet.npz ./data/unet.ot
use clap::Parser;
use diffusers::{
    models::{unet_2d, vae},
    schedulers::ddim,
    transformers::clip,
};
use tch::{nn, nn::Module, Device, Kind, Tensor};

const HEIGHT: i64 = 512;
const WIDTH: i64 = 512;
const GUIDANCE_SCALE: f64 = 7.5;

// TODO: LMSDiscreteScheduler
// https://github.com/huggingface/diffusers/blob/32bf4fdc4386809c870528cb261028baae012d27/src/diffusers/schedulers/scheduling_lms_discrete.py#L47

fn build_clip_transformer(device: Device) -> anyhow::Result<clip::ClipTextTransformer> {
    let mut vs = nn::VarStore::new(device);
    let text_model = clip::ClipTextTransformer::new(vs.root());
    vs.load("data/pytorch_model.ot")?;
    Ok(text_model)
}

fn build_vae(device: Device) -> anyhow::Result<vae::AutoEncoderKL> {
    let mut vs_ae = nn::VarStore::new(device);
    // https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/vae/config.json
    let autoencoder_cfg = vae::AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
    };
    let autoencoder = vae::AutoEncoderKL::new(vs_ae.root(), 3, 3, autoencoder_cfg);
    vs_ae.load("data/vae.ot")?;
    Ok(autoencoder)
}

fn build_unet(device: Device) -> anyhow::Result<unet_2d::UNet2DConditionModel> {
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
    };
    let unet = unet_2d::UNet2DConditionModel::new(vs_unet.root(), 4, 4, unet_cfg);
    vs_unet.load("data/unet.ot")?;
    Ok(unet)
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(long)]
    prompt: Option<String>,

    /// When set, use the CPU even if some CUDA devices are available.
    #[arg(long)]
    cpu: bool,

    /// The number of steps to run the diffusion for.
    #[arg(long, default_value_t = 30)]
    n_steps: usize,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE")]
    final_image: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tch::maybe_init_cuda();
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    let device = if args.cpu { Device::Cpu } else { Device::cuda_if_available() };
    let scheduler = ddim::DDIMScheduler::new(args.n_steps, 1000, Default::default());

    let tokenizer = clip::Tokenizer::create("data/bpe_simple_vocab_16e6.txt")?;
    let prompt =
        args.prompt.unwrap_or("A rusty robot holding a fire torch in its hand".to_string());
    let tokens = tokenizer.encode(&prompt)?;
    let str = tokenizer.decode(&tokens);
    println!("Str: {}", str);
    let tokens: Vec<i64> = tokens.iter().map(|x| *x as i64).collect();
    let tokens = Tensor::of_slice(&tokens).view((1, -1)).to(device);
    let uncond_tokens = tokenizer.encode("")?;
    let uncond_tokens: Vec<i64> = uncond_tokens.iter().map(|x| *x as i64).collect();
    let uncond_tokens = Tensor::of_slice(&uncond_tokens).view((1, -1)).to(device);
    println!("Tokens: {:?}", tokens);

    let no_grad_guard = tch::no_grad_guard();
    println!("Building the Clip transformer.");
    let text_model = build_clip_transformer(device)?;
    let text_embeddings = text_model.forward(&tokens);
    let uncond_embeddings = text_model.forward(&uncond_tokens);
    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0);
    println!("Text embeddings: {:?}", text_embeddings);

    println!("Building the autoencoder.");
    let vae = build_vae(device)?;
    println!("Building the unet.");
    let unet = build_unet(device)?;

    let bsize = 1;
    // DETERMINISTIC SEEDING
    tch::manual_seed(32);
    let mut latents = Tensor::randn(&[bsize, 4, HEIGHT / 8, WIDTH / 8], (Kind::Float, device));

    for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
        println!("Timestep {} {} {:?}", timestep_index, timestep, latents);
        let latent_model_input = Tensor::cat(&[&latents, &latents], 0);
        let noise_pred = unet.forward(&latent_model_input, timestep as f64, &text_embeddings);
        let noise_pred = noise_pred.chunk(2, 0);
        let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
        let noise_pred = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * GUIDANCE_SCALE;
        latents = scheduler.step(&noise_pred, timestep, &latents);
    }

    let image = vae.decode(&(&latents / 0.18215));
    let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
    let image = (image * 255.).to_kind(Kind::Uint8);
    let final_image = args.final_image.unwrap_or("sd_final.png".to_string());
    tch::vision::image::save(&image, final_image)?;

    drop(no_grad_guard);
    Ok(())
}
