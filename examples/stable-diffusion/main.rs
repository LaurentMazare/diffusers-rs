// Stable Diffusion implementation inspired:
// - Huggingface's amazing diffuser Python api: https://huggingface.co/blog/annotated-diffusion
// - Huggingface's (also amazing) blog post: https://huggingface.co/blog/annotated-diffusion
// - The "Grokking Stable Diffusion" notebook by Jonathan Whitaker.
// https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing
//
// In order to run this, the weights first have to be downloaded and converted by following
// the instructions below.
//
// mkdir -p data && cd data
// wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
// gunzip bpe_simple_vocab_16e6.txt.gz
//
// Getting the weights then depend on the stable diffusion version (1.5 or 2.1).
//
// # How to get the weights for Stable Diffusion 2.1.
//
// 1. Clip Encoding Weights
// wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/fp16/text_encoder/pytorch_model.bin -O clip.bin
// From python, extract the weights and save them as a .npz file.
//   import numpy as np
//   import torch
//   model = torch.load("./clip.bin")
//   np.savez("./clip_v2.1.npz", **{k: v.numpy() for k, v in model.items() if "text_model" in k})
//
// Then use tensor-tools from the tch-rs repo to convert this to a .ot file that tch can use.
// In the tch-rs repo (https://github.com/LaurentMazare/tch-rs):
//   cargo run --release --example tensor-tools cp ./data/clip_v2.1.npz ./data/clip_v2.1.ot
//
// 2. VAE and Unet Weights
// wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/fp16/vae/diffusion_pytorch_model.bin -O vae.bin
// wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/fp16/unet/diffusion_pytorch_model.bin -O unet.bin
//
//   import numpy as np
//   import torch
//   model = torch.load("./vae.bin")
//   np.savez("./vae_v2.1.npz", **{k: v.numpy() for k, v in model.items()})
//   model = torch.load("./unet.bin")
//   np.savez("./unet_v2.1.npz", **{k: v.numpy() for k, v in model.items()})
//
//   cargo run --release --example tensor-tools cp ./data/vae_v2.1.npz ./data/vae_v2.1.ot
//   cargo run --release --example tensor-tools cp ./data/unet_v2.1.npz ./data/unet_v2.1.ot
//
// # How to get the weights for Stable Diffusion 1.5.
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
// https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/diffusion_pytorch_model.bin
// https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/unet/diffusion_pytorch_model.bin
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
use diffusers::pipelines::stable_diffusion;
use diffusers::transformers::clip;
use tch::{nn::Module, Device, Kind, Tensor};

const GUIDANCE_SCALE: f64 = 7.5;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    /// When set, use the CPU for the listed devices, can be 'all', 'unet', 'clip', etc.
    /// Multiple values can be set.
    #[arg(long)]
    cpu: Vec<String>,

    /// The UNet weight file, in .ot format.
    #[arg(long, value_name = "FILE")]
    unet_weights: Option<String>,

    /// The CLIP weight file, in .ot format.
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// The VAE weight file, in .ot format.
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<String>,

    #[arg(long, value_name = "FILE", default_value = "data/bpe_simple_vocab_16e6.txt")]
    /// The file specifying the vocabulary to used for tokenization.
    vocab_file: String,

    /// The size of the sliced attention or 0 to disable slicing (default)
    #[arg(long)]
    sliced_attention_size: Option<i64>,

    /// The number of steps to run the diffusion for.
    #[arg(long, default_value_t = 30)]
    n_steps: usize,

    /// The random seed to be used for the generation.
    #[arg(long, default_value_t = 32)]
    seed: i64,

    /// The number of samples to generate.
    #[arg(long, default_value_t = 1)]
    num_samples: i64,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "sd_final.png")]
    final_image: String,

    /// Use autocast (disabled by default as it may use more memory in some cases).
    #[arg(long, action)]
    autocast: bool,

    #[arg(long, value_enum, default_value = "v2-1")]
    sd_version: StableDiffusionVersion,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum StableDiffusionVersion {
    V1_5,
    V2_1,
}

impl Args {
    fn clip_weights(&self) -> String {
        match &self.clip_weights {
            Some(w) => w.clone(),
            None => match self.sd_version {
                StableDiffusionVersion::V1_5 => "data/pytorch_model.ot".to_string(),
                StableDiffusionVersion::V2_1 => "data/clip_v2.1.ot".to_string(),
            },
        }
    }

    fn vae_weights(&self) -> String {
        match &self.vae_weights {
            Some(w) => w.clone(),
            None => match self.sd_version {
                StableDiffusionVersion::V1_5 => "data/vae.ot".to_string(),
                StableDiffusionVersion::V2_1 => "data/vae_v2.1.ot".to_string(),
            },
        }
    }

    fn unet_weights(&self) -> String {
        match &self.unet_weights {
            Some(w) => w.clone(),
            None => match self.sd_version {
                StableDiffusionVersion::V1_5 => "data/unet.ot".to_string(),
                StableDiffusionVersion::V2_1 => "data/unet_v2.1.ot".to_string(),
            },
        }
    }
}

fn run(args: Args) -> anyhow::Result<()> {
    let clip_weights = args.clip_weights();
    let vae_weights = args.vae_weights();
    let unet_weights = args.unet_weights();
    let Args {
        prompt,
        cpu,
        n_steps,
        seed,
        vocab_file,
        final_image,
        sliced_attention_size,
        num_samples,
        sd_version,
        ..
    } = args;
    tch::maybe_init_cuda();
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    let sd_config = match sd_version {
        StableDiffusionVersion::V1_5 => {
            stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size)
        }
        StableDiffusionVersion::V2_1 => {
            stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size)
        }
    };
    let cuda_device = Device::cuda_if_available();
    let get_device = |name: &str| match &cuda_device {
        Device::Cpu => Device::Cpu,
        Device::Cuda(_) => {
            if cpu.iter().any(|c| c == "all" || c == name) {
                Device::Cpu
            } else {
                cuda_device.clone()
            }
        }
        Device::Mps => {
            if cpu.iter().any(|c| c == "all" || c == name) {
                Device::Cpu
            } else {
                Device::Mps
            }
        }
    };
    let clip_device = get_device("clip");
    let vae_device = get_device("vae");
    let unet_device = get_device("unet");
    let scheduler = sd_config.build_scheduler(n_steps);

    let tokenizer = clip::Tokenizer::create(vocab_file, &sd_config.clip)?;
    println!("Running with prompt \"{prompt}\".");
    let tokens = tokenizer.encode(&prompt)?;
    let tokens: Vec<i64> = tokens.into_iter().map(|x| x as i64).collect();
    let tokens = Tensor::of_slice(&tokens).view((1, -1)).to(clip_device);
    let uncond_tokens = tokenizer.encode("")?;
    let uncond_tokens: Vec<i64> = uncond_tokens.into_iter().map(|x| x as i64).collect();
    let uncond_tokens = Tensor::of_slice(&uncond_tokens).view((1, -1)).to(clip_device);

    let no_grad_guard = tch::no_grad_guard();

    println!("Building the Clip transformer.");
    let text_model = sd_config.build_clip_transformer(&clip_weights, clip_device)?;
    let text_embeddings = text_model.forward(&tokens);
    let uncond_embeddings = text_model.forward(&uncond_tokens);
    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0).to(unet_device);

    println!("Building the autoencoder.");
    let vae = sd_config.build_vae(&vae_weights, vae_device)?;
    println!("Building the unet.");
    let unet = sd_config.build_unet(&unet_weights, unet_device, 4)?;

    let bsize = 1;
    for idx in 0..num_samples {
        tch::manual_seed(seed + idx);
        let mut latents = Tensor::randn(
            &[bsize, 4, sd_config.height / 8, sd_config.width / 8],
            (Kind::Float, unet_device),
        );

        for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
            println!("Timestep {timestep_index}/{n_steps}");
            let latent_model_input = Tensor::cat(&[&latents, &latents], 0);
            let noise_pred = unet.forward(&latent_model_input, timestep as f64, &text_embeddings);
            let noise_pred = noise_pred.chunk(2, 0);
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            let noise_pred =
                noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * GUIDANCE_SCALE;
            latents = scheduler.step(&noise_pred, timestep, &latents);
        }

        println!("Generating the final image for sample {}/{}.", idx + 1, num_samples);
        let latents = latents.to(vae_device);
        let image = vae.decode(&(&latents / 0.18215));
        let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
        let image = (image * 255.).to_kind(Kind::Uint8);
        let final_image = if num_samples > 1 {
            match final_image.rsplit_once('.') {
                None => format!("{}.{}.png", final_image, idx + 1),
                Some((filename_no_extension, extension)) => {
                    format!("{}.{}.{}", filename_no_extension, idx + 1, extension)
                }
            }
        } else {
            final_image.clone()
        };
        tch::vision::image::save(&image, final_image)?;
    }

    drop(no_grad_guard);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if !args.autocast {
        run(args)
    } else {
        tch::autocast(true, || run(args))
    }
}
