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
use diffusers::{pipelines::stable_diffusion, schedulers::ddim, transformers::clip};
use tch::{nn::Module, Device, Kind, Tensor};

const HEIGHT: i64 = 512;
const WIDTH: i64 = 512;
const GUIDANCE_SCALE: f64 = 7.5;

// TODO: LMSDiscreteScheduler
// https://github.com/huggingface/diffusers/blob/32bf4fdc4386809c870528cb261028baae012d27/src/diffusers/schedulers/scheduling_lms_discrete.py#L47

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: Option<String>,

    /// When set, use the CPU for the listed devices, can be 'all', 'unet', 'clip', etc.
    /// Multiple values can be set.
    #[arg(long)]
    cpu: Vec<String>,

    /// The UNet weight file, in .ot format.
    #[arg(long, value_name = "FILE", default_value = "data/unet.ot")]
    unet_weights: String,

    /// The CLIP weight file, in .ot format.
    #[arg(long, value_name = "FILE", default_value = "data/pytorch_model.ot")]
    clip_weights: String,

    /// The VAE weight file, in .ot format.
    #[arg(long, value_name = "FILE", default_value = "data/vae.ot")]
    vae_weights: String,

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
}

fn run(args: Args) -> anyhow::Result<()> {
    let Args {
        prompt,
        cpu,
        n_steps,
        seed,
        final_image,
        vae_weights,
        clip_weights,
        unet_weights,
        sliced_attention_size,
        num_samples,
        autocast: _,
    } = args;
    tch::maybe_init_cuda();
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    let cuda_device = Device::cuda_if_available();
    let cpu_or_cuda = |name: &str| {
        if cpu.iter().any(|c| c == "all" || c == name) {
            Device::Cpu
        } else {
            cuda_device
        }
    };
    let clip_device = cpu_or_cuda("clip");
    let vae_device = cpu_or_cuda("vae");
    let unet_device = cpu_or_cuda("unet");
    let scheduler = ddim::DDIMScheduler::new(n_steps, 1000, Default::default());

    let clip_config = clip::Config::v1_5();
    let tokenizer = clip::Tokenizer::create("data/bpe_simple_vocab_16e6.txt", &clip_config)?;
    let prompt = prompt.unwrap_or_else(|| {
        "A very realistic photo of a rusty robot walking on a sandy beach".to_string()
    });
    println!("Running with prompt \"{prompt}\".");
    let tokens = tokenizer.encode(&prompt)?;
    let tokens: Vec<i64> = tokens.into_iter().map(|x| x as i64).collect();
    let tokens = Tensor::of_slice(&tokens).view((1, -1)).to(clip_device);
    let uncond_tokens = tokenizer.encode("")?;
    let uncond_tokens: Vec<i64> = uncond_tokens.into_iter().map(|x| x as i64).collect();
    let uncond_tokens = Tensor::of_slice(&uncond_tokens).view((1, -1)).to(clip_device);

    let no_grad_guard = tch::no_grad_guard();

    println!("Building the Clip transformer.");
    let text_model =
        stable_diffusion::build_clip_transformer(&clip_weights, &clip_config, clip_device)?;
    let text_embeddings = text_model.forward(&tokens);
    let uncond_embeddings = text_model.forward(&uncond_tokens);
    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0).to(unet_device);

    println!("Building the autoencoder.");
    let vae = stable_diffusion::build_vae(&vae_weights, vae_device)?;
    println!("Building the unet.");
    let unet = stable_diffusion::build_unet(&unet_weights, unet_device, 4, sliced_attention_size)?;

    let bsize = 1;
    for idx in 0..num_samples {
        tch::manual_seed(seed + idx);
        let mut latents =
            Tensor::randn(&[bsize, 4, HEIGHT / 8, WIDTH / 8], (Kind::Float, unet_device));

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
        tch::vision::image::save(&image, &final_image)?;
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
