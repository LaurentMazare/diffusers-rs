use clap::Parser;
use diffusers::pipelines::stable_diffusion;
use diffusers::transformers::clip;
use tch::{nn, nn::Module, Device, Kind, Tensor};

const GUIDANCE_SCALE: f64 = 7.5;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The input image.
    #[arg(long, value_name = "FILE")]
    input_image: String,

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

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<i64>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<i64>,

    /// The UNet weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    unet_weights: Option<String>,

    /// The CLIP weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// The VAE weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<String>,

    #[arg(long, value_name = "FILE", default_value = "data/bpe_simple_vocab_16e6.txt")]
    /// The file specifying the vocabulary to used for tokenization.
    vocab_file: String,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
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

    /// Generate intermediary images at each step.
    #[arg(long, action)]
    intermediary_images: bool,
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
                StableDiffusionVersion::V1_5 => "data/pytorch_model.safetensors".to_string(),
                StableDiffusionVersion::V2_1 => "data/clip_v2.1.safetensors".to_string(),
            },
        }
    }

    fn vae_weights(&self) -> String {
        match &self.vae_weights {
            Some(w) => w.clone(),
            None => match self.sd_version {
                StableDiffusionVersion::V1_5 => "data/vae.safetensors".to_string(),
                StableDiffusionVersion::V2_1 => "data/vae_v2.1.safetensors".to_string(),
            },
        }
    }

    fn unet_weights(&self) -> String {
        match &self.unet_weights {
            Some(w) => w.clone(),
            None => match self.sd_version {
                StableDiffusionVersion::V1_5 => "data/unet.safetensors".to_string(),
                StableDiffusionVersion::V2_1 => "data/unet_v2.1.safetensors".to_string(),
            },
        }
    }
}

fn output_filename(
    basename: &str,
    sample_idx: i64,
    num_samples: i64,
    timestep_idx: Option<usize>,
) -> String {
    let filename = if num_samples > 1 {
        match basename.rsplit_once('.') {
            None => format!("{basename}.{sample_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}.{sample_idx}.{extension}")
            }
        }
    } else {
        basename.to_string()
    };
    match timestep_idx {
        None => filename,
        Some(timestep_idx) => match filename.rsplit_once('.') {
            None => format!("{filename}-{timestep_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}-{timestep_idx}.{extension}")
            }
        },
    }
}

fn image_preprocess<T: AsRef<std::path::Path>>(path: T) -> anyhow::Result<Tensor> {
    let image = tch::vision::image::load(path)?;
    let (_num_channels, height, width) = image.size3()?;
    let height = height - height % 32;
    let width = width - width % 32;
    let image = tch::vision::image::resize(&image, width, height)?;
    Ok((image / 255. * 2. - 1.).unsqueeze(0))
}

fn run(args: Args) -> anyhow::Result<()> {
    let clip_weights = args.clip_weights();
    let vae_weights = args.vae_weights();
    let unet_weights = args.unet_weights();
    let Args {
        prompt,
        cpu,
        height,
        width,
        n_steps,
        seed,
        vocab_file,
        final_image,
        sliced_attention_size,
        num_samples,
        sd_version,
        input_image,
        ..
    } = args;
    tch::maybe_init_cuda();
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    println!("MPS available: {}", tch::utils::has_mps());

    let sd_config = match sd_version {
        StableDiffusionVersion::V1_5 => {
            stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::V2_1 => {
            stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
        }
    };

    let image = image_preprocess(input_image)?;
    let device_setup = diffusers::utils::DeviceSetup::new(cpu);
    let clip_device = device_setup.get("clip");
    let vae_device = device_setup.get("vae");
    let unet_device = device_setup.get("unet");
    let scheduler = sd_config.build_scheduler(n_steps);

    let tokenizer = clip::Tokenizer::create(vocab_file, &sd_config.clip)?;
    println!("Running with prompt \"{prompt}\".");
    let tokens = tokenizer.encode(&prompt)?;
    let tokens: Vec<i64> = tokens.into_iter().map(|x| x as i64).collect();
    let tokens = Tensor::from_slice(&tokens).view((1, -1)).to(clip_device);
    let uncond_tokens = tokenizer.encode("")?;
    let uncond_tokens: Vec<i64> = uncond_tokens.into_iter().map(|x| x as i64).collect();
    let uncond_tokens = Tensor::from_slice(&uncond_tokens).view((1, -1)).to(clip_device);

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
    println!("Building the controlnet.");
    let vs_controlnet = nn::VarStore::new(unet_device);
    let controlnet =
        diffusers::models::controlnet::ControlNet::new(vs_controlnet.root(), 4, Default::default());

    let bsize = 1;
    for idx in 0..num_samples {
        tch::manual_seed(seed + idx);
        let mut latents = Tensor::randn(
            [bsize, 4, sd_config.height / 8, sd_config.width / 8],
            (Kind::Float, unet_device),
        );

        // scale the initial noise by the standard deviation required by the scheduler
        latents *= scheduler.init_noise_sigma();

        for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
            println!("Timestep {timestep_index}/{n_steps}");
            let latent_model_input = Tensor::cat(&[&latents, &latents], 0);

            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep);
            let (down_block_additional_residuals, mid_block_additional_residuals) = controlnet
                .forward(&latent_model_input, timestep as f64, &text_embeddings, &image, 1.);
            let noise_pred = unet.forward_with_additional_residuals(
                &latent_model_input,
                timestep as f64,
                &text_embeddings,
                Some(&down_block_additional_residuals),
                Some(&mid_block_additional_residuals),
            );
            let noise_pred = noise_pred.chunk(2, 0);
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            let noise_pred =
                noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * GUIDANCE_SCALE;
            latents = scheduler.step(&noise_pred, timestep, &latents);

            if args.intermediary_images {
                let latents = latents.to(vae_device);
                let image = vae.decode(&(&latents / 0.18215));
                let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
                let image = (image * 255.).to_kind(Kind::Uint8);
                let final_image =
                    output_filename(&final_image, idx + 1, num_samples, Some(timestep_index + 1));
                tch::vision::image::save(&image, final_image)?;
            }
        }

        println!("Generating the final image for sample {}/{}.", idx + 1, num_samples);
        let latents = latents.to(vae_device);
        let image = vae.decode(&(&latents / 0.18215));
        let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
        let image = (image * 255.).to_kind(Kind::Uint8);
        let final_image = output_filename(&final_image, idx + 1, num_samples, None);
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
