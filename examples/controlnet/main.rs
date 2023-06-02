// The additional weight files can be found on HuggingFace hub:
// https://huggingface.co/lllyasviel/sd-controlnet-canny/blob/main/diffusion_pytorch_model.safetensors
// This has to be copied in data/controlnet.safetensors
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
    #[arg(long, value_name = "FILE", default_value = "data/unet.safetensors")]
    unet_weights: String,

    /// The ControlNet weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE", default_value = "data/controlnet.safetensors")]
    controlnet_weights: String,

    /// The CLIP weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE", default_value = "data/pytorch_model.safetensors")]
    clip_weights: String,

    /// The VAE weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE", default_value = "data/vae.safetensors")]
    vae_weights: String,

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

    /// Generate intermediary images at each step.
    #[arg(long, action)]
    intermediary_images: bool,

    /// The type of ControlNet model to be used.
    #[arg(long, value_enum, default_value = "canny")]
    control_type: ControlType,
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

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum ControlType {
    Canny,
}

impl ControlType {
    fn image_preprocess<T: AsRef<std::path::Path>>(&self, path: T) -> anyhow::Result<Tensor> {
        match self {
            Self::Canny => {
                // TODO: Use an implementation of the Canny edge detector in PyTorch
                // and remove this dependency.
                use image::EncodableLayout;
                let image = image::open(path)?.to_luma8();
                let edges = imageproc::edges::canny(&image, 50., 100.);
                let tensor = Tensor::f_from_data_size(
                    edges.as_bytes(),
                    &[1, 1, edges.height() as i64, edges.width() as i64],
                    Kind::Uint8,
                )?;
                let tensor = Tensor::f_concat(&[&tensor, &tensor, &tensor], 1)?;
                // In order to look at the detected edges, uncomment the following line:
                // tch::vision::image::save(&tensor.squeeze(), "/tmp/edges.png").unwrap();
                let tensor = Tensor::f_concat(&[&tensor, &tensor], 0)?;
                Ok(tensor.to_kind(Kind::Float) / 255.)
            }
        }
    }
}

fn run(args: Args) -> anyhow::Result<()> {
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
        input_image,
        unet_weights,
        vae_weights,
        clip_weights,
        controlnet_weights,
        control_type,
        ..
    } = args;
    tch::maybe_init_cuda();
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    println!("MPS available: {}", tch::utils::has_mps());

    let sd_config =
        stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width);

    let image = control_type.image_preprocess(input_image)?;
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
    let mut vs_controlnet = nn::VarStore::new(unet_device);
    let controlnet =
        diffusers::models::controlnet::ControlNet::new(vs_controlnet.root(), 4, Default::default());
    vs_controlnet.load(controlnet_weights)?;

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
