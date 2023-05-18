// Stable diffusion inpainting pipeline.
// See the main stable-diffusion example for how to get the weights.
//
// This has been mostly adapted from looking at the diff between the sample
// diffusion standard and inpaint pipelines in the diffusers library.
// patdiff src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion{,_inpaint}.py
//
// The unet weights should be downloaded from:
// https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/unet/diffusion_pytorch_model.bin
// Or for the fp16 version:
// https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/fp16/unet/diffusion_pytorch_model.bin
//
// Sample input image:
// https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png
// Sample mask:
// https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png
use clap::Parser;
use diffusers::pipelines::stable_diffusion;
use diffusers::transformers::clip;
use tch::{nn::Module, Device, Kind, Tensor};

const GUIDANCE_SCALE: f64 = 7.5;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The input image that will be inpainted.
    #[arg(long, value_name = "FILE")]
    input_image: String,

    /// The mask image to be used for inpainting, white pixels are repainted whereas black pixels
    /// are preserved.
    #[arg(long, value_name = "FILE")]
    mask_image: String,

    /// The prompt to be used for image generation.
    #[arg(long, default_value = "Face of a yellow cat, high resolution, sitting on a park bench")]
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

    #[arg(long, value_name = "FILE", default_value = "data/bpe_simple_vocab_16e6.txt")]
    /// The file specifying the vocabulary to used for tokenization.
    vocab_file: String,

    /// The UNet weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    unet_weights: Option<String>,

    /// The CLIP weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// The VAE weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<String>,

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

    #[arg(long, value_enum, default_value = "v1-5")]
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
                StableDiffusionVersion::V1_5 => "data/unet-inpaint.safetensors".to_string(),
                StableDiffusionVersion::V2_1 => "data/unet-inpaint_v2.1.safetensors".to_string(),
            },
        }
    }
}

fn prepare_mask_and_masked_image<T: AsRef<std::path::Path>>(
    path_input: T,
    path_mask: T,
) -> anyhow::Result<(Tensor, Tensor)> {
    let image = tch::vision::image::load(path_input)?;
    let image = image / 255. * 2. - 1.;

    let mask = tch::vision::image::load(path_mask)?;
    let mask = mask.mean_dim(Some([0].as_slice()), true, Kind::Float);
    let mask = mask.ge(122.5).totype(Kind::Float);
    let masked_image: Tensor = image * (1 - &mask);
    Ok((mask.unsqueeze(0), masked_image.unsqueeze(0)))
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
        final_image,
        sliced_attention_size,
        num_samples,
        input_image,
        mask_image,
        vocab_file,
        sd_version,
        ..
    } = args;
    tch::maybe_init_cuda();
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    let sd_config = match sd_version {
        StableDiffusionVersion::V1_5 => {
            stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::V2_1 => stable_diffusion::StableDiffusionConfig::v2_1_inpaint(
            sliced_attention_size,
            height,
            width,
        ),
    };
    let (mask, masked_image) = prepare_mask_and_masked_image(input_image, mask_image)?;
    println!("Loaded input image and mask, {:?} {:?}.", masked_image.size(), mask.size());
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
    let unet = sd_config.build_unet(&unet_weights, unet_device, 9)?;

    let mask = mask.upsample_nearest2d([sd_config.height / 8, sd_config.width / 8], None, None);
    let mask = Tensor::cat(&[&mask, &mask], 0).to_device(unet_device);
    let masked_image_dist = vae.encode(&masked_image.to_device(vae_device));

    let bsize = 1;
    for idx in 0..num_samples {
        tch::manual_seed(seed + idx);
        let masked_image_latents = (masked_image_dist.sample() * 0.18215).to(unet_device);
        let masked_image_latents = Tensor::cat(&[&masked_image_latents, &masked_image_latents], 0);
        let mut latents = Tensor::randn(
            [bsize, 4, sd_config.height / 8, sd_config.width / 8],
            (Kind::Float, unet_device),
        );

        // scale the initial noise by the standard deviation required by the scheduler
        latents *= scheduler.init_noise_sigma();

        for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
            println!("Timestep {timestep_index}/{n_steps}");
            let latent_model_input = Tensor::cat(&[&latents, &latents], 0);

            // concat latents, mask, masked_image_latents in the channel dimension
            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep);
            let latent_model_input =
                Tensor::cat(&[&latent_model_input, &mask, &masked_image_latents], 1);
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
    run(args)
}
