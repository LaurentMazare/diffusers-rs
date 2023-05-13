//! # Denoising Diffusion Implicit Models
//!
//! The Denoising Diffusion Implicit Models (DDIM) is a simple scheduler
//! similar to Denoising Diffusion Probabilistic Models (DDPM). The DDPM
//! generative process is the reverse of a Markovian process, DDIM generalizes
//! this to non-Markovian guidance.
//!
//! Denoising Diffusion Implicit Models, J. Song et al, 2020.
//! https://arxiv.org/abs/2010.02502
use super::{betas_for_alpha_bar, BetaSchedule, PredictionType};
use tch::{kind, Kind, Tensor};

/// The configuration for the DDIM scheduler.
#[derive(Debug, Clone, Copy)]
pub struct DDIMSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// The amount of noise to be added at each step.
    pub eta: f64,
    /// Adjust the indexes of the inference schedule by this value.
    pub steps_offset: usize,
    /// prediction type of the scheduler function, one of `epsilon` (predicting
    /// the noise of the diffusion process), `sample` (directly predicting the noisy sample`)
    /// or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)
    pub prediction_type: PredictionType,
    /// number of diffusion steps used to train the model
    pub train_timesteps: usize,
}

impl Default for DDIMSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085f64,
            beta_end: 0.012f64,
            beta_schedule: BetaSchedule::ScaledLinear,
            eta: 0.,
            steps_offset: 1,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
        }
    }
}

/// The DDIM scheduler.
#[derive(Debug, Clone)]
pub struct DDIMScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
    step_ratio: usize,
    init_noise_sigma: f64,
    pub config: DDIMSchedulerConfig,
}

// clip_sample: False, set_alpha_to_one: False
impl DDIMScheduler {
    /// Creates a new DDIM scheduler given the number of steps to be
    /// used for inference as well as the number of steps that was used
    /// during training.
    pub fn new(inference_steps: usize, config: DDIMSchedulerConfig) -> Self {
        let step_ratio = config.train_timesteps / inference_steps;
        let timesteps: Vec<usize> =
            (0..(inference_steps)).map(|s| s * step_ratio + config.steps_offset).rev().collect();
        let betas = match config.beta_schedule {
            BetaSchedule::ScaledLinear => Tensor::linspace(
                config.beta_start.sqrt(),
                config.beta_end.sqrt(),
                config.train_timesteps as i64,
                kind::FLOAT_CPU,
            )
            .square(),
            BetaSchedule::Linear => Tensor::linspace(
                config.beta_start,
                config.beta_end,
                config.train_timesteps as i64,
                kind::FLOAT_CPU,
            ),
            BetaSchedule::SquaredcosCapV2 => betas_for_alpha_bar(config.train_timesteps, 0.999),
        };
        let alphas: Tensor = 1.0 - betas;
        let alphas_cumprod = Vec::<f64>::try_from(alphas.cumprod(0, Kind::Double)).unwrap();
        Self { alphas_cumprod, timesteps, step_ratio, init_noise_sigma: 1., config }
    }

    pub fn timesteps(&self) -> &[usize] {
        self.timesteps.as_slice()
    }

    ///  Ensures interchangeability with schedulers that need to scale the denoising model input
    /// depending on the current timestep.
    pub fn scale_model_input(&self, sample: Tensor, _timestep: usize) -> Tensor {
        sample
    }

    /// Performs a backward step during inference.
    pub fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Tensor {
        let timestep = if timestep >= self.alphas_cumprod.len() { timestep - 1 } else { timestep };
        // https://github.com/huggingface/diffusers/blob/6e099e2c8ce4c4f5c7318e970a8c093dc5c7046e/src/diffusers/schedulers/scheduling_ddim.py#L195
        let prev_timestep = if timestep > self.step_ratio { timestep - self.step_ratio } else { 0 };

        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = self.alphas_cumprod[prev_timestep];
        let beta_prod_t = 1. - alpha_prod_t;
        let beta_prod_t_prev = 1. - alpha_prod_t_prev;

        let (pred_original_sample, pred_epsilon) = match self.config.prediction_type {
            PredictionType::Epsilon => {
                let pred_original_sample =
                    (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt();
                (pred_original_sample, model_output.shallow_clone())
            }
            PredictionType::VPrediction => {
                let pred_original_sample =
                    alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output;
                let pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample;
                (pred_original_sample, pred_epsilon)
            }
            PredictionType::Sample => {
                let pred_original_sample = model_output.shallow_clone();
                let pred_epsilon =
                    (sample - alpha_prod_t.sqrt() * &pred_original_sample) / beta_prod_t.sqrt();
                (pred_original_sample, pred_epsilon)
            }
        };

        let variance = (beta_prod_t_prev / beta_prod_t) * (1. - alpha_prod_t / alpha_prod_t_prev);
        let std_dev_t = self.config.eta * variance.sqrt();

        let pred_sample_direction =
            (1. - alpha_prod_t_prev - std_dev_t * std_dev_t).sqrt() * pred_epsilon;
        let prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction;
        if self.config.eta > 0. {
            &prev_sample + Tensor::randn_like(&prev_sample) * std_dev_t
        } else {
            prev_sample
        }
    }

    pub fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Tensor {
        let timestep = if timestep >= self.alphas_cumprod.len() { timestep - 1 } else { timestep };
        let sqrt_alpha_prod = self.alphas_cumprod[timestep].sqrt();
        let sqrt_one_minus_alpha_prod = (1.0 - self.alphas_cumprod[timestep]).sqrt();
        sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
    }

    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }
}
