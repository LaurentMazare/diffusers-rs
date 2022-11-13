//! # Denoising Diffusion Implicit Models
//!
//! The Denoising Diffusion Implicit Models (DDIM) is a simple scheduler
//! similar to Denoising Diffusion Probabilistic Models (DDPM). The DDPM
//! generative process is the reverse of a Markovian process, DDIM generalizes
//! this to non-Markovian guidance.
//!
//! Denoising Diffusion Implicit Models, J. Song et al, 2020.
//! https://arxiv.org/abs/2010.02502
use tch::{kind, Kind, Tensor};

/// This represents how beta ranges from its minimum value to the maximum
/// during training.
#[derive(Debug, Clone, Copy)]
pub enum BetaSchedule {
    /// Linear interpolation.
    Linear,
    /// Linear interpolation of the square root of beta.
    ScaledLinear,
}

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
}

impl Default for DDIMSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085f64,
            beta_end: 0.012f64,
            beta_schedule: BetaSchedule::ScaledLinear,
            eta: 0.,
        }
    }
}

/// The DDIM scheduler.
#[derive(Debug, Clone)]
pub struct DDIMScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
    step_ratio: usize,
    pub config: DDIMSchedulerConfig,
}

// clip_sample: False, set_alpha_to_one: False
impl DDIMScheduler {
    /// Creates a new DDIM scheduler given the number of steps to be
    /// used for inference as well as the number of steps that was used
    /// during training.
    pub fn new(
        inference_steps: usize,
        train_timesteps: usize,
        config: DDIMSchedulerConfig,
    ) -> Self {
        let step_ratio = train_timesteps / inference_steps;
        // TODO: Remove this hack which aimed at matching the behavior of diffusers==0.2.4
        let timesteps = (0..(inference_steps + 1)).map(|s| s * step_ratio).rev().collect();
        let betas = match config.beta_schedule {
            BetaSchedule::ScaledLinear => Tensor::linspace(
                config.beta_start.sqrt(),
                config.beta_end.sqrt(),
                train_timesteps as i64,
                kind::FLOAT_CPU,
            )
            .square(),
            BetaSchedule::Linear => Tensor::linspace(
                config.beta_start,
                config.beta_end,
                train_timesteps as i64,
                kind::FLOAT_CPU,
            ),
        };
        let alphas: Tensor = 1.0 - betas;
        let alphas_cumprod = Vec::<f64>::from(alphas.cumprod(0, Kind::Double));
        Self { alphas_cumprod, timesteps, step_ratio, config }
    }

    pub fn timesteps(&self) -> &[usize] {
        self.timesteps.as_slice()
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

        let pred_original_sample =
            (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt();

        let variance = (beta_prod_t_prev / beta_prod_t) * (1. - alpha_prod_t / alpha_prod_t_prev);
        let std_dev_t = self.config.eta * variance.sqrt();

        let pred_sample_direction =
            (1. - alpha_prod_t_prev - std_dev_t * std_dev_t).sqrt() * model_output;
        let prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction;
        if self.config.eta > 0. {
            &prev_sample + Tensor::randn_like(&prev_sample) * std_dev_t
        } else {
            prev_sample
        }
    }
}
