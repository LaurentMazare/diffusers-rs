use super::{betas_for_alpha_bar, BetaSchedule, PredictionType};
use tch::{kind, Kind, Tensor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DDPMVarianceType {
    FixedSmall,
    FixedSmallLog,
    FixedLarge,
    FixedLargeLog,
    Learned,
}

impl Default for DDPMVarianceType {
    fn default() -> Self {
        Self::FixedSmall
    }
}

#[derive(Debug, Clone)]
pub struct DDPMSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// Option to predicted sample between -1 and 1 for numerical stability.
    pub clip_sample: bool,
    /// Option to clip the variance used when adding noise to the denoised sample.
    pub variance_type: DDPMVarianceType,
    pub prediction_type: PredictionType,
    pub train_timesteps: usize,
}

impl Default for DDPMSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.0001,
            beta_end: 0.02,
            beta_schedule: BetaSchedule::Linear,
            clip_sample: true,
            variance_type: DDPMVarianceType::FixedSmall,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
        }
    }
}

pub struct DDPMScheduler {
    alphas: Vec<f64>,
    betas: Vec<f64>,
    alphas_cumprod: Vec<f64>,
    init_noise_sigma: f64,
    timesteps: Vec<usize>,
    pub config: DDPMSchedulerConfig,
}

impl DDPMScheduler {
    pub fn new(inference_steps: usize, config: DDPMSchedulerConfig) -> Self {
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

        // &betas to avoid moving it
        let alphas: Tensor = 1. - &betas;
        let alphas_cumprod = Vec::<f64>::from(alphas.cumprod(0, Kind::Double));

        // min(train_timesteps, inference_steps)
        // https://github.com/huggingface/diffusers/blob/8331da46837be40f96fbd24de6a6fb2da28acd11/src/diffusers/schedulers/scheduling_ddpm.py#L187
        let inference_steps = inference_steps.min(config.train_timesteps);
        // arange the number of the scheduler's timesteps
        let step_ratio = config.train_timesteps / inference_steps;
        let timesteps: Vec<usize> =
            (0..(config.train_timesteps)).step_by(step_ratio).rev().collect();

        Self {
            alphas: Vec::<f64>::from(alphas),
            betas: Vec::<f64>::from(betas),
            alphas_cumprod,
            init_noise_sigma: 1.0,
            timesteps,
            config,
        }
    }

    fn get_variance(&self, timestep: usize) -> f64 {
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if timestep > 0 { self.alphas_cumprod[timestep - 1] } else { 1.0 };

        let variance = (1. - alpha_prod_t_prev) / (1. - alpha_prod_t) * self.betas[timestep];

        match self.config.variance_type {
            DDPMVarianceType::FixedSmall => variance.max(1e-20),
            DDPMVarianceType::FixedSmallLog => variance.max(1e-20).ln(),
            DDPMVarianceType::FixedLarge => self.betas[timestep],
            DDPMVarianceType::FixedLargeLog => self.betas[timestep].ln(),
            DDPMVarianceType::Learned => variance,
        }
    }

    pub fn timesteps(&self) -> &[usize] {
        self.timesteps.as_slice()
    }

    pub fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Tensor {
        // https://github.com/huggingface/diffusers/blob/df2b548e893ccb8a888467c2508756680df22821/src/diffusers/schedulers/scheduling_ddpm.py#L272
        // 1. compute alphas, betas
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if timestep > 0 { self.alphas_cumprod[timestep - 1] } else { 1.0 };
        let beta_prod_t = 1. - alpha_prod_t;
        let beta_prod_t_prev = 1. - alpha_prod_t_prev;

        // 2. compute predicted original sample from predicted noise also called "predicted x_0" of formula (15)
        let mut pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => {
                (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
            }
            PredictionType::Sample => model_output.shallow_clone(),
            PredictionType::VPrediction => {
                alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            }
        };

        // 3. clip predicted x_0
        if self.config.clip_sample {
            pred_original_sample = pred_original_sample.clamp(-1., 1.);
        }

        // 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        // See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        let pred_original_sample_coeff =
            (alpha_prod_t_prev.sqrt() * self.betas[timestep]) / beta_prod_t;
        let current_sample_coeff = self.alphas[timestep].sqrt() * beta_prod_t_prev / beta_prod_t;

        // 5. Compute predicted previous sample µ_t
        // See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        let pred_prev_sample =
            pred_original_sample_coeff * &pred_original_sample + current_sample_coeff * sample;

        // https://github.com/huggingface/diffusers/blob/df2b548e893ccb8a888467c2508756680df22821/src/diffusers/schedulers/scheduling_ddpm.py#L305
        // 6. Add noise
        let mut variance = Tensor::zeros(&pred_prev_sample.size(), kind::FLOAT_CPU);
        if timestep > 0 {
            let variance_noise = Tensor::randn_like(model_output);
            if self.config.variance_type == DDPMVarianceType::FixedSmallLog {
                variance = self.get_variance(timestep) * variance_noise;
            } else {
                variance = self.get_variance(timestep).sqrt() * variance_noise;
            }
        }
        &pred_prev_sample + variance
    }

    pub fn add_noise(&self, original_samples: &Tensor, noise: Tensor, timestep: usize) -> Tensor {
        self.alphas_cumprod[timestep].sqrt() * original_samples
            + (1. - self.alphas_cumprod[timestep]).sqrt() * noise
    }

    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }
}
