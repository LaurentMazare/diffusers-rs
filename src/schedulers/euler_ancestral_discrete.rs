use super::{interp, BetaSchedule, PredictionType};
use tch::{kind, Kind, Tensor};

#[derive(Debug, Clone)]
pub struct EulerAncestralDiscreteSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// number of diffusion steps used to train the model.
    pub train_timesteps: usize,
    /// prediction type of the scheduler function
    pub prediction_type: PredictionType,
}

impl Default for EulerAncestralDiscreteSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            train_timesteps: 1000,
            prediction_type: PredictionType::Epsilon,
        }
    }
}

/// Ancestral sampling with Euler method steps.
/// Based on the original k-diffusion implementation by Katherine Crowson:
///
/// https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L72
#[derive(Clone)]
pub struct EulerAncestralDiscreteScheduler {
    timesteps: Vec<f64>,
    sigmas: Vec<f64>,
    init_noise_sigma: f64,
    pub config: EulerAncestralDiscreteSchedulerConfig,
}

impl EulerAncestralDiscreteScheduler {
    pub fn new(inference_steps: usize, config: EulerAncestralDiscreteSchedulerConfig) -> Self {
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
            _ => unimplemented!(
                "EulerAncestralDiscreteScheduler only implements linear and scaled_linear betas."
            ),
        };

        let alphas: Tensor = 1. - betas;
        let alphas_cumprod = alphas.cumprod(0, Kind::Double);

        let timesteps = Tensor::linspace(
            (config.train_timesteps - 1) as f64,
            0.,
            inference_steps as i64,
            kind::FLOAT_CPU,
        );

        let sigmas = ((1. - &alphas_cumprod) as Tensor / &alphas_cumprod).sqrt();
        let sigmas = interp(
            &timesteps, // x-coordinates at which to evaluate the interpolated values
            Tensor::range(0, sigmas.size1().unwrap() - 1, kind::FLOAT_CPU),
            sigmas,
        );

        let sigmas = Tensor::concat(&[sigmas, Tensor::of_slice(&[0.0])], 0);

        // standard deviation of the initial noise distribution
        let init_noise_sigma: f64 = sigmas.max().try_into().unwrap();
        Self {
            timesteps: timesteps.try_into().unwrap(),
            sigmas: sigmas.try_into().unwrap(),
            init_noise_sigma,
            config,
        }
    }

    pub fn timesteps(&self) -> &[f64] {
        self.timesteps.as_slice()
    }

    pub fn scale_model_input(&self, sample: Tensor, timestep: f64) -> Tensor {
        let step_index = self.timesteps.iter().position(|&t| t == timestep).unwrap();
        let sigma = self.sigmas[step_index];

        // https://github.com/huggingface/diffusers/blob/aba2a65d6ab47c0d1c12fa47e9b238c1d3e34512/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py#L132
        sample / (sigma.powi(2) + 1.).sqrt()
    }

    pub fn step(&self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Tensor {
        let step_index = self.timesteps.iter().position(|&t| t == timestep).unwrap();
        let sigma = self.sigmas[step_index];

        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => sample - sigma * model_output,
            PredictionType::VPrediction => {
                model_output * (-sigma / (sigma.powi(2) + 1.).sqrt())
                    + (sample / (sigma.powi(2) + 1.))
            }
            _ => unimplemented!("Prediction type must be one of `epsilon` or `v_prediction`"),
        };

        let sigma_from = self.sigmas[step_index];
        let sigma_to = self.sigmas[step_index + 1];
        let sigma_up = (sigma_to.powi(2) * (sigma_from.powi(2) - sigma_to.powi(2))
            / sigma_from.powi(2))
        .sqrt();
        let sigma_down = (sigma_to.powi(2) - sigma_up.powi(2)).sqrt();

        // 2. Convert to an ODE derivative
        let derivative = (sample - pred_original_sample) / sigma;
        let dt = sigma_down - sigma;

        let prev_sample = sample + derivative * dt;
        let noise = Tensor::randn_like(model_output);

        prev_sample + noise * sigma_up
    }

    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    pub fn add_noise(&self, original_samples: &Tensor, noise: Tensor, timestep: f64) -> Tensor {
        let step_index = self.timesteps.iter().position(|&t| t == timestep).unwrap();
        let sigma = self.sigmas[step_index];

        // noisy samples
        original_samples + noise * sigma
    }
}
