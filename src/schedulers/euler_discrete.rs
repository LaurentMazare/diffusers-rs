use super::{interp, BetaSchedule, PredictionType};
use tch::{kind, Kind, Tensor};

#[derive(Debug, Clone)]
pub struct EulerDiscreteSchedulerConfig {
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

impl Default for EulerDiscreteSchedulerConfig {
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

/// Euler scheduler (Algorithm 2) from Karras et al. (2022) https://arxiv.org/abs/2206.00364.
/// Based on the original
/// k-diffusion implementation by Katherine Crowson:
/// https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L51
#[derive(Clone)]
pub struct EulerDiscreteScheduler {
    timesteps: Vec<f64>,
    sigmas: Vec<f64>,
    init_noise_sigma: f64,
    pub config: EulerDiscreteSchedulerConfig,
}

impl EulerDiscreteScheduler {
    pub fn new(inference_steps: usize, config: EulerDiscreteSchedulerConfig) -> Self {
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
                "EulerDiscreteScheduler only implements linear and scaled_linear betas."
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

        // https://github.com/huggingface/diffusers/blob/2bd53a940c60d13421d9e8887af96b30a53c1b95/src/diffusers/schedulers/scheduling_euler_discrete.py#L133
        sample / (sigma.powi(2) + 1.).sqrt()
    }

    pub fn step(&self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Tensor {
        let (s_churn, s_tmin, s_tmax, s_noise) = (0.0, 0.0, f64::INFINITY, 1.0);

        let step_index = self.timesteps.iter().position(|&t| t == timestep).unwrap();
        let sigma = self.sigmas[step_index];

        let gamma = if s_tmin <= sigma && sigma <= s_tmax {
            (s_churn / (self.sigmas.len() as f64 - 1.)).min(2.0_f64.sqrt() - 1.)
        } else {
            0.0
        };

        let noise = Tensor::randn_like(model_output);
        let eps = noise * s_noise;
        let sigma_hat = sigma * (gamma + 1.);

        let sample = if gamma > 0.0 {
            sample + eps * (sigma_hat.powi(2) - sigma.powi(2)).sqrt()
        } else {
            sample.shallow_clone()
        };

        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => &sample - sigma_hat * model_output,
            PredictionType::VPrediction => {
                model_output * (-sigma / (sigma.powi(2) + 1.).sqrt())
                    + (&sample / (sigma.powi(2) + 1.))
            }
            _ => unimplemented!("Prediction type must be one of `epsilon` or `v_prediction`"),
        };

        // 2. Convert to an ODE derivative
        let derivative = (&sample - pred_original_sample) / sigma_hat;
        let dt = self.sigmas[step_index + 1] - sigma_hat;

        sample + derivative * dt
    }

    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    pub fn add_noise(&self, original_samples: &Tensor, noise: Tensor, timestep: f64) -> Tensor {
        let step_index = self.timesteps.iter().position(|&t| t == timestep).unwrap();
        let sigma = self.sigmas[step_index];

        original_samples + noise * sigma
    }
}
