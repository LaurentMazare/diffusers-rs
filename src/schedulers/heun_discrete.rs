use super::{interp, BetaSchedule, PredictionType};
use tch::{kind, IndexOp, Kind, Tensor};

#[derive(Debug, Clone)]
pub struct HeunDiscreteSchedulerConfig {
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

impl Default for HeunDiscreteSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085, // sensible defaults
            beta_end: 0.012,
            beta_schedule: BetaSchedule::Linear,
            train_timesteps: 1000,
            prediction_type: PredictionType::Epsilon,
        }
    }
}

pub struct HeunDiscreteScheduler {
    timesteps: Vec<f64>,
    sigmas: Vec<f64>,
    init_noise_sigma: f64,
    prev_derivative: Option<Tensor>,
    sample: Option<Tensor>,
    dt: Option<f64>,
    pub config: HeunDiscreteSchedulerConfig,
}

impl HeunDiscreteScheduler {
    pub fn new(inference_steps: usize, config: HeunDiscreteSchedulerConfig) -> Self {
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
                "HeunDiscreteScheduler only implements linear and scaled_linear betas."
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

        // https://github.com/huggingface/diffusers/blob/aba2a65d6ab47c0d1c12fa47e9b238c1d3e34512/src/diffusers/schedulers/scheduling_heun_discrete.py#L132-L134
        let sigmas = Tensor::cat(
            &[
                // sigmas[:1]
                sigmas.i(..1),
                // sigmas[1:].repeat_interleave(2)
                sigmas.i(1..).repeat_interleave_self_int(2, 0, None),
                // append 0.0
                Tensor::from_slice(&[0.0]),
            ],
            0,
        );

        let init_noise_sigma: f64 = sigmas.max().try_into().unwrap();

        // https://github.com/huggingface/diffusers/blob/aba2a65d6ab47c0d1c12fa47e9b238c1d3e34512/src/diffusers/schedulers/scheduling_heun_discrete.py#L140
        let timesteps = Tensor::cat(
            &[
                // timesteps[:1]
                timesteps.i(..1),
                // timesteps[1:].repeat_interleave(2)
                timesteps.i(1..).repeat_interleave_self_int(2, 0, None),
            ],
            0,
        );

        Self {
            timesteps: timesteps.try_into().unwrap(),
            sigmas: sigmas.try_into().unwrap(),
            prev_derivative: None,
            dt: None,
            sample: None,
            init_noise_sigma,
            config,
        }
    }

    pub fn timesteps(&self) -> &[f64] {
        self.timesteps.as_slice()
    }

    fn index_for_timestep(&self, timestep: f64) -> usize {
        // find all the positions of the timesteps corresponding to timestep
        let indices = self
            .timesteps
            .iter()
            .enumerate()
            .filter_map(|(idx, &t)| (t == timestep).then_some(idx))
            .collect::<Vec<_>>();

        if self.state_in_first_order() {
            *indices.last().unwrap()
        } else {
            indices[0]
        }
    }

    /// Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
    /// current timestep.
    pub fn scale_model_input(&self, sample: Tensor, timestep: f64) -> Tensor {
        let step_index = self.index_for_timestep(timestep);
        let sigma = self.sigmas[step_index];

        // https://github.com/huggingface/diffusers/blob/aba2a65d6ab47c0d1c12fa47e9b238c1d3e34512/src/diffusers/schedulers/scheduling_heun_discrete.py#L106
        sample / (sigma.powi(2) + 1.).sqrt()
    }

    fn state_in_first_order(&self) -> bool {
        self.dt.is_none()
    }

    pub fn step(&mut self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Tensor {
        let step_index = self.index_for_timestep(timestep);

        let (sigma, sigma_next) = if self.state_in_first_order() {
            (self.sigmas[step_index], self.sigmas[step_index + 1])
        } else {
            // 2nd order / Heun's method
            (self.sigmas[step_index - 1], self.sigmas[step_index])
        };

        // currently only gamma=0 is supported. This usually works best anyways.
        // We can support gamma in the future but then need to scale the timestep before
        //  passing it to the model which requires a change in API
        let gamma = 0.0;
        let sigma_hat = sigma * (gamma + 1.); // sigma_hat == sigma for now

        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        let sigma_input = if self.state_in_first_order() { sigma_hat } else { sigma_next };

        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => sample - sigma_input * model_output,
            PredictionType::VPrediction => {
                model_output * (-sigma_input / (sigma_input.powi(2) + 1.).sqrt())
                    + (sample / (sigma_input.powi(2) + 1.))
            }
            _ => unimplemented!("Prediction type must be one of `epsilon` or `v_prediction`"),
        };

        let (derivative, dt, sample) = if self.state_in_first_order() {
            // 2. Convert to an ODE derivative for 1st order
            (
                (sample - pred_original_sample) / sigma_hat,
                sigma_next - sigma_hat,
                sample.shallow_clone(),
            )
        } else {
            // 2. 2nd order / Heun's method
            let derivative = (sample - &pred_original_sample) / sigma_next;
            (
                (self.prev_derivative.as_ref().unwrap() + derivative) / 2.,
                self.dt.unwrap(),
                self.sample.as_ref().unwrap().shallow_clone(),
            )
        };

        if self.state_in_first_order() {
            // store for 2nd order step
            self.prev_derivative = Some(derivative.shallow_clone());
            self.dt = Some(dt);
            self.sample = Some(sample.shallow_clone());
        } else {
            // free dt and derivative
            // Note, this puts the scheduler in "first order mode"
            self.prev_derivative = None;
            self.dt = None;
            self.sample = None;
        }

        sample + derivative * dt
    }

    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    pub fn add_noise(&self, original_samples: &Tensor, noise: Tensor, timestep: f64) -> Tensor {
        let step_index = self.index_for_timestep(timestep);
        let sigma = self.sigmas[step_index];

        // noisy samples
        original_samples + noise * sigma
    }
}
