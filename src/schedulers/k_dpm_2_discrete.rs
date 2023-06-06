use super::{interp, BetaSchedule, PredictionType};
use tch::{kind, IndexOp, Kind, Tensor};

#[derive(Debug, Clone)]
pub struct KDPM2DiscreteSchedulerConfig {
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

impl Default for KDPM2DiscreteSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085, // sensible defaults
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            train_timesteps: 1000,
            prediction_type: PredictionType::Epsilon,
        }
    }
}

///  Scheduler created by @crowsonkb in [k_diffusion](https://github.com/crowsonkb/k-diffusion), see:
/// https://github.com/crowsonkb/k-diffusion/blob/5b3af030dd83e0297272d861c19477735d0317ec/k_diffusion/sampling.py#L188
///
/// Scheduler inspired by DPM-Solver-2 and Algorthim 2 from Karras et al. (2022).
pub struct KDPM2DiscreteScheduler {
    timesteps: Vec<f64>,
    sigmas: Vec<f64>,
    sigmas_interpol: Vec<f64>,
    init_noise_sigma: f64,
    sample: Option<Tensor>,
    pub config: KDPM2DiscreteSchedulerConfig,
}

impl KDPM2DiscreteScheduler {
    pub fn new(inference_steps: usize, config: KDPM2DiscreteSchedulerConfig) -> Self {
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
                "KDPM2DiscreteScheduler only implements linear and scaled_linear betas."
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

        let sigmas = ((1. - &alphas_cumprod) as Tensor / alphas_cumprod).sqrt();
        let log_sigmas = sigmas.log();

        let sigmas = interp(
            &timesteps, // x-coordinates at which to evaluate the interpolated values
            Tensor::range(0, sigmas.size1().unwrap() - 1, kind::FLOAT_CPU),
            sigmas,
        );
        // append 0.0
        let sigmas = Tensor::concat(&[sigmas, Tensor::from_slice(&[0.0])], 0);

        // interpolate sigmas
        let sigmas_interpol = sigmas.log().lerp(&sigmas.roll([1], [0]).log(), 0.5).exp();

        // https://github.com/huggingface/diffusers/blob/9b37ed33b5fa09e594b38e4e6f7477beff3bd66a/src/diffusers/schedulers/scheduling_k_dpm_2_discrete.py#L145
        let sigmas = Tensor::cat(
            &[
                // sigmas[:1]
                sigmas.i(..1),
                // sigmas[1:].repeat_interleave(2)
                sigmas.i(1..).repeat_interleave_self_int(2, 0, None),
                //sigmas[-1:]
                sigmas.i(-1..0),
            ],
            0,
        );

        let init_noise_sigma: f64 = sigmas.max().try_into().unwrap();

        // interpolate timesteps
        let timesteps_interpol = Self::sigma_to_t(&sigmas_interpol, log_sigmas);
        let interleaved_timesteps = Tensor::stack(
            &[
                // timesteps_interpol[1:-1, None]
                timesteps_interpol.slice(0, 1, -1, 1).unsqueeze(-1),
                // timesteps[1:, None]
                timesteps.i(1..).unsqueeze(-1),
            ],
            -1,
        )
        .flatten(0, -1);

        // https://github.com/huggingface/diffusers/blob/9b37ed33b5fa09e594b38e4e6f7477beff3bd66a/src/diffusers/schedulers/scheduling_k_dpm_2_discrete.py#L146-L148
        let sigmas_interpol = Tensor::cat(
            &[
                // sigmas_interpol[:1]
                sigmas_interpol.i(..1),
                // sigmas_interpol[1:].repeat_interleave(2)
                sigmas_interpol.i(1..).repeat_interleave_self_int(2, 0, None),
                //sigmas_interpol[-1:]
                sigmas_interpol.i(-1..0),
            ],
            0,
        );
        // https://github.com/huggingface/diffusers/blob/9b37ed33b5fa09e594b38e4e6f7477beff3bd66a/src/diffusers/schedulers/scheduling_k_dpm_2_discrete.py#L158
        let timesteps = Tensor::cat(
            &[
                // timesteps[:1]
                timesteps.i(..1),
                interleaved_timesteps,
            ],
            0,
        );

        Self {
            timesteps: timesteps.try_into().unwrap(),
            sigmas: sigmas.try_into().unwrap(),
            sigmas_interpol: sigmas_interpol.try_into().unwrap(),
            init_noise_sigma,
            sample: None,
            config,
        }
    }

    fn sigma_to_t(sigma: &Tensor, log_sigmas: Tensor) -> Tensor {
        // get log sigma
        let log_sigma = sigma.log();

        // get distribution
        let dists = &log_sigma - log_sigmas.unsqueeze(-1);

        // get sigmas range
        let low_idx = dists
            .ge(0)
            .cumsum(0, Kind::Int64)
            .argmax(0, false)
            .clamp_max(log_sigmas.size1().unwrap() - 2);
        let high_idx = &low_idx + 1;

        let low = log_sigmas.index_select(0, &low_idx);
        let high = log_sigmas.index_select(0, &high_idx);

        // interpolate sigmas
        let w = (&low - log_sigma) / (low - high);
        let w = w.clamp(0., 1.);

        // transform interpolation to time range
        let t: Tensor = (1 - &w) * low_idx + w * high_idx;

        t.view(sigma.size().as_slice())
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

    /// Scales model input by (sigma^2 + 1) ^ .5
    pub fn scale_model_input(&self, sample: Tensor, timestep: f64) -> Tensor {
        let step_index = self.index_for_timestep(timestep);

        let sigma = if self.state_in_first_order() {
            self.sigmas[step_index]
        } else {
            self.sigmas_interpol[step_index]
        };

        sample / (sigma.powi(2) + 1.).sqrt()
    }

    fn state_in_first_order(&self) -> bool {
        self.sample.is_none()
    }

    pub fn step(&mut self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Tensor {
        let step_index = self.index_for_timestep(timestep);

        let (sigma, sigma_interpol, sigma_next) = if self.state_in_first_order() {
            (
                self.sigmas[step_index],
                self.sigmas_interpol[step_index + 1],
                self.sigmas[step_index + 1],
            )
        } else {
            // 2nd order / KDPM2's method
            (
                self.sigmas[step_index - 1],
                self.sigmas_interpol[step_index + 1],
                self.sigmas[step_index],
            )
        };

        // currently only gamma=0 is supported. This usually works best anyways.
        // We can support gamma in the future but then need to scale the timestep before
        //  passing it to the model which requires a change in API
        let gamma = 0.0;
        let sigma_hat = sigma * (gamma + 1.); // sigma_hat == sigma for now

        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        let sigma_input = if self.state_in_first_order() { sigma_hat } else { sigma_interpol };
        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => sample - sigma_input * model_output,
            PredictionType::VPrediction => {
                model_output * (-sigma_input / (sigma_input.powi(2) + 1.).sqrt())
                    + (sample / (sigma_input.powi(2) + 1.))
            }
            _ => unimplemented!("Prediction type must be one of `epsilon` or `v_prediction`"),
        };

        let (derivative, dt, sample) = if self.state_in_first_order() {
            (
                // 2. Convert to an ODE derivative for 1st order
                (sample - pred_original_sample) / sigma_hat,
                // 3. delta timestep
                sigma_interpol - sigma_hat,
                sample.shallow_clone(),
            )
        } else {
            (
                // DPM-Solver-2
                // 2. Convert to an ODE derivative for 2nd order
                (sample - pred_original_sample) / sigma_interpol,
                // 3. delta timestep
                sigma_next - sigma_hat,
                self.sample.as_ref().unwrap().shallow_clone(),
            )
        };

        self.sample = if self.state_in_first_order() {
            // store for 2nd order step
            sample.shallow_clone().into()
        } else {
            None
        };

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
