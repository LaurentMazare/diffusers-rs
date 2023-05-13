use super::{betas_for_alpha_bar, BetaSchedule, PredictionType};
use std::iter;
use tch::{kind, Kind, Tensor};

/// The algorithm type for the solver.
///
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub enum DPMSolverAlgorithmType {
    /// Implements the algorithms defined in <https://arxiv.org/abs/2211.01095>.
    #[default]
    DPMSolverPlusPlus,
    /// Implements the algorithms defined in <https://arxiv.org/abs/2206.00927>.
    DPMSolver,
}

/// The solver type for the second-order solver.
/// The solver type slightly affects the sample quality, especially for
/// small number of steps.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub enum DPMSolverType {
    #[default]
    Midpoint,
    Heun,
}

#[derive(Debug, Clone)]
pub struct DPMSolverMultistepSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// number of diffusion steps used to train the model.
    pub train_timesteps: usize,
    /// the order of DPM-Solver; can be `1` or `2` or `3`. We recommend to use `solver_order=2` for guided
    /// sampling, and `solver_order=3` for unconditional sampling.
    pub solver_order: usize,
    /// prediction type of the scheduler function
    pub prediction_type: PredictionType,
    /// whether to use the "dynamic thresholding" method (introduced by Imagen, https://arxiv.org/abs/2205.11487).
    /// For pixel-space diffusion models, you can set both `algorithm_type=dpmsolver++` and `thresholding=True` to
    /// use the dynamic thresholding. Note that the thresholding method is unsuitable for latent-space diffusion
    /// models (such as stable-diffusion).
    pub thresholding: bool,
    /// the ratio for the dynamic thresholding method. Default is `0.995`, the same as Imagen
    /// (https://arxiv.org/abs/2205.11487).
    pub dynamic_thresholding_ratio: f64,
    /// The threshold value for dynamic thresholding. Valid only when `thresholding: true` and
    /// `algorithm_type: DPMSolverAlgorithmType::DPMSolverPlusPlus`.
    pub sample_max_value: f64,
    /// The algorithm type for the solver
    pub algorithm_type: DPMSolverAlgorithmType,
    /// The solver type for the second-order solver.
    pub solver_type: DPMSolverType,
    /// Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. We empirically
    /// find this can stabilize the sampling of DPM-Solver for `steps < 15`, especially for steps <= 10.
    pub lower_order_final: bool,
}

impl Default for DPMSolverMultistepSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            train_timesteps: 1000,
            solver_order: 2,
            prediction_type: PredictionType::Epsilon,
            thresholding: false,
            dynamic_thresholding_ratio: 0.995,
            sample_max_value: 1.0,
            algorithm_type: DPMSolverAlgorithmType::DPMSolverPlusPlus,
            solver_type: DPMSolverType::Midpoint,
            lower_order_final: true,
        }
    }
}

pub struct DPMSolverMultistepScheduler {
    alphas_cumprod: Vec<f64>,
    alpha_t: Vec<f64>,
    sigma_t: Vec<f64>,
    lambda_t: Vec<f64>,
    init_noise_sigma: f64,
    lower_order_nums: usize,
    model_outputs: Vec<Tensor>,
    timesteps: Vec<usize>,
    pub config: DPMSolverMultistepSchedulerConfig,
}

impl DPMSolverMultistepScheduler {
    pub fn new(inference_steps: usize, config: DPMSolverMultistepSchedulerConfig) -> Self {
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
        let alphas: Tensor = 1. - betas;
        let alphas_cumprod = alphas.cumprod(0, Kind::Double);

        let alpha_t = alphas_cumprod.sqrt();
        let sigma_t = ((1. - &alphas_cumprod) as Tensor).sqrt();
        let lambda_t = alpha_t.log() - sigma_t.log();

        let step = (config.train_timesteps - 1) as f64 / inference_steps as f64;
        // https://github.com/huggingface/diffusers/blob/e4fe9413121b78c4c1f109b50f0f3cc1c320a1a2/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L199-L204
        let timesteps: Vec<usize> = (0..inference_steps + 1)
            .map(|i| (i as f64 * step).round() as usize)
            // discards the 0.0 element
            .skip(1)
            .rev()
            .collect();

        // creates a vector of `solver_order` empty tensors
        // https://github.com/huggingface/diffusers/blob/e4fe9413121b78c4c1f109b50f0f3cc1c320a1a2/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L206-L208
        let model_outputs = iter::repeat_with(Tensor::new).take(config.solver_order).collect();

        Self {
            alphas_cumprod: alphas_cumprod.try_into().unwrap(),
            alpha_t: alpha_t.try_into().unwrap(),
            sigma_t: sigma_t.try_into().unwrap(),
            lambda_t: lambda_t.try_into().unwrap(),
            init_noise_sigma: 1.,
            lower_order_nums: 0,
            model_outputs,
            timesteps,
            config,
        }
    }

    /// Convert the model output to the corresponding type that the algorithm (DPM-Solver / DPM-Solver++) needs.
    ///
    /// DPM-Solver is designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to
    /// discretize an integral of the data prediction model. So we need to first convert the model output to the
    /// corresponding type to match the algorithm.
    ///
    /// Note that the algorithm type and the model type is decoupled. That is to say, we can use either DPM-Solver or
    /// DPM-Solver++ for both noise prediction model and data prediction model.
    fn convert_model_output(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> Tensor {
        match self.config.algorithm_type {
            DPMSolverAlgorithmType::DPMSolverPlusPlus => {
                let mut x0_pred = match self.config.prediction_type {
                    PredictionType::Epsilon => {
                        let alpha_t = self.alpha_t[timestep];
                        let sigma_t = self.sigma_t[timestep];
                        (sample - sigma_t * model_output) / alpha_t
                    }
                    PredictionType::Sample => model_output.shallow_clone(),
                    PredictionType::VPrediction => {
                        let alpha_t = self.alpha_t[timestep];
                        let sigma_t = self.sigma_t[timestep];
                        alpha_t * sample - sigma_t * model_output
                    }
                };
                if self.config.thresholding {
                    // Dynamic thresholding in https://arxiv.org/abs/2205.11487
                    let dynamic_max_val =
                        x0_pred.abs().reshape([x0_pred.size()[0], -1]).quantile_scalar(
                            self.config.dynamic_thresholding_ratio,
                            1,
                            false,
                            // default in torch.quantile
                            "linear",
                        );
                    // this converts the following indexing pattern: (...,) + (None,) * (x0_pred.ndim-1)
                    // https://github.com/huggingface/diffusers/blob/ed616bd8a8740927770eebe017aedb6204c6105f/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L266
                    let shape = [dynamic_max_val.size(), vec![1; x0_pred.dim() - 1]].concat();
                    let dynamic_max_val = dynamic_max_val
                        .maximum(
                            &(dynamic_max_val.ones_like().to(dynamic_max_val.device())
                                * self.config.sample_max_value),
                        )
                        .view(shape.as_slice());

                    x0_pred = x0_pred.clamp_tensor(
                        Some(-dynamic_max_val.shallow_clone()),
                        Some(dynamic_max_val.shallow_clone()),
                    ) / dynamic_max_val;
                }

                x0_pred
            }
            DPMSolverAlgorithmType::DPMSolver => match self.config.prediction_type {
                PredictionType::Epsilon => model_output.shallow_clone(),
                PredictionType::Sample => {
                    let alpha_t = self.alpha_t[timestep];
                    let sigma_t = self.sigma_t[timestep];
                    (sample - alpha_t * model_output) / sigma_t
                }
                PredictionType::VPrediction => {
                    let alpha_t = self.alpha_t[timestep];
                    let sigma_t = self.sigma_t[timestep];
                    alpha_t * model_output + sigma_t * sample
                }
            },
        }
    }

    ///  One step for the first-order DPM-Solver (equivalent to DDIM).
    ///  See https://arxiv.org/abs/2206.00927 for the detailed derivation.
    fn dpm_solver_first_order_update(
        &self,
        model_output: Tensor,
        timestep: usize,
        prev_timestep: usize,
        sample: &Tensor,
    ) -> Tensor {
        let (lambda_t, lambda_s) = (self.lambda_t[prev_timestep], self.lambda_t[timestep]);
        let (alpha_t, alpha_s) = (self.alpha_t[prev_timestep], self.alpha_t[timestep]);
        let (sigma_t, sigma_s) = (self.sigma_t[prev_timestep], self.sigma_t[timestep]);
        let h = lambda_t - lambda_s;
        match self.config.algorithm_type {
            DPMSolverAlgorithmType::DPMSolverPlusPlus => {
                (sigma_t / sigma_s) * sample - (alpha_t * ((-h).exp() - 1.0)) * model_output
            }
            DPMSolverAlgorithmType::DPMSolver => {
                (alpha_t / alpha_s) * sample - (sigma_t * (h.exp() - 1.0)) * model_output
            }
        }
    }

    ///  One step for the second-order multistep DPM-Solver.
    fn multistep_dpm_solver_second_order_update(
        &self,
        model_output_list: &Vec<Tensor>,
        timestep_list: [usize; 2],
        prev_timestep: usize,
        sample: &Tensor,
    ) -> Tensor {
        let (t, s0, s1) = (
            prev_timestep,
            timestep_list[timestep_list.len() - 1],
            timestep_list[timestep_list.len() - 2],
        );
        let (m0, m1) = (
            model_output_list[model_output_list.len() - 1].as_ref(),
            model_output_list[model_output_list.len() - 2].as_ref(),
        );
        let (lambda_t, lambda_s0, lambda_s1) =
            (self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]);
        let (alpha_t, alpha_s0) = (self.alpha_t[t], self.alpha_t[s0]);
        let (sigma_t, sigma_s0) = (self.sigma_t[t], self.sigma_t[s0]);
        let (h, h_0) = (lambda_t - lambda_s0, lambda_s0 - lambda_s1);
        let r0 = h_0 / h;
        let (d0, d1) = (m0, (1.0 / r0) * (m0 - m1));
        match self.config.algorithm_type {
            DPMSolverAlgorithmType::DPMSolverPlusPlus => match self.config.solver_type {
                // See https://arxiv.org/abs/2211.01095 for detailed derivations
                DPMSolverType::Midpoint => {
                    (sigma_t / sigma_s0) * sample
                        - (alpha_t * ((-h).exp() - 1.0)) * d0
                        - 0.5 * (alpha_t * ((-h).exp() - 1.0)) * d1
                }
                DPMSolverType::Heun => {
                    (sigma_t / sigma_s0) * sample - (alpha_t * ((-h).exp() - 1.0)) * d0
                        + (alpha_t * (((-h).exp() - 1.0) / h + 1.0)) * d1
                }
            },
            DPMSolverAlgorithmType::DPMSolver => match self.config.solver_type {
                // See https://arxiv.org/abs/2206.00927 for detailed derivations
                DPMSolverType::Midpoint => {
                    (alpha_t / alpha_s0) * sample
                        - (sigma_t * (h.exp() - 1.0)) * d0
                        - 0.5 * (sigma_t * (h.exp() - 1.0)) * d1
                }
                DPMSolverType::Heun => {
                    (alpha_t / alpha_s0) * sample
                        - (sigma_t * (h.exp() - 1.0)) * d0
                        - (sigma_t * ((h.exp() - 1.0) / h - 1.0)) * d1
                }
            },
        }
    }

    /// One step for the third-order multistep DPM-Solver
    fn multistep_dpm_solver_third_order_update(
        &self,
        model_output_list: &Vec<Tensor>,
        timestep_list: [usize; 3],
        prev_timestep: usize,
        sample: &Tensor,
    ) -> Tensor {
        let (t, s0, s1, s2) = (
            prev_timestep,
            timestep_list[timestep_list.len() - 1],
            timestep_list[timestep_list.len() - 2],
            timestep_list[timestep_list.len() - 3],
        );
        let (m0, m1, m2) = (
            model_output_list[model_output_list.len() - 1].as_ref(),
            model_output_list[model_output_list.len() - 2].as_ref(),
            model_output_list[model_output_list.len() - 3].as_ref(),
        );
        let (lambda_t, lambda_s0, lambda_s1, lambda_s2) =
            (self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1], self.lambda_t[s2]);
        let (alpha_t, alpha_s0) = (self.alpha_t[t], self.alpha_t[s0]);
        let (sigma_t, sigma_s0) = (self.sigma_t[t], self.sigma_t[s0]);
        let (h, h_0, h_1) = (lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2);
        let (r0, r1) = (h_0 / h, h_1 / h);
        let d0 = m0;
        let (d1_0, d1_1) = ((1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2));
        let d1 = &d1_0 + (r0 / (r0 + r1)) * (&d1_0 - &d1_1);
        let d2 = (1.0 / (r0 + r1)) * (d1_0 - d1_1);

        match self.config.algorithm_type {
            DPMSolverAlgorithmType::DPMSolverPlusPlus => {
                // See https://arxiv.org/abs/2206.00927 for detailed derivations
                (sigma_t / sigma_s0) * sample - (alpha_t * ((-h).exp() - 1.0)) * d0
                    + (alpha_t * (((-h).exp() - 1.0) / h + 1.0)) * d1
                    - (alpha_t * (((-h).exp() - 1.0 + h) / h.powi(2) - 0.5)) * d2
            }
            DPMSolverAlgorithmType::DPMSolver => {
                (alpha_t / alpha_s0) * sample
                    - (sigma_t * (h.exp() - 1.0)) * d0
                    - (sigma_t * ((h.exp() - 1.0) / h - 1.0)) * d1
                    - (sigma_t * ((h.exp() - 1.0 - h) / h.powi(2) - 0.5)) * d2
            }
        }
    }

    pub fn timesteps(&self) -> &[usize] {
        self.timesteps.as_slice()
    }

    ///  Ensures interchangeability with schedulers that need to scale the denoising model input
    /// depending on the current timestep.
    pub fn scale_model_input(&self, sample: Tensor, _timestep: usize) -> Tensor {
        sample
    }

    pub fn step(&mut self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Tensor {
        // https://github.com/huggingface/diffusers/blob/e4fe9413121b78c4c1f109b50f0f3cc1c320a1a2/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L457
        let step_index = self.timesteps.iter().position(|&t| t == timestep).unwrap();

        let prev_timestep =
            if step_index == self.timesteps.len() - 1 { 0 } else { self.timesteps[step_index + 1] };
        let lower_order_final = (step_index == self.timesteps.len() - 1)
            && self.config.lower_order_final
            && self.timesteps.len() < 15;
        let lower_order_second = (step_index == self.timesteps.len() - 2)
            && self.config.lower_order_final
            && self.timesteps.len() < 15;

        let model_output = self.convert_model_output(model_output, timestep, sample);
        for i in 0..self.config.solver_order - 1 {
            self.model_outputs[i] = self.model_outputs[i + 1].shallow_clone();
        }
        // self.model_outputs[-1] = model_output
        let m = self.model_outputs.len();
        self.model_outputs[m - 1] = model_output.shallow_clone();

        let prev_sample = if self.config.solver_order == 1
            || self.lower_order_nums < 1
            || lower_order_final
        {
            self.dpm_solver_first_order_update(model_output, timestep, prev_timestep, sample)
        } else if self.config.solver_order == 2 || self.lower_order_nums < 2 || lower_order_second {
            let timestep_list = [self.timesteps[step_index - 1], timestep];
            self.multistep_dpm_solver_second_order_update(
                &self.model_outputs,
                timestep_list,
                prev_timestep,
                sample,
            )
        } else {
            let timestep_list =
                [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep];
            self.multistep_dpm_solver_third_order_update(
                &self.model_outputs,
                timestep_list,
                prev_timestep,
                sample,
            )
        };

        if self.lower_order_nums < self.config.solver_order {
            self.lower_order_nums += 1;
        }

        prev_sample
    }

    pub fn add_noise(&self, original_samples: &Tensor, noise: Tensor, timestep: usize) -> Tensor {
        self.alphas_cumprod[timestep].sqrt() * original_samples.to_owned()
            + (1.0 - self.alphas_cumprod[timestep]).sqrt() * noise
    }

    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }
}
