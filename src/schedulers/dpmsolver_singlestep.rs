use std::iter::repeat;

use super::{
    betas_for_alpha_bar,
    dpmsolver::{DPMSolverAlgorithmType, DPMSolverSchedulerConfig, DPMSolverType},
    BetaSchedule, PredictionType,
};
use tch::{kind, Kind, Tensor};

pub struct DPMSolverSinglestepScheduler {
    alphas_cumprod: Vec<f64>,
    alpha_t: Vec<f64>,
    sigma_t: Vec<f64>,
    lambda_t: Vec<f64>,
    init_noise_sigma: f64,
    order_list: Vec<usize>,
    /// Direct outputs from learned diffusion model at current and latter timesteps
    model_outputs: Vec<Tensor>,
    /// List of current discrete timesteps in the diffusion chain
    timesteps: Vec<usize>,
    /// Current instance of sample being created by diffusion process
    sample: Option<Tensor>,
    pub config: DPMSolverSchedulerConfig,
}

impl DPMSolverSinglestepScheduler {
    pub fn new(inference_steps: usize, config: DPMSolverSchedulerConfig) -> Self {
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
        // https://github.com/huggingface/diffusers/blob/e4fe9413121b78c4c1f109b50f0f3cc1c320a1a2/src/diffusers/schedulers/scheduling_dpmsolver_singlestep.py#L172-L173
        let timesteps: Vec<usize> = (0..inference_steps + 1)
            .map(|i| (i as f64 * step).round() as usize)
            // discards the 0.0 element
            .skip(1)
            .rev()
            .collect();

        let mut model_outputs = Vec::<Tensor>::new();
        for _ in 0..config.solver_order {
            model_outputs.push(Tensor::new());
        }

        let order_list = get_order_list(inference_steps, config.solver_order, true);

        Self {
            alphas_cumprod: Vec::<f64>::from(alphas_cumprod),
            alpha_t: Vec::<f64>::from(alpha_t),
            sigma_t: Vec::<f64>::from(sigma_t),
            lambda_t: Vec::<f64>::from(lambda_t),
            init_noise_sigma: 1.,
            order_list,
            model_outputs,
            timesteps,
            config,
            sample: None,
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
    ///
    /// # Arguments
    ///
    /// * `model_output` - direct output from learned diffusion mode
    /// * `timestep` - current discrete timestep in the diffusion chain
    /// * `sample` - current instance of sample being created by diffusion process
    fn convert_model_output(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> Tensor {
        match self.config.algorithm_type {
            DPMSolverAlgorithmType::DPMSolverPlusPlus => {
                match self.config.prediction_type {
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
                }
                // TODO: implement Dynamic thresholding
                // https://arxiv.org/abs/2205.11487
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

    /// One step for the first-order DPM-Solver (equivalent to DDIM).
    /// See https://arxiv.org/abs/2206.00927 for the detailed derivation.
    ///
    /// # Arguments
    ///
    /// * `model_output` - direct output from learned diffusion model
    /// * `timestep` - current discrete timestep in the diffusion chain
    /// * `prev_timestep` - previous discrete timestep in the diffusion chain
    /// * `sample` - current instance of sample being created by diffusion process
    fn dpm_solver_first_order_update(
        &self,
        model_output: &Tensor,
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

    /// One step for the second-order multistep DPM-Solver.
    /// It computes the solution at time `prev_timestep` from the time `timestep_list[-2]`.
    ///
    /// # Arguments
    ///
    /// * `model_output_list` - direct outputs from learned diffusion model at current and latter timesteps
    /// * `timestep_list` - current and latter discrete timestep in the diffusion chain
    /// * `prev_timestep` - previous discrete timestep in the diffusion chain
    /// * `sample` - current instance of sample being created by diffusion process
    fn singlestep_dpm_solver_second_order_update(
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
    /// It computes the solution at time `prev_timestep` from the time `timestep_list[-3]`.
    ///
    /// # Arguments
    ///
    /// * `model_output_list` - direct outputs from learned diffusion model at current and latter timesteps
    /// * `timestep_list` - current and latter discrete timestep in the diffusion chain
    /// * `prev_timestep` - previous discrete timestep in the diffusion chain
    /// * `sample` - current instance of sample being created by diffusion process
    fn singlestep_dpm_solver_third_order_update(
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
        let d1 = &d1_0 + (r0 / (r0 + r1)) * (&d1_0 - &d1_1.shallow_clone());
        let d2 = (1.0 / (r0 + r1)) * (d1_0 - d1_1.shallow_clone());

        match self.config.algorithm_type {
            DPMSolverAlgorithmType::DPMSolverPlusPlus => match self.config.solver_type {
                DPMSolverType::Midpoint => {
                    (sigma_t / sigma_s0) * sample - (alpha_t * ((-h).exp() - 1.0)) * d0
                        + (alpha_t * (((-h).exp() - 1.0) / h + 1.0)) * d1_1
                }
                DPMSolverType::Heun => {
                    // See https://arxiv.org/abs/2206.00927 for detailed derivations
                    (sigma_t / sigma_s0) * sample - (alpha_t * ((-h).exp() - 1.0)) * d0
                        + (alpha_t * (((-h).exp() - 1.0) / h + 1.0)) * d1
                        - (alpha_t * (((-h).exp() - 1.0 + h) / h.powi(2) - 0.5)) * d2
                }
            },
            DPMSolverAlgorithmType::DPMSolver => match self.config.solver_type {
                DPMSolverType::Midpoint => {
                    (alpha_t / alpha_s0) * sample
                        - (sigma_t * (h.exp() - 1.0)) * d0
                        - (sigma_t * ((h.exp() - 1.0) / h - 1.0)) * d1_1
                }
                DPMSolverType::Heun => {
                    (alpha_t / alpha_s0) * sample
                        - (sigma_t * (h.exp() - 1.0)) * d0
                        - (sigma_t * ((h.exp() - 1.0) / h - 1.0)) * d1
                        - (sigma_t * ((h.exp() - 1.0 - h) / h.powi(2) - 0.5)) * d2
                }
            },
        }
    }

    pub fn timesteps(&self) -> &[usize] {
        self.timesteps.as_slice()
    }

    /// Ensures interchangeability with schedulers that need to scale the denoising model input
    /// depending on the current timestep.
    pub fn scale_model_input(&self, sample: Tensor, _timestep: usize) -> Tensor {
        sample
    }

    /// Step function propagating the sample with the singlestep DPM-Solver
    ///
    /// # Arguments
    ///
    /// * `model_output` - direct output from learned diffusion model
    /// * `timestep` - current discrete timestep in the diffusion chain
    /// * `sample` - current instance of sample being created by diffusion process
    pub fn step(&mut self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Tensor {
        // https://github.com/huggingface/diffusers/blob/e4fe9413121b78c4c1f109b50f0f3cc1c320a1a2/src/diffusers/schedulers/scheduling_dpmsolver_singlestep.py#L535
        let step_index: usize = self.timesteps.iter().position(|&t| t == timestep).unwrap();

        let prev_timestep =
            if step_index == self.timesteps.len() - 1 { 0 } else { self.timesteps[step_index + 1] };

        let model_output = self.convert_model_output(model_output, timestep, &sample);
        for i in 0..self.config.solver_order - 1 {
            self.model_outputs[i] = self.model_outputs[i + 1].shallow_clone();
        }
        let m = self.model_outputs.len();
        self.model_outputs[m - 1] = model_output.shallow_clone();

        let order = self.order_list[step_index];

        // For single-step solvers, we use the initial value at each time with order = 1.
        if order == 1 {
            self.sample = Some(sample.shallow_clone());
        };

        match order {
            1 => self.dpm_solver_first_order_update(
                &self.model_outputs[self.model_outputs.len() - 1],
                timestep,
                prev_timestep,
                &self.sample.as_ref().unwrap(),
            ),
            2 => self.singlestep_dpm_solver_second_order_update(
                &self.model_outputs,
                [self.timesteps[step_index - 1], self.timesteps[step_index]],
                prev_timestep,
                &self.sample.as_ref().unwrap(),
            ),
            3 => self.singlestep_dpm_solver_third_order_update(
                &self.model_outputs,
                [
                    self.timesteps[step_index - 2],
                    self.timesteps[step_index - 1],
                    self.timesteps[step_index],
                ],
                prev_timestep,
                &self.sample.as_ref().unwrap(),
            ),
            _ => {
                panic!("invalid order");
            }
        }
    }

    pub fn add_noise(&self, original_samples: &Tensor, noise: Tensor, timestep: usize) -> Tensor {
        self.alphas_cumprod[timestep].sqrt() * original_samples.to_owned()
            + (1.0 - self.alphas_cumprod[timestep]).sqrt() * noise
    }

    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }
}

/// Computes the solver order at each time step.
///
/// # Arguments
///
/// * `steps` - the number of diffusion steps used when generating samples with a pre-trained model
/// * `solver_order` - the order of DPM-Solver; can be `1` or `2` or `3`. We recommend to use `solver_order=2` for guided
///     sampling, and `solver_order=3` for unconditional sampling.
/// * `lower_order_final` - whether to use lower-order solvers in the final steps. For singlestep schedulers, we recommend to enable
///     this to use up all the function evaluations.
fn get_order_list(steps: usize, solver_order: usize, lower_order_final: bool) -> Vec<usize> {
    if lower_order_final {
        if solver_order == 3 {
            if steps % 3 == 0 {
                repeat(&[1, 2, 3][..])
                    .take((steps / 3) - 1)
                    .chain([&[1, 2][..], &[1][..]])
                    .flatten()
                    .map(|v| *v)
                    .collect()
            } else if steps % 3 == 1 {
                repeat(&[1, 2, 3][..])
                    .take(steps / 3)
                    .chain([&[1][..]])
                    .flatten()
                    .map(|v| *v)
                    .collect()
            } else {
                repeat(&[1, 2, 3][..])
                    .take(steps / 3)
                    .chain([&[1][..], &[2][..]])
                    .flatten()
                    .map(|v| *v)
                    .collect()
            }
        } else if solver_order == 2 {
            if steps % 2 == 0 {
                repeat(&[1, 2][..]).take(steps / 2).flatten().map(|v| *v).collect()
            } else {
                repeat(&[1, 2][..])
                    .take(steps / 2)
                    .chain([&[1][..]])
                    .flatten()
                    .map(|v| *v)
                    .collect()
            }
        } else if solver_order == 1 {
            repeat(&[1][..]).take(steps).flatten().map(|v| *v).collect()
        } else {
            panic!("invalid solver_order");
        }
    } else {
        if solver_order == 3 {
            repeat(&[1, 2, 3][..]).take(steps / 3).flatten().map(|v| *v).collect()
        } else if solver_order == 2 {
            repeat(&[1, 2][..]).take(steps / 2).flatten().map(|v| *v).collect()
        } else if solver_order == 1 {
            repeat(&[1][..]).take(steps).flatten().map(|v| *v).collect()
        } else {
            panic!("invalid solver_order");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::get_order_list;

    #[test]
    fn order_list() {
        assert_eq!(get_order_list(15, 2, false), vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],);

        assert_eq!(
            get_order_list(16, 2, false),
            vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        );

        assert_eq!(
            get_order_list(16, 1, false),
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        );

        assert_eq!(get_order_list(16, 3, false), vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]);
        assert_eq!(
            get_order_list(16, 3, true),
            vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
        );
        assert_eq!(
            get_order_list(16, 1, true),
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        );
        assert_eq!(
            get_order_list(25, 1, true),
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        );

        assert_eq!(get_order_list(1, 1, true), vec![1]);

        assert_eq!(get_order_list(2, 2, true), vec![1, 2]);
    }
}