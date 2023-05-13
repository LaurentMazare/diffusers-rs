use super::{betas_for_alpha_bar, BetaSchedule, PredictionType};
use tch::{kind, Kind, Tensor};

#[derive(Debug, Clone)]
pub struct PNDMSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// each diffusion step uses the value of alphas product at that step and
    /// at the previous one. For the final step there is no previous alpha.
    /// When this option is `True` the previous alpha product is fixed to `1`,
    /// otherwise it uses the value of alpha at step 0.
    pub set_alpha_to_one: bool,
    /// prediction type of the scheduler function
    pub prediction_type: PredictionType,
    /// an offset added to the inference steps.
    pub steps_offset: usize,
    /// number of diffusion steps used to train the model.
    pub train_timesteps: usize,
}

impl Default for PNDMSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            set_alpha_to_one: false,
            prediction_type: PredictionType::Epsilon,
            steps_offset: 1,
            train_timesteps: 1000,
        }
    }
}

/// Pseudo numerical methods for diffusion models (PNDM) proposes using more advanced ODE
/// integration techniques, namely Runge-Kutta method and a linear multi-step method.
pub struct PNDMScheduler {
    alphas_cumprod: Vec<f64>,
    final_alpha_cumprod: f64,
    step_ratio: usize,
    init_noise_sigma: f64,
    counter: usize,
    cur_sample: Option<Tensor>,
    ets: Vec<Tensor>,
    timesteps: Vec<usize>,
    pub config: PNDMSchedulerConfig,
}

impl PNDMScheduler {
    pub fn new(inference_steps: usize, config: PNDMSchedulerConfig) -> Self {
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
        let alphas: Tensor = 1. - betas;
        let alphas_cumprod = Vec::<f64>::try_from(alphas.cumprod(0, Kind::Double)).unwrap();

        let final_alpha_cumprod = if config.set_alpha_to_one { 1.0 } else { alphas_cumprod[0] };
        // creates integer timesteps by multiplying by ratio
        // casting to int to avoid issues when num_inference_step is power of 3
        let step_ratio = config.train_timesteps / inference_steps;
        let timesteps: Vec<usize> =
            (0..(inference_steps)).map(|s| s * step_ratio + config.steps_offset).collect();

        let n_ts = timesteps.len();
        // https://github.com/huggingface/diffusers/blob/8f581591598255eff72cce8858f365eace47481f/src/diffusers/schedulers/scheduling_pndm.py#L173
        let plms_timesteps =
            [&timesteps[..n_ts - 2], &[timesteps[n_ts - 2]], &timesteps[n_ts - 2..]]
                .concat()
                .into_iter()
                .rev()
                .collect();

        Self {
            alphas_cumprod,
            final_alpha_cumprod,
            step_ratio,
            init_noise_sigma: 1.0,
            counter: 0,
            cur_sample: None,
            ets: vec![],
            timesteps: plms_timesteps,
            config,
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
        self.step_plms(model_output, timestep, sample)
    }

    /// Step function propagating the sample with the linear multi-step method.
    /// This has one forward pass with multiple times to approximate the solution.
    fn step_plms(&mut self, model_output: &Tensor, mut timestep: usize, sample: &Tensor) -> Tensor {
        let mut prev_timestep = timestep as isize - self.step_ratio as isize;

        if self.counter != 1 {
            // make sure `ets` has at least size 3 before
            // taking a slice of the last 3
            if self.ets.len() > 3 {
                // self.ets = self.ets[-3:]
                self.ets.drain(0..self.ets.len() - 3);
            }
            self.ets.push(model_output.shallow_clone());
        } else {
            prev_timestep = timestep as isize;
            timestep += self.step_ratio;
        }

        let (ets_last, n_ets) = (self.ets.last().unwrap(), self.ets.len());
        let (mut model_output, mut sample) = (model_output.shallow_clone(), sample.shallow_clone());

        if n_ets == 1 && self.counter == 0 {
            self.cur_sample = Some(sample.shallow_clone());
        } else if n_ets == 1 && self.counter == 1 {
            sample = self.cur_sample.as_ref().unwrap().shallow_clone();
            self.cur_sample = None;
            model_output = (model_output + ets_last) / 2.;
        } else if n_ets == 2 {
            model_output = (3. * ets_last - &self.ets[n_ets - 2]) / 2.;
        } else if n_ets == 3 {
            model_output =
                (23. * ets_last - 16. * &self.ets[n_ets - 2] + 5. * &self.ets[n_ets - 3]) / 12.;
        } else {
            model_output = (1. / 24.)
                * (55. * ets_last - 59. * &self.ets[n_ets - 2] + 37. * &self.ets[n_ets - 3]
                    - 9. * &self.ets[n_ets - 4]);
        }

        let prev_sample = self.get_prev_sample(sample, timestep, prev_timestep, model_output);
        self.counter += 1;

        prev_sample
    }

    fn get_prev_sample(
        &self,
        sample: Tensor,
        timestep: usize,
        prev_timestep: isize,
        model_output: Tensor,
    ) -> Tensor {
        //  See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
        //  this function computes x_(t−δ) using the formula of (9)
        //  Note that x_t needs to be added to both sides of the equation
        //
        //  Notation (<variable name> -> <name in paper>
        //  alpha_prod_t -> α_t
        //  alpha_prod_t_prev -> α_(t−δ)
        //  beta_prod_t -> (1 - α_t)
        //  beta_prod_t_prev -> (1 - α_(t−δ))
        //  sample -> x_t
        //  model_output -> e_θ(x_t, t)
        //  prev_sample -> x_(t−δ)
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_timestep >= 0 {
            self.alphas_cumprod[prev_timestep as usize]
        } else {
            self.final_alpha_cumprod
        };

        let beta_prod_t = 1. - alpha_prod_t;
        let beta_prod_t_prev = 1. - alpha_prod_t_prev;

        let model_output = match self.config.prediction_type {
            PredictionType::VPrediction => {
                alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * &sample
            }
            PredictionType::Epsilon => model_output.shallow_clone(),
            _ => unimplemented!("Prediction type must be one of `epsilon` or `v_prediction"),
        };

        // corresponds to (α_(t−δ) - α_t) divided by
        // denominator of x_t in formula (9) and plus 1
        // Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        // sqrt(α_(t−δ)) / sqrt(α_t))
        let sample_coeff = (alpha_prod_t_prev / alpha_prod_t).sqrt();

        // corresponds to denominator of e_θ(x_t, t) in formula (9)
        let model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev.sqrt()
            + (alpha_prod_t * beta_prod_t * alpha_prod_t_prev).sqrt();

        // full formula (9)
        // prev sample
        sample_coeff * sample
            - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
    }

    pub fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Tensor {
        let timestep = if timestep >= self.alphas_cumprod.len() { timestep - 1 } else { timestep };
        let sqrt_alpha_prod = self.alphas_cumprod[timestep].sqrt();
        let sqrt_one_minus_alpha_prod = (1.0 - self.alphas_cumprod[timestep]).sqrt();
        // noisy samples
        sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
    }

    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }
}
