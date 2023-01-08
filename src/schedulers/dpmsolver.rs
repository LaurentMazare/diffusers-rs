use tch::Tensor;

use crate::schedulers::BetaSchedule;
use crate::schedulers::PredictionType;

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
pub struct DPMSolverSchedulerConfig {
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
    /// The threshold value for dynamic thresholding. Valid only when `thresholding: true` and
    /// `algorithm_type: DPMSolverAlgorithmType::DPMSolverPlusPlus`.
    pub sample_max_value: f32,
    /// The algorithm type for the solver
    pub algorithm_type: DPMSolverAlgorithmType,
    /// The solver type for the second-order solver.
    pub solver_type: DPMSolverType,
    /// Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. We empirically
    /// find this can stabilize the sampling of DPM-Solver for `steps < 15`, especially for steps <= 10.
    pub lower_order_final: bool,
}

impl Default for DPMSolverSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.0001,
            beta_end: 0.02,
            beta_schedule: BetaSchedule::Linear,
            train_timesteps: 1000,
            solver_order: 2,
            prediction_type: PredictionType::Epsilon,
            sample_max_value: 1.0,
            algorithm_type: DPMSolverAlgorithmType::DPMSolverPlusPlus,
            solver_type: DPMSolverType::Midpoint,
            lower_order_final: true,
        }
    }
}

pub trait DPMSolverScheduler {
    fn new(inference_steps: usize, config: DPMSolverSchedulerConfig) -> Self;
    fn convert_model_output(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> Tensor;

    fn first_order_update(
        &self,
        model_output: Tensor,
        timestep: usize,
        prev_timestep: usize,
        sample: &Tensor,
    ) -> Tensor;

    fn second_order_update(
        &self,
        model_output_list: &Vec<Tensor>,
        timestep_list: [usize; 2],
        prev_timestep: usize,
        sample: &Tensor,
    ) -> Tensor;

    fn third_order_update(
        &self,
        model_output_list: &Vec<Tensor>,
        timestep_list: [usize; 3],
        prev_timestep: usize,
        sample: &Tensor,
    ) -> Tensor;

    fn step(&mut self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Tensor;

    fn timesteps(&self) -> &[usize];
    fn scale_model_input(&self, sample: Tensor, timestep: usize) -> Tensor;


    fn add_noise(&self, original_samples: &Tensor, noise: Tensor, timestep: usize) -> Tensor;
    fn init_noise_sigma(&self) -> f64;
}
