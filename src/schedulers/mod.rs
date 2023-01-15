//! # Diffusion pipelines and models
//!
//! Noise schedulers can be used to set the trade-off between
//! inference speed and quality.

use tch::{Kind, Tensor};

pub mod ddim;
pub mod ddpm;
pub mod dpmsolver_multistep;
pub mod euler_ancestral_discrete;
pub mod euler_discrete;
pub mod lms_discrete;
mod integrate;

/// This represents how beta ranges from its minimum value to the maximum
/// during training.
#[derive(Debug, Clone, Copy)]
pub enum BetaSchedule {
    /// Linear interpolation.
    Linear,
    /// Linear interpolation of the square root of beta.
    ScaledLinear,
    /// Glide cosine schedule
    SquaredcosCapV2,
}

#[derive(Debug, Clone, Copy)]
pub enum PredictionType {
    Epsilon,
    VPrediction,
    Sample,
}

/// Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
/// `(1-beta)` over time from `t = [0,1]`.
///
/// Contains a function `alpha_bar` that takes an argument `t` and transforms it to the cumulative product of `(1-beta)`
/// up to that part of the diffusion process.
pub(crate) fn betas_for_alpha_bar(num_diffusion_timesteps: usize, max_beta: f64) -> Tensor {
    let alpha_bar = |time_step: usize| {
        f64::cos((time_step as f64 + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2).powi(2)
    };
    let mut betas = Vec::with_capacity(num_diffusion_timesteps);
    for i in 0..num_diffusion_timesteps {
        let t1 = i / num_diffusion_timesteps;
        let t2 = (i + 1) / num_diffusion_timesteps;
        betas.push((1.0 - alpha_bar(t2) / alpha_bar(t1)).min(max_beta));
    }

    Tensor::of_slice(&betas)
}

/// One-dimensional linear interpolation for monotonically increasing sample
/// points, mimicking np.interp().
///
/// Based on https://github.com/pytorch/pytorch/issues/50334#issuecomment-1000917964
pub fn interp(x: &[f64], xp: Tensor, yp: Tensor) -> Tensor {
    // (yp[1:] - yp[:-1]) / (xp[1:] - xp[:-1])
    let m = (yp.slice(0, 1, None, 1) - yp.slice(0, 0, -1, 1))
        / (xp.slice(0, 1, None, 1) - xp.slice(0, 0, -1, 1));

    // yp[:-1] - (m * xp[:-1])
    let b = yp.slice(0, 0, -1, 1) - (&m * xp.slice(0, 0, -1, 1));

    let mut tensors = vec![];
    for &t in x.iter() {
        tensors.push(xp.le(t).sum(Kind::Int64) - 1);
    }
    let indices = Tensor::stack(&tensors, 0).clamp(0, m.size1().unwrap() - 1);

    m.take(&indices) * Tensor::of_slice(x) + b.take(&indices)
}
