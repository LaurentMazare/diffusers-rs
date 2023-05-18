//! # Diffusion pipelines and models
//!
//! Noise schedulers can be used to set the trade-off between
//! inference speed and quality.

use tch::{IndexOp, Kind, Tensor};

pub mod ddim;
pub mod ddpm;
pub mod dpmsolver_multistep;
pub mod euler_ancestral_discrete;
pub mod euler_discrete;
pub mod heun_discrete;
mod integrate;
pub mod k_dpm_2_ancestral_discrete;
pub mod k_dpm_2_discrete;
pub mod lms_discrete;
pub mod pndm;

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
    Tensor::from_slice(&betas)
}

/// One-dimensional linear interpolation for monotonically increasing sample
/// points, mimicking np.interp().
///
/// Based on https://github.com/pytorch/pytorch/issues/50334#issuecomment-1000917964
pub fn interp(x: &Tensor, xp: Tensor, yp: Tensor) -> Tensor {
    assert_eq!(xp.size(), yp.size());
    let sz = xp.size1().unwrap();

    // (yp[1:] - yp[:-1]) / (xp[1:] - xp[:-1])
    let m = (yp.i(1..) - yp.i(..sz - 1)) / (xp.i(1..) - xp.i(..sz - 1));

    // yp[:-1] - (m * xp[:-1])
    let b = yp.i(..sz - 1) - (&m * xp.i(..sz - 1));

    // torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    let indices = x.unsqueeze(-1).ge_tensor(&xp.unsqueeze(0));
    let indices = indices.sum_dim_intlist(1, false, Kind::Int64) - 1;
    // torch.clamp(indices, 0, len(m) - 1)
    let indices = indices.clamp(0, m.size1().unwrap() - 1);

    m.take(&indices) * x + b.take(&indices)
}
