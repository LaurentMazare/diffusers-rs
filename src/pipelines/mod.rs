//! # Pipelines

pub mod stable_diffusion_v1_5;

pub mod stable_diffusion {
    pub use super::stable_diffusion_v1_5 as v1_5;
}
