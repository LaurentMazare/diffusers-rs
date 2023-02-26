//! # Diffusion pipelines and models
//!
//! This is a Rust port of Hugging Face's [diffusers](https://github.com/huggingface/diffusers) Python api using Torch via the [tch-rs](https://github.com/LaurentMazare/tch-rs).
//!
//! This library includes:
//! - Multiple type of UNet based models, with a ResNet backend.
//! - Training examples including version 1.5 of Stable Diffusion.
//! - Some basic transformers implementation for handling user prompts.
//!
//! The models can used pre-trained weights adapted from the Python
//! implementation.

pub mod models;
pub mod pipelines;
pub mod schedulers;
pub mod transformers;
pub mod utils;
