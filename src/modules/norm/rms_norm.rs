//! Root-mean-square normalisation over the last dimension.
//!
//! `RMSNorm(x) = x / rms(x) · γ` where `rms(x) = √(mean(x²))`. Unlike
//! LayerNorm, there is no mean subtraction and no bias: only a learnable
//! per-channel scale `γ`. It is the Pre-LN of every residual block, and also a
//! **QK-Norm** on the key/query-like projections of a block.
//!
//! An f16 or bf16 input computes in f32, with the `ε` of f32. So an f16, bf16
//! or f32 input computes the same function. An f64 input computes in f64,
//! with the `ε` of f64. See [`rms_norm_gated`] for the SiLU-gated variant.
//!
//! [`rms_norm_gated`]: crate::modules::norm::rms_norm_gated

use super::rms;
use crate::utils::{downcast, upcast};
use burn::module::{Content, DisplaySettings, ModuleDisplay, Param};
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::FloatDType;

/// Configuration to create a [`RmsNorm`] layer.
#[derive(Config, Debug)]
pub struct RmsNormConfig {
    /// The size of the input features.
    pub d_model: usize,
}

impl RmsNormConfig {
    /// Initialize a new [`RmsNorm`] module.
    pub fn init(&self, device: &Device) -> RmsNorm {
        let gamma = Initializer::Ones.init([self.d_model], device);
        RmsNorm { gamma }
    }
}

/// Applies RMS normalisation over an input tensor along the last dimension:
/// `y = x / √(mean(x²)) · γ`.
///
/// Create it with [`RmsNormConfig`].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct RmsNorm {
    /// The learnable per-channel scale `γ`, shape `[d_model]`.
    pub gamma: Param<Tensor<1>>,
}

impl RmsNorm {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    /// - input `x`: `[..., d_model]`
    /// - output: `[..., d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let (x, half) = upcast(x);
        let gamma = match half {
            Some(_) => self.gamma.val().cast(FloatDType::F32),
            None => self.gamma.val(),
        };
        downcast(x.clone() / rms(x) * gamma.unsqueeze(), half)
    }
}

impl ModuleDisplay for RmsNorm {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_model] = self.gamma.shape().dims();
        content.add("d_model", &d_model).optional()
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
