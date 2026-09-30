//! RMS normalisation fused with a SiLU(z) gate: the output norm of a gated
//! block.
//!
//! `norm_before_gate` selects the order of the two operations:
//! - `true`: normalise, then gate: `y = (x / rms(x) · γ) · SiLU(z)`
//! - `false`: gate, then normalise: `y = rms(x · SiLU(z)) · γ` applied to
//!   `x · SiLU(z)`
//!
//! The epsilon is the `div_eps` of the dtype that the norm computes in, so
//! there is no configurable epsilon. As in
//! [`RmsNorm`](crate::modules::norm::rms_norm::RmsNorm), an f16 or bf16 input
//! computes in f32, the gate included, and an f64 input computes in f64.

use super::rms;
use crate::utils::{downcast, upcast};
use crate::modules::Silu;
use burn::module::{Content, DisplaySettings, ModuleDisplay, Param};
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::FloatDType;

/// Configuration to create a [`RmsNormGated`] layer.
#[derive(Config, Debug)]
pub struct RmsNormGatedConfig {
    /// The size of the input features.
    pub d_model: usize,
    /// Whether to apply normalization before gating. Default: true
    #[config(default = true)]
    pub norm_before_gate: bool,
}

impl RmsNormGatedConfig {
    /// Initialize a new [`RmsNormGated`] module.
    pub fn init(&self, device: &Device) -> RmsNormGated {
        let gamma = Initializer::Ones.init([self.d_model], device);
        RmsNormGated {
            gamma,
            norm_before_gate: self.norm_before_gate,
        }
    }
}

/// Applies Gated Rms Normalization over an input tensor along the last dimension.
///
/// - If `norm_before_gate=true`: `Y = (X / sqrt(mean(X^2) + eps) * gamma) * SiLU(z)`
/// - If `norm_before_gate=false`: `Y = (X * SiLU(z)) / sqrt(mean((X * SiLU(z))^2) + eps) * gamma`
///
/// Where:
/// - `X` is the input tensor
/// - `Y` is the output tensor
/// - `z` is the gating tensor
/// - `gamma` is the learnable weight
/// - `mean` is the mean operation
/// - `eps` is a small value to avoid division by zero.
///
/// Create it with [`RmsNormGatedConfig`].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct RmsNormGated {
    /// The learnable per-channel scale `γ`, shape `[d_model]`.
    pub gamma: Param<Tensor<1>>,
    /// Whether to normalize before applying the gating.
    pub norm_before_gate: bool,
}

impl RmsNormGated {
    /// Applies the forward pass on the input tensor with gating.
    ///
    /// # Shapes
    /// - input `x`: `[..., any, d_model]`
    /// - input `z`: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<D>, z: Tensor<D>) -> Tensor<D> {
        let silu = Silu::new();
        let (x, half) = upcast(x);
        let (z, gamma) = match half {
            Some(_) => (z.cast(FloatDType::F32), self.gamma.val().cast(FloatDType::F32)),
            None => (z, self.gamma.val()),
        };

        let x = if self.norm_before_gate {
            // gate will be applied later
            x
        } else {
            // gate before norm
            x * silu.forward(z.clone())
        };

        let normalized = x.clone() / rms(x) * gamma.unsqueeze();

        let y = if self.norm_before_gate {
            // gate gets applied late (now)
            normalized * silu.forward(z)
        } else {
            // gate already got applied before
            normalized
        };
        downcast(y, half)
    }
}

impl ModuleDisplay for RmsNormGated {
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
