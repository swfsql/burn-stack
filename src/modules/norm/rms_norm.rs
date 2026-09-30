//! Root-mean-square normalisation over the last dimension.
//!
//! `RMSNorm(x) = x / rms(x) · γ` where `rms(x) = √(mean(x²))`. Unlike
//! LayerNorm, there is no mean subtraction and no bias: only a learnable
//! per-channel scale `γ`. It is the Pre-LN of every residual block, and also a
//! **QK-Norm** on the key/query-like projections of a block.
//!
//! The fp16 path does not form `x²` directly (it overflows for moderately
//! large activations, e.g. 256·256). It first divides each row by its own
//! `max(|x|)`, so the squared values stay `≤ 1` (`rescaled_rms_f16`). A row
//! does not read the other rows. See [`rms_norm_gated`] for the SiLU-gated
//! variant.
//!
//! [`rms_norm_gated`]: crate::modules::norm::rms_norm_gated

use super::rescaled_rms_f16;
use crate::utils::div_eps;
use burn::module::{Content, DisplaySettings, ModuleDisplay, Param};
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::DType;

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
        let normalized = match x.dtype() {
            DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
                let div_eps = div_eps(x.dtype());
                // eps *inside* the root. It guards both the forward division
                // and the `1/(2√·)` backward of the `sqrt` node, which is
                // otherwise singular for a zero-norm slice (see
                // `tests::rms_norm_gradient_finite_on_collapsed_slice`).
                let rms = ((x.clone() * x.clone()).mean_dim(D - 1) + div_eps).sqrt();
                let normalized = (x / rms) * self.gamma.val().unsqueeze();
                normalized
            }
            DType::F16 => {
                // The same formula as the main branch, on each row rescaled
                // by its own `max(|x|)` (a direct `x²` overflows, e.g. at
                // 256 · 256).
                let (x_, rms_, _) = rescaled_rms_f16(x, div_eps(DType::F16));
                x_ / rms_ * self.gamma.val().unsqueeze()
            }
            DType::I64
            | DType::I32
            | DType::I16
            | DType::I8
            | DType::U64
            | DType::U32
            | DType::U16
            | DType::U8 => {
                unreachable!()
            }
            DType::Bool(_) => {
                unreachable!()
            }
            DType::QFloat(_) => {
                unimplemented!()
            }
        };
        normalized
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
