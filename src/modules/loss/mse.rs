//! Mean squared error loss.
//!
//! The fp16 path avoids forming `(logits − targets)²` directly (which overflows
//! for large differences) by factoring out `s = max(|diff|) + eps` before
//! squaring, then multiplying the same `s` back in after the reduction.

use crate::utils::div_eps;
use burn::module::Module;
use burn::nn::loss::Reduction;
use burn::tensor::{DType, Tensor, f16};

/// Calculate the mean squared error loss from the input logits and the targets.
#[derive(Module, Debug)]
pub struct MseLoss;

impl Default for MseLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl MseLoss {
    /// Create the criterion.
    pub fn new() -> Self {
        Self
    }

    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// - logits: `[batch_size, num_targets]`
    /// - targets: `[batch_size, num_targets]`
    pub fn forward(
        &self,
        logits: Tensor<2>,
        targets: Tensor<2>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let [batch_size, _num_targets] = logits.dims();
        match logits.dtype() {
            DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
                let tensor = self.forward_no_reduction(logits, targets);
                match reduction {
                    Reduction::Mean | Reduction::Auto => tensor.mean(),
                    Reduction::BatchMean => tensor.mean() / batch_size as f32,
                    Reduction::Sum => tensor.sum(),
                }
            }
            DType::F16 => {
                use burn::tensor::ElementConversion;
                let div_eps: f16 = f16::from_elem(div_eps(logits.dtype())) * f16::from_f32(2.);
                // avoid calculating sub² directly (due to overflow e.g. on 256 * 256)
                let sub = logits.sub(targets);
                // `s = max(|sub|) + eps`, off autodiff (`eps` guards an all-zero
                // `sub`). `mean(sub · sub/s) · s = mean(sub²)` for each `s > 0`,
                // so the division and the product use the same `s`. The max is
                // global because the loss is one sum over all elements: `s`
                // changes only the rounding. An output per row would need the
                // max of each row.
                let scale = sub.clone().without_autodiff().abs().max() + div_eps;
                let sub_ = sub.clone() / scale.clone().expand(sub.shape()); // sub_.abs() <= 1
                let partial = sub * sub_; // sub² = partial * s
                let reduced_partial = match reduction {
                    Reduction::Mean | Reduction::Auto => partial.mean(),
                    Reduction::BatchMean => partial.mean() / batch_size as f32,
                    Reduction::Sum => partial.sum(),
                };
                reduced_partial * scale
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
        }
    }

    /// Compute the criterion on the input tensor without reducing.
    pub fn forward_no_reduction<const D: usize>(
        &self,
        logits: Tensor<D>,
        targets: Tensor<D>,
    ) -> Tensor<D> {
        logits.sub(targets).square()
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
