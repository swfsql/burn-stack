//! Mean squared error loss.
//!
//! An f16 or bf16 input computes in f32, and the loss goes back to the dtype
//! of the input. In f16, `(logits − targets)²` overflows at a difference of
//! about 256. f32 and f64 compute in their own dtype.

use crate::utils::{downcast, upcast};
use burn::module::Module;
use burn::nn::loss::Reduction;
use burn::tensor::Tensor;

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
        let (logits, half) = upcast(logits);
        let targets = targets.cast(logits.dtype());
        let tensor = self.forward_no_reduction(logits, targets);
        let loss = match reduction {
            Reduction::Mean | Reduction::Auto => tensor.mean(),
            Reduction::BatchMean => tensor.mean() / batch_size as f32,
            Reduction::Sum => tensor.sum(),
        };
        downcast(loss, half)
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
