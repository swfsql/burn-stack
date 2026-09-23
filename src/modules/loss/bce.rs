//! Binary cross-entropy loss.
//!
//! When `logits = true`, the loss is computed in a numerically stable way from
//! raw logits, with [`log_sigmoid`](crate::modules::log_sigmoid). Otherwise
//! the inputs are probabilities, and a dtype-aware epsilon (added *inside* the
//! log) floors the logs to avoid `−∞`.

use crate::modules::log_sigmoid;
use crate::utils::div_eps;
use burn::module::Module;
use burn::prelude::*;

/// Configuration to create a [`BinaryCrossEntropyLoss`] using the [`BinaryCrossEntropyLossConfig::init`].
#[derive(Config, Debug)]
pub struct BinaryCrossEntropyLossConfig {
    /// Treat the inputs as logits, applying a sigmoid activation when computing the loss.
    #[config(default = false)]
    pub logits: bool,
}

impl BinaryCrossEntropyLossConfig {
    /// Initialize [`BinaryCrossEntropyLoss`].
    pub fn init(&self) -> BinaryCrossEntropyLoss {
        BinaryCrossEntropyLoss {
            logits: self.logits,
        }
    }
}

/// Calculate the binary cross entropy loss from the input logits and the targets.
///
/// Create it with [`BinaryCrossEntropyLossConfig`].
#[derive(Module, Debug)]
pub struct BinaryCrossEntropyLoss {
    /// Treat the inputs as logits
    pub logits: bool,
}

impl BinaryCrossEntropyLoss {
    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// Binary:
    /// - logits: `[batch_size]`
    /// - targets: `[batch_size]`
    ///
    /// Multi-label:
    /// - logits: `[batch_size, num_classes]`
    /// - targets: `[batch_size, num_classes]`
    pub fn forward<const D: usize>(&self, logits: Tensor<D>, targets: Tensor<D>) -> Tensor<1> {
        let loss = if self.logits {
            // Numerically stable by combining `log(sigmoid(x))` with `log_sigmoid(x)`
            (targets.neg() + 1.) * logits.clone() - log_sigmoid(logits)
        } else {
            // - (target * log(input) + (1 - target) * log(1 - input))
            // eps *inside* each log (dtype-aware `div_eps`, so f16-safe) floors
            // both the value and the `1/x` backward at a zero-probability class.
            // An outer clamp on the log output does not: it leaves `1/x` to blow
            // up (a −100 floor gives a `1/x` of `≈e¹⁰⁰`, which overflows f32).
            let eps = div_eps(logits.dtype());
            (targets.clone() - 1) * (logits.clone().neg() + eps).log1p()
                - targets * (logits + eps).log()
        };

        loss.mean()
    }
}
