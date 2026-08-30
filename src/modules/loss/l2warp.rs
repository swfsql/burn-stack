//! The L2 penalty on the largest logit ("L2 warp"), a regulariser the reference
//! linear-attention language models train with (`use_l2warp`).
//!
//! It penalises confidence: at each position only the **winning** logit is
//! pulled toward zero, by `∂/∂z_max = c/(B·T) · z_max`, which is the gradient of
//!
//! ```text
//!   ½ · c · mean over positions of (max over the vocabulary of z)²
//! ```
//!
//! Nothing else in the distribution is touched, so the model is free to keep the
//! *margins* it has learned while its absolute scale stays bounded — which is
//! what keeps the softmax accurate once logits are stored in bf16, the reason
//! the trick exists.
//!
//! The value returned is the **unmodified** loss: the penalty enters through the
//! gradient alone (`p − p.detach()` is zero-valued and unit-derivative), exactly
//! as the reference's hand-written backward does, so a training curve stays
//! comparable to a run without it. The reference's version is a custom
//! `autograd.Function` only to avoid keeping the full logits tensor alive; here
//! the closed form above is written out and the same gradient falls out of
//! autodiff.

use burn::prelude::*;

/// The reference's `l2_penalty_factor` default.
pub const DEFAULT_L2_PENALTY: f64 = 1e-4;

/// Add the max-logit L2 penalty's gradient to a reduced `loss`, leaving its
/// value alone.
///
/// # Shapes
/// - `logits`: `[..., vocab]` — the last dimension is reduced over.
/// - `loss`: `[1]`, the already-reduced loss to wrap.
pub fn l2_warp<const D: usize>(loss: Tensor<1>, logits: Tensor<D>, factor: f64) -> Tensor<1> {
    let penalty = logits.max_dim(D - 1).square().mean().mul_scalar(factor / 2.);
    // Value-zero, gradient-one: the penalty reaches the parameters without
    // moving the number the caller reports.
    loss + (penalty.clone() - penalty.detach())
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
