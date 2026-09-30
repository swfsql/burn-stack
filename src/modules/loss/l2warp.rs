//! The L2 penalty on the largest logit ("L2 warp"): a regulariser that the
//! reference linear-attention language models train with (`use_l2warp`).
//!
//! It penalises confidence. At each position, only the **winning** logit is
//! pulled toward zero, by `∂/∂z_max = c/(B·T) · z_max`, which is the gradient
//! of
//!
//! ```text
//!   ½ · c · mean over positions of (max over the vocabulary of z)²
//! ```
//!
//! Nothing else in the distribution is touched. So the model can keep the
//! *margins* that it has learned, while its absolute scale stays bounded. This
//! keeps the softmax accurate when logits are stored in bf16, which is the
//! reason for the trick.
//!
//! The returned value is the **unmodified** loss. The penalty enters through
//! the gradient alone (`p − p.detach()` has value zero and derivative one),
//! exactly as the hand-written backward of the reference does. So a training
//! curve stays comparable to a run without it. The version of the reference is
//! a custom `autograd.Function` only to avoid keeping the full logits tensor
//! alive. Here, the closed form above is written out, and autodiff gives the
//! same gradient.
//!
//! For f16 or bf16 logits, the penalty computes in f32, as the norms do. f32
//! and f64 compute in their own dtype. In f16, `z_max²` overflows above 256,
//! and the backward chain (`c/2`, then `/(B·T)`) underflows to 0 for any real
//! batch. The gradient that reaches f16 logits, `c/(B·T) · z_max`, is still
//! subnormal (about `10⁻⁷` at `B·T = 8192`). So it keeps only 1–2 bits, and it
//! is 0 for a small `z_max`. In f16, the penalty needs loss scaling to have an
//! effect. bf16 has the range of f32, so it has neither problem.

use crate::utils::upcast;
use burn::prelude::*;

/// The `l2_penalty_factor` default of the reference.
pub const DEFAULT_L2_PENALTY: f64 = 1e-4;

/// Add the gradient of the max-logit L2 penalty to a reduced `loss`, and leave
/// its value unchanged.
///
/// The result has the dtype of `loss`.
///
/// # Shapes
/// - `logits`: `[..., vocab]` — the last dimension is reduced over.
/// - `loss`: `[1]`, the already-reduced loss to wrap.
pub fn l2_warp<const D: usize>(loss: Tensor<1>, logits: Tensor<D>, factor: f64) -> Tensor<1> {
    let dtype = loss.dtype();
    // The max is exact in each dtype, so only `z_max` goes to f32.
    let (z_max, _) = upcast(logits.max_dim(D - 1));
    let penalty = z_max.square().mean().mul_scalar(factor / 2.);
    // Value zero, gradient one: the penalty reaches the parameters without a
    // change to the number that the caller reports.
    loss + (penalty.clone() - penalty.detach()).cast(dtype)
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
