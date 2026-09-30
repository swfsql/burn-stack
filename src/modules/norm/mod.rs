/// Root-mean-square normalisation (last-dim), also a QK-Norm.
pub mod rms_norm;
/// RMSNorm followed by a SiLU(z) gate (a gated block's output norm).
pub mod rms_norm_gated;
/// The RMSNorm-then-dot score of the Multi-Gate mixer and aggregator.
pub mod rms_score;

use crate::utils::div_eps;
use burn::prelude::*;

/// `√(mean(x²) + ε)` along the last axis, shape `[‥, 1]`. Give it the output
/// of [`upcast`](crate::utils::upcast), so an f16 or bf16 input computes in
/// f32. In f16, `x²` overflows at `|x| ≈ 256`.
///
/// `ε` is the [`div_eps`] of the dtype that the norm computes in: the `ε` of
/// f32 for an f16, bf16 or f32 input, and the `ε` of f64 for an f64 input. So
/// an f16, bf16 or f32 model computes the same function. The [`div_eps`] of a
/// half dtype is not small against the activations. In f16 (`7.1e-4`), it
/// changes the pre-LN norms of a trained model by 0.2–0.5%, and at init the
/// gate-first norm of a gated block by 6–9%.
///
/// `ε` is inside the root. It guards both the forward division and the
/// `1/(2√·)` backward of the `sqrt`, which is otherwise singular for a zero
/// row. A row with `mean(x²) ≪ ε` multiplies its gradient by about `1/√ε`
/// (`≈ 3500` in f32). For an f16 input, the gradient goes back to f16, so an
/// upstream gradient above about 18 at such a row overflows. A row of a real
/// activation is far above `ε`, and its gain is `1/rms(x)`.
pub(crate) fn rms<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let eps = div_eps(x.dtype());
    ((x.clone() * x).mean_dim(D - 1) + eps).sqrt()
}
