/// Root-mean-square normalisation (last-dim), also a QK-Norm.
pub mod rms_norm;
/// RMSNorm followed by a SiLU(z) gate (a gated block's output norm).
pub mod rms_norm_gated;
/// The RMSNorm-then-dot score of the Multi-Gate β gate and aggregator.
pub mod rms_score;

use crate::utils::div_eps;
use burn::prelude::*;

/// `√(max(mean(x²), ε))` along the last axis, shape `[‥, 1]` (ADR-0010). Give
/// it the output of [`upcast`](crate::utils::upcast), so an f16 or bf16 input
/// computes in f32. In f16, `x²` overflows at `|x| ≈ 256`. In the compute
/// dtype, `x²` can overflow at the top of its range (ADR-0010).
///
/// `ε` is the [`div_eps`] of the dtype that the norm computes in. That is the
/// `ε` of f32 for an f16, bf16 or f32 input, and the `ε` of f64 for an f64
/// input. So an f16, bf16 or f32 model computes the same function. The
/// [`div_eps`] of a half dtype is not small against the activations. With the
/// [`div_eps`] of f16 (`7.1e-4`), the clamp would hold each row with `rms(x)`
/// below about `0.027`.
///
/// The clamp acts only where `mean(x²) < ε`. Each other row gets `√mean(x²)`,
/// with no `ε` in it. The clamp is before the root, so it guards both the
/// forward division by `rms` and the `1/(2√·)` backward of the `sqrt`.
///
/// Below `ε`, `rms` is `√ε`, and the clamp gives no gradient to `mean(x²)`.
/// So the norm `x / rms(x)` of such a row is `x/√ε`, and its gradient is
/// `h/√ε` (`≈ 3500·h` in f32). For an f16 input, the backward casts the
/// gradient to f16, so an upstream gradient above about 18 at such a row
/// overflows. A row of a real activation is far above `ε`, and its gain is
/// `1/rms(x)`.
pub(crate) fn rms<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let eps = div_eps(x.dtype());
    (x.clone() * x).mean_dim(D - 1).clamp_min(eps).sqrt()
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
