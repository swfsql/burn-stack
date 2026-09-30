/// Root-mean-square normalisation (last-dim), also a QK-Norm.
pub mod rms_norm;
/// RMSNorm followed by a SiLU(z) gate (a gated block's output norm).
pub mod rms_norm_gated;
/// The RMSNorm-then-dot score of the Multi-Gate mixer and aggregator.
pub mod rms_score;

use crate::utils::div_eps;
use burn::prelude::*;
use burn::tensor::{DType, FloatDType};

/// The `ε` of `√(mean(x²) + ε)` in every RMS norm: the [`div_eps`] of f32, in
/// every dtype.
///
/// So a model computes the same function in each dtype. The [`div_eps`] of a
/// half dtype is not small against the activations. In f16 (`7.1e-4`), it
/// changes the pre-LN norms of a trained model by 0.2–0.5%, and at init the
/// gate-first norm of a Mamba-2 block by 6–9%.
///
/// A row with `mean(x²) ≪ ε` multiplies its gradient by about `1/√ε ≈ 3500`,
/// in every dtype. For an f16 input, the gradient goes back to f16, so an
/// upstream gradient above about 18 at such a row overflows. A row of a real
/// activation is far above `ε`, and its gain is `1/rms(x)`.
pub(crate) fn norm_eps() -> f32 {
    div_eps(DType::F32)
}

/// `x` in the dtype that the norms compute in, and the dtype of `x` if the
/// result must go back to it.
///
/// A half-precision input (f16, bf16) goes to f32. In f16, `x²` overflows at
/// `|x| ≈ 256`, and [`norm_eps`] is below the f16 range of a safe backward.
/// The official kernels also compute their norms in f32
/// (`mamba_ssm/ops/triton/layer_norm.py`). f32 and f64 stay as they are.
pub(crate) fn upcast<const D: usize>(x: Tensor<D>) -> (Tensor<D>, Option<DType>) {
    match x.dtype() {
        dtype @ (DType::F16 | DType::BF16) => (x.cast(FloatDType::F32), Some(dtype)),
        _ => (x, None),
    }
}

/// `x` cast back to the dtype that [`upcast`] returned, if any.
pub(crate) fn downcast<const D: usize>(x: Tensor<D>, dtype: Option<DType>) -> Tensor<D> {
    match dtype {
        Some(dtype) => x.cast(dtype),
        None => x,
    }
}

/// `√(mean(x²) + ε)` along the last axis, shape `[‥, 1]`, with `ε` =
/// [`norm_eps`]. Give it the output of [`upcast`].
///
/// `ε` is inside the root. It guards both the forward division and the
/// `1/(2√·)` backward of the `sqrt`, which is otherwise singular for a zero
/// row.
pub(crate) fn rms<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    ((x.clone() * x).mean_dim(D - 1) + norm_eps()).sqrt()
}
