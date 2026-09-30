/// Root-mean-square normalisation (last-dim, fp16-safe), also a QK-Norm.
pub mod rms_norm;
/// RMSNorm followed by a SiLU(z) gate (a gated block's output norm).
pub mod rms_norm_gated;
/// The RMSNorm-then-dot score of the Multi-Gate mixer and aggregator.
pub mod rms_score;

use burn::prelude::*;

/// The fp16 form of `x / √(mean(x²) + eps)` along the last axis, as the
/// parts `(x_, rms_, m)` with `x / √(mean(x²) + eps) = x_ / rms_`.
///
/// - `m`: the `max(|x|)` of each row, off autodiff, shape `[‥, 1]`.
/// - `x_ = x / m`, so `|x_| ≤ 1` and `x_²` cannot overflow.
/// - `rms_ = √(mean(x_²) + eps/m²)`, and `mean(x_²) ≥ 1/width`.
///
/// The identity holds for each positive constant `m`. Thus `m` changes only
/// the rounding. A floor on `m` keeps `eps/m² ≤ 128²`. For a row below the
/// floor, `mean(x²)` is much smaller than `eps`, so the value stays correct.
/// Each row reads only itself: the other tokens and batch rows of `x` do not
/// change its result.
pub(crate) fn rescaled_rms_f16<const D: usize>(
    x: Tensor<D>,
    eps: f32,
) -> (Tensor<D>, Tensor<D>, Tensor<D>) {
    let sqrt_eps = eps.sqrt();
    let m = x
        .clone()
        .without_autodiff()
        .abs()
        .max_dim(D - 1)
        .clamp_min(sqrt_eps / 128.0);
    let k = m.clone().recip() * sqrt_eps; // √eps / m ≤ 128
    let x_ = x / m.clone();
    let rms_ = ((x_.clone() * x_.clone()).mean_dim(D - 1) + k.clone() * k).sqrt();
    (x_, rms_, m)
}
