//! The RMSNorm-then-dot **score**: a parameter-free RMS normalisation folded
//! into a dot product against a learnable query.
//!
//! It is the scoring primitive of
//! [`MultiGateResidual`](crate::modules::MultiGateResidual), which scores the
//! depth-*streams* that it pools. It is public, so that a downstream container
//! that mixes some other set of parallel items (a pool of streaming caches,
//! for example) can weight them by the same rule, not by a new projection from
//! `d_model`. A query dotted against RMS-normalised *content* is scale-free in
//! the content and needs no temperature. The normalisation fixes the scale at
//! which the query reads, and the query itself learns how sharp the resulting
//! mixture is.
//!
//! The RMS denominator is constant along the feature axis, so it is folded
//! *out* of the reduction. [`normed_score`](crate::modules::normed_score)
//! never materialises the full-width normalised tensor, and computes exactly
//! `Σ_feat(rms_norm(x) · w) · scale`.

use super::rescaled_rms_f16;
use crate::utils::div_eps;
use burn::prelude::*;
use burn::tensor::DType;

/// The parameter-free RMS denominator `d(x) ∈ [‥, 1]` such that the RMSNorm
/// (matching [`RmsNorm`] math with `γ ≡ 1`) is `x / d(x)`, shape `[‥, 1]`.
///
/// The fp16 path keeps the same overflow-safe per-row max-rescale as
/// [`RmsNorm`], folded into the same scalar denominator.
///
/// [`RmsNorm`]: crate::modules::RmsNorm
pub fn rms_denom<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    match x.dtype() {
        DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
            let eps = div_eps(x.dtype());
            // eps *inside* the root (matches `RmsNorm`): the `sqrt` backward
            // is otherwise singular for a zero-norm slice.
            ((x.clone() * x).mean_dim(D - 1) + eps).sqrt()
        }
        DType::F16 => {
            // `m · rms_ = √(mean(x²) + eps)`, with `m` the `max(|x|)` of each
            // row.
            let (_, rms_, m) = rescaled_rms_f16(x, div_eps(DType::F16));
            rms_ * m
        }
        _ => unreachable!("rms_denom expects a float dtype"),
    }
}

/// The RMSNorm-then-dot score `scale · Σ_feat(x · w) / (rms(x)+eps)`, shape
/// `[‥, 1]`.
///
/// `w` broadcasts against `x` on every axis but the feature one (`D-1`), where
/// it must be full width. `scale` is the `1/√width` temperature of the query.
pub fn normed_score<const D: usize>(x: Tensor<D>, w: Tensor<D>, scale: f64) -> Tensor<D> {
    let dot = (x.clone() * w).sum_dim(D - 1);
    dot * scale / rms_denom(x)
}

/// `1/√width`: the temperature that keeps a `width`-wide dot product `O(1)`.
pub fn score_scale(width: usize) -> f64 {
    (width as f64).powf(-0.5)
}
