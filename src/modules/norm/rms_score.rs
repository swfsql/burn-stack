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

use super::{downcast, rms, upcast};
use burn::prelude::*;
use burn::tensor::FloatDType;

/// The parameter-free RMS denominator `d(x) ∈ [‥, 1]` such that the RMSNorm
/// (matching [`RmsNorm`] math with `γ ≡ 1`) is `x / d(x)`, shape `[‥, 1]`.
///
/// As in [`RmsNorm`], an f16 or bf16 input computes in f32. The result is in
/// the dtype of `x`.
///
/// [`RmsNorm`]: crate::modules::RmsNorm
pub fn rms_denom<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let (x, half) = upcast(x);
    downcast(rms(x), half)
}

/// The RMSNorm-then-dot score `scale · Σ_feat(x · w) / (rms(x)+eps)`, shape
/// `[‥, 1]`.
///
/// `w` broadcasts against `x` on every axis but the feature one (`D-1`), where
/// it must be full width. `scale` is the `1/√width` temperature of the query.
/// An f16 or bf16 input computes in f32, the dot product included.
pub fn normed_score<const D: usize>(x: Tensor<D>, w: Tensor<D>, scale: f64) -> Tensor<D> {
    let (x, half) = upcast(x);
    let w = match half {
        Some(_) => w.cast(FloatDType::F32),
        None => w,
    };
    let dot = (x.clone() * w).sum_dim(D - 1);
    downcast(dot * scale / rms(x), half)
}

/// `1/√width`: the temperature that keeps a `width`-wide dot product `O(1)`.
pub fn score_scale(width: usize) -> f64 {
    (width as f64).powf(-0.5)
}
