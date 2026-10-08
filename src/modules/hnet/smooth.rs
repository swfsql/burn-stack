//! The smoothing module of the dechunking layer: an exponential moving
//! average over the chunks.
//!
//! ```text
//!   z̄ₖ = Pₖ·ẑₖ + (1 − Pₖ)·z̄ₖ₋₁        z̄₋₁ = the carry of the previous call
//! ```
//!
//! `Pₖ` is the router probability of the row that started chunk `k`, clamped
//! to `[10⁻⁴, 1 − 10⁻⁴]`. A confident chunk (`P ≈ 1`) passes through. A weak
//! chunk (`P ≈ ½`) mixes with the chunk before it. This is the path on which
//! the gradient reaches the boundary decision, which is discrete.
//!
//! The reference runs this recurrence through the Mamba-2 SSD kernel (state
//! size 1, `A = −1`, `Δ = −ln(1 − P)`). Here it is the same scan in plain
//! tensor ops, chunked like SSD:
//!
//! 1. Split the `K` chunks into blocks of `l` rows. In a block, the decay from
//!    row `j` to row `i` is `exp(segsum(ln(1 − P))[i, j])`. One matmul gives
//!    every in-block output from a zero start.
//! 2. The state that enters block `c` is the carry and the end state of every
//!    earlier block, each decayed through the blocks between. That is the
//!    same segsum one level up: an `n × n` matmul over the block totals.
//! 3. Add the entering state to each row of the block, decayed through the
//!    rows before it.
//!
//! `P = 0` is a valid row that holds the state (decay 1, no write). An absent
//! row (the inner padding) gets it, with its input set to zero. The output of
//! a padded row is unspecified, and `0·NaN` is not zero. So the last column of
//! `z̄` is the last real chunk of each slot. Every exponent is a sum of
//! `ln(1 − P) ≤ 0` terms, so nothing overflows.
//!
//! The clamp is in f32. In f16, `1 − 10⁻⁴` rounds to `1`, and `ln(1 − 1)` is
//! `−∞`.

use crate::modules::segsum;
use crate::utils::{downcast, upcast};
use burn::prelude::*;

/// The lower clamp of `P` (the reference value).
pub const P_MIN: f64 = 1e-4;
/// The upper clamp of `P` (the reference value).
pub const P_MAX: f64 = 1.0 - 1e-4;

/// The EMA over `x_bkd` (`[batch, K, d]`) from `carry_bd` (`[batch, d]`), in
/// blocks of `block` rows. Out: `z̄`, `[batch, K, d]`.
///
/// `p_bk` (`[batch, K]`) is the router probability of each chunk. It is
/// clamped to `[P_MIN, P_MAX]`. `pad_bk` (`None` ⇒ none) marks the absent
/// chunks, which hold the state. It computes in f32 when the inputs are
/// f16/bf16.
pub fn smooth(
    x_bkd: Tensor<3>,
    p_bk: Tensor<2>,
    pad_bk: Option<Tensor<2, Bool>>,
    carry_bd: Tensor<2>,
    block: usize,
) -> Tensor<3> {
    assert!(block > 0, "the smoothing block must hold at least 1 row");
    let (x_bkd, dtype) = upcast(x_bkd);
    let (p_bk, _) = upcast(p_bk);
    let (carry_bd, _) = upcast(carry_bd);
    let p_bk = p_bk.clamp(P_MIN, P_MAX);
    let [batch, k, d] = x_bkd.dims();
    let (x_bkd, p_bk) = match pad_bk {
        None => (x_bkd, p_bk),
        Some(pad_bk) => (
            x_bkd.mask_fill(pad_bk.clone().unsqueeze_dim::<3>(2).expand([batch, k, d]), 0.0),
            p_bk.mask_fill(pad_bk, 0.0),
        ),
    };
    let device = x_bkd.device();
    let n = k.div_ceil(block);
    let l = if n == 1 { k } else { block };
    let pad = n * l - k;
    // Pad with rows that hold (P = 0, x = 0). They are after every real row,
    // so they change no output. The end narrows them off.
    let (x_bkd, p_bk) = if pad == 0 {
        (x_bkd, p_bk)
    } else {
        let x_pad = Tensor::zeros([batch, pad, d], (&device, x_bkd.dtype()));
        let p_pad = Tensor::zeros([batch, pad], (&device, p_bk.dtype()));
        (Tensor::cat(vec![x_bkd, x_pad], 1), Tensor::cat(vec![p_bk, p_pad], 1))
    };

    // ln(1 − P) and the write P·x, split into blocks.
    let la_bnl = p_bk.clone().neg().log1p().reshape([batch, n, l]);
    let w_bnld = (p_bk.unsqueeze_dim::<3>(2) * x_bkd).reshape([batch, n, l, d]);

    // 1. In-block outputs from a zero start.
    let decay_bnll = segsum::<3, 4>(la_bnl.clone()).exp();
    let intra_bnld = decay_bnll.matmul(w_bnld);

    // The decay from the start of a block through row i of it.
    let cum_bnl = la_bnl.cumsum(2);

    // 2. The state that enters each block.
    let enter_bnd = if n == 1 {
        carry_bd.unsqueeze_dim::<3>(1)
    } else {
        // h = [carry, end₀, …, end_{n−2}] and u = [0, T₀, …, T_{n−2}], with
        // T the total log-decay of a block. Then the state that enters block c
        // is Σ_{j ≤ c} exp(segsum(u)[c, j]) · h[j].
        let total_bn = cum_bnl.clone().narrow(2, l - 1, 1).reshape([batch, n]);
        let end_bnd = intra_bnld.clone().narrow(2, l - 1, 1).reshape([batch, n, d]);
        let h_bnd = Tensor::cat(vec![carry_bd.unsqueeze_dim(1), end_bnd.narrow(1, 0, n - 1)], 1);
        let u_bn = Tensor::cat(
            vec![
                Tensor::zeros([batch, 1], (&device, total_bn.dtype())),
                total_bn.narrow(1, 0, n - 1),
            ],
            1,
        );
        segsum::<2, 3>(u_bn).exp().matmul(h_bnd)
    };

    // 3. Each row adds the entering state, decayed through the rows before it
    //    (its own row included).
    let out_bnld = intra_bnld + cum_bnl.exp().unsqueeze_dim::<4>(3) * enter_bnd.unsqueeze_dim(2);
    let out_bkd = out_bnld.reshape([batch, n * l, d]);
    let out_bkd = if pad == 0 { out_bkd } else { out_bkd.narrow(1, 0, k) };
    downcast(out_bkd, dtype)
}

/// One step of the EMA: `P·x + (1 − P)·carry` on the rows where `take_b` is
/// `true`, and the carry unchanged (bit for bit) on the other rows. `p_b`
/// (`[batch]`) is clamped as in [`smooth`]. It computes in f32 when the
/// inputs are f16/bf16, as [`smooth`] does.
pub fn smooth_step(x_bd: Tensor<2>, p_b: Tensor<1>, take_b: Tensor<1, Bool>, carry_bd: Tensor<2>) -> Tensor<2> {
    let [batch, d] = carry_bd.dims();
    let take_bd = take_b.reshape([batch, 1]).expand([batch, d]);
    let (x_bd, dtype) = upcast(x_bd);
    let (p_b, _) = upcast(p_b);
    let (carry_f_bd, _) = upcast(carry_bd.clone());
    let p_b1 = p_b.clamp(P_MIN, P_MAX).reshape([batch, 1]);
    let mixed_bd = downcast(p_b1.clone() * x_bd + (p_b1.neg() + 1.0) * carry_f_bd, dtype);
    carry_bd.mask_where(take_bd, mixed_bd)
}
