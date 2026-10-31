//! The index math of the chunking (the downsampler) and of the upsampler of
//! the dechunking.
//!
//! ```text
//!   upto[t]   = Σ_{j ≤ t} b[j]             chunk boundaries at or before t
//!   src[k]    = the position t with b[t] = 1 and upto[t] = k + 1
//!   x'[k]     = x̂[src[k]]                  chunk:   keep the chunk boundaries
//!   z̃[t]      = Z[upto[t]]                 dechunk: repeat the latest chunk
//! ```
//!
//! `Z = [carry, z̄₀, z̄₁, …]` puts the smoothed chunk of the previous call in
//! front. Only a call that continues a cache can have positions before its
//! first chunk boundary. Such a position reads the latest chunk of the
//! previous call.
//!
//! The chunk count differs from row to row. So the chunked sequence is
//! right-padded to the largest count `K`, and the inner H-Net runs on it with
//! the padding contract of [`Block`](crate::modules::Block). `K` is a shape,
//! so [`plan`] reads the counts to the host once per call.

use crate::modules::prefix_sum;
use burn::prelude::*;
use burn::tensor::IndexingUpdateOp;

/// Where the chunks of one call are.
pub(crate) struct ChunkPlan {
    /// The source position of each chunk, `[batch, K]`. Past the count of a
    /// row, it is `0` (any valid position: the inner padding hides it). With
    /// `K = 0`, it is `[batch, 1]` and unused.
    pub src_bk: Tensor<2, Int>,
    /// The chunk boundaries at or before each position, `[batch, sequence]`.
    pub upto_bs: Tensor<2, Int>,
    /// The inner padding: `true` past the count of the row, `[batch, K]`.
    /// `None` when every row has `K` chunks.
    pub pad_bk: Option<Tensor<2, Bool>>,
    /// `K`: the largest chunk count, rounded up to the multiple of the call.
    pub len: usize,
}

/// The [`ChunkPlan`] of a chunk-boundary mask `[batch, sequence]` (`false` on
/// every absent position). `K` is rounded up to a multiple of `multiple`, so
/// that a run sees fewer distinct inner shapes. This reads the chunk counts to
/// the host.
pub(crate) fn plan(chunk_boundary_bs: Tensor<2, Bool>, multiple: usize) -> ChunkPlan {
    let [batch, sequence] = chunk_boundary_bs.dims();
    assert!(sequence > 0, "an H-Net call needs at least one position");
    assert!(multiple > 0, "the chunk-length multiple must be at least 1");
    let device = chunk_boundary_bs.device();
    let upto_bs = prefix_sum::<2, 3, Int>(chunk_boundary_bs.clone().int(), 1);
    let count_b1 = upto_bs.clone().narrow(1, sequence - 1, 1);
    let counts: Vec<i64> = count_b1.clone().into_data().iter::<i64>().collect();
    let max = counts.iter().copied().max().unwrap_or(0) as usize;
    let len = max.div_ceil(multiple) * multiple;
    if len == 0 {
        // No chunk at all. The caller skips the inner H-Net and reads no
        // chunk, so `src_bk` only keeps a valid shape.
        return ChunkPlan { src_bk: Tensor::zeros([batch, 1], &device), upto_bs, pad_bk: None, len };
    }

    // Each chunk boundary writes its own position index into its chunk slot
    // `k`. Each other position writes into a dump slot of its own (`K + t`).
    // So every index is unique, and an additive scatter onto zeros is an
    // assignment.
    let t_bs = Tensor::<1, Int>::arange(0..sequence as i64, &device)
        .reshape([1, sequence])
        .expand([batch, sequence]);
    let slot_bs = (t_bs.clone() + len as i64).mask_where(chunk_boundary_bs, upto_bs.clone() - 1);
    let src_bk = Tensor::<2, Int>::zeros([batch, len + sequence], &device)
        .scatter(1, slot_bs, t_bs, IndexingUpdateOp::Add)
        .narrow(1, 0, len);

    let pad_bk = counts.iter().any(|&c| c as usize != len).then(|| {
        Tensor::<1, Int>::arange(0..len as i64, &device)
            .reshape([1, len])
            .expand([batch, len])
            .greater_equal(count_b1.expand([batch, len]))
    });
    ChunkPlan { src_bk, upto_bs, pad_bk, len }
}

/// The positions `src_bk` of `x_bsd`: `[batch, K, d]`.
pub(crate) fn gather_positions(x_bsd: Tensor<3>, src_bk: Tensor<2, Int>) -> Tensor<3> {
    let [batch, _sequence, d] = x_bsd.dims();
    let [_batch, k] = src_bk.dims();
    x_bsd.gather(1, src_bk.unsqueeze_dim::<3>(2).expand([batch, k, d]))
}

/// The upsampler: position `t` reads `z_all_bKd[upto[t]]`, where
/// `z_all_bKd` is `[carry, z̄₀, …, z̄_{K−1}]` (`[batch, K + 1, d]`). Out:
/// `[batch, sequence, d]`.
#[allow(non_snake_case)]
pub(crate) fn upsample(z_all_bKd: Tensor<3>, upto_bs: Tensor<2, Int>) -> Tensor<3> {
    let [batch, _k1, d] = z_all_bKd.dims();
    let [_batch, sequence] = upto_bs.dims();
    z_all_bKd.gather(1, upto_bs.unsqueeze_dim::<3>(2).expand([batch, sequence, d]))
}
