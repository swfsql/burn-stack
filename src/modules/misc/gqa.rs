//! Grouped-Query Attention (GQA) dimension expansion.
//!
//! A block can project its key/value-like tensors per group (size `ngroups`),
//! while the kernel that consumes them wants one per head (size `nheads`).
//! This helper connects the two: it replicates the vector of each group
//! across the `heads_per_group = nheads / ngroups` heads of that group.

use burn::prelude::*;

/// Expand the `ngroups` dim of a tensor at `group_dim` into an `nheads` dim.
/// It replicates the slice of each group across the
/// `heads_per_group = nheads / ngroups` heads of that group.
///
/// The const generic `DP1` must equal `D + 1` (the rank of the intermediate
/// `unsqueeze`+`expand`). Rust cannot yet express that constraint directly,
/// so the caller must satisfy it. A wrong value gives a compile-time rank
/// mismatch from `unsqueeze_dim::<DP1>` / `reshape`.
///
/// # Panics
/// Panics if `nheads % ngroups != 0` (i.e. `nheads` is not a multiple of the
/// current group count).
///
/// # Example
/// ```ignore
/// // k: [batch, sequence, ngroups, dim] (D=4)
/// // group_dim = 2 (the ngroups axis)
/// // result: [batch, sequence, nheads, dim]
/// let k = gqa_expand_to_heads::<4, 5>(k, 2, nheads);
/// ```
pub fn gqa_expand_to_heads<const D: usize, const DP1: usize>(
    t: Tensor<D>,
    group_dim: usize,
    nheads: usize,
) -> Tensor<D> {
    let dims = t.dims();
    let ngroups = dims[group_dim];
    assert!(
        nheads.is_multiple_of(ngroups),
        "nheads ({nheads}) must be a multiple of ngroups ({ngroups})"
    );
    let heads_per_group = nheads / ngroups;

    // Expanded shape: insert `heads_per_group` immediately after `group_dim`.
    let mut expanded = [0usize; DP1];
    expanded[..=group_dim].copy_from_slice(&dims[..=group_dim]);
    expanded[group_dim + 1] = heads_per_group;
    expanded[group_dim + 2..].copy_from_slice(&dims[group_dim + 1..]);

    // Final shape: collapse `(ngroups, heads_per_group)` back into `nheads`.
    let mut final_shape = dims;
    final_shape[group_dim] = nheads;

    t.unsqueeze_dim::<DP1>(group_dim + 1)
        .expand(expanded)
        .reshape(final_shape)
}
