//! A blocked inclusive prefix sum: the values of
//! [`Tensor::cumsum`](burn::tensor::Tensor::cumsum), at a cost that does not
//! grow quadratically with the scanned length.

use burn::prelude::*;
use burn::tensor::kind::Numeric;

/// The in-block length that [`prefix_sum`] scans directly, for an axis of
/// `len` rows.
///
/// A `cumsum` of length `L` over `N` elements costs `∝ N·L` on the cubecl
/// backends. So a blocked scan costs `∝ N·(block + len/block²)`: one pass in
/// each block, and one pass over the `len/block` block totals. The minimum is
/// at `block = ∛(2·len)`. This function approaches it on a power-of-two grid.
/// The floor of 16 amortises the fixed cost of the blocking on short axes.
/// The ceiling bounds the in-block cost.
///
/// `scan_block(len) < len` means "this length takes the blocked branch".
pub fn scan_block(len: usize) -> usize {
    let mut block = 16;
    while block < 256 && block * block * block < 2 * len {
        block *= 2;
    }
    block
}

/// Inclusive prefix sum along `dim`: `out[i] = Σ_{j ≤ i} t[j]`.
///
/// The values of [`Tensor::cumsum`], but **blocked**:
///
/// 1. Split the scanned axis into runs of [`scan_block`] rows.
/// 2. Scan each run alone.
/// 3. Add the exclusive prefix of the run totals.
///
/// Burn's `cumsum` is quadratic in the scanned length on the cubecl backends.
/// This scan does the same work in a few full-tensor ops for any length. On a
/// backend whose `cumsum` is already linear, it costs about the same.
///
/// It takes any numeric kind. An `Int` scan (for example, a count of the
/// `true` rows of a mask) is exact. A float scan associates the additions
/// differently from a sequential scan, so the two agree to rounding, not
/// bit-for-bit.
///
/// # Shapes
/// - `t`   : any rank, scanned along `dim`. `DP1` is `D + 1`.
/// - out   : the shape of `t`
pub fn prefix_sum<const D: usize, const DP1: usize, K: Numeric>(
    t: Tensor<D, K>,
    dim: usize,
) -> Tensor<D, K> {
    assert_eq!(D + 1, DP1, "DP1 must be D + 1");
    let len = t.dims()[dim];
    let block = scan_block(len);
    if len <= block {
        return t.cumsum(dim);
    }
    let device = t.device();

    // Pad up to a whole number of blocks. Zero is the additive identity, so the
    // padding adds to no prefix. The end narrows it off.
    let nblocks = len.div_ceil(block);
    let pad = nblocks * block - len;
    let t = if pad == 0 {
        t
    } else {
        let mut pad_dims = t.dims();
        pad_dims[dim] = pad;
        let zeros = Tensor::<D, K>::zeros(pad_dims, (&device, t.dtype()));
        Tensor::cat(vec![t, zeros], dim)
    };

    // Split the scanned axis into (which block, where in it). The layout is
    // row-major, so the reshape moves no data.
    let dims = t.dims();
    let mut split = [0usize; DP1];
    split[..dim].copy_from_slice(&dims[..dim]);
    split[dim] = nblocks;
    split[dim + 1] = block;
    split[dim + 2..].copy_from_slice(&dims[dim + 1..]);

    let inner = t.reshape(split).cumsum(dim + 1);
    // The total of a block is its last in-block prefix. It keeps width 1 on
    // that axis, so the carry broadcasts back over the block.
    let totals = inner.clone().narrow(dim + 1, block - 1, 1);
    // The exclusive prefix of the totals. The block axis is short
    // (`len / block ≤ block²/2`), so a plain `cumsum` is cheap here.
    let carry = totals.clone().cumsum(dim) - totals;

    let joined = (inner + carry).reshape(dims);
    if pad == 0 { joined } else { joined.narrow(dim, 0, len) }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests {
    use super::*;
    use crate::utils::test_helpers::test_device;

    /// The blocked branch gives the exact values of `cumsum` on an `Int`
    /// mask count, for lengths below, at, and above whole blocks.
    #[test]
    fn int_prefix_sum_equals_cumsum() {
        let device = test_device();
        for len in [5, 16, 17, 100, 257, 1000] {
            assert!(len <= 16 || scan_block(len) < len, "len {len} must take the blocked branch");
            let values: Vec<i64> = (0..2 * len).map(|i| ((i * 7 + 3) % 5 == 0) as i64).collect();
            let t = Tensor::<2, Int>::from_data(TensorData::new(values, [2, len]), &device);
            let want: Vec<i64> = t.clone().cumsum(1).into_data().iter::<i64>().collect();
            let got: Vec<i64> = prefix_sum::<2, 3, Int>(t, 1).into_data().iter::<i64>().collect();
            assert_eq!(got, want, "len {len}");
        }
    }
}
