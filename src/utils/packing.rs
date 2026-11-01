//! Packed rows: in each row of a batch, several sequences one after another.
//!
//! A batch of sequences of different lengths wastes the padding of its short
//! rows. A packed batch fills each row with whole sequences instead. Each
//! sequence of a packed row is a **segment**, and it starts at a **reset**.
//! There, every block starts again from a zero cache (see
//! [`Block::block_forward_packed`]). So each segment gives the outputs that
//! it gives alone.
//!
//! The class latents of a stack (all `Start`) open every segment, not only
//! the row. A splice would move every later position of the row. So a packed
//! row **reserves** the opening positions instead. The first `lead` positions
//! of each segment are slots. The stack puts its latent `k` into slot `k`, in
//! place of the input at that position. The reset is at the first slot. The
//! shapes stay fixed.
//!
//! A block can accept a reset only at some positions (for example, only at
//! the start of a chunk). The caller that packs the rows puts the resets
//! there. The positions after the last segment of a row are free. They reach
//! only the returned cache.
//!
//! [`Block::block_forward_packed`]: crate::modules::Block::block_forward_packed

use burn::prelude::*;

/// The layout of a packed batch: where each segment starts, and its opening
/// slots. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct Packed {
    /// `[batch, sequence]`: `true` at the first position of each segment.
    pub reset_bs: Tensor<2, Bool>,
    /// `[batch, sequence]`: `k` at the `k`-th opening slot of a segment, `-1`
    /// at every other position.
    pub latent_bs: Tensor<2, Int>,
}

impl Packed {
    /// The layout of a batch whose rows have `sequence` positions each.
    /// `starts[b]` holds the first position of each segment of row `b`, and
    /// each segment opens with `lead` slots.
    ///
    /// # Panics
    /// - If a start is past the end of the row.
    /// - If the opening slots of a segment reach the next start or the end
    ///   of the row.
    pub fn from_starts(starts: &[Vec<usize>], lead: usize, sequence: usize, device: &Device) -> Self {
        let batch = starts.len();
        let mut reset = vec![false; batch * sequence];
        let mut latent = vec![-1i64; batch * sequence];
        for (b, starts) in starts.iter().enumerate() {
            let mut end = 0;
            for &start in starts {
                assert!(
                    start >= end,
                    "the segments of a row must start in order, after the opening slots"
                );
                assert!(start + lead.max(1) <= sequence, "a segment starts past the end of the row");
                reset[b * sequence + start] = true;
                for k in 0..lead {
                    latent[b * sequence + start + k] = k as i64;
                }
                end = start + lead.max(1);
            }
        }
        Self {
            reset_bs: Tensor::<1, Bool>::from_bool(TensorData::new(reset, [batch * sequence]), device)
                .reshape([batch, sequence]),
            latent_bs: Tensor::<1, Int>::from_ints(latent.as_slice(), device).reshape([batch, sequence]),
        }
    }

    /// `[batch, sequence]`.
    pub fn dims(&self) -> [usize; 2] {
        self.reset_bs.dims()
    }

    /// Move the layout to `device`.
    pub fn to_device(self, device: &Device) -> Self {
        Self {
            reset_bs: bool_to_device(self.reset_bs, device),
            latent_bs: int_to_device(self.latent_bs, device),
        }
    }

    /// The layout on the inner backend. No gradient flows through it.
    pub fn inner(&self) -> Self {
        Self {
            reset_bs: self.reset_bs.clone().inner(),
            latent_bs: self.latent_bs.clone().inner(),
        }
    }

    /// Put the rows of `table` (`[lead, width]`) into the opening slots of `x`
    /// (`[batch, sequence, width]`): row `k` at every slot `k`. The other
    /// positions of `x` pass unchanged. The gradient of a slot goes to its row
    /// of `table`.
    pub fn place(&self, x: Tensor<3>, table: Tensor<2>) -> Tensor<3> {
        let [batch, sequence, width] = x.dims();
        assert_eq!(self.dims(), [batch, sequence], "the layout of a packed batch has its shape");
        let [lead, table_width] = table.dims();
        assert_eq!(table_width, width, "the table must have one row per latent, with the width of `x`");
        let device = x.device();
        // Row `lead` of the extended table is the zero row of every other
        // position. The mask below discards it.
        let table = Tensor::cat(vec![table, Tensor::zeros([1, width], &device)], 0);
        let slot_bs = self.latent_bs.clone().to_device(&device);
        let is_slot_bs = slot_bs.clone().greater_equal_elem(0);
        let index = slot_bs
            .mask_fill(is_slot_bs.clone().bool_not(), lead as i64)
            .reshape([batch * sequence]);
        let slots = table.select(0, index).reshape([batch, sequence, width]);
        let is_slot = is_slot_bs.unsqueeze_dim::<3>(2).expand([batch, sequence, width]);
        x.mask_where(is_slot, slots)
    }
}

/// An int tensor moved to `device`, in its default int dtype.
fn int_to_device<const D: usize>(t: Tensor<D, Int>, device: &Device) -> Tensor<D, Int> {
    t.to_device(device).cast(device.settings().int_dtype)
}

/// A bool tensor moved to `device`. It moves as ints: a native bool from
/// another backend can fail to load (the cubecl backends take no
/// `Bool(Native)` data).
pub fn bool_to_device<const D: usize>(t: Tensor<D, Bool>, device: &Device) -> Tensor<D, Bool> {
    int_to_device(t.int(), device).equal_elem(1)
}
