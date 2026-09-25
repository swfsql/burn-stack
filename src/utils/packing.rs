//! Packed rows: in each slot of a batch, several sequences one after another.
//!
//! A batch of sequences of different lengths wastes the padding of its short
//! slots. A packed batch fills each slot with whole sequences instead. Each
//! sequence starts at a **reset**. There, every block restarts from a zero
//! cache (see [`Block::block_forward_packed`]). So each sequence gives the
//! outputs that it gives alone.
//!
//! The class latents of a stack (all `Start`) open every sequence, not only
//! the slot. A splice would move every later row of the slot. So a packed row
//! **reserves** the opening rows instead. The first `lead` rows of each
//! sequence are slots. The stack puts its latent `k` into slot `k`, in place
//! of the input row. The reset is at the first slot. The shapes stay fixed.
//!
//! A block can accept a reset only at some rows (for example, only at a chunk
//! start). The caller that packs the rows places the resets there. The rows
//! after the last sequence of a slot are free. They reach only the returned
//! cache.
//!
//! [`Block::block_forward_packed`]: crate::modules::Block::block_forward_packed

use burn::prelude::*;

/// The layout of a packed batch: where each sequence starts, and its opening
/// slots. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct Packed {
    /// `[batch, rows]`: `true` at the first row of each sequence.
    pub reset_bs: Tensor<2, Bool>,
    /// `[batch, rows]`: `k` at the `k`-th opening slot of a sequence, `-1` at
    /// every other row.
    pub latent_bs: Tensor<2, Int>,
}

impl Packed {
    /// The layout of `rows` rows per slot. `starts[b]` holds the first row of
    /// each sequence of slot `b`, and each sequence opens with `lead` slots.
    ///
    /// # Panics
    /// - If a start is past the end of the row.
    /// - If the opening slots of a sequence reach the next start or the end
    ///   of the row.
    pub fn from_starts(starts: &[Vec<usize>], lead: usize, rows: usize, device: &Device) -> Self {
        let batch = starts.len();
        let mut reset = vec![false; batch * rows];
        let mut latent = vec![-1i64; batch * rows];
        for (b, starts) in starts.iter().enumerate() {
            let mut end = 0;
            for &start in starts {
                assert!(start >= end, "the sequences of a slot start in order, after the opening slots");
                assert!(start + lead.max(1) <= rows, "a sequence starts past the end of the row");
                reset[b * rows + start] = true;
                for k in 0..lead {
                    latent[b * rows + start + k] = k as i64;
                }
                end = start + lead.max(1);
            }
        }
        Self {
            reset_bs: Tensor::<1, Bool>::from_bool(TensorData::new(reset, [batch * rows]), device)
                .reshape([batch, rows]),
            latent_bs: Tensor::<1, Int>::from_ints(latent.as_slice(), device).reshape([batch, rows]),
        }
    }

    /// `[batch, rows]`.
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
    /// (`[batch, rows, width]`): row `k` at every slot `k`. The other rows of
    /// `x` pass unchanged. The gradient of a slot goes to its row of `table`.
    pub fn place(&self, x: Tensor<3>, table: Tensor<2>) -> Tensor<3> {
        let [batch, rows, width] = x.dims();
        assert_eq!(self.dims(), [batch, rows], "the layout of a packed batch has its shape");
        let [lead, table_width] = table.dims();
        assert_eq!(table_width, width, "one opening row per latent, of the width of the rows");
        let device = x.device();
        // Row `lead` of the extended table is the zero row of every other
        // position. The mask below discards it.
        let table = Tensor::cat(vec![table, Tensor::zeros([1, width], &device)], 0);
        let slot_bs = self.latent_bs.clone().to_device(&device);
        let is_slot_bs = slot_bs.clone().greater_equal_elem(0);
        let index = slot_bs
            .mask_fill(is_slot_bs.clone().bool_not(), lead as i64)
            .reshape([batch * rows]);
        let slots = table.select(0, index).reshape([batch, rows, width]);
        let is_slot = is_slot_bs.unsqueeze_dim::<3>(2).expand([batch, rows, width]);
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
