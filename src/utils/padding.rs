//! Right-padded batches of sequences of different lengths.
//!
//! A caller that batches sequences of different lengths pads each one on the
//! **right** to a common length, and says where the padding is: a
//! `[batch, sequence]` mask, `true` at padding. A padded position is
//! **absent**. Every container and [`Block`](crate::modules::Block) keeps this
//! contract: to run a padded batch is to run the sequence of each row alone,
//! on the outputs of its real tokens and on the final cache. The output at a
//! padded position is unspecified.
//!
//! A block only ever sees a right-padded mask, and this module keeps it
//! right-padded. This is necessary because one plan splices the class markers
//! ([`ClassLatent`], [`ClassToken`]) for the whole batch, at batch-wide
//! positions. But their place in the sequence of a row depends on the length
//! of that row:
//!
//! - `Start` and `Custom(k)` precede a user token, so they sit where every
//!   row would put them. Each takes the padding of the token that it
//!   precedes. So a `Custom(k)` past the end of a short row is spliced, and
//!   absent there, exactly as the `step` of that row never emits it. The mask
//!   stays right-padded.
//! - `Middle` and `End` sit at `len / 2` and `len` of each row. They are
//!   spliced at their batch-wide position like the rest, and [`Padding`]
//!   records where each position sits in the sequence of its own row. A block
//!   then runs on the positions gathered into that order
//!   ([`Padding::in_row_order`]), right-padded again, and its output is
//!   scattered back.
//! - `End` also works across windows. It lands in the closing window, after
//!   the rows that ended earlier carried their state through padding
//!   untouched. `Middle` needs the whole sequence in one call: its position is
//!   half of the full length of each row, which no single window knows.
//!
//! Everything is computed on the device from the mask, at fixed shapes. So a
//! padded forward stays capturable, and its launch shapes do not change with
//! the lengths of the rows.

use crate::utils::class::ClassMarker;
use burn::prelude::*;
use burn::tensor::IndexingUpdateOp;

/// Which positions of a batch of sequences are padding. The place of some
/// class markers depends on the length of each row. After a container splices
/// one of them, it also records where every position sits in the sequence of
/// its own row.
///
/// It is built from a right-padded mask ([`Self::new`]). The containers
/// advance it through every class-marker splice ([`Self::splice`]), and run
/// each block through [`Self::in_row_order`]. Its tensors live on the inner
/// backend: no gradient ever flows through them.
#[derive(Clone, Debug)]
pub struct Padding {
    /// `[batch, sequence]`, `true` at padding, in the batch-wide order.
    pub pad_bs: Tensor<2, Bool>,
    /// `[batch, sequence]`: the place of each position in the sequence of its
    /// own row (its real tokens first, in order, then its padding). `None` ⇒
    /// the batch-wide order is already the order of every row.
    local_bs: Option<Tensor<2, Int>>,
}

impl Padding {
    /// A right-padded batch: `pad_bs` is `[batch, sequence]`, `true` at padding,
    /// and in every row the padding follows all of its real tokens.
    pub fn new(pad_bs: Tensor<2, Bool>) -> Self {
        Self {
            pad_bs: pad_bs.inner(),
            local_bs: None,
        }
    }

    /// `[batch, sequence]`.
    pub fn dims(&self) -> [usize; 2] {
        self.pad_bs.dims()
    }

    /// The count of real tokens of each row, `[batch, 1]`.
    fn real_b1(&self) -> Tensor<2, Int> {
        self.pad_bs.clone().bool_not().int().sum_dim(1)
    }

    /// `[batch, sequence]`: `0, 1, …` in every row.
    fn arange_bs(&self) -> Tensor<2, Int> {
        let [batch, sequence] = self.dims();
        Tensor::<1, Int>::arange(0..sequence as i64, &self.pad_bs.device())
            .reshape([1, sequence])
            .expand([batch, sequence])
    }

    /// Splice the positions that a class-marker plan inserts (`(at, marker)`
    /// pairs, as [`class_window_plan`](crate::utils::class::class_window_plan)
    /// returns them): the padding half of
    /// [`insert_class_markers_padded`](crate::utils::class::insert_class_markers_padded).
    ///
    /// `whole` says that the spliced window is the entire sequence (its cursor
    /// at the start, and its full length equal to the window length).
    ///
    /// # Panics
    /// A `Middle` marker in a window that is not the whole sequence.
    pub fn splice<M: ClassMarker>(
        self,
        plan: &[(usize, usize)],
        markers: &[M],
        whole: bool,
        who: &str,
    ) -> Self {
        if plan.is_empty() {
            return self;
        }
        let [batch, sequence] = self.dims();
        let device = self.pad_bs.device();
        let k = plan.len();
        let n_real_b1 = self.real_b1();
        let is_middle = |i: usize| markers[i].needs_full_len() && !markers[i].closes_sequence();
        assert!(
            whole || !plan.iter().any(|&(_, i)| is_middle(i)),
            "{who}: a Middle class marker needs the whole padded sequence in one call. Its \
             place is half of the length of each row.",
        );

        // Where each marker enters the sequence of its row (before position
        // `p` of that row, `p = len` ⇒ after its last position), and whether it
        // lands there at all. `Start`/`Custom` precede the position that the
        // plan put them before. `Middle`/`End` follow the length of the row.
        // `End` always lands: it closes the sequence, however early the real
        // tokens of a row ran out.
        let (p_bk, real_bk): (Vec<Tensor<2, Int>>, Vec<Tensor<2, Bool>>) = plan
            .iter()
            .map(|&(at, i)| {
                let p_b1 = match markers[i].needs_full_len() {
                    true if markers[i].closes_sequence() => n_real_b1.clone(),
                    true => n_real_b1.clone().div_scalar(2),
                    false => n_real_b1.clone().zeros_like().add_scalar(at as i64),
                };
                let real_b1 = match markers[i].closes_sequence() {
                    true => n_real_b1.clone().zeros_like().equal_elem(0),
                    false => p_b1.clone().lower(n_real_b1.clone()),
                };
                (p_b1, real_b1)
            })
            .unzip();
        let p_bk = Tensor::cat(p_bk, 1);
        let real_bk = Tensor::cat(real_bk, 1);

        let spliced_pad = splice_columns(self.pad_bs.clone(), plan, real_bk.clone().bool_not());
        let moves = plan.iter().any(|&(_, i)| markers[i].needs_full_len());
        let Some(local_bs) = self.local_bs.clone().or_else(|| moves.then(|| self.arange_bs())) else {
            // Every position is already where its row puts it. A marker
            // precedes the position that it is spliced before, and takes the
            // padding of that position.
            return Self {
                pad_bs: spliced_pad,
                local_bs: None,
            };
        };

        // The order of each row. A real token moves down by the real markers
        // that enter at or before it. A padded position moves down by all of
        // them, because the padding follows every real token. An absent marker
        // goes last.
        let real_i_bk = real_bk.clone().int();
        let m_real_b1 = real_i_bk.clone().sum_dim(1);
        let p_bsk = p_bk.clone().unsqueeze_dim::<3>(1).expand([batch, sequence, k]);
        let before_bs = local_bs
            .clone()
            .unsqueeze_dim::<3>(2)
            .expand([batch, sequence, k])
            .greater_equal(p_bsk)
            .int()
            .mul(real_i_bk.clone().unsqueeze_dim::<3>(1).expand([batch, sequence, k]))
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        let positions_local_bs = local_bs.clone().add(
            before_bs.mask_where(self.pad_bs.clone(), m_real_b1.clone().expand([batch, sequence])),
        );

        // The place of a marker: its entry point, plus the real markers ordered
        // before it there. The order is by entry point, then kind (`Start` <
        // `Middle` < `End` < `Custom`), then declaration. Absent markers follow
        // all the rest, in plan order.
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by_key(|&j| (markers[plan[j].1].group_rank(), plan[j].1));
        let mut rank = vec![0i64; k];
        for (r, &j) in order.iter().enumerate() {
            rank[j] = r as i64;
        }
        let rank_1k = Tensor::<1, Int>::from_ints(rank.as_slice(), &device).reshape([1, k]);
        let key_bk = p_bk.clone().mul_scalar(k as i64).add(rank_1k.expand([batch, k]));
        let key_j_bkk = key_bk.clone().unsqueeze_dim::<3>(1).expand([batch, k, k]);
        let key_i_bkk = key_bk.unsqueeze_dim::<3>(2).expand([batch, k, k]);
        let real_j_bkk = real_i_bk.clone().unsqueeze_dim::<3>(1).expand([batch, k, k]);
        let ahead_bk = key_j_bkk
            .lower(key_i_bkk)
            .int()
            .mul(real_j_bkk.clone())
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        let earlier: Vec<i64> = (0..k * k).map(|ij| i64::from(ij % k < ij / k)).collect();
        let earlier_1kk = Tensor::<1, Int>::from_ints(earlier.as_slice(), &device).reshape([1, k, k]);
        let absent_ahead_bk = earlier_1kk
            .expand([batch, k, k])
            .mul(real_j_bkk.neg().add_scalar(1))
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        let markers_local_bk = p_bk.add(ahead_bk).mask_where(
            real_bk.bool_not(),
            absent_ahead_bk.add(m_real_b1.add_scalar(sequence as i64).expand([batch, k])),
        );

        Self {
            pad_bs: spliced_pad,
            local_bs: Some(splice_columns(positions_local_bs, plan, markers_local_bk)),
        }
    }

    /// Run `f` (a block) on `x_bsd`, with its positions in the order of each
    /// row and the matching right-padded mask. Return its output in the
    /// batch-wide order again (the second output of `f`, a cache, is
    /// untouched).
    ///
    /// While every position is already in the order of its row, this is a
    /// no-op wrapper. Otherwise it is a gather on the way in and one on the
    /// way out.
    pub fn in_row_order<C>(
        &self,
        x_bsd: Tensor<3>,
        f: impl FnOnce(Tensor<3>, Tensor<2, Bool>) -> (Tensor<3>, C),
    ) -> (Tensor<3>, C) {
        let Some(local_bs) = &self.local_bs else {
            return f(x_bsd, self.pad_bs.clone());
        };
        let [batch, sequence, d] = x_bsd.dims();
        assert_eq!(
            [batch, sequence],
            self.dims(),
            "the padding must have the batch and sequence shape of the input"
        );
        // Position `r` goes to `local[r]`. So position `q` of the row order
        // reads the position whose `local` is `q`: the inverse permutation.
        let src_bs = local_bs.clone().zeros_like().scatter(
            1,
            local_bs.clone(),
            self.arange_bs(),
            IndexingUpdateOp::Add,
        );
        let x_bsd = x_bsd.gather(1, src_bs.unsqueeze_dim::<3>(2).expand([batch, sequence, d]));
        let pad_bs = self
            .arange_bs()
            .greater_equal(self.real_b1().expand([batch, sequence]));
        let (y_bsd, c) = f(x_bsd, pad_bs);
        let y_bsd = y_bsd.gather(1, local_bs.clone().unsqueeze_dim::<3>(2).expand([batch, sequence, d]));
        (y_bsd, c)
    }

    /// The same positions, read right to left in each row (its real tokens
    /// reversed, its padding still last), as the reversed pass of a
    /// bidirectional pair reads them (see
    /// [`BidiLayers`](crate::modules::BidiLayers)).
    ///
    /// Its batch-wide order does not change. Only the place of each position
    /// in its row changes.
    pub fn reversed(&self) -> Self {
        let [batch, sequence] = self.dims();
        let local_bs = self.local_bs.clone().unwrap_or_else(|| self.arange_bs());
        let mirrored_bs = self
            .real_b1()
            .expand([batch, sequence])
            .sub_scalar(1)
            .sub(local_bs.clone());
        Self {
            pad_bs: self.pad_bs.clone(),
            local_bs: Some(mirrored_bs.mask_where(self.pad_bs.clone(), local_bs)),
        }
    }
}

/// Splice one column per plan entry (`cols_bk`, `[batch, plan.len()]`, in plan
/// order) into `x_bs` along its sequence axis, before the position that each
/// entry names. This is the counterpart of
/// [`splice_class_embeddings`](crate::utils::class::splice_class_embeddings)
/// with a value for each row.
fn splice_columns<K>(x_bs: Tensor<2, K>, plan: &[(usize, usize)], cols_bk: Tensor<2, K>) -> Tensor<2, K>
where
    K: burn::tensor::kind::Basic,
{
    let [_batch, sequence] = x_bs.dims();
    let mut parts = Vec::with_capacity(2 * plan.len() + 1);
    let mut taken = 0usize;
    for (j, &(at, _)) in plan.iter().enumerate() {
        if at > taken {
            parts.push(x_bs.clone().narrow(1, taken, at - taken));
            taken = at;
        }
        parts.push(cols_bk.clone().narrow(1, j, 1));
    }
    if taken < sequence {
        parts.push(x_bs.narrow(1, taken, sequence - taken));
    }
    Tensor::cat(parts, 1)
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
