//! Right-padded batches of sequences of different lengths.
//!
//! A caller batching sequences of different lengths pads each one on the
//! **right** to a common length and says where the padding is: a
//! `[batch, sequence]` mask, `true` at padding. A padded row is **absent** — the
//! contract every container and [`Block`](crate::modules::Block) keeps is that
//! running a padded batch is running each slot's own sequence alone, on the
//! outputs of its real rows and on the final cache. What a padded row itself
//! outputs is unspecified.
//!
//! A block only ever sees a right-padded mask. Keeping it one is this module's
//! job, because class markers ([`ClassLatent`](crate::utils::ClassLatent),
//! [`ClassToken`](crate::utils::ClassToken)) are spliced by one plan for the
//! whole batch, at batch-wide positions, while their place in a slot's own
//! sequence depends on that slot's length:
//!
//! - `Start` and `Custom(k)` precede a user token, so they sit where every slot
//!   would put them; each takes the padding of the token it precedes (so a
//!   `Custom(k)` past a short slot's end is spliced, and absent there — exactly
//!   as that slot's own `step` never emits it). The mask stays right-padded.
//! - `Middle` and `End` sit at each slot's own `len / 2` and `len`. They are
//!   spliced at their batch-wide position like the rest, and [`Padding`]
//!   records where each row sits in its own slot's sequence instead; a block is
//!   then run on the rows gathered into that order ([`Padding::in_slot_order`]),
//!   right-padded again, and its output scattered back. `End` also works across
//!   chunks — it lands in the closing chunk, after the slots that ended earlier
//!   have carried their state through padding untouched — while `Middle` needs
//!   the whole sequence in one call (its position is half of each slot's full
//!   length, which no single chunk knows).
//!
//! Everything is computed on the device from the mask, at fixed shapes, so a
//! padded forward stays capturable and its launch shapes do not vary with the
//! slots' lengths.

use crate::utils::class::ClassMarker;
use burn::prelude::*;
use burn::tensor::IndexingUpdateOp;

/// Which rows of a batch of sequences are padding, and — once a container has
/// spliced a row that belongs at a per-slot position — where every row sits in
/// its own slot's sequence.
///
/// Built from a right-padded mask ([`Self::new`]); the containers advance it
/// through every class-marker splice ([`Self::splice`]) and run each block
/// through [`Self::in_slot_order`]. Its tensors live on the inner backend: no
/// gradient ever flows through them.
#[derive(Clone, Debug)]
pub struct Padding {
    /// `[batch, rows]`, `true` at padding, in the batch-wide row order.
    pub pad_bs: Tensor<2, Bool>,
    /// `[batch, rows]`: each row's position in its own slot's sequence — its
    /// real rows first, in order, then its padding. `None` ⇒ the batch-wide order
    /// already is every slot's own.
    local_bs: Option<Tensor<2, Int>>,
}

impl Padding {
    /// A right-padded batch: `pad_bs` is `[batch, sequence]`, `true` at padding,
    /// and in every slot the padding follows all of its real rows.
    pub fn new(pad_bs: Tensor<2, Bool>) -> Self {
        Self {
            pad_bs: pad_bs.inner(),
            local_bs: None,
        }
    }

    /// `[batch, rows]`.
    pub fn dims(&self) -> [usize; 2] {
        self.pad_bs.dims()
    }

    /// Real rows per slot, `[batch, 1]`.
    fn real_b1(&self) -> Tensor<2, Int> {
        self.pad_bs.clone().bool_not().int().sum_dim(1)
    }

    /// `[batch, rows]`: `0, 1, …` in every slot.
    fn arange_bs(&self) -> Tensor<2, Int> {
        let [batch, rows] = self.dims();
        Tensor::<1, Int>::arange(0..rows as i64, &self.pad_bs.device())
            .reshape([1, rows])
            .expand([batch, rows])
    }

    /// Splice the rows a class-marker plan inserts (`(at, marker)` pairs, as
    /// [`class_chunk_plan`](crate::utils::class::class_chunk_plan) returns them)
    /// — the padding half of
    /// [`insert_class_markers`](crate::utils::class::insert_class_markers).
    ///
    /// `whole` says the chunk being spliced is the entire sequence (its
    /// cursor at the start, its full length the chunk's).
    ///
    /// # Panics
    /// A `Middle` marker in a chunk that is not the whole sequence.
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
        let [batch, rows] = self.dims();
        let device = self.pad_bs.device();
        let k = plan.len();
        let n_real_b1 = self.real_b1();
        let is_middle = |i: usize| markers[i].needs_full_len() && !markers[i].closes_sequence();
        assert!(
            whole || !plan.iter().any(|&(_, i)| is_middle(i)),
            "{who}: a Middle class marker needs the whole padded sequence in one call — its \
             place is half of each slot's own length",
        );

        // Where each marker enters its slot's own sequence (before that slot's
        // row `p`, `p = len` ⇒ after its last one), and whether it lands there
        // at all. `Start`/`Custom` precede the row the plan put them before;
        // `Middle`/`End` follow the slot's own length. `End` always lands — it
        // closes the sequence, however early a slot's own real rows ran out.
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
            // Every row is already where its slot puts it: a marker precedes the
            // row it is spliced before, and takes that row's padding with it.
            return Self {
                pad_bs: spliced_pad,
                local_bs: None,
            };
        };

        // Each slot's own order. A real row moves down by the real markers
        // entering at or before it; a padded one by all of them, the padding
        // following every real row; an absent marker goes last.
        let real_i_bk = real_bk.clone().int();
        let m_real_b1 = real_i_bk.clone().sum_dim(1);
        let p_bsk = p_bk.clone().unsqueeze_dim::<3>(1).expand([batch, rows, k]);
        let before_bs = local_bs
            .clone()
            .unsqueeze_dim::<3>(2)
            .expand([batch, rows, k])
            .greater_equal(p_bsk)
            .int()
            .mul(real_i_bk.clone().unsqueeze_dim::<3>(1).expand([batch, rows, k]))
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        let rows_local_bs = local_bs.clone().add(
            before_bs.mask_where(self.pad_bs.clone(), m_real_b1.clone().expand([batch, rows])),
        );

        // A marker's own place: its entry point, plus the real markers ordered
        // before it there — by entry point, then kind (`Start` < `Middle` < `End`
        // < `Custom`), then declaration. Absent ones follow all the rest, in plan
        // order.
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
            absent_ahead_bk.add(m_real_b1.add_scalar(rows as i64).expand([batch, k])),
        );

        Self {
            pad_bs: spliced_pad,
            local_bs: Some(splice_columns(rows_local_bs, plan, markers_local_bk)),
        }
    }

    /// Run `f` — a block — on `x_bsd` with its rows in each slot's own order and
    /// the matching right-padded mask, and return its output in the batch-wide
    /// order again (`f`'s second output, a cache, untouched).
    ///
    /// A no-op wrapper while every row is already in its slot's order; otherwise
    /// a gather on the way in and one on the way out.
    pub fn in_slot_order<C>(
        &self,
        x_bsd: Tensor<3>,
        f: impl FnOnce(Tensor<3>, Tensor<2, Bool>) -> (Tensor<3>, C),
    ) -> (Tensor<3>, C) {
        let Some(local_bs) = &self.local_bs else {
            return f(x_bsd, self.pad_bs.clone());
        };
        let [batch, rows, d] = x_bsd.dims();
        assert_eq!([batch, rows], self.dims(), "the padding covers every row");
        // Row `r` goes to `local[r]`, so slot position `q` reads the row whose
        // `local` is `q` — the inverse permutation.
        let src_bs = local_bs.clone().zeros_like().scatter(
            1,
            local_bs.clone(),
            self.arange_bs(),
            IndexingUpdateOp::Add,
        );
        let x_bsd = x_bsd.gather(1, src_bs.unsqueeze_dim::<3>(2).expand([batch, rows, d]));
        let pad_bs = self
            .arange_bs()
            .greater_equal(self.real_b1().expand([batch, rows]));
        let (y_bsd, c) = f(x_bsd, pad_bs);
        let y_bsd = y_bsd.gather(1, local_bs.clone().unsqueeze_dim::<3>(2).expand([batch, rows, d]));
        (y_bsd, c)
    }

    /// The same rows read right to left in each slot — its real rows reversed,
    /// its padding still last — as the reversed direction of a bidirectional
    /// pair reads them (see [`BidiLayers`](crate::modules::BidiLayers)).
    ///
    /// Its batch-wide order is the forward one: only where each row sits in its
    /// slot moves.
    pub fn reversed(&self) -> Self {
        let [batch, rows] = self.dims();
        let local_bs = self.local_bs.clone().unwrap_or_else(|| self.arange_bs());
        let mirrored_bs = self
            .real_b1()
            .expand([batch, rows])
            .sub_scalar(1)
            .sub(local_bs.clone());
        Self {
            pad_bs: self.pad_bs.clone(),
            local_bs: Some(mirrored_bs.mask_where(self.pad_bs.clone(), local_bs)),
        }
    }
}

/// Splice one column per plan entry (`cols_bk`, `[batch, plan.len()]`, in plan
/// order) into `x_bs` along its row axis, before the row each entry names — the
/// per-slot counterpart of
/// [`splice_class_rows`](crate::utils::class::splice_class_rows).
fn splice_columns<K>(x_bs: Tensor<2, K>, plan: &[(usize, usize)], cols_bk: Tensor<2, K>) -> Tensor<2, K>
where
    K: burn::tensor::kind::Basic,
{
    let [_batch, rows] = x_bs.dims();
    let mut segments = Vec::with_capacity(2 * plan.len() + 1);
    let mut taken = 0usize;
    for (j, &(at, _)) in plan.iter().enumerate() {
        if at > taken {
            segments.push(x_bs.clone().narrow(1, taken, at - taken));
            taken = at;
        }
        segments.push(cols_bk.clone().narrow(1, j, 1));
    }
    if taken < rows {
        segments.push(x_bs.narrow(1, taken, rows - taken));
    }
    Tensor::cat(segments, 1)
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
