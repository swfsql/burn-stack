//! The caches of an [`HNet`](super::HNet), and the per-row operations that
//! its `step` does on the cache of a whole inner network.
//!
//! A `step` runs the inner network only on the rows that start a chunk. So it
//! needs three things that do not depend on the block family:
//!
//! - `select_rows`: per row, the new cache or the old one.
//! - `gather_rows`: some rows of a cache, as a smaller batch.
//! - `merge_rows`: a smaller batch written back into the full one.
//!
//! All three go through
//! [`CacheTensors`](crate::modules::CacheTensors). They read axis 0 of every
//! cache tensor as the batch axis, which is a contract of that trait.

use crate::modules::{Block, CacheStack, CacheTensors, TensorZip, lift};
use crate::modules::hnet::routing::RouterCache;
use burn::prelude::*;

/// The cache of one H-Net stage.
pub struct HNetStageCache<E: Block> {
    /// The caches of the encoder stack.
    pub encoder: E::Caches,
    /// The state of the router.
    pub router: RouterCache,
    /// The smoothed chunk `z̄` of the last chunk of each slot, `[batch,
    /// d_model]`: the carry of the EMA of the dechunking layer.
    pub smooth_bd: Tensor<2>,
    /// The caches of the decoder stack.
    pub decoder: E::Caches,
}

/// The caches of a whole [`HNet`](super::HNet): one [`HNetStageCache`] per
/// stage (outermost first), and the caches of the main network.
pub struct HNetCaches<E: Block, M: Block> {
    /// One cache per stage, outermost first.
    pub stages: Vec<HNetStageCache<E>>,
    /// The caches of the main network.
    pub main: M::Caches,
}

impl<E: Block> Clone for HNetStageCache<E>
where
    E::Caches: Clone,
{
    fn clone(&self) -> Self {
        Self {
            encoder: self.encoder.clone(),
            router: self.router.clone(),
            smooth_bd: self.smooth_bd.clone(),
            decoder: self.decoder.clone(),
        }
    }
}

impl<E: Block, M: Block> Clone for HNetCaches<E, M>
where
    E::Caches: Clone,
    M::Caches: Clone,
{
    fn clone(&self) -> Self {
        Self { stages: self.stages.clone(), main: self.main.clone() }
    }
}

impl<E: Block> CacheTensors for HNetStageCache<E>
where
    E::Caches: CacheTensors,
{
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
        Self {
            encoder: self.encoder.zip_tensors(other.encoder, z),
            router: RouterCache {
                last_bd: z.zip(self.router.last_bd, other.router.last_bd),
                seen_b: z.zip(self.router.seen_b, other.router.seen_b),
            },
            smooth_bd: z.zip(self.smooth_bd, other.smooth_bd),
            decoder: self.decoder.zip_tensors(other.decoder, z),
        }
    }
}

impl<E: Block, M: Block> CacheTensors for HNetCaches<E, M>
where
    E::Caches: CacheTensors,
    M::Caches: CacheTensors,
{
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
        assert_eq!(self.stages.len(), other.stages.len(), "the two caches differ in stage count");
        Self {
            stages: self
                .stages
                .into_iter()
                .zip(other.stages)
                .map(|(a, b)| a.zip_tensors(b, z))
                .collect(),
            main: self.main.zip_tensors(other.main, z),
        }
    }
}

impl<E: Block, M: Block> HNetCaches<E, M> {
    /// The same values, cut off the autodiff graph: every block cache goes
    /// through [`CacheStack::detach`], and every other tensor through the
    /// same backend hop. So a training loop can carry the state into the next
    /// window without the activations of this one.
    pub fn detach(self) -> Self {
        let hop = |t: Tensor<2>| {
            let device = t.device();
            lift(t.inner(), &device)
        };
        let hop1 = |t: Tensor<1>| {
            let device = t.device();
            lift(t.inner(), &device)
        };
        Self {
            stages: self
                .stages
                .into_iter()
                .map(|s| HNetStageCache {
                    encoder: s.encoder.detach(),
                    router: RouterCache { last_bd: hop(s.router.last_bd), seen_b: hop1(s.router.seen_b) },
                    smooth_bd: hop(s.smooth_bd),
                    decoder: s.decoder.detach(),
                })
                .collect(),
            main: self.main.detach(),
        }
    }
}

/// The row mask `take_b` (`[batch]`) as a mask of shape `dims`: on axis 0,
/// and broadcast over the other axes.
fn rows_mask<const D: usize>(take_b: &Tensor<1, Bool>, dims: [usize; D]) -> Tensor<D, Bool> {
    let mut shape = [1usize; D];
    shape[0] = dims[0];
    take_b.clone().reshape(shape).expand(dims)
}

/// Per row: `new` where `take_b` is `true`, else `old`.
pub(crate) fn select_rows<C: CacheTensors>(old: C, new: C, take_b: &Tensor<1, Bool>) -> C {
    struct Select<'a>(&'a Tensor<1, Bool>);
    impl TensorZip for Select<'_> {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            let mask = rows_mask(self.0, a.dims());
            a.mask_where(mask, b)
        }
    }
    old.zip_tensors(new, &mut Select(take_b))
}

/// The rows `idx_r` of `cache`, as a batch of `r` rows.
pub(crate) fn gather_rows<C: CacheTensors>(cache: C, idx_r: &Tensor<1, Int>) -> C {
    struct Gather<'a>(&'a Tensor<1, Int>);
    impl TensorZip for Gather<'_> {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            drop(b);
            a.select(0, self.0.clone())
        }
    }
    cache.clone().zip_tensors(cache, &mut Gather(idx_r))
}

/// The full batch `old` with some rows from `sub`. Row `i` of the result is
/// row `map_b[i]` of `[old; sub]` (the rows of `old`, then those of `sub`). A
/// gather, not an add, so the kept rows are bit-exact.
pub(crate) fn merge_rows<C: CacheTensors>(old: C, sub: C, map_b: &Tensor<1, Int>) -> C {
    struct Merge<'a>(&'a Tensor<1, Int>);
    impl TensorZip for Merge<'_> {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            Tensor::cat(vec![a, b], 0).select(0, self.0.clone())
        }
    }
    old.zip_tensors(sub, &mut Merge(map_b))
}
