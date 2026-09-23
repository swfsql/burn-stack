//! The per-network cache collection trait, and the pairwise tensor visitor
//! that a captured step uses to write its cache back.
//!
//! A [`Block`](crate::modules::Block) family supplies its own `Caches` type.
//! The generic containers need only indexed access to one slot per (virtual)
//! layer, plus the backend hop of `grad_horizon`.
//! [`CapturedStep`](crate::utils::graph::CapturedStep) also needs
//! [`CacheTensors`]: every tensor of a cache, visited pairwise with the
//! matching tensor of another cache.

use burn::prelude::*;

/// The uniform interface of a per-network cache collection for the generic
/// [`Layers`](crate::modules::Layers) loop: `slot_count`, plus move-in and
/// move-out of the per-layer slots.
pub trait CacheStack: Sized {
    /// The per-layer cache element.
    type Cache;
    /// Number of per-(virtual-)layer slots.
    fn slot_count(&self) -> usize;
    /// Move each slot into an `Option`, so the loop can `take` without a clone.
    fn into_slots(self) -> Vec<Option<Self::Cache>>;
    /// Inverse of [`Self::into_slots`].
    fn from_slots(slots: Vec<Option<Self::Cache>>) -> Self;
    /// Move one cache slot **to** the inner (non-autodiff) backend.
    ///
    /// [`Layers::grad_horizon`](crate::modules::Layers::grad_horizon) needs it,
    /// because its untracked segments run on the inner backend. A cache from a
    /// tracked segment must come down with them, for two reasons: the segment
    /// must build no graph, and the dispatch of Burn cannot mix backends
    /// within one op.
    ///
    /// Each family writes it by hand, not with a derive, because
    /// [`Module::map`] is a **no-op on plain `Tensor` fields** (Burn implements
    /// `Module for Tensor` as a constant). Caches hold bare tensors, not
    /// `Param`s. So a `Module`-based conversion would silently skip every one
    /// of them.
    ///
    /// `Tensor::inner` returns a tensor that is already off the autodiff
    /// backend unchanged. So this does nothing there, and is not an error. A
    /// caller that wants to skip the round-trip asks
    /// [`Device::is_autodiff`](burn::prelude::Device::is_autodiff) itself.
    fn cache_to_inner(cache: Self::Cache) -> Self::Cache;

    /// Lift one cache slot back **from** the inner backend, as a new graph
    /// root. The inverse of [`Self::cache_to_inner`] (see its notes).
    fn cache_from_inner(cache: Self::Cache) -> Self::Cache;

    /// Round-trip **every** slot through the inner backend. The cache keeps
    /// its values and loses the graph that produced it. So a caller can carry
    /// state across a gradient boundary (truncated BPTT) without the
    /// activations of the previous segment.
    ///
    /// `detach` alone would cut the gradients but free nothing, because Burn
    /// still registers an untracked op (see
    /// [`detach_params`](crate::utils::detach_params)). The backend hop drops
    /// the graph.
    ///
    /// Does nothing off the autodiff backend (see [`Self::cache_to_inner`]).
    fn detach(self) -> Self {
        let slots = self
            .into_slots()
            .into_iter()
            .map(|slot| slot.map(|cache| Self::cache_from_inner(Self::cache_to_inner(cache))))
            .collect();
        Self::from_slots(slots)
    }
}

/// A rank-generic binary map over the matching tensors of two caches: the
/// visitor that [`CacheTensors::zip_tensors`] drives. A map is the zip of a
/// cache with itself.
pub trait TensorZip {
    /// Combine `a` (from the receiving cache) with `b` (the matching tensor of
    /// the other cache). The result takes the place of `a`.
    fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D>;
}

/// A cache whose tensors can be visited pairwise: one hand-written traversal
/// per cache type (for the same reason that [`CacheStack::cache_to_inner`] is
/// hand-written). The two operations that a captured step needs follow from
/// it.
///
/// A captured graph replays against the exact device buffers that its closure
/// touched. But `step` is functional: it returns new cache tensors, so a
/// replayed step would keep reading the capture-time state. So the captured
/// closure ends with a call of [`assign_in_place`](Self::assign_in_place) on
/// the stable cache, which copies the new cache into the stable buffers.
/// `slice_assign` runs in place on a buffer that no other tensor shares. So
/// the graph records the copies, and each replay advances the state itself
/// (see [`CapturedStep`](crate::utils::graph::CapturedStep)).
///
/// It is in place **only while** the destination buffers are unshared. This
/// is what [`into_owned_buffers`](Self::into_owned_buffers) is for: the cache
/// of a step can hold views into a shared buffer (a `narrow` of a wider
/// tensor). A shared destination does not give a wrong value (`slice_assign`
/// copies). But a replayed graph keeps writing the buffer that the capture
/// saw.
pub trait CacheTensors: Clone {
    /// Pair every tensor of `self` with the matching one of `other` through
    /// `z`, and keep the structure of `self`.
    ///
    /// # Panics
    ///
    /// If the two differ in structure (slot count, which optional tensors they
    /// hold, or whatever variants the cache type has).
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self;

    /// The same values, each tensor in a new contiguous buffer of its own.
    fn into_owned_buffers(self) -> Self {
        struct Own;
        impl TensorZip for Own {
            fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
                drop(b);
                let whole = a.dims().map(|d| 0..d);
                a.empty_like().slice_assign(whole, a)
            }
        }
        self.clone().zip_tensors(self, &mut Own)
    }

    /// `values` written into the buffers of `self`, in place while those
    /// buffers are unshared. The returned cache holds the buffers of `self`.
    fn assign_in_place(self, values: Self) -> Self {
        struct Assign;
        impl TensorZip for Assign {
            fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
                let whole = a.dims().map(|d| 0..d);
                a.slice_assign(whole, b)
            }
        }
        self.zip_tensors(values, &mut Assign)
    }
}

/// No cache: a stateless call (e.g. a fixed-shape `forward`). Its captured
/// replays only refresh the input.
impl CacheTensors for () {
    fn zip_tensors(self, _other: Self, _z: &mut impl TensorZip) -> Self {}
}

impl<C: CacheTensors> CacheTensors for Vec<C> {
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
        assert_eq!(self.len(), other.len(), "the two caches differ in slot count");
        self.into_iter()
            .zip(other)
            .map(|(a, b)| a.zip_tensors(b, z))
            .collect()
    }
}

/// A lone tensor of state, e.g. what a step carries beside the cache of a
/// model.
impl<const D: usize> CacheTensors for Tensor<D> {
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
        z.zip(self, other)
    }
}

/// Two states stepped together (e.g. the cache of a model and a decoded
/// token).
impl<A: CacheTensors, B: CacheTensors> CacheTensors for (A, B) {
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
        (self.0.zip_tensors(other.0, z), self.1.zip_tensors(other.1, z))
    }
}
