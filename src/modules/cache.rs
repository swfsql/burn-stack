//! The per-network cache collection trait, and the pairwise tensor visitor a
//! captured step writes its cache back with.
//!
//! A [`Block`](crate::modules::Block) family supplies its own `Caches` type; all
//! the generic containers need from it is indexed access to one slot per
//! (virtual) layer, plus the backend hop `grad_horizon` performs.
//! [`CacheTensors`] is what [`CapturedStep`](crate::utils::graph::CapturedStep)
//! needs on top: every tensor of a cache, visited pairwise with another cache's.

use burn::prelude::*;

/// The uniform interface a per-network cache collection exposes for the generic
/// [`Layers`](crate::modules::Layers) loop: `slot_count` + move-in/move-out of
/// the per-layer slots.
pub trait CacheStack: Sized {
    /// The per-layer cache element.
    type Cache;
    /// Number of per-(virtual-)layer slots.
    fn slot_count(&self) -> usize;
    /// Move each slot into an `Option` so the loop can `take` without cloning.
    fn into_slots(self) -> Vec<Option<Self::Cache>>;
    /// Inverse of [`Self::into_slots`].
    fn from_slots(slots: Vec<Option<Self::Cache>>) -> Self;
    /// Move one cache slot **to** the inner (non-autodiff) backend.
    ///
    /// Needed by [`Layers::grad_horizon`](crate::modules::Layers::grad_horizon),
    /// whose no-grad prefix runs on the inner backend: a cache carried in from a
    /// tracked segment has to come down with it, both so the prefix builds no
    /// graph and because Burn's dispatch cannot mix backends within one op.
    ///
    /// Spelled out per family rather than derived, because
    /// [`Module::map`] is a **no-op on plain `Tensor`
    /// fields** (Burn implements `Module for Tensor` as a constant) and caches
    /// hold bare tensors, not `Param`s — a `Module`-based conversion would
    /// silently skip every one of them.
    ///
    /// `Tensor::inner` returns a tensor that is already off the autodiff backend
    /// unchanged, so this is inert there rather than an error — a caller that
    /// wants to skip the round-trip asks
    /// [`Device::is_autodiff`](burn::prelude::Device::is_autodiff) itself.
    fn cache_to_inner(cache: Self::Cache) -> Self::Cache;

    /// Lift one cache slot back **from** the inner backend, as a fresh graph
    /// root. The inverse of [`Self::cache_to_inner`]; see its notes.
    fn cache_from_inner(cache: Self::Cache) -> Self::Cache;

    /// Round-trip **every** slot through the inner backend: the cache keeps its
    /// values and loses the graph that produced it, so a caller can carry state
    /// across a gradient boundary (truncated BPTT) without retaining the
    /// previous segment's activations.
    ///
    /// `detach` alone would cut the gradients but free nothing — an untracked op
    /// is still registered (see [`detach_params`](crate::utils::detach_params));
    /// the backend hop is what drops the graph.
    ///
    /// Inert off the autodiff backend — see [`Self::cache_to_inner`].
    fn detach(self) -> Self {
        let slots = self
            .into_slots()
            .into_iter()
            .map(|slot| slot.map(|cache| Self::cache_from_inner(Self::cache_to_inner(cache))))
            .collect();
        Self::from_slots(slots)
    }
}

/// A rank-generic binary map over two caches' matching tensors — the visitor
/// [`CacheTensors::zip_tensors`] drives. A map is the zip of a cache with itself.
pub trait TensorZip {
    /// Combine `a` (from the receiving cache) with `b` (the matching tensor of
    /// the other one); the result takes `a`'s place.
    fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D>;
}

/// A cache whose tensors can be visited pairwise: one hand-written traversal per
/// cache type (for the reason [`CacheStack::cache_to_inner`] is hand-written),
/// from which the two operations a captured step needs follow.
///
/// A captured graph replays against the exact device buffers its closure
/// touched, while `step` is functional — it returns fresh cache tensors, so a
/// replayed step would keep reading the capture-time state.
/// [`assign_in_place`](Self::assign_in_place), called on the stable cache at the
/// end of the captured closure, copies the new cache into the stable buffers;
/// `slice_assign` runs in place on a buffer no other tensor shares, so the copies
/// are recorded into the graph and each replay advances the state itself (see
/// [`CapturedStep`](crate::utils::graph::CapturedStep)).
///
/// In place **only while** the destination buffers are unshared, which is what
/// [`into_owned_buffers`](Self::into_owned_buffers) is for: a step's cache may
/// hold views into a shared buffer (a `narrow` of a wider tensor). A shared
/// destination is not a wrong value — `slice_assign` copies — but a replayed
/// graph keeps writing the buffer the capture saw.
pub trait CacheTensors: Clone {
    /// Pair every tensor of `self` with the matching one of `other` through `z`,
    /// keeping `self`'s structure.
    ///
    /// # Panics
    ///
    /// If the two differ in structure (slot count, which optional tensors they
    /// hold, or whatever variants the cache type has).
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self;

    /// The same values, each tensor in a fresh contiguous buffer of its own.
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

    /// `values` written into `self`'s buffers — in place while those are
    /// unshared; the returned cache holds `self`'s buffers.
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

/// No cache: a stateless call (e.g. a fixed-shape `forward`), whose captured
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
