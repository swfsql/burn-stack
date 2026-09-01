//! The per-network cache collection trait.
//!
//! A [`Block`](crate::modules::Block) family supplies its own `Caches` type; all
//! the generic containers need from it is indexed access to one slot per
//! (virtual) layer, plus the backend hop `grad_horizon` performs.

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
    /// [`Module::map`](burn::module::Module::map) is a **no-op on plain `Tensor`
    /// fields** (Burn implements `Module for Tensor` as a constant) and caches
    /// hold bare tensors, not `Param`s — a `Module`-based conversion would
    /// silently skip every one of them.
    ///
    /// # Panics
    /// `Tensor::inner` panics on a tensor that is already off the autodiff
    /// backend, so the caller must have checked
    /// [`Device::is_autodiff`](burn::prelude::Device::is_autodiff) first.
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
    /// # Panics
    /// See [`Self::cache_to_inner`]: the cache must be on an autodiff device.
    fn detach(self) -> Self {
        let slots = self
            .into_slots()
            .into_iter()
            .map(|slot| slot.map(|cache| Self::cache_from_inner(Self::cache_to_inner(cache))))
            .collect();
        Self::from_slots(slots)
    }
}
