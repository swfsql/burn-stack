//! The per-network cache collection trait, the pairwise tensor traversal of a
//! cache, and the backend hop that follows from that traversal.
//!
//! A family supplies its own `Caches` type. The generic containers need only
//! indexed access to one cache slot per virtual layer ([`CacheStack`]). Each
//! slot implements [`CacheTensors`]: every tensor of the cache, visited
//! pairwise with the matching tensor of another cache. This traversal is the
//! one hand-written list of the fields of a cache. These operations follow
//! from it:
//!
//! - the backend hop of `grad_horizon`: [`cache_to_inner`] and
//!   [`cache_from_inner`],
//! - the detach of a cache, the round trip through that hop onto the device
//!   of the cache: [`cache_detach`] and [`cache_device`],
//! - the cache write-back of a
//!   [`CapturedStep`](crate::utils::capture::CapturedStep),
//! - the per-row select, gather and merge of [`HNet::step`].

use burn::prelude::*;

/// `tensor` lifted onto the autodiff backend of `device`, as a new graph root
/// with the gradient-checkpointing strategy of `device`. On a device without
/// autodiff, `tensor` unchanged. This is the way back from `Tensor::inner`,
/// and like `inner`, it does nothing off the autodiff backend.
///
/// Use it, not a bare `Tensor::from_inner` (an alias of `Tensor::autodiff`).
/// `from_inner` enables autodiff also on a plain tensor, and it always sets
/// the `Disabled` strategy. An op on two tensors with different strategies
/// panics. So on a device with gradient checkpointing, the first op that mixes
/// such a tensor with the other tensors of the pass fails.
pub fn lift<const D: usize>(tensor: Tensor<D>, device: &Device) -> Tensor<D> {
    #[cfg(feature = "autodiff")]
    if let Some(strategy) = device.gradient_checkpointing_strategy() {
        return Tensor::from_inner(tensor).with_gradient_checkpointing_strategy(strategy);
    }
    #[cfg(not(feature = "autodiff"))]
    let _ = device;
    tensor
}

/// The uniform interface of a per-network cache collection for the generic
/// [`Layers`](crate::modules::Layers) loop: `slot_count`, plus move-in and
/// move-out of the per-layer slots.
pub trait CacheStack: Sized {
    /// The per-layer cache element. Its traversal ([`CacheTensors`]) gives
    /// the backend hop of [`Self::detach`] and of
    /// [`Layers::grad_horizon`](crate::modules::Layers::grad_horizon).
    type Cache: CacheTensors;
    /// Number of slots: one per virtual layer.
    fn slot_count(&self) -> usize;
    /// Move each slot into an `Option`, so the loop can `take` without a clone.
    fn into_slots(self) -> Vec<Option<Self::Cache>>;
    /// Inverse of [`Self::into_slots`].
    fn from_slots(slots: Vec<Option<Self::Cache>>) -> Self;

    /// Detach **every** slot ([`cache_detach`]). The cache keeps its values
    /// and loses the graph that produced it. So a caller can carry the state
    /// into the next window without the activations of the previous window.
    ///
    /// Each slot goes back onto its own device ([`cache_device`]), with the
    /// checkpointing strategy of that device. On a device without autodiff,
    /// this does nothing.
    fn detach(self) -> Self {
        let slots = self.into_slots().into_iter().map(|slot| slot.map(cache_detach)).collect();
        Self::from_slots(slots)
    }
}

/// `cache` cut off the autodiff graph: the round trip [`cache_to_inner`],
/// then [`cache_from_inner`] onto the device of `cache` ([`cache_device`]).
/// The cache keeps its values and loses the graph that produced it. Each
/// tensor gets the checkpointing strategy of that device. On a device without
/// autodiff, the cache stays unchanged.
///
/// `Tensor::detach` alone would cut the gradients but free nothing, because
/// Burn still registers an untracked op (see
/// [`detach_params`](crate::utils::detach_params)). The backend hop drops the
/// graph.
///
/// The detach follows from [`CacheTensors::zip_tensors`]. So a cache type
/// cannot override it, and its fields stay in one list.
///
/// The rule: all the tensors of `cache` are on one device. Each tensor goes
/// onto the device of the first tensor.
///
/// # Panics
///
/// If `cache` holds no tensor (see [`cache_device`]).
pub fn cache_detach<C: CacheTensors>(cache: C) -> C {
    let device = cache_device(&cache);
    cache_from_inner(cache_to_inner(cache), &device)
}

/// `cache` moved **to** the inner (non-autodiff) backend: each tensor goes
/// through `Tensor::inner`.
///
/// [`Layers::grad_horizon`](crate::modules::Layers::grad_horizon) needs it,
/// because its untracked spans run on the inner backend. A tracked cache must
/// go to the inner backend with them, for two reasons. The span must build no
/// graph, and the dispatch of Burn cannot mix backends within one op.
///
/// `Tensor::inner` (an alias of `Tensor::without_autodiff`) returns a tensor
/// that is already off the autodiff backend unchanged. So this does nothing
/// there, and is not an error. [`cache_from_inner`] also does nothing there.
///
/// The hop follows from [`CacheTensors::zip_tensors`]. So a cache type cannot
/// override it, and its fields stay in one list. The map is the zip of `cache`
/// with its clone: one refcount per tensor, and no tensor op.
pub fn cache_to_inner<C: CacheTensors>(cache: C) -> C {
    struct ToInner;
    impl TensorZip for ToInner {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            drop(b);
            a.inner()
        }
    }
    cache.clone().zip_tensors(cache, &mut ToInner)
}

/// `cache` lifted back **from** the inner backend onto the autodiff backend of
/// `device`, as a new graph root: the inverse of [`cache_to_inner`]. Each
/// tensor goes through [`lift`], not through a bare `Tensor::from_inner` (see
/// [`lift`]). So the cache gets the checkpointing strategy of `device`. On a
/// device without autodiff, the cache stays unchanged.
pub fn cache_from_inner<C: CacheTensors>(cache: C, device: &Device) -> C {
    struct Lift<'a>(&'a Device);
    impl TensorZip for Lift<'_> {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            drop(b);
            lift(a, self.0)
        }
    }
    cache.clone().zip_tensors(cache, &mut Lift(device))
}

/// The device of `cache`: the device of its first tensor, with its autodiff
/// context. [`cache_detach`] lifts the cache back onto it.
///
/// The rule: a per-layer cache ([`CacheStack::Cache`]) holds at least one
/// tensor. Each other cache that goes through [`cache_detach`] also holds at
/// least one tensor.
///
/// # Panics
///
/// If `cache` holds no tensor (for example, `()`).
pub fn cache_device<C: CacheTensors>(cache: &C) -> Device {
    struct First(Option<Device>);
    impl TensorZip for First {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            drop(b);
            self.0.get_or_insert_with(|| a.device());
            a
        }
    }
    let mut first = First(None);
    let _ = cache.clone().zip_tensors(cache.clone(), &mut first);
    first
        .0
        .expect("the cache holds no tensor, so it has no device. A per-layer cache holds at least one tensor.")
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
/// per cache type. This traversal is the one list of the fields of a cache.
/// The backend hop ([`cache_to_inner`], [`cache_from_inner`],
/// [`cache_device`]), the detach ([`cache_detach`]) and the two operations
/// that a captured step needs follow from it.
///
/// The traversal is hand-written, not derived from `Module`, because
/// [`Module::map`] is a **no-op on plain `Tensor` fields** (Burn implements
/// `Module for Tensor` as a constant). Caches hold bare tensors, not
/// `Param`s. So a `Module`-based traversal would silently skip every one of
/// them.
///
/// A captured step replays against the exact device buffers that its closure
/// touched. But `step` is functional: it returns new cache tensors, so a
/// replay would keep reading the capture-time state. So the captured closure
/// ends with a call of [`assign_in_place`](Self::assign_in_place) on the
/// stable cache, which copies the new cache into the stable buffers.
/// `slice_assign` runs in place on a buffer that no other tensor shares. So
/// the captured step records the copies, and each replay advances the state
/// itself (see [`CapturedStep`](crate::utils::capture::CapturedStep)).
///
/// It is in place **only while** the destination buffers are unshared. This
/// is what [`into_owned_buffers`](Self::into_owned_buffers) is for: the cache
/// of a step can hold views into a shared buffer (a `narrow` of a wider
/// tensor). A shared destination does not give a wrong value (`slice_assign`
/// copies). But each replay keeps writing the buffer that the capture saw.
///
/// **Axis 0 of every tensor is the batch.** A container that runs a network on
/// some rows only (the `step` of [`HNet`](crate::modules::HNet)) selects,
/// gathers and merges the rows of a cache through this traversal. So a cache
/// must hold no tensor that is shared by the rows of a batch. The two caches
/// that a traversal pairs can then also differ in batch size.
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

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
