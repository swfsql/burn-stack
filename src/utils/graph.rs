//! Graph capture/replay of a recurrent step ([`burn::tensor::capture`]): one
//! `(input, caches) → (output, caches')` call recorded once, then replayed as a
//! single dispatch per step.
//!
//! A small model on CUDA is host-bound: a step spends its time enqueueing
//! hundreds of tiny launches while the device idles, and a replay collapses them
//! into one. A replay reads and writes the exact device buffers the capture
//! touched, which is what shapes [`CapturedStep`]:
//!
//! - the input — one tensor or a tuple of them, a [`StepInput`] — and the caches
//!   live in **stable** buffers it owns, refreshed in place: the input by an
//!   in-place `slice_assign` from device tensors (or, for one tensor, a host
//!   write into the same pointer), the caches by the captured closure itself,
//!   which ends with
//!   [`CacheTensors::assign_in_place`](crate::modules::CacheTensors::assign_in_place) so that every
//!   replay advances the state;
//! - the closure *runs* before it is recorded — once eagerly, then
//!   [`capture`](burn::tensor::capture)'s warm-ups (and on a backend without
//!   graphs the recorded run too) — so the caches are snapshotted first and
//!   restored after: the first [`step`](CapturedStep::step) continues from the
//!   caches it was given;
//! - the eager run is what lets a *cold* capture succeed: `capture`'s warm-ups
//!   hold a second handle on every buffer they allocate, so none of their ops
//!   runs in place, while the recorded run does — and a kernel variant first
//!   compiled inside the window loads a module mid-capture, which invalidates
//!   it. The eager run compiles the recorded run's variants first (a workaround:
//!   <https://github.com/tracel-ai/burn/issues/5772>);
//! - a replay is correct only if every write landed in place, which is checked
//!   once, by comparing buffer ids across the capture. Where that cannot be
//!   confirmed — no hardware graph (flex, ndarray, …), or a primitive the check
//!   cannot see into (fusion) — the graph is dropped and the step runs eagerly:
//!   the same results, without the speed-up.
//!   [`is_captured`](CapturedStep::is_captured) says which.
//!
//! Shapes and the launch sequence are frozen at capture: one batch size, and a
//! step whose host-side control flow does not change between calls (e.g. no
//! class marker left to land). Kernels should be compiled and autotuned before
//! it — see [`WARMUP_STEPS`](crate::utils::graph::WARMUP_STEPS). Nor may a step
//! read back to the host: on CUDA the read fails inside the recording and
//! leaves the stream capturing, so every later read in the process fails too.
//!
//! A stateless call — a fixed-shape `forward` — is the step with caches `()`:
//! `|x, ()| (f(x), ())`. A training step is the step whose state is the model's
//! own parameters, [`Weights`]: forward, backward and an optimizer whose update a
//! replay can advance — [`SgdConfig::step`](crate::optim::SgdConfig::step), the
//! learning rate an input.

#[cfg(test)]
mod tests;
#[cfg(feature = "autodiff")]
mod weights;

#[cfg(feature = "autodiff")]
pub use weights::Weights;

use crate::modules::{CacheTensors, TensorZip};
use burn::prelude::*;
use burn::tensor::{Graph, TensorData, kind::Basic};
use std::cell::RefCell;
use std::rc::Rc;

/// Eager steps to run before [`CapturedStep::capture`], on top of the runs it
/// makes itself (one eager, then the 3 warm-ups of
/// [`capture`](burn::tensor::capture), all rolled back) — real steps, whose
/// outputs are used like any other.
///
/// None are needed by default. Fusion and autotune get one more warm-up: their
/// first runs build and tune the fused/tuned variants — a different launch
/// sequence from the one a replay should record — and autotune's benchmark
/// buffers, allocated inside the capture's window, would be pinned to the graph.
pub const WARMUP_STEPS: usize = if cfg!(any(feature = "fusion", feature = "dev-autotune")) {
    1
} else {
    0
};

/// The tensor kinds a [`CapturedStep`] input may be (`Float`, `Int`, `Bool`).
#[cfg(all(feature = "cubecl", not(feature = "fusion")))]
pub trait InputKind:
    Basic
    + burn::tensor::BackendPrimitive<burn_cubecl::Cube, Primitive = burn_cubecl::tensor::CubeTensor>
{
}
#[cfg(all(feature = "cubecl", not(feature = "fusion")))]
impl<K> InputKind for K where
    K: Basic
        + burn::tensor::BackendPrimitive<
            burn_cubecl::Cube,
            Primitive = burn_cubecl::tensor::CubeTensor,
        >
{
}
/// The tensor kinds a [`CapturedStep`] input may be (`Float`, `Int`, `Bool`).
#[cfg(not(all(feature = "cubecl", not(feature = "fusion"))))]
pub trait InputKind: Basic {}
#[cfg(not(all(feature = "cubecl", not(feature = "fusion"))))]
impl<K: Basic> InputKind for K {}

/// `t`'s cubecl primitive, when it has one (never under fusion).
#[cfg(all(feature = "cubecl", not(feature = "fusion")))]
fn cube<const D: usize, K: InputKind>(t: &Tensor<D, K>) -> Option<burn_cubecl::tensor::CubeTensor> {
    t.clone().try_into_primitive::<burn_cubecl::Cube>().ok()
}

/// The id of `t`'s device buffer, where one can be read.
fn buffer_id<const D: usize, K: InputKind>(t: &Tensor<D, K>) -> Option<usize> {
    #[cfg(all(feature = "cubecl", not(feature = "fusion")))]
    return cube(t).map(|c| c.handle.memory.descriptor().id.value);
    #[cfg(not(all(feature = "cubecl", not(feature = "fusion"))))]
    {
        let _ = t;
        None
    }
}

/// What a [`CapturedStep`] takes as its input: one tensor, or a tuple of them
/// (a batch, its targets and a learning rate, say), each held in a stable buffer
/// and refreshed in place before every call.
pub trait StepInput: Clone {
    /// The same values, each tensor in a fresh buffer of its own.
    fn into_owned(self) -> Self;

    /// `values` written into `self`'s buffers — in place while those are
    /// unshared; the result holds `self`'s buffers.
    fn assign(self, values: Self) -> Self;

    /// Whether `other` has `self`'s shapes.
    fn same_shape(&self, other: &Self) -> bool;

    /// Append every tensor's buffer id to `ids`, which turns `None` once one
    /// cannot be read.
    fn buffer_ids(&self, ids: &mut Option<Vec<usize>>);
}

impl<const D: usize, K: InputKind> StepInput for Tensor<D, K> {
    fn into_owned(self) -> Self {
        let range = whole(&self);
        self.empty_like().slice_assign(range, self)
    }

    fn assign(self, values: Self) -> Self {
        let range = whole(&self);
        self.slice_assign(range, values)
    }

    fn same_shape(&self, other: &Self) -> bool {
        self.dims() == other.dims()
    }

    fn buffer_ids(&self, ids: &mut Option<Vec<usize>>) {
        *ids = ids.take().zip(buffer_id(self)).map(|(mut ids, id)| {
            ids.push(id);
            ids
        });
    }
}

impl<A: StepInput, B: StepInput> StepInput for (A, B) {
    fn into_owned(self) -> Self {
        (self.0.into_owned(), self.1.into_owned())
    }

    fn assign(self, values: Self) -> Self {
        (self.0.assign(values.0), self.1.assign(values.1))
    }

    fn same_shape(&self, other: &Self) -> bool {
        self.0.same_shape(&other.0) && self.1.same_shape(&other.1)
    }

    fn buffer_ids(&self, ids: &mut Option<Vec<usize>>) {
        self.0.buffer_ids(ids);
        self.1.buffer_ids(ids);
    }
}

impl<A: StepInput, B: StepInput, C: StepInput> StepInput for (A, B, C) {
    fn into_owned(self) -> Self {
        (self.0.into_owned(), self.1.into_owned(), self.2.into_owned())
    }

    fn assign(self, values: Self) -> Self {
        (
            self.0.assign(values.0),
            self.1.assign(values.1),
            self.2.assign(values.2),
        )
    }

    fn same_shape(&self, other: &Self) -> bool {
        self.0.same_shape(&other.0) && self.1.same_shape(&other.1) && self.2.same_shape(&other.2)
    }

    fn buffer_ids(&self, ids: &mut Option<Vec<usize>>) {
        self.0.buffer_ids(ids);
        self.1.buffer_ids(ids);
        self.2.buffer_ids(ids);
    }
}

/// The input's buffer ids — `None` if any cannot be read.
fn input_ids<I: StepInput>(input: &I) -> Option<Vec<usize>> {
    let mut ids = Some(Vec::new());
    input.buffer_ids(&mut ids);
    ids
}

/// The buffer ids of the input and every cache tensor — `None` if any cannot
/// be read.
fn stable_ids<I: StepInput, C: CacheTensors>(input: &I, caches: &C) -> Option<Vec<usize>> {
    struct Ids(Option<Vec<usize>>);
    impl TensorZip for Ids {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            drop(b);
            let id = buffer_id(&a);
            self.0 = self.0.take().zip(id).map(|(mut ids, id)| {
                ids.push(id);
                ids
            });
            a
        }
    }
    let mut ids = Ids(input_ids(input));
    let _ = caches.clone().zip_tensors(caches.clone(), &mut ids);
    ids.0
}

/// The whole-tensor slice of `t`.
fn whole<const D: usize, K: Basic>(t: &Tensor<D, K>) -> [core::ops::Range<usize>; D] {
    t.dims().map(|d| 0..d)
}

type StepFn<'a, I, Y, C> = dyn FnMut(I, C) -> (Y, C) + 'a;
type Shared<T> = Rc<RefCell<Option<T>>>;

/// A recurrent step, captured once and replayed per call — see the
/// [module docs](self).
///
/// `I` is the input (e.g. token ids `Tensor<1, Int>`, or a tuple of tensors),
/// `Y` the step's output, `C` its caches. Not `Send`: every refresh, replay and
/// read has to be issued on the stream the graph was captured on, i.e. from
/// this thread.
pub struct CapturedStep<'a, I: StepInput + 'a, Y: 'a, C: CacheTensors + 'a> {
    // Declared first, so it drops first: destroying the graph waits for its
    // replays before the buffers below can go.
    graph: Option<Graph<Y, Box<dyn FnMut() -> Y + 'a>>>,
    step: Rc<RefCell<Box<StepFn<'a, I, Y, C>>>>,
    input: Shared<I>,
    caches: Shared<C>,
    /// The last eager step's output (only without a graph).
    output: Option<Y>,
    device: Device,
}

impl<'a, I: StepInput + 'a, Y: 'a, C: CacheTensors + 'a> CapturedStep<'a, I, Y, C> {
    /// Capture `step` at `input` and `caches` (the shapes every later call
    /// keeps). Nothing is stepped: the first [`step`](Self::step) continues
    /// from `caches`.
    ///
    /// # Safety
    ///
    /// Every tensor `step` reads other than through its two arguments — the
    /// model's weights, typically — must stay the same device buffer while the
    /// returned value lives: a replay reads the buffers the capture saw, with
    /// nothing tracking them. A model `step` borrows for `'a` satisfies this (it
    /// can be neither dropped nor mutated meanwhile); state behind interior
    /// mutability that `step` replaces does not.
    pub unsafe fn capture(
        device: &Device,
        input: I,
        caches: C,
        step: impl FnMut(I, C) -> (Y, C) + 'a,
    ) -> Self {
        let input = input.into_owned();
        let caches = caches.into_owned_buffers();
        let snapshot = caches.clone().into_owned_buffers();
        let ids = stable_ids(&input, &caches);

        let step: Rc<RefCell<Box<StepFn<'a, I, Y, C>>>> = Rc::new(RefCell::new(Box::new(step)));
        let input = Rc::new(RefCell::new(Some(input)));
        let caches = Rc::new(RefCell::new(Some(caches)));
        let mut run: Box<dyn FnMut() -> Y + 'a> = {
            let (step, input, caches) = (step.clone(), input.clone(), caches.clone());
            Box::new(move || {
                let stable = caches.borrow_mut().take().expect("caches are always put back");
                let x = input.borrow().as_ref().expect("the input is always present").clone();
                let (y, new) = (&mut *step.borrow_mut())(x, stable.clone());
                *caches.borrow_mut() = Some(stable.assign_in_place(new));
                y
            })
        };
        // Eager, before `capture`'s warm-ups (see the module docs).
        drop(run());
        let graph = burn::tensor::capture(device, run);

        // The closure ran: put the caches it was given back.
        {
            let mut slot = caches.borrow_mut();
            let stable = slot.take().expect("caches are always put back");
            *slot = Some(stable.assign_in_place(snapshot));
        }
        let kept = ids.is_some()
            && ids
                == stable_ids(
                    input.borrow().as_ref().expect("the input is always present"),
                    caches.borrow().as_ref().expect("caches are always put back"),
                );
        Self {
            graph: (graph.is_hardware() && kept).then_some(graph),
            step,
            input,
            caches,
            output: None,
            device: device.clone(),
        }
    }

    /// Whether calls replay a hardware graph (`false`: they step eagerly, see
    /// the [module docs](self)).
    pub fn is_captured(&self) -> bool {
        self.graph.is_some()
    }

    /// Whether `x` has the captured input's shapes, which every call keeps.
    pub fn accepts(&self, x: &I) -> bool {
        self.input.borrow().as_ref().expect("the input is always present").same_shape(x)
    }

    /// Step on the device tensor(s) `x`, which must have the captured input's
    /// shapes. The output is overwritten by the next call on a captured graph:
    /// read or copy it first.
    pub fn step(&mut self, x: I) -> &Y {
        if self.graph.is_none() {
            return self.eager(x);
        }
        {
            let mut slot = self.input.borrow_mut();
            let stable = slot.take().expect("the input is always present");
            assert!(stable.same_shape(&x), "a captured step keeps its input shape");
            let before = input_ids(&stable);
            let stable = stable.assign(x);
            assert_eq!(input_ids(&stable), before, "the input refresh must land in place");
            *slot = Some(stable);
        }
        self.replay()
    }

    /// A copy of the current caches, in buffers of its own (the stable ones stay
    /// the graph's).
    pub fn caches(&self) -> C {
        self.caches
            .borrow()
            .as_ref()
            .expect("caches are always put back")
            .clone()
            .into_owned_buffers()
    }

    /// Overwrite the current caches with `caches` (e.g. a fresh opening), in
    /// place.
    pub fn set_caches(&mut self, caches: C) {
        let mut slot = self.caches.borrow_mut();
        let stable = slot.take().expect("caches are always put back");
        let input = self.input.borrow();
        let input = input.as_ref().expect("the input is always present");
        let before = self.graph.as_ref().and(stable_ids(input, &stable));
        let stable = stable.assign_in_place(caches);
        if before.is_some() {
            assert_eq!(stable_ids(input, &stable), before, "the caches must be set in place");
        }
        *slot = Some(stable);
    }

    /// Release the graph and hand the current caches back.
    pub fn into_caches(mut self) -> C {
        self.graph = None;
        self.caches.borrow_mut().take().expect("caches are always put back")
    }

    fn eager(&mut self, x: I) -> &Y {
        let caches = self.caches.borrow_mut().take().expect("caches are always put back");
        let (y, caches) = (&mut *self.step.borrow_mut())(x, caches);
        *self.caches.borrow_mut() = Some(caches);
        self.output.insert(y)
    }

    fn replay(&mut self) -> &Y {
        let graph = self.graph.as_mut().expect("only called with a graph");
        // Safety: the buffers the graph touches are the stable input and caches
        // (held by `self` and by the closure the graph owns, and only ever
        // written in place — checked at capture and at every refresh), those
        // `step` borrows for `'a` (the contract of `capture`), and those the
        // graph retains itself. `self` is not `Send`, so every refresh, replay
        // and read is issued from this thread, on the capture's stream.
        unsafe { graph.replay() }
    }
}

/// A step on one tensor, which the host can also feed directly.
impl<'a, const D: usize, K: InputKind + 'a, Y: 'a, C: CacheTensors + 'a>
    CapturedStep<'a, Tensor<D, K>, Y, C>
{
    /// The captured input's shape, which every call keeps.
    pub fn input_dims(&self) -> [usize; D] {
        self.input.borrow().as_ref().expect("the input is always present").dims()
    }

    /// [`step`](Self::step) on host data (converted to the input's dtype): on a
    /// captured graph a host write straight into the input's buffer.
    pub fn step_data(&mut self, data: TensorData) -> &Y {
        let dtype = self.input.borrow().as_ref().expect("the input is always present").dtype();
        let data = data.convert_dtype(dtype);
        if self.graph.is_none() {
            let x = Tensor::from_data(data, &self.device);
            return self.eager(x);
        }
        #[cfg(all(feature = "cubecl", not(feature = "fusion")))]
        {
            {
                let slot = self.input.borrow();
                let stable = slot.as_ref().expect("the input is always present");
                assert_eq!(stable.shape(), data.shape, "a captured step keeps its input shape");
                // A graph is only kept once the input's id was read, so it has one.
                let c = cube(stable).expect("a captured input is a cubecl tensor");
                c.client.write(&c.handle, data.into_bytes());
            }
            self.replay()
        }
        #[cfg(not(all(feature = "cubecl", not(feature = "fusion"))))]
        unreachable!("a graph is only kept where buffer ids can be read: {data:?}")
    }
}
