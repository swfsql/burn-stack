//! Graph capture/replay of a recurrent step ([`burn::tensor::capture`]). One
//! `(input, caches) → (output, caches')` call is recorded once, then replayed
//! as a single dispatch per step.
//!
//! A small model on CUDA is host-bound. A step spends its time on the enqueue
//! of hundreds of tiny launches while the device idles, and a replay collapses
//! them into one. A replay reads and writes the exact device buffers that the
//! capture touched. This shapes [`CapturedStep`]:
//!
//! - The input (one tensor or a tuple of them, a [`StepInput`]) and the caches
//!   live in **stable** buffers that it owns, refreshed in place. An in-place
//!   `slice_assign` from device tensors refreshes the input (or, for one
//!   tensor, a host write into the same pointer). The captured closure
//!   refreshes the caches itself. It ends with
//!   [`CacheTensors::assign_in_place`](crate::modules::CacheTensors::assign_in_place),
//!   so that every replay advances the state.
//! - The closure *runs* before it is recorded: once eagerly, then the warm-ups
//!   of [`capture`](burn::tensor::capture) (and on a backend without graphs,
//!   also the recorded run). So the caches are saved first and restored
//!   after. The first [`step`](CapturedStep::step) continues from the caches
//!   that it was given.
//! - The eager run lets a *cold* capture succeed. The warm-ups of `capture`
//!   hold a second handle on every buffer that they allocate, so none of their
//!   ops runs in place, while the recorded run does. Also, a kernel variant
//!   that first compiles inside the window loads a module mid-capture, which
//!   invalidates the capture. The eager run compiles the variants of the
//!   recorded run first (a workaround:
//!   <https://github.com/tracel-ai/burn/issues/5772>).
//! - A replay is correct only if every write landed in place. The capture
//!   checks this once, with a comparison of buffer ids across the capture.
//!   Where this cannot be confirmed (no hardware graph: flex, ndarray, …, or a
//!   primitive that the check cannot see into: fusion), the graph is dropped
//!   and the step runs eagerly: the same results, without the speed-up.
//!   [`is_captured`](CapturedStep::is_captured) says which.
//!
//! Shapes and the launch sequence are frozen at capture: one batch size, and a
//! step whose host-side control flow does not change between calls (e.g. no
//! class marker left to land). Compile and autotune the kernels before the
//! capture (see [`WARMUP_STEPS`](crate::utils::graph::WARMUP_STEPS)). A step
//! also must not read back to the host. On CUDA, the read fails inside the
//! recording and leaves the stream in capture mode, so every later read in
//! the process fails too.
//!
//! A stateless call (a fixed-shape `forward`) is the step with caches `()`:
//! `|x, ()| (f(x), ())`. A training step is the step whose state is the
//! parameters of the model, [`Weights`]: forward, backward, and an optimizer
//! whose update a replay can advance
//! ([`SgdConfig::step`](crate::optim::SgdConfig::step), with the learning rate
//! as an input).

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

/// Eager steps to run before [`CapturedStep::capture`], in addition to the
/// runs that it makes itself (one eager, then the 3 warm-ups of
/// [`capture`](burn::tensor::capture), all rolled back). These are real steps,
/// and their outputs are used like any other.
///
/// By default, none are necessary. Fusion and autotune get one more warm-up.
/// Their first runs build and tune the fused/tuned variants, which is a
/// different launch sequence from the one that a replay should record. Also,
/// the benchmark buffers of autotune, allocated inside the capture window,
/// would be pinned to the graph.
pub const WARMUP_STEPS: usize = if cfg!(any(feature = "fusion", feature = "dev-autotune")) {
    1
} else {
    0
};

/// The tensor kinds that a [`CapturedStep`] input can be (`Float`, `Int`,
/// `Bool`).
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
/// The tensor kinds that a [`CapturedStep`] input can be (`Float`, `Int`,
/// `Bool`).
#[cfg(not(all(feature = "cubecl", not(feature = "fusion"))))]
pub trait InputKind: Basic {}
#[cfg(not(all(feature = "cubecl", not(feature = "fusion"))))]
impl<K: Basic> InputKind for K {}

/// The cubecl primitive of `t`, when it has one (never under fusion).
#[cfg(all(feature = "cubecl", not(feature = "fusion")))]
fn cube<const D: usize, K: InputKind>(t: &Tensor<D, K>) -> Option<burn_cubecl::tensor::CubeTensor> {
    t.clone().try_into_primitive::<burn_cubecl::Cube>().ok()
}

/// The id of the device buffer of `t`, where it can be read.
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
/// (for example a batch, its targets and a learning rate). Each tensor is held
/// in a stable buffer and refreshed in place before every call.
pub trait StepInput: Clone {
    /// The same values, each tensor in a new buffer of its own.
    fn into_owned(self) -> Self;

    /// `values` written into the buffers of `self`, in place while those
    /// buffers are unshared. The result holds the buffers of `self`.
    fn assign(self, values: Self) -> Self;

    /// Whether `other` has the shapes of `self`.
    fn same_shape(&self, other: &Self) -> bool;

    /// Append the buffer id of every tensor to `ids`. `ids` becomes `None`
    /// when an id cannot be read.
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

/// The buffer ids of the input. `None` if any cannot be read.
fn input_ids<I: StepInput>(input: &I) -> Option<Vec<usize>> {
    let mut ids = Some(Vec::new());
    input.buffer_ids(&mut ids);
    ids
}

/// The buffer ids of the input and of every cache tensor. `None` if any
/// cannot be read.
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

/// A recurrent step, captured once and replayed per call (see the
/// [module docs](self)).
///
/// `I` is the input (e.g. token ids `Tensor<1, Int>`, or a tuple of tensors),
/// `Y` the output of the step, `C` its caches. Not `Send`: every refresh,
/// replay and read must be issued on the stream where the graph was captured,
/// that is, from this thread.
pub struct CapturedStep<'a, I: StepInput + 'a, Y: 'a, C: CacheTensors + 'a> {
    // Declared first, so it drops first. The destruction of the graph waits
    // for its replays, before the buffers below can go.
    graph: Option<Graph<Y, Box<dyn FnMut() -> Y + 'a>>>,
    step: Rc<RefCell<Box<StepFn<'a, I, Y, C>>>>,
    input: Shared<I>,
    caches: Shared<C>,
    /// The output of the last eager step (only without a graph).
    output: Option<Y>,
    device: Device,
}

impl<'a, I: StepInput + 'a, Y: 'a, C: CacheTensors + 'a> CapturedStep<'a, I, Y, C> {
    /// Capture `step` at `input` and `caches` (every later call keeps these
    /// shapes). Nothing is stepped: the first [`step`](Self::step) continues
    /// from `caches`.
    ///
    /// # Safety
    ///
    /// Every tensor that `step` reads other than through its two arguments
    /// (typically the weights of the model) must stay the same device buffer
    /// while the returned value lives. A replay reads the buffers that the
    /// capture saw, and nothing tracks them. A model that `step` borrows for
    /// `'a` satisfies this, because it can be neither dropped nor mutated in
    /// that time. State behind interior mutability that `step` replaces does
    /// not satisfy it.
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
        // Eager, before the warm-ups of `capture` (see the module docs).
        drop(run());
        let graph = burn::tensor::capture(device, run);

        // The closure ran: put back the caches that it was given.
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

    /// Whether `x` has the shapes of the captured input, which every call
    /// keeps.
    pub fn accepts(&self, x: &I) -> bool {
        self.input.borrow().as_ref().expect("the input is always present").same_shape(x)
    }

    /// Step on the device tensor(s) `x`, which must have the shapes of the
    /// captured input. On a captured graph, the next call overwrites the
    /// output: read or copy it first.
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

    /// A copy of the current caches, in buffers of its own (the stable buffers
    /// stay with the graph).
    pub fn caches(&self) -> C {
        self.caches
            .borrow()
            .as_ref()
            .expect("caches are always put back")
            .clone()
            .into_owned_buffers()
    }

    /// Overwrite the current caches with `caches` (e.g. a new opening), in
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

    /// Release the graph and give back the current caches.
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
        // Safety: the graph touches three sets of buffers:
        // - the stable input and caches, held by `self` and by the closure
        //   that the graph owns, and only written in place (checked at capture
        //   and at every refresh),
        // - the buffers that `step` borrows for `'a` (the contract of
        //   `capture`),
        // - the buffers that the graph retains itself.
        // `self` is not `Send`, so every refresh, replay and read comes from
        // this thread, on the stream of the capture.
        unsafe { graph.replay() }
    }
}

/// A step on one tensor, which the host can also feed directly.
impl<'a, const D: usize, K: InputKind + 'a, Y: 'a, C: CacheTensors + 'a>
    CapturedStep<'a, Tensor<D, K>, Y, C>
{
    /// The shape of the captured input, which every call keeps.
    pub fn input_dims(&self) -> [usize; D] {
        self.input.borrow().as_ref().expect("the input is always present").dims()
    }

    /// [`step`](Self::step) on host data (converted to the dtype of the
    /// input). On a captured graph, this is a host write straight into the
    /// buffer of the input.
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
                // A graph is kept only after the id of the input was read, so
                // the input has one.
                let c = cube(stable).expect("a captured input is a cubecl tensor");
                c.client.write(&c.handle, data.into_bytes());
            }
            self.replay()
        }
        #[cfg(not(all(feature = "cubecl", not(feature = "fusion"))))]
        unreachable!("a graph is only kept where buffer ids can be read: {data:?}")
    }
}
