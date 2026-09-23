//! [`Trainer`]: an example's whole training step — forward, backward and the
//! optimizer's update — replayed from a captured graph under plain SGD, and
//! stepped eagerly under any other optimizer.
//!
//! The step is one [`LossFn`] either way: the module's loss on a batch, and the
//! outputs the metrics read. Eagerly, its gradients go to the
//! [`ModuleOptimizer`] each call is handed. Under plain SGD (unless
//! `--no-graph`) the weights move into a [`CapturedStep`] at the first call:
//! forward, backward and [`SgdConfig::step`], captured at that batch's shapes
//! and replayed for every later batch of them — a batch of any other shape (a
//! short last one) steps eagerly, into the same weights. The capture's own runs
//! are rolled back and the learning rate is an input, so it trains exactly as
//! the eager steps would. No other optimizer can be replayed
//! (tracel-ai/burn#5779), Muon + SGD included, so they step eagerly.
//!
//! A replay freezes the launch sequence, so the loss must keep the batch's
//! shapes: a mask, not a gather of the positions it scores, and no read back to
//! the host.

#[cfg(test)]
mod tests;

use crate::examples::training::OptimizerConfig;
use crate::optim::SgdConfig;
use crate::utils::{CapturedStep, StepInput, Weights};
use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::prelude::*;

/// A training step's forward: `module`'s loss on one batch, and the outputs the
/// metrics read.
///
/// The batch arrives on the inner backend (lift it with `.autodiff()`); the
/// loss is the tracked `[1]` mean the gradients come from, and the outputs go
/// back to the inner backend (`.inner()`).
pub type LossFn<M, I, Y> = fn(&M, I) -> (Tensor<1>, Y);

/// A module under training, one [`step`](Self::step) per batch — see the
/// [module docs](self).
///
/// `M` is the module (on the autodiff backend), `I` a batch (one tensor or a
/// tuple, on the inner backend), `Y` the outputs its [`LossFn`] returns beside
/// the loss. Not `Send`, like the [`CapturedStep`] inside.
pub struct Trainer<M, I, Y>
where
    M: Module + Clone + 'static,
    I: StepInput + 'static,
    Y: StepInput + 'static,
{
    /// `None` only inside [`step`](Self::step).
    state: Option<State<M, I, Y>>,
    loss: LossFn<M, I, Y>,
    /// The SGD a captured step replays; `None` ⇒ eager steps through the
    /// module optimizer.
    capture: Option<SgdConfig>,
    /// The inner device the weights live on.
    device: Device,
}

/// Where a [`Trainer`]'s weights are.
enum State<M, I, Y>
where
    M: Module + Clone + 'static,
    I: StepInput + 'static,
    Y: StepInput + 'static,
{
    Eager(M),
    Captured(Captured<M, I, Y>),
}

/// The captured step: `(batch, lr)` → `(loss, outputs)`, the weights its state.
type Captured<M, I, Y> = CapturedStep<'static, (I, Tensor<1>), (Tensor<1>, Y), Weights<M>>;

impl<M, I, Y> Trainer<M, I, Y>
where
    M: Module + Clone + 'static,
    I: StepInput + 'static,
    Y: StepInput + 'static,
{
    /// `module`, trained through `loss` under `optimizer`: from a captured
    /// graph if that is plain SGD and `graphs` (the examples'
    /// [`AppArgs::graphs`](crate::examples::cli::AppArgs::graphs)), else
    /// eagerly.
    pub fn new(module: M, loss: LossFn<M, I, Y>, optimizer: &OptimizerConfig, graphs: bool) -> Self {
        let device = module.devices().into_iter().next().expect("a module with parameters");
        Self {
            state: Some(State::Eager(module)),
            loss,
            capture: graphs.then(|| optimizer.plain_sgd().cloned()).flatten(),
            device: device.inner(),
        }
    }

    /// One training step on `batch` (inner backend) at learning rate `lr`: its
    /// loss and outputs, on the inner backend, in buffers of their own. `optim`
    /// updates the weights of an eager step; a captured one runs its own SGD.
    pub fn step(&mut self, batch: I, optim: &mut ModuleOptimizer, lr: f64) -> (Tensor<1>, Y) {
        let state = self.state.take().expect("a step leaves its state behind");
        let (state, output) = match (state, self.capture.clone()) {
            (State::Eager(module), None) => {
                let (loss, y) = (self.loss)(&module, batch);
                let grads = GradientsParams::from_grads(loss.backward(), &module);
                (State::Eager(optim.step(lr, module, grads)), (loss.inner(), y))
            }
            (state, Some(sgd)) => {
                let input = (batch, Tensor::<1>::from_floats([lr], &self.device));
                let mut captured = match state {
                    State::Captured(captured) => captured,
                    State::Eager(module) => {
                        let (loss, sgd) = (self.loss, sgd.clone());
                        // Safety: the step reads nothing but its arguments and
                        // `sgd`, which it owns — `loss` is a `fn`, capturing
                        // nothing.
                        unsafe {
                            CapturedStep::capture(&self.device, input.clone(), Weights(module), move |x, w| {
                                sgd_step(&sgd, loss, x, w)
                            })
                        }
                    }
                };
                let output = if captured.accepts(&input) {
                    // Copied out of the graph's output buffers, which the next
                    // replay overwrites.
                    captured.step(input).clone().into_owned()
                } else {
                    let (output, weights) = sgd_step(&sgd, self.loss, input, captured.caches());
                    captured.set_caches(weights);
                    output
                };
                (State::Captured(captured), output)
            }
            (State::Captured(_), None) => unreachable!("only plain SGD is captured"),
        };
        self.state = Some(state);
        output
    }

    /// The current weights (a copy, when a graph holds them).
    pub fn module(&self) -> M {
        match self.state.as_ref().expect("a step leaves its state behind") {
            State::Eager(module) => module.clone(),
            State::Captured(captured) => captured.caches().0,
        }
    }

    /// The inner device the weights live on, which a batch must be on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether steps replay a hardware graph: plain SGD, graphs on, a first
    /// step taken, and a backend that keeps graphs (see
    /// [`CapturedStep::is_captured`]).
    pub fn is_captured(&self) -> bool {
        matches!(&self.state, Some(State::Captured(captured)) if captured.is_captured())
    }
}

/// One SGD step of `loss` on `batch` at the rate `lr` (inner backend): the loss
/// and outputs, and the updated weights.
fn sgd_step<M: Module + Clone, I, Y>(
    sgd: &SgdConfig,
    loss: LossFn<M, I, Y>,
    (batch, lr): (I, Tensor<1>),
    weights: Weights<M>,
) -> ((Tensor<1>, Y), Weights<M>) {
    let module = weights.0;
    let (loss, y) = loss(&module, batch);
    let grads = GradientsParams::from_grads(loss.backward(), &module);
    ((loss.inner(), y), Weights(sgd.step(module, grads, lr)))
}
