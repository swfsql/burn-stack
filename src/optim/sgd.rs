//! Plain SGD that a captured training step can replay.
//!
//! Burn's optimizers do not survive a replay (tracel-ai/burn#5779): the graph
//! bakes their host scalars in — the learning rate, Adam's bias correction — and
//! their update lands back in its input's buffer only by operand order, so a
//! replay can read state that no longer advances. Plain SGD has no state, which
//! leaves the learning rate: [`SgdConfig::step`] takes it as a `[1]` device
//! tensor, an input the host refreshes before each replay, so a schedule keeps
//! moving. The weights themselves are written back by the capture
//! ([`Weights`](crate::utils::graph::Weights)).
//!
//! The step is Burn's `Sgd` without momentum, op for op — clip, decay, scale,
//! subtract — so a captured run trains exactly as [`SgdConfig::init`]'s eager
//! optimizer does.

use burn::grad_clipping::{GradientClipping, GradientClippingConfig};
use burn::module::{ModuleMapper, Param};
use burn::optim::{GradientsParams, ModuleOptimizer, decay::WeightDecayConfig};
use burn::prelude::*;

/// Plain SGD: optional weight decay and gradient clipping, no momentum (see the
/// [module docs](self)).
#[derive(Config, Debug)]
pub struct SgdConfig {
    /// The L2 penalty `λ`, added to the gradient as `grad + λ·w` (Burn's
    /// `WeightDecay`).
    pub weight_decay: Option<f32>,
    /// Gradient clipping, applied to each parameter's gradient first.
    pub grad_clipping: Option<GradientClippingConfig>,
}

impl SgdConfig {
    /// Burn's `Sgd` with these settings: the eager optimizer, and the one a
    /// checkpoint saves (it holds no state).
    pub fn init(&self) -> ModuleOptimizer {
        burn::optim::SgdConfig::new()
            .with_weight_decay(self.weight_decay.map(WeightDecayConfig::new))
            .with_gradient_clipping(self.grad_clipping.clone())
            .init()
    }

    /// One step on `module` at learning rate `lr` (`[1]`, on the gradients'
    /// device, in the parameters' float dtype): every parameter with a gradient
    /// in `grads` becomes `w − lr·g`. Parameters without one are left as they
    /// are. [`init`](Self::init)'s step, op for op, with the rate a tensor.
    pub fn step<M: Module>(&self, module: M, grads: GradientsParams, lr: Tensor<1>) -> M {
        module.map(&mut Step {
            weight_decay: self.weight_decay,
            clipping: self.grad_clipping.as_ref().map(|config| config.init()),
            grads,
            lr,
        })
    }
}

struct Step {
    weight_decay: Option<f32>,
    clipping: Option<GradientClipping>,
    grads: GradientsParams,
    lr: Tensor<1>,
}

impl ModuleMapper for Step {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        let Some(grad) = self.grads.remove::<D>(id) else {
            return Param::from_mapped_value(id, tensor, mapper);
        };
        // What `ModuleOptimizer` does around `Sgd::step`: the update runs on the
        // inner backend, the autodiff state is put back after.
        let autodiff = tensor.is_autodiff();
        let strategy = tensor.gradient_checkpointing_strategy();
        let require_grad = tensor.is_require_grad();
        let w = tensor.inner();
        let grad = match &self.clipping {
            Some(clipping) => clipping.clip_gradient(grad),
            None => grad,
        };
        let grad = match self.weight_decay {
            Some(penalty) => w.clone().mul_scalar(penalty).add(grad),
            None => grad,
        };
        let mut w = w - grad.mul(self.lr.clone().unsqueeze::<D>());
        if autodiff {
            w = Tensor::from_inner(w);
            if let Some(strategy) = strategy {
                w = w.with_gradient_checkpointing_strategy(strategy);
            }
            if require_grad {
                w = w.require_grad();
            }
        }
        Param::from_mapped_value(id, w, mapper)
    }
}
