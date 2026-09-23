//! Plain SGD that a captured training step can replay.
//!
//! The optimizers of Burn do not survive a replay (tracel-ai/burn#5779). The
//! graph bakes in their host scalars (the learning rate, the bias correction
//! of Adam). Also, their update lands back in the buffer of its input only by
//! operand order, so a replay can read state that stops advancing. Plain SGD
//! has no state, which leaves the learning rate. [`SgdConfig::step`] takes it
//! as a `[1]` device tensor: an input that the host refreshes before each
//! replay, so a schedule keeps moving. The capture writes back the weights
//! themselves ([`Weights`](crate::utils::graph::Weights)).
//!
//! The step is the `Sgd` of Burn without momentum, op for op (clip, decay,
//! scale, subtract). So a captured run trains exactly as the eager optimizer
//! of [`SgdConfig::init`] does.

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
    /// Gradient clipping, applied first to the gradient of each parameter.
    pub grad_clipping: Option<GradientClippingConfig>,
}

impl SgdConfig {
    /// The `Sgd` of Burn with these settings: the eager optimizer, and the one
    /// that a checkpoint saves (it holds no state).
    pub fn init(&self) -> ModuleOptimizer {
        self.burn().init()
    }

    /// The bare optimizer that [`init`](Self::init) wraps, for one parameter
    /// group (or one [`Segmented`](super::Segmented) block): no gradient
    /// clipping, no state.
    pub fn build(&self) -> burn::optim::Sgd {
        self.burn().build()
    }

    fn burn(&self) -> burn::optim::SgdConfig {
        burn::optim::SgdConfig::new()
            .with_weight_decay(self.weight_decay.map(WeightDecayConfig::new))
            .with_gradient_clipping(self.grad_clipping.clone())
    }

    /// One step on `module` at learning rate `lr` (`[1]`, on the device of the
    /// gradients, in the float dtype of the parameters). Every parameter with
    /// a gradient in `grads` becomes `w − lr·g`. Parameters without one stay as
    /// they are. This is the step of [`init`](Self::init), op for op, with the
    /// rate as a tensor.
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
        // inner backend, and the autodiff state is put back after it.
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
