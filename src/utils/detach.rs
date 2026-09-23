//! Detach the parameters of a module from the autodiff graph.
//!
//! This cuts the gradient flow through a module. **It does not save memory**
//! (see the warning below).
//! [`Layers::grad_horizon`](crate::modules::Layers::grad_horizon) moves its
//! untracked segments to the inner backend instead.
//!
//! # Why a detach of the *activations* is not sufficient
//!
//! Burn tracks gradients **per tensor**, not ambiently: there is no
//! `torch.no_grad()`. An op is untracked only when *every* input node has
//! `Requirement::None`. So `layer.forward(x.detach())` still builds a full
//! graph: the weights of the layer are tracked `Param`s, and one tracked input
//! is sufficient. Burn registers untracked ops with **unit state**, so they
//! retain no activation. That is where the memory saving comes from.
//!
//! So a cut of the gradient flow through a prefix takes *both* halves:
//!
//! - [`detach_params`] on a **clone** of the module. The original keeps its
//!   tracked params, which lets a weight-shared layer receive gradient from
//!   its tracked applications only.
//! - `.detach()` on every tensor that enters the prefix from outside.
//!
//! # Warning: this frees no memory
//!
//! An untracked op is still **registered in the graph**. Burn keeps an
//! `UntrackedOpsStep` per op, so that a memory-bound op with an untracked
//! parent can still retrieve it. So its output stays retained. Measured on a
//! 64-virtual-layer `Layers` of a recurrent block (flex, `d_model` 128, batch
//! 4, seq 512), with a 2-layer gradient horizon: the peak RSS was 2765 MB with
//! a detached prefix, and 550 MB with an inner-backend prefix. The latter is
//! the floor (the same forward with no autodiff at all costs 501 MB). A detach
//! buys the gradient semantics, and almost none of the memory.
//!
//! # What this does *not* reach
//!
//! [`Module::map`](burn::module::Module::map) is a **no-op on plain `Tensor`
//! fields**: Burn implements `Module for Tensor` as a constant (`map` returns
//! `self`). So only [`Param`](burn::module::Param)-wrapped tensors go through
//! the mapper. Weights are `Param`s and are covered. Bare tensors (notably the
//! caches) are not.

use burn::module::{Module, ModuleMapper, Param};
use burn::prelude::*;

/// Clears `require_grad` and cuts the graph on every parameter that it visits.
struct DetachParams;

impl ModuleMapper for DetachParams {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        // `set_require_grad` fixes the *leaf* requirement (and, through
        // `Param::from_mapped_value`, the `require_grad` flag of the param,
        // which is read again from the tensor). `detach` also re-roots the
        // tensor, so nothing upstream of it is reachable. Both are documented
        // no-ops off the autodiff backend, so this is safe on every backend.
        Param::from_mapped_value(id, tensor.set_require_grad(false).detach(), mapper)
    }
}

/// Return `module` with every parameter detached from the autodiff graph.
///
/// Call it on a **clone**. The no-grad part runs the detached copy, and the
/// original keeps the tracked parameters that the back-propagated part needs.
/// Under weight sharing, both refer to the same weights, and the gradient then
/// accumulates from the tracked applications alone.
///
/// In effect a no-op when the module is not on an autodiff backend. And, as
/// the module warning says, **not** a way to save memory.
pub fn detach_params<T: Module>(module: T) -> T {
    module.map(&mut DetachParams)
}
