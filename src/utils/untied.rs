//! # Untied parameters
//!
//! A virtual-layer stack ties the weights of a real layer across every
//! application of it. An **untied** parameter holds one copy per application
//! instead. The copies lie side by side along one of its existing axes, so it
//! keeps its rank and its name. A real layer built for a single application
//! also keeps its stock shape. Application `k` reads its own copy back through
//! [`view`](crate::utils::untied::view), so the module that owns the
//! parameter never sees the tiling.
//!
//! A family names its untied parameters, each with the axis of its copies
//! ([`Block::untied_params`](crate::modules::Block::untied_params)). Its
//! config builds the block for one application
//! ([`BlockConfig::init_block`](crate::modules::BlockConfig::init_block)).
//! The layer tiles the untied parameters when it is built for several
//! applications.
//! [`Layer::application`](crate::modules::Layer::application) builds the view
//! that the containers run.
//!
//! [`tile`](crate::utils::untied::tile) and
//! [`retie`](crate::utils::untied::retie) share one mapper. It repeats the
//! copy of the first application of each untied parameter along its axis.
//! `tile` reads each untied parameter. So Burn draws a lazy untied parameter
//! in `tile`. With one application, `tile` changes nothing, so a lazy
//! parameter stays lazy.
//!
//! Three rules follow from the layout:
//!
//! - **The copies start tied.** [`tile`](crate::utils::untied::tile) repeats
//!   one draw into every application. So an untied stack at init computes
//!   exactly the tied one, and the gradients of its copies sum to that of the
//!   tied weight. Only training makes them differ. A post-build
//!   [`InitPolicy`] redraws a 2-D `weight` element by
//!   element, and [`retie`](crate::utils::untied::retie) undoes that.
//! - **Every application must train.** Only its own application reads a copy.
//!   So an untracked application leaves it without a gradient, and
//!   [`Layers::grad_horizon`](crate::modules::Layers::grad_horizon) panics
//!   when it leaves an application untracked in a layer that unties anything.
//! - **The count is fixed at the build.** A real layer that runs more often
//!   than it was built for has no copy for the extra applications, and
//!   [`view`](crate::utils::untied::view) panics.

use burn::module::{Module, ModuleMapper, Param, ParamId};
use burn::prelude::*;

/// A parameter stored once per application: which one, and the axis of its
/// side-by-side copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UntiedParam {
    /// The parameter.
    pub id: ParamId,
    /// The axis along which its per-application copies are concatenated.
    pub axis: usize,
}

impl UntiedParam {
    /// `param`, with its copies along `axis`.
    pub fn new<const D: usize>(param: &Param<Tensor<D>>, axis: usize) -> Self {
        assert!(axis < D, "untied axis {axis} is out of range for a {D}-D parameter");
        Self { id: param.id, axis }
    }
}

/// `module` with each parameter of `params` repeated `n_applications` times
/// along its axis: the untied layout of a module built for one application.
/// So every application starts from the same draw. With one application,
/// this is `module` unchanged.
///
/// Each parameter keeps its [`ParamId`], its `require_grad` flag and its
/// `is_active` flag.
pub fn tile<M: Module>(module: M, params: &[UntiedParam], n_applications: usize) -> M {
    assert!(n_applications >= 1, "a real layer is built for at least one application");
    if n_applications == 1 || params.is_empty() {
        return module;
    }
    module.map(&mut Repeat {
        params,
        stored_copies: 1,
        n_applications,
    })
}

/// `module` as its `application`-th application sees it: a clone whose
/// `params` are narrowed to the copy of that application, with everything else
/// shared.
///
/// A copy is *read*, not detached, so the gradient of each application lands
/// on its own slice of the stored parameter.
pub fn view<M: Module>(
    module: &M,
    params: &[UntiedParam],
    application: usize,
    n_applications: usize,
) -> M {
    assert!(
        application < n_applications,
        "application {application} of a real layer built for {n_applications}: an untied \
         parameter has no copy for an application added after the stack was built",
    );
    module.clone().map(&mut Narrow {
        params,
        application,
        n_applications,
    })
}

/// `module` with every one of `params` reset to copies of its first
/// application: [`tile`] again, over a module that is already tiled.
pub fn retie<M: Module>(module: M, params: &[UntiedParam], n_applications: usize) -> M {
    if n_applications == 1 || params.is_empty() {
        return module;
    }
    module.map(&mut Repeat {
        params,
        stored_copies: n_applications,
        n_applications,
    })
}

/// The untied axis of the parameter `id`, if it is untied.
fn axis_of(params: &[UntiedParam], id: ParamId) -> Option<usize> {
    params.iter().find(|p| p.id == id).map(|p| p.axis)
}

/// Length of the copy of one application along `axis`.
fn copy_len(dims: &[usize], axis: usize, n_applications: usize) -> usize {
    assert_eq!(
        dims[axis] % n_applications,
        0,
        "an untied parameter of shape {dims:?} does not hold {n_applications} copies along axis {axis}",
    );
    dims[axis] / n_applications
}

/// The mapper of [`view`].
struct Narrow<'a> {
    params: &'a [UntiedParam],
    application: usize,
    n_applications: usize,
}

impl ModuleMapper for Narrow<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let Some(axis) = axis_of(self.params, param.id) else {
            return param;
        };
        let tensor = param.val();
        let len = copy_len(&tensor.dims(), axis, self.n_applications);
        Param::initialized(param.id, tensor.narrow(axis, self.application * len, len))
    }
}

/// The mapper of [`tile`] and [`retie`]: the copy of the first application of
/// each parameter of `params`, repeated `n_applications` times along its axis.
struct Repeat<'a> {
    params: &'a [UntiedParam],
    /// The copies that the stored parameter holds: 1 for [`tile`], and
    /// `n_applications` for [`retie`].
    stored_copies: usize,
    n_applications: usize,
}

impl ModuleMapper for Repeat<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let Some(axis) = axis_of(self.params, param.id) else {
            return param;
        };
        let (stored, n) = (self.stored_copies, self.n_applications);
        param.map(|tensor| {
            // Detached after the repeat, so the copies are one new leaf, not a
            // graph that hangs off the draw. `detach` keeps the backend of the
            // tensor. `Param::map` keeps the id and `is_active`, but it reads
            // the `require_grad` flag again from the tensor that it gets. So
            // the new leaf gets the flag explicitly, as in `InitPolicy`.
            let require_grad = tensor.is_require_grad();
            let len = copy_len(&tensor.dims(), axis, stored);
            let first = tensor.narrow(axis, 0, len);
            first.repeat_dim(axis, n).detach().set_require_grad(require_grad)
        })
    }
}
