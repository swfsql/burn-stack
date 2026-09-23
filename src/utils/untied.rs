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
//! A family says *which* of its parameters are untied
//! ([`Block::untied_params`](crate::modules::Block::untied_params)), and tiles
//! them when it is built for several applications
//! ([`BlockConfig::init_block`](crate::modules::BlockConfig::init_block)).
//! [`Layer::application`](crate::modules::Layer::application) builds the view
//! that the containers run. Three rules follow from the layout:
//!
//! - **The copies start tied.** [`tile`](crate::utils::untied::tile) copies one
//!   initialisation into every application. So an untied stack at init
//!   computes exactly the tied one, and the gradients of its copies sum to
//!   that of the tied weight. Only training makes them differ. A post-build
//!   [`InitPolicy`](crate::utils::InitPolicy) redraws a 2-D `weight` element by
//!   element, and [`retie`](crate::utils::untied::retie) undoes that.
//! - **Every application must train.** Only its own application reads a copy.
//!   So an untracked application leaves it without a gradient, and
//!   [`Layers::grad_horizon`](crate::modules::Layers::grad_horizon) panics on a
//!   cut through a layer that unties anything.
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

/// `param` copied `n_applications` times along `axis`: the untied layout of a
/// newly initialised parameter, with every application from the same draw. A
/// single application is `param` itself.
pub fn tile<const D: usize>(
    param: Param<Tensor<D>>,
    axis: usize,
    n_applications: usize,
) -> Param<Tensor<D>> {
    assert!(n_applications >= 1, "a real layer is built for at least one application");
    if n_applications == 1 {
        return param;
    }
    // Built from the inner value, so the copies are one new leaf, not a graph
    // that hangs off the draw (`detach` keeps the `require_grad` of a leaf).
    let tiled = param.val().inner().repeat_dim(axis, n_applications);
    Param::from_tensor(Tensor::from_inner(tiled))
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
/// application: [`tile`] again, over a module that is already built.
pub fn retie<M: Module>(module: M, params: &[UntiedParam], n_applications: usize) -> M {
    if n_applications == 1 || params.is_empty() {
        return module;
    }
    module.map(&mut Retie {
        params,
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

/// The mapper of [`retie`].
struct Retie<'a> {
    params: &'a [UntiedParam],
    n_applications: usize,
}

impl ModuleMapper for Retie<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let Some(axis) = axis_of(self.params, param.id) else {
            return param;
        };
        let n = self.n_applications;
        param.map(|tensor| {
            // `Param::map` reads the flag again from the tensor that it gets.
            // So the new leaf built from the inner value (see `tile`) must get
            // the flag explicitly, as in `InitPolicy`.
            let require_grad = tensor.is_require_grad();
            let len = copy_len(&tensor.dims(), axis, n);
            let first = tensor.inner().narrow(axis, 0, len);
            Tensor::from_inner(first.repeat_dim(axis, n)).set_require_grad(require_grad)
        })
    }
}
