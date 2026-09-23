//! [`Weights`]: a module's float parameters as the state a captured step
//! advances.

use crate::modules::{CacheTensors, TensorZip};
use burn::module::{ModuleMapper, ModuleVisitor, Param};
use burn::prelude::*;
use std::collections::VecDeque;

/// The float parameters of `M` as the caches of a
/// [`CapturedStep`](super::CapturedStep): what a captured *training* step
/// advances. Int and bool parameters come along untouched.
///
/// The closure of the step receives the module, trains it one step, and gives
/// back the updated module. `CapturedStep` then writes those values into the
/// parameters that it holds (in place, so a replay advances them), and rolls
/// back the steps that `capture` ran. The optimizer inside must be one that a
/// replay can advance: stateless, with its learning rate as an input tensor
/// ([`SgdConfig::step`]).
///
/// Every tensor goes through the zip on the inner backend, flattened:
///
/// - A `slice_assign` on an autodiff parameter would be a tracked op, and
///   would make the parameter a graph node.
/// - The flattening lets one rank-1 zip serve every rank (the reshape of a
///   contiguous parameter is a view of the same buffer).
///
/// After the zip, each parameter gets back its autodiff association, its
/// checkpointing strategy and its `require_grad`.
///
/// [`SgdConfig::step`]: crate::optim::SgdConfig::step
#[derive(Clone, Debug)]
pub struct Weights<M>(pub M);

impl<M: Module + Clone> CacheTensors for Weights<M> {
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
        /// The float parameters of `other`, flattened, in visit order.
        struct Collect(VecDeque<Tensor<1>>);
        impl ModuleVisitor for Collect {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                let t = param.val().inner();
                let n = t.shape().num_elements();
                self.0.push_back(t.reshape([n]));
            }
        }
        struct Zip<'z, Z> {
            other: VecDeque<Tensor<1>>,
            z: &'z mut Z,
        }
        impl<Z: TensorZip> ModuleMapper for Zip<'_, Z> {
            fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
                let (id, t, mapper) = param.consume();
                let autodiff = t.is_autodiff();
                let strategy = t.gradient_checkpointing_strategy();
                let require_grad = t.is_require_grad();
                let dims = t.dims();
                let t = t.inner();
                let n = t.shape().num_elements();
                let other = self.other.pop_front().expect("the two differ in parameter count");
                assert_eq!(other.dims(), [n], "the two differ in a parameter's size");
                let mut t = self.z.zip(t.reshape([n]), other).reshape(dims);
                if autodiff {
                    t = Tensor::from_inner(t);
                    if let Some(strategy) = strategy {
                        t = t.with_gradient_checkpointing_strategy(strategy);
                    }
                    if require_grad {
                        t = t.require_grad();
                    }
                }
                Param::from_mapped_value(id, t, mapper)
            }
        }
        let mut collect = Collect(VecDeque::new());
        other.0.visit(&mut collect);
        drop(other);
        let mut zip = Zip {
            other: collect.0,
            z,
        };
        let module = self.0.map(&mut zip);
        assert!(zip.other.is_empty(), "the two differ in parameter count");
        Weights(module)
    }
}
