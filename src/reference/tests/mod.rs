//! The tests of the block-generic containers, composed over
//! [`RefBlock`](super::RefBlock).
//!
//! The point is coverage that is *independent of any family*. If these
//! tests pass, the containers do not secretly depend on something that a real
//! block happens to provide. They pin the properties that every family relies
//! on. The block contracts on `RefBlock` (`src/block_contract/tests.rs`)
//! check the block itself. Each file holds the tests of one mechanism:
//!
//! - `layers.rs`: forward/step parity through a `Layer`, a `Layers` stack and
//!   a `LatentNetwork`, and the feed-forward of a layer. It also holds cache
//!   threading, virtual-layer weight sharing, the ignored residuals and the
//!   record of a network,
//! - `grad_horizon.rs`: gradient reachability across a `grad_horizon` cut,
//!   the gradients of a cut stack against one window and against the step,
//!   and a probe of the memory that a cut saves,
//! - `cache.rs`: the detach and the cache traversal of the caches of a layer
//!   stack,
//! - `multi_gate.rs`: Multi-Gate Residuals in `Layers` and in a bidirectional
//!   stack, and with the class markers,
//! - `bidi.rs`: bidirectional pairing, and the ignored residuals of a pair,
//! - `class.rs`: class-marker placement in a forward, in a step and in a
//!   prime,
//! - `untied.rs`: the untied copies of a parameter,
//! - `packed.rs`: packed rows against each segment alone, in the outputs and
//!   in the gradients, the origins and their gradients, and the checks of
//!   their layout,
//! - `muon.rs`: the Muon allowlist, the plans of the containers, the module
//!   optimizer, the report, and the update of an untied projection.
//!
//! The helpers below and the imports of this file serve every file.

use super::*;
use crate::modules::bidi::{BidiLayersBuilder, OutputMergeConfig};
use crate::modules::{
    GatedMlpConfig, LatentNetworkBuilder, LayerUntied, Layers, LayersBuilder, ResidualsConfig,
    VocabNetworkBuilder, cache_detach,
};
use crate::utils::test_helpers::{dtype_tol, max_abs_diff, max_rel_diff};
use crate::utils::{BidiSchedule, ClassLatent, ClassToken, GradHorizon, Packed, Schedule};
use burn::module::{ModuleMapper, ModuleVisitor, Param};
use burn::tensor::{Distribution, Gradients};
use crate::utils::test_helpers::test_device;

mod bidi;
mod cache;
mod class;
mod grad_horizon;
mod layers;
#[cfg(feature = "optim")]
mod muon;
mod multi_gate;
mod packed;
mod untied;

const D_MODEL: usize = 8;
const TOL: f32 = 1e-4;

fn block_config() -> RefBlockConfig {
    RefBlockConfig::new(D_MODEL)
}

fn layers_builder(n_real: usize) -> LayersBuilder<RefBlockConfig> {
    LayersBuilder::new(n_real, block_config())
}

fn layers(n_real: usize, device: &Device) -> crate::modules::Layers<RefBlock> {
    layers_builder(n_real).init(device)
}

fn randn3(batch: usize, sequence: usize, device: &Device) -> Tensor<3> {
    Tensor::random([batch, sequence, D_MODEL], Distribution::Normal(0.0, 1.0), device)
}

/// `Layers::step` unrolled over each token of `x`, from no cache and with no
/// class cursors: the outputs `[batch, sequence, d_model]`.
fn step_unrolled(layers: &Layers<RefBlock>, x: Tensor<3>) -> Tensor<3> {
    step_unrolled_with_caches(layers, x).0
}

/// [`step_unrolled`], with the final caches of the last step.
fn step_unrolled_with_caches(layers: &Layers<RefBlock>, x: Tensor<3>) -> (Tensor<3>, Vec<RefCache>) {
    let [_batch, sequence, _d_model] = x.dims();
    let mut caches = None;
    let mut ys = Vec::with_capacity(sequence);
    for t in 0..sequence {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = layers.step(x_t, caches, None);
        caches = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }
    (Tensor::cat(ys, 1), caches.expect("the sequence has at least one token"))
}

/// Panic unless each final cache in `got` matches its cache in `want`.
fn assert_same_caches(want: &[RefCache], got: &[RefCache], label: &str) {
    assert_eq!(want.len(), got.len(), "{label}: the counts of the cache slots differ");
    for (i, (want, got)) in want.iter().zip(got).enumerate() {
        let diff = max_rel_diff(got.state_bd.clone(), want.state_bd.clone());
        assert!(diff < dtype_tol(TOL), "{label}: the final caches of virtual layer {i} differ by {diff}");
    }
}

/// Panic unless the outputs `got` match the outputs `want`.
fn assert_same_outputs<const D: usize>(want: Tensor<D>, got: Tensor<D>, label: &str) {
    let diff = max_rel_diff(got, want);
    assert!(diff < dtype_tol(TOL), "{label}: the outputs differ by {diff}");
}

/// The gradient of each float parameter of `module` in `grads`, flattened,
/// with the path of the parameter. `None`: the parameter has no gradient.
///
/// `src/modules/hnet/tests.rs` has a copy, and `src/utils/padding/tests.rs`
/// has a copy without the paths.
fn param_grads(module: &impl Module, grads: &Gradients) -> Vec<(String, Option<Tensor<1>>)> {
    struct Collect<'a> {
        grads: &'a Gradients,
        path: Vec<String>,
        out: Vec<(String, Option<Tensor<1>>)>,
    }
    impl ModuleVisitor for Collect<'_> {
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.path.push(name.to_string());
        }
        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let grad = param.val().grad(self.grads).map(|g| g.reshape([-1]));
            self.out.push((self.path.join("."), grad));
        }
    }
    let mut collect = Collect { grads, path: Vec::new(), out: Vec::new() };
    module.visit(&mut collect);
    collect.out
}

/// Panic unless each float parameter of `module` has a gradient in `want`
/// and in `got`, and the two gradients match.
fn assert_same_grads(module: &impl Module, want: &Gradients, got: &Gradients, label: &str) {
    for (path, grad) in param_grads(module, want) {
        assert!(grad.is_some(), "{label}: `{path}` gets no gradient in the reference");
    }
    assert_matching_grads(module, want, got, label);
}

/// Panic unless each float parameter of `module` has a gradient in both
/// `want` and `got`, or in neither, and the two gradients match.
fn assert_matching_grads(module: &impl Module, want: &Gradients, got: &Gradients, label: &str) {
    let want = param_grads(module, want);
    let got = param_grads(module, got);
    for ((path, want), (_, got)) in want.into_iter().zip(got) {
        match (want, got) {
            (Some(want), Some(got)) => {
                let diff = max_rel_diff(got, want);
                assert!(diff < dtype_tol(TOL), "{label}: the gradients of `{path}` differ by {diff}");
            }
            (None, None) => {}
            (Some(_), None) => panic!("{label}: `{path}` gets a gradient in the reference only"),
            (None, Some(_)) => panic!("{label}: `{path}` gets a gradient in the run under test only"),
        }
    }
}

/// A [`LatentNetwork`](crate::modules::LatentNetwork) of 2 real layers, with
/// an input of 3 features and an output of 2 features. It has the class
/// tokens `class_tokens`, and its stack has the class latents
/// `class_latents`.
fn latent_network(
    class_tokens: Vec<ClassToken>,
    class_latents: Vec<ClassLatent>,
    device: &Device,
) -> crate::modules::LatentNetwork<RefBlock> {
    LatentNetworkBuilder {
        input_size: 3,
        layers: LayersBuilder { class_latents, ..layers_builder(2) },
        output_size: 2,
        final_norm: false,
        class_tokens,
    }
    .init(device)
}

/// A random input of [`latent_network`]: `[batch, sequence, 3]`.
fn latent_input(batch: usize, sequence: usize, device: &Device) -> Tensor<3> {
    Tensor::random([batch, sequence, 3], Distribution::Normal(0.0, 1.0), device)
}
