//! The tests of the block-generic containers, composed over
//! [`RefBlock`](super::RefBlock).
//!
//! The point is coverage that is *independent of any family*. If these
//! tests pass, the containers do not secretly depend on something that a real
//! block happens to provide. They pin the properties that every family relies
//! on. Each file holds the tests of one mechanism:
//!
//! - `block.rs`: the forward/step parity of the block itself, and its windows
//!   through the cache,
//! - `layers.rs`: forward/step parity through a `Layer`, a `Layers` stack and
//!   a `LatentNetwork`, the feed-forward of a layer, cache threading,
//!   virtual-layer weight sharing, the ignored residuals,
//! - `grad_horizon.rs`: gradient reachability across a `grad_horizon` cut,
//!   the gradients of a cut stack against one window and against the step,
//!   and a probe of the memory that a cut saves,
//! - `cache.rs`: the detach of the cache stack,
//! - `multi_gate.rs`: Multi-Gate Residuals in `Layers` and in a bidirectional
//!   stack, and with the class markers,
//! - `bidi.rs`: bidirectional pairing, and the ignored residuals of a pair,
//! - `class.rs`: class-marker placement in a forward, in a step and in a
//!   prime,
//! - `untied.rs`: the untied copies of a parameter,
//! - `packed.rs`: packed rows against each segment alone, and the checks of
//!   their layout,
//! - `muon.rs`: the Muon allowlist, the plans of the containers, the module
//!   optimizer, the report, and the update of an untied projection.
//!
//! The helpers below and the imports of this file serve every file.

use super::*;
use crate::modules::bidi::{BidiLayersBuilder, OutputMergeConfig};
use crate::modules::{
    GatedMlpConfig, LatentNetworkBuilder, LayerUntied, Layers, LayersBuilder, ResidualsConfig,
    VocabNetworkBuilder,
};
use crate::utils::test_helpers::{dtype_tol, max_abs_diff, max_rel_diff};
use crate::utils::{BidiSchedule, ClassLatent, ClassToken, GradHorizon, Packed, Schedule};
use burn::module::{ModuleMapper, Param};
use burn::tensor::Distribution;
use crate::utils::test_helpers::test_device;

mod bidi;
mod block;
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
fn step_unrolled_with_caches(layers: &Layers<RefBlock>, x: Tensor<3>) -> (Tensor<3>, RefCaches) {
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
fn assert_same_caches(want: &RefCaches, got: &RefCaches, label: &str) {
    assert_eq!(want.caches.len(), got.caches.len(), "{label}: the counts of the cache slots differ");
    for (i, (want, got)) in want.caches.iter().zip(&got.caches).enumerate() {
        let diff = max_rel_diff(got.state_bd.clone(), want.state_bd.clone());
        assert!(diff < dtype_tol(TOL), "{label}: the final caches of virtual layer {i} differ by {diff}");
    }
}

/// Panic unless the outputs `got` match the outputs `want`.
fn assert_same_outputs<const D: usize>(want: Tensor<D>, got: Tensor<D>, label: &str) {
    let diff = max_rel_diff(got, want);
    assert!(diff < dtype_tol(TOL), "{label}: the outputs differ by {diff}");
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
