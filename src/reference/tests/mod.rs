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
//! - `layers.rs`: forward/step parity through a `Layer` and a `Layers` stack,
//!   cache threading, virtual-layer weight sharing, the ignored residuals,
//! - `grad_horizon.rs`: gradient reachability across a `grad_horizon` cut,
//! - `cache.rs`: the detach of the cache stack,
//! - `multi_gate.rs`: Multi-Gate Residuals in `Layers` and in a bidirectional
//!   stack, and with the class markers,
//! - `bidi.rs`: bidirectional pairing,
//! - `class.rs`: class-marker placement,
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
    let [_batch, sequence, _d_model] = x.dims();
    let mut caches = None;
    let mut ys = Vec::with_capacity(sequence);
    for t in 0..sequence {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = layers.step(x_t, caches, None);
        caches = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }
    Tensor::cat(ys, 1)
}
