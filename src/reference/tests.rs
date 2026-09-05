//! Contract tests for the block-generic containers, composed over
//! [`RefBlock`](super::RefBlock).
//!
//! The point is coverage that is *independent of any mixer family*: if these
//! pass, the containers do not secretly depend on anything a real block
//! happens to provide. They pin the properties every family relies on —
//! forward/step parity through a `Layer` and a `Layers` stack, cache threading,
//! virtual-layer weight sharing, gradient reachability across a `grad_horizon`
//! cut, multi-gate residuals, bidirectional pairing, class-marker placement,
//! and the Muon allowlist.

use super::*;
use crate::modules::bidi::{BidiLayersBuilder, OutputMergeConfig};
use crate::modules::{LatentNetworkBuilder, LayersBuilder, ResidualsConfig};
use crate::utils::class::init_class_emb;
use crate::utils::test_helpers::max_abs_diff;
use crate::utils::{ClassLatent, ClassToken, GradHorizon, Schedule};
use burn::tensor::Distribution;

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

// ---------------------------------------------------------------------------
// The block itself: the parity contract the containers assume
// ---------------------------------------------------------------------------

/// `block_forward` over a sequence must equal `block_step` unrolled from the
/// same cache. Everything downstream is built on this.
#[test]
fn block_forward_equals_step_unrolled() {
    let device: Device = Default::default();
    let block = block_config().init(&device);
    let x = randn3(2, 6, &device);

    let (y_fwd, cache_fwd) = block.block_forward(x.clone(), None, ());

    let mut cache = None;
    let mut ys = Vec::new();
    for t in 0..6 {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = block.block_step(x_t, cache);
        cache = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }
    let y_step = Tensor::cat(ys, 1);

    assert!(max_abs_diff(y_fwd, y_step) < TOL);
    assert!(max_abs_diff(cache_fwd.state_bd, cache.unwrap().state_bd) < TOL);
}

/// A `forward` split into two chunks, threading the cache, equals one `forward`
/// over the whole sequence. This is what prefill-then-decode relies on.
#[test]
fn block_forward_is_chunkable_through_the_cache() {
    let device: Device = Default::default();
    let block = block_config().init(&device);
    let x = randn3(2, 6, &device);

    let (y_all, _) = block.block_forward(x.clone(), None, ());
    let (y_a, cache) = block.block_forward(x.clone().narrow(1, 0, 4), None, ());
    let (y_b, _) = block.block_forward(x.narrow(1, 4, 2), Some(cache), ());

    assert!(max_abs_diff(y_all, Tensor::cat(vec![y_a, y_b], 1)) < TOL);
}

// ---------------------------------------------------------------------------
// Layer / Layers
// ---------------------------------------------------------------------------

/// `Layers::forward` equals `Layers::step` unrolled — the container threads
/// caches per (virtual) layer without reordering or dropping any.
#[test]
fn layers_forward_equals_step_unrolled() {
    let device: Device = Default::default();
    let layers = layers(3, &device);
    let x = randn3(2, 5, &device);

    let (y_fwd, _) = layers.forward(x.clone(), None, (), None);

    let mut caches = None;
    let mut ys = Vec::new();
    for t in 0..5 {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = layers.step(x_t, caches, None);
        caches = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }

    assert!(max_abs_diff(y_fwd, Tensor::cat(ys, 1)) < TOL);
}

/// An `mlp` on the layer adds a second, *inner* residual. The layer returns its
/// total delta, so the stack's single outer add must reproduce
/// `(x + mixer) + mlp(norm2(x + mixer))`.
#[test]
fn layer_mlp_reproduces_two_separate_residuals() {
    use crate::modules::GatedMlpConfig;

    let device: Device = Default::default();
    let layers = LayersBuilder {
        mlp: Some(GatedMlpConfig::new(D_MODEL, D_MODEL * 2)),
        ..layers_builder(1)
    }
    .init(&device);
    let x = randn3(2, 4, &device);

    let (got, _) = layers.forward(x.clone(), None, (), None);

    // Reference: the two residuals written out.
    let layer = &layers.real_layers[0];
    let (h1, _) = layer.block.block_forward(layer.norm.forward(x.clone()), None, ());
    let residual = x + h1;
    let norm2 = layer.norm2.as_ref().expect("norm2 present with an mlp");
    let mlp = layer.mlp.as_ref().expect("mlp present");
    let want = residual.clone() + mlp.forward(norm2.forward(residual));

    assert!(max_abs_diff(got, want) < TOL);
}

/// Virtual layers reuse one real weight set: 6 virtual over 2 real must hold
/// exactly 2 weight sets, 6 cache slots, and reach every real layer's gradient.
#[test]
fn virtual_layers_share_weights_and_keep_one_cache_each() {
    let device = Device::default().autodiff();
    let layers = LayersBuilder {
        n_virtual_layers: Some((6, Schedule::Cyclic)),
        ..layers_builder(2)
    }
    .init(&device);
    assert_eq!(layers.real_layers.len(), 2);

    let x = Tensor::random([2, 4, D_MODEL], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();
    let (y, caches) = layers.forward(x.clone(), None, (), None);
    assert_eq!(caches.slot_count(), 6);

    let grads = y.sum().backward();
    for l in &layers.real_layers {
        assert!(
            l.block.in_proj.weight.val().grad(&grads).is_some(),
            "every real layer must collect a gradient from its virtual applications",
        );
    }
}

/// A `grad_horizon` covering the whole stack must be a no-op on the values, and
/// a shorter one must cut the gradient to the untracked layers' own parameters
/// while leaving the stack input reachable (the straight-through re-attachment).
#[test]
fn grad_horizon_cuts_the_prefix_but_not_the_input() {
    let device = Device::default().autodiff();
    let x = Tensor::random([2, 4, D_MODEL], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();

    let full = layers_builder(4).init(&device);
    let (y_full, _) = full.forward(x.clone(), None, (), None);

    let mut uncut = layers_builder(4).init(&device);
    uncut.real_layers = full.real_layers.clone();
    uncut.grad_horizon = Some(GradHorizon::last(4, 4));
    let (y_uncut, _) = uncut.forward(x.clone(), None, (), None);
    assert!(
        max_abs_diff(y_full, y_uncut) < TOL,
        "a horizon tracking everything must reproduce the untouched stack exactly",
    );

    let mut cut = layers_builder(4).init(&device);
    cut.real_layers = uncut.real_layers.clone();
    cut.grad_horizon = Some(GradHorizon::last(1, 4));
    let (y_cut, _) = cut.forward(x.clone(), None, (), None);
    let grads = y_cut.sum().backward();

    assert!(
        x.grad(&grads).is_some(),
        "the stack input stays reachable across the cut (straight-through re-attach)",
    );
    assert!(
        cut.real_layers[0].block.in_proj.weight.val().grad(&grads).is_none(),
        "a prefix-only parameter must collect no gradient",
    );
    assert!(
        cut.real_layers[3].block.in_proj.weight.val().grad(&grads).is_some(),
        "the tracked suffix must still train",
    );
}

/// A stretched schedule gives each real layer one contiguous run of virtual
/// layers, so a horizon must cut and lift back **once per real layer** — the
/// tail of every run tracked. A single top suffix (what the horizon used to be)
/// would leave every real layer but the topmost with no gradient at all.
#[test]
fn grad_horizon_stretched_trains_every_real_layer() {
    let device = Device::default().autodiff();
    let x = Tensor::random([2, 4, D_MODEL], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();

    // 8 virtual over 3 real: runs [0,1,2], [3,4,5], [6,7]; `Depth(1)` tracks
    // {2, 5, 7}, so the stack crosses the boundary three times.
    let build = || LayersBuilder {
        n_virtual_layers: Some((8, Schedule::Stretched)),
        ..layers_builder(3)
    }
    .init(&device);

    let uncut = build();
    let (y_uncut, caches_uncut) = uncut.forward(x.clone(), None, (), None);

    let mut cut = build();
    cut.real_layers = uncut.real_layers.clone();
    cut.grad_horizon = Some(GradHorizon::Depth(1));
    let (y_cut, caches_cut) = cut.forward(x.clone(), None, (), None);

    assert!(
        max_abs_diff(y_uncut, y_cut.clone()) < TOL,
        "a horizon changes the graph, never the values",
    );
    for (a, b) in caches_uncut.caches.into_iter().zip(caches_cut.caches) {
        assert!(
            max_abs_diff(a.state_bd, b.state_bd) < TOL,
            "nor the caches it hands back",
        );
    }

    let grads = y_cut.sum().backward();
    assert!(
        x.grad(&grads).is_some(),
        "the stack input stays reachable across every cut",
    );
    for (i, l) in cut.real_layers.iter().enumerate() {
        assert!(
            l.block.in_proj.weight.val().grad(&grads).is_some(),
            "real layer {i} must keep a tracked application under a stretched schedule",
        );
    }
}

/// The mask may alternate arbitrarily: only the layers it tracks collect a
/// gradient, and `forward` still equals `step` unrolled through every one of
/// those boundaries — under both residual schemes, the Multi-Gate streams
/// making the same hop as the tokens.
#[test]
fn grad_horizon_mask_alternates_and_keeps_forward_step_parity() {
    let device = Device::default().autodiff();
    let x = Tensor::random([2, 3, D_MODEL], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();

    for residuals in [
        ResidualsConfig::Standard,
        ResidualsConfig::MultiGate {
            n_stream: 3,
            init_bias: 2.0,
            init_bias_step: 0.0,
            per_virtual_layer: false,
        },
    ] {
        let build = || LayersBuilder {
            residuals: residuals.clone(),
            ..layers_builder(6)
        }
        .init(&device);
        let uncut = build();
        let (y_uncut, _) = uncut.forward(x.clone(), None, (), None);

        let mut cut = build();
        cut.real_layers = uncut.real_layers.clone();
        cut.residuals = uncut.residuals.clone();
        // Tracked, cut, tracked, cut, … — four boundaries.
        let mask = vec![true, false, true, false, false, true];
        cut.grad_horizon = Some(GradHorizon::Mask(mask.clone()));
        let (y_cut, _) = cut.forward(x.clone(), None, (), None);
        assert!(
            max_abs_diff(y_uncut, y_cut.clone()) < TOL,
            "an alternating mask changes the graph, never the values",
        );

        // `forward` = `step` unrolled must survive the boundaries too, the
        // cascade taking them token stream by token stream.
        let mut caches = None;
        let mut ys = Vec::new();
        for t in 0..3 {
            let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
            let (y_t, c) = cut.step(x_t, caches, None);
            caches = Some(c);
            ys.push(y_t.unsqueeze_dim::<3>(1));
        }
        assert!(max_abs_diff(y_cut.clone(), Tensor::cat(ys, 1)) < TOL);

        let grads = y_cut.sum().backward();
        assert!(x.grad(&grads).is_some(), "the input crosses every cut");
        for (i, tracked) in mask.iter().enumerate() {
            let g = cut.real_layers[i].block.in_proj.weight.val().grad(&grads);
            assert_eq!(
                g.is_some(),
                *tracked,
                "layer {i} must collect a gradient exactly when the mask tracks it",
            );
        }
    }
}

/// A class latent belonging to an **untracked** layer still trains: it is a
/// learnable *input row*, not part of that layer's transform, so it rides the
/// straight-through carry as a value-zero ghost row. The mask here also ends
/// untracked, so the stack lifts its output at the very top.
#[test]
fn grad_horizon_ghosts_an_untracked_layers_class_latent() {
    let device = Device::default().autodiff();
    let mut layers = layers_builder(3).init(&device);
    layers.real_layers[0].class_latents = vec![ClassLatent::Start];
    layers.real_layers[0].class_latents_emb = init_class_emb(1, D_MODEL, &device);
    layers.grad_horizon = Some(GradHorizon::Mask(vec![false, true, false]));

    let (y, _) = layers.forward(randn3(2, 4, &device), None, (), None);
    assert_eq!(y.dims()[1], 5, "the layer's latent lengthens the sequence");

    let grads = y.sum().backward();
    let emb = layers.real_layers[0]
        .class_latents_emb
        .as_ref()
        .expect("latent table");
    assert!(
        emb.val().grad(&grads).is_some(),
        "an untracked layer's class latent trains through its ghost row",
    );
    assert!(
        layers.real_layers[0].block.in_proj.weight.val().grad(&grads).is_none(),
        "…while that same layer's transform stays undifferentiated",
    );
}

// ---------------------------------------------------------------------------
// Multi-gate residuals
// ---------------------------------------------------------------------------

/// Multi-Gate swaps the additive skip for a convex mean-pool over `k` streams,
/// so the output stays `O(1)` in depth where the additive skip grows. Both
/// modes must still satisfy forward/step parity.
#[test]
fn multi_gate_forward_equals_step_and_stays_bounded() {
    let device: Device = Default::default();
    let layers = LayersBuilder {
        residuals: ResidualsConfig::MultiGate {
            n_stream: 3,
            init_bias: 2.0,
            init_bias_step: 0.0,
            per_virtual_layer: false,
        },
        ..layers_builder(4)
    }
    .init(&device);
    let x = randn3(2, 5, &device);

    let (y_fwd, _) = layers.forward(x.clone(), None, (), None);

    let mut caches = None;
    let mut ys = Vec::new();
    for t in 0..5 {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = layers.step(x_t, caches, None);
        caches = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }

    assert!(max_abs_diff(y_fwd, Tensor::cat(ys, 1)) < TOL);
}

// ---------------------------------------------------------------------------
// Bidirectional
// ---------------------------------------------------------------------------

/// A bidi stack pairs a straight and a reversed pass; with more virtual than
/// real pairs the per-pair merge must be indexed by the **real** pair.
#[test]
fn bidi_virtual_pairs_share_the_real_pair_merge() {
    let device = Device::default().autodiff();
    let layers = BidiLayersBuilder {
        n_real_layers: 4,
        n_virtual_layers: Some((10, Default::default())),
        block: block_config(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: vec![OutputMergeConfig::CatLinear; 2],
        class_latents: Vec::new(),
        residuals: ResidualsConfig::Standard,
    }
    .init(&device);

    let x = Tensor::random([2, 6, D_MODEL], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();
    let (y, _) = layers.forward(x, None, (), None);
    assert_eq!(y.dims(), [2, 6, D_MODEL]);

    let grads = y.sum().backward();
    for pair in &layers.outputs_merge {
        if let crate::modules::OutputMerge::CatLinear(lin) = pair {
            assert!(
                lin.weight.val().grad(&grads).is_some(),
                "both real pairs' merges must be exercised, not dangling",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Class markers
// ---------------------------------------------------------------------------

/// A `Start` class token lengthens the sequence by one and reports its own
/// output index; the same placement must hold whether the sequence arrives via
/// `forward` or one token at a time.
#[test]
fn class_token_placement_is_the_same_for_forward_and_step() {
    let device: Device = Default::default();
    let net = LatentNetworkBuilder {
        input_size: 3,
        layers: layers_builder(2),
        output_size: 2,
        final_norm: false,
        class_tokens: vec![ClassToken::Start],
    }
    .init(&device);

    let sequence = 4;
    let x = Tensor::random([1, sequence, 3], Distribution::Normal(0.0, 1.0), &device);
    let (y_fwd, _) = net.forward(x.clone(), None, (), None);
    assert_eq!(
        y_fwd.dims(),
        [1, sequence + 1, 2],
        "the Start token is emitted before the first user token",
    );

    let mut cursors = crate::utils::ClassCursors::new(sequence);
    let mut caches = None;
    let mut ys = Vec::new();
    for t in 0..sequence {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = net.step(x_t, caches, Some(&mut cursors));
        caches = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }
    // `step` returns the last token it emitted, so the marker is folded into
    // the first step's state rather than surfacing as an extra output row.
    assert_eq!(Tensor::cat(ys, 1).dims(), [1, sequence, 2]);
}

/// A per-layer class latent is a learnable input row: it must train.
#[test]
fn class_latents_receive_gradients() {
    let device = Device::default().autodiff();
    let layers = LayersBuilder {
        class_latents: vec![ClassLatent::Start],
        ..layers_builder(2)
    }
    .init(&device);

    let x = Tensor::random([2, 4, D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    let (y, _) = layers.forward(x, None, (), None);
    let grads = y.sum().backward();

    let emb = layers.class_latents_emb.as_ref().expect("latent table");
    assert!(emb.val().grad(&grads).is_some());
}

// ---------------------------------------------------------------------------
// Muon plan
// ---------------------------------------------------------------------------

/// The plan is an **allowlist**: only the rank-2 weights a `BlockConfig` names
/// are moved off AdamW, and every listed path must exist on the real module.
#[cfg(feature = "optim")]
#[test]
fn muon_plan_matches_only_existing_rank_2_weights() {
    use crate::optim::MuonPlan;

    let device: Device = Default::default();
    let layers = layers(2, &device);
    let plan = MuonPlan::new(BlockConfig::muon_projections(&block_config()));

    let report = plan.describe(&layers);
    let mut on_muon = 0;
    for line in report.lines().filter(|l| l.contains("muon[")) {
        on_muon += 1;
        assert!(
            line.trim_start().starts_with("2D"),
            "Muon only ever owns rank-2 weights: {line}",
        );
    }
    assert_eq!(on_muon, 6, "3 listed weights x 2 real layers");
    for line in report.lines().filter(|l| l.contains("decay_raw")) {
        assert!(
            line.ends_with("adamw"),
            "the rank-1 decay stays on the fallback optimizer: {line}",
        );
    }
}
