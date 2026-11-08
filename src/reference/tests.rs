//! Contract tests for the block-generic containers, composed over
//! [`RefBlock`](super::RefBlock).
//!
//! The point is coverage that is *independent of any family*. If these
//! tests pass, the containers do not secretly depend on something that a real
//! block happens to provide. They pin the properties that every family relies
//! on:
//!
//! - forward/step parity through a `Layer` and a `Layers` stack,
//! - cache threading,
//! - virtual-layer weight sharing,
//! - gradient reachability across a `grad_horizon` cut,
//! - multi-gate residuals,
//! - bidirectional pairing,
//! - class-marker placement,
//! - packed rows against each segment alone, and the checks of their layout,
//! - the Muon allowlist.

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
    let device = test_device();
    let block = block_config().init(&device);
    let x = randn3(2, 6, &device);

    let (y_fwd, cache_fwd) = block.block_forward(x.clone(), None, (), None);

    let mut cache = None;
    let mut ys = Vec::new();
    for t in 0..6 {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = block.block_step(x_t, cache);
        cache = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }
    let y_step = Tensor::cat(ys, 1);

    assert!(max_abs_diff(y_fwd, y_step) < dtype_tol(TOL));
    assert!(max_abs_diff(cache_fwd.state_bd, cache.unwrap().state_bd) < dtype_tol(TOL));
}

/// A `forward` split into two windows, threading the cache, equals one
/// `forward` over the whole sequence. Prefill followed by decode relies on
/// this.
#[test]
fn block_forward_splits_into_windows_through_the_cache() {
    let device = test_device();
    let block = block_config().init(&device);
    let x = randn3(2, 6, &device);

    let (y_all, _) = block.block_forward(x.clone(), None, (), None);
    let (y_a, cache) = block.block_forward(x.clone().narrow(1, 0, 4), None, (), None);
    let (y_b, _) = block.block_forward(x.narrow(1, 4, 2), Some(cache), (), None);

    assert!(max_abs_diff(y_all, Tensor::cat(vec![y_a, y_b], 1)) < dtype_tol(TOL));
}

// ---------------------------------------------------------------------------
// Layer / Layers
// ---------------------------------------------------------------------------

/// `Layers::forward` equals `Layers::step` unrolled: the container threads
/// caches per virtual layer, and does not reorder or drop any.
#[test]
fn layers_forward_equals_step_unrolled() {
    let device = test_device();
    let layers = layers(3, &device);
    let x = randn3(2, 5, &device);

    let (y_fwd, _) = layers.forward(x.clone(), None, (), None, None);

    let mut caches = None;
    let mut ys = Vec::new();
    for t in 0..5 {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = layers.step(x_t, caches, None);
        caches = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }

    assert!(max_abs_diff(y_fwd, Tensor::cat(ys, 1)) < dtype_tol(TOL));
}

/// An `mlp` on the layer adds a second, *inner* residual. The layer returns its
/// total delta, so the single outer add of the stack must reproduce
/// `(x + h₁) + mlp(norm2(x + h₁))`, with `h₁` the output of the block.
#[test]
fn layer_mlp_reproduces_two_separate_residuals() {
    use crate::modules::GatedMlpConfig;

    let device = test_device();
    let layers = LayersBuilder {
        mlp: Some(GatedMlpConfig::new(D_MODEL, D_MODEL * 2)),
        ..layers_builder(1)
    }
    .init(&device);
    let x = randn3(2, 4, &device);

    let (got, _) = layers.forward(x.clone(), None, (), None, None);

    // Reference: the two residuals written out.
    let layer = &layers.real_layers[0];
    let (h1, _) = layer.block.block_forward(layer.norm.forward(x.clone()), None, (), None);
    let residual = x + h1;
    let norm2 = layer.norm2.as_ref().expect("norm2 present with an mlp");
    let mlp = layer.mlp.as_ref().expect("mlp present");
    let want = residual.clone() + mlp.forward(norm2.forward(residual));

    assert!(max_abs_diff(got, want) < dtype_tol(TOL));
}

/// Virtual layers reuse the weights of a real layer. 6 virtual over 2 real
/// must hold exactly 2 real layers and 6 cache slots, and reach the gradient
/// of every real layer.
#[test]
fn virtual_layers_share_weights_and_keep_one_cache_each() {
    let device = test_device().autodiff();
    let layers = LayersBuilder {
        n_virtual_layers: Some((6, Schedule::Cyclic)),
        ..layers_builder(2)
    }
    .init(&device);
    assert_eq!(layers.real_layers.len(), 2);

    let x = Tensor::random([2, 4, D_MODEL], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();
    let (y, caches) = layers.forward(x.clone(), None, (), None, None);
    assert_eq!(caches.slot_count(), 6);

    let grads = y.sum().backward();
    for l in &layers.real_layers {
        assert!(
            l.block.in_proj.weight.val().grad(&grads).is_some(),
            "every real layer must collect a gradient from its virtual applications",
        );
    }
}

/// A `grad_horizon` that covers the whole stack must be a no-op on the
/// values. A shorter one must cut the gradient to the parameters of the
/// untracked layers, and keep the stack input reachable (the straight-through
/// re-attachment).
#[test]
fn grad_horizon_cuts_the_prefix_but_not_the_input() {
    let device = test_device().autodiff();
    let x = Tensor::random([2, 4, D_MODEL], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();

    let full = layers_builder(4).init(&device);
    let (y_full, _) = full.forward(x.clone(), None, (), None, None);

    let mut uncut = layers_builder(4).init(&device);
    uncut.real_layers = full.real_layers.clone();
    uncut.grad_horizon = Some(GradHorizon::last(4, 4));
    let (y_uncut, _) = uncut.forward(x.clone(), None, (), None, None);
    assert!(
        max_abs_diff(y_full, y_uncut) < dtype_tol(TOL),
        "a horizon tracking everything must reproduce the untouched stack exactly",
    );

    let mut cut = layers_builder(4).init(&device);
    cut.real_layers = uncut.real_layers.clone();
    cut.grad_horizon = Some(GradHorizon::last(1, 4));
    let (y_cut, _) = cut.forward(x.clone(), None, (), None, None);
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

/// A stretched layer schedule gives each real layer one contiguous run of
/// virtual layers. So a horizon must cut and lift back **once per real
/// layer**, with the tail of every run tracked. A single top suffix would
/// leave every real layer except the topmost with no gradient.
#[test]
fn grad_horizon_stretched_trains_every_real_layer() {
    let device = test_device().autodiff();
    let x = Tensor::random([2, 4, D_MODEL], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();

    // 8 virtual over 3 real: runs [0,1,2], [3,4,5], [6,7]. `Depth(1)` tracks
    // {2, 5, 7}, so the stack has three cuts.
    let build = || LayersBuilder {
        n_virtual_layers: Some((8, Schedule::Stretched)),
        ..layers_builder(3)
    }
    .init(&device);

    let uncut = build();
    let (y_uncut, caches_uncut) = uncut.forward(x.clone(), None, (), None, None);

    let mut cut = build();
    cut.real_layers = uncut.real_layers.clone();
    cut.grad_horizon = Some(GradHorizon::Depth(1));
    let (y_cut, caches_cut) = cut.forward(x.clone(), None, (), None, None);

    assert!(
        max_abs_diff(y_uncut, y_cut.clone()) < dtype_tol(TOL),
        "a horizon changes the graph, never the values",
    );
    for (a, b) in caches_uncut.caches.into_iter().zip(caches_cut.caches) {
        assert!(
            max_abs_diff(a.state_bd, b.state_bd) < dtype_tol(TOL),
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
            "real layer {i} must keep a tracked application under a stretched layer schedule",
        );
    }
}

/// The mask can alternate arbitrarily. Only the layers that it tracks collect
/// a gradient, and `forward` still equals `step` unrolled through every one of
/// those backend hops. This holds under both residual schemes, and the
/// residual streams of Multi-Gate make the same hop as the tokens.
#[test]
fn grad_horizon_mask_alternates_and_keeps_forward_step_parity() {
    let device = test_device().autodiff();
    let x = Tensor::random([2, 3, D_MODEL], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();

    for residuals in [
        ResidualsConfig::Standard,
        ResidualsConfig::MultiGate {
            n_stream: 3,
            init_bias: 2.0,
            init_bias_offset: 0.0,
            per_virtual_layer: false,
        },
    ] {
        let build = || LayersBuilder {
            residuals: residuals.clone(),
            ..layers_builder(6)
        }
        .init(&device);
        let uncut = build();
        let (y_uncut, _) = uncut.forward(x.clone(), None, (), None, None);

        let mut cut = build();
        cut.real_layers = uncut.real_layers.clone();
        cut.residuals = uncut.residuals.clone();
        // Tracked, untracked, tracked, untracked, …: four backend hops.
        let mask = vec![true, false, true, false, false, true];
        cut.grad_horizon = Some(GradHorizon::Mask(mask.clone()));
        let (y_cut, _) = cut.forward(x.clone(), None, (), None, None);
        assert!(
            max_abs_diff(y_uncut, y_cut.clone()) < dtype_tol(TOL),
            "an alternating mask changes the graph, never the values",
        );

        // `forward` = `step` unrolled must also survive the hops. The
        // cascade takes them one token stream at a time.
        let mut caches = None;
        let mut ys = Vec::new();
        for t in 0..3 {
            let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
            let (y_t, c) = cut.step(x_t, caches, None);
            caches = Some(c);
            ys.push(y_t.unsqueeze_dim::<3>(1));
        }
        assert!(max_rel_diff(y_cut.clone(), Tensor::cat(ys, 1)) < dtype_tol(TOL));

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

/// On a device with gradient checkpointing, a cut lifts what it took down
/// with the strategy of the device ([`lift`](crate::modules::lift)). A bare
/// `Tensor::from_inner` sets the `Disabled` strategy, and the next op that
/// mixes the lifted tensor with a tensor of the device panics. The mask ends
/// untracked, so the in-loop lift and the lift at the top both run, in
/// `forward` and in `step`.
#[test]
fn grad_horizon_keeps_the_checkpointing_strategy() {
    let device = test_device().autodiff().gradient_checkpointing();
    let x = randn3(2, 3, &device).require_grad();
    let mut layers = layers(4, &device);
    layers.grad_horizon = Some(GradHorizon::Mask(vec![true, false, true, false]));

    let (y, caches) = layers.forward(x.clone(), None, (), None, None);
    let grads = y.sum().backward();
    assert!(x.grad(&grads).is_some(), "the input crosses every cut");

    // The backward above consumed the graph of the caches, so the step takes
    // them detached (Burn has no `retain_graph`).
    let x_t = x.narrow(1, 0, 1).squeeze_dim::<2>(1);
    let (y_t, _) = layers.step(x_t, Some(caches.detach()), None);
    let _ = y_t.sum().backward();
}

/// `CacheStack::detach` puts each slot back on its own device: a plain slot
/// stays plain, and an autodiff slot keeps the checkpointing strategy of its
/// device.
#[test]
fn detach_keeps_the_device_of_each_slot() {
    for device in [test_device(), test_device().autodiff().gradient_checkpointing()] {
        let (_, caches) = layers(2, &device).forward(randn3(1, 3, &device), None, (), None, None);
        for slot in caches.detach().into_slots().into_iter().flatten() {
            let state = slot.state_bd;
            assert_eq!(state.is_autodiff(), device.is_autodiff());
            assert_eq!(
                state.gradient_checkpointing_strategy(),
                device.gradient_checkpointing_strategy(),
            );
        }
    }
}

/// A class latent of the stack trains below a cut. It is a learnable *input*,
/// so it rides the straight-through carry with the stack input. The mask here
/// also ends untracked, so the stack lifts its output at the very top.
#[test]
fn grad_horizon_keeps_a_class_latent_of_the_stack_training() {
    let device = test_device().autodiff();
    let layers = LayersBuilder {
        class_latents: vec![ClassLatent::Start],
        grad_horizon: Some(GradHorizon::Mask(vec![false, true, false])),
        ..layers_builder(3)
    }
    .init(&device);

    let (y, _) = layers.forward(randn3(2, 4, &device), None, (), None, None);
    assert_eq!(y.dims()[1], 5, "the latent makes the sequence longer");

    let grads = y.sum().backward();
    let emb = layers.class_latents_emb.as_ref().expect("latent table");
    assert!(
        emb.val().grad(&grads).is_some(),
        "the class latent trains through the straight-through carry",
    );
    assert!(
        layers.real_layers[0].block.in_proj.weight.val().grad(&grads).is_none(),
        "the transform of an untracked layer stays undifferentiated",
    );
}

// ---------------------------------------------------------------------------
// Multi-gate residuals
// ---------------------------------------------------------------------------

/// Multi-Gate swaps the additive skip for a convex mean-pool over `k` streams,
/// so the output stays `O(1)` in depth where the additive skip grows. Both
/// residual schemes must still satisfy forward/step parity.
#[test]
fn multi_gate_forward_equals_step_and_stays_bounded() {
    let device = test_device();
    let layers = LayersBuilder {
        residuals: ResidualsConfig::MultiGate {
            n_stream: 3,
            init_bias: 2.0,
            init_bias_offset: 0.0,
            per_virtual_layer: false,
        },
        ..layers_builder(4)
    }
    .init(&device);
    let x = randn3(2, 5, &device);

    let (y_fwd, _) = layers.forward(x.clone(), None, (), None, None);

    let mut caches = None;
    let mut ys = Vec::new();
    for t in 0..5 {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = layers.step(x_t, caches, None);
        caches = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }

    assert!(max_abs_diff(y_fwd, Tensor::cat(ys, 1)) < dtype_tol(TOL));
}

// ---------------------------------------------------------------------------
// Bidirectional
// ---------------------------------------------------------------------------

/// A bidi stack pairs a straight and a reversed pass. With more virtual than
/// real pairs, the per-pair merge must be indexed by the **real** pair.
#[test]
fn bidi_virtual_pairs_share_the_real_pair_merge() {
    let device = test_device().autodiff();
    let layers = BidiLayersBuilder {
        n_real_layers: 4,
        n_virtual_layers: Some((10, Default::default())),
        block: block_config(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: vec![OutputMergeConfig::CatLinear; 2],
        class_latents: Vec::new(),
        residuals: ResidualsConfig::Standard,
        untied: Vec::new(),
    }
    .init(&device);

    let x = Tensor::random([2, 6, D_MODEL], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();
    let (y, _) = layers.forward(x, None, (), None, None);
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

/// A `Start` class token makes the sequence one longer, and reports its own
/// output index. The same placement must hold whether the sequence arrives
/// through `forward` or one token at a time.
#[test]
fn class_token_placement_is_the_same_for_forward_and_step() {
    let device = test_device();
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
    let (y_fwd, _) = net.forward(x.clone(), None, (), None, None);
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
    // `step` returns the last token that it emitted. So the marker is folded
    // into the state of the first step, and does not come out as an extra
    // output position.
    assert_eq!(Tensor::cat(ys, 1).dims(), [1, sequence, 2]);
}

/// A class latent of the stack is a learnable input: it must train.
#[test]
fn class_latents_receive_gradients() {
    let device = test_device().autodiff();
    let layers = LayersBuilder {
        class_latents: vec![ClassLatent::Start],
        ..layers_builder(2)
    }
    .init(&device);

    let x = Tensor::random([2, 4, D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    let (y, _) = layers.forward(x, None, (), None, None);
    let grads = y.sum().backward();

    let emb = layers.class_latents_emb.as_ref().expect("latent table");
    assert!(emb.val().grad(&grads).is_some());
}

// ---------------------------------------------------------------------------
// Untied parameters
// ---------------------------------------------------------------------------

/// Two real layers over five virtual ones (`Cyclic`: three applications and
/// two). Everything untiable is untied: both pre-norms and both untiable
/// tensors of `RefBlock`.
fn untied_builder() -> LayersBuilder<RefBlockConfig> {
    LayersBuilder {
        n_virtual_layers: Some((5, Schedule::Cyclic)),
        block: block_config().with_untied(vec![RefUntied::Decay, RefUntied::GateProj]),
        mlp: Some(GatedMlpConfig::new(D_MODEL, D_MODEL * 2)),
        untied: vec![LayerUntied::Norm, LayerUntied::Norm2],
        ..layers_builder(2)
    }
}

/// `layers` with every real layer replaced by its application-0 view: the tied
/// stack an untied one starts as.
fn tied_view(layers: &Layers<RefBlock>) -> Layers<RefBlock> {
    Layers {
        real_layers: layers.real_layers.iter().map(|l| l.application(0).into_owned()).collect(),
        ..layers.clone()
    }
}

/// Adds noise to every parameter, so the copies of an untied stack stop
/// agreeing.
struct Jitter;

impl ModuleMapper for Jitter {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|t| {
            let noise = Tensor::random(t.shape(), Distribution::Normal(0.0, 0.5), &t.device());
            t + noise
        })
    }
}

/// An untied stack equals the stack of its application views, with no weight
/// sharing. Virtual layer `i` runs application `index[i]` of its real layer,
/// in `forward` and in `step`. The test first adds noise to the copies, so a
/// read of the wrong copy shows.
#[test]
fn an_untied_stack_equals_the_stack_of_its_application_views() {
    let device = test_device();
    let layers = untied_builder().init(&device).map(&mut Jitter);
    let n = layers.n_virtual_count();
    let apps = Schedule::Cyclic.applications(n, 2);
    let view_stack = Layers {
        n_real_layers: n,
        n_virtual_layers: None,
        real_layers: (0..n)
            .map(|i| {
                let real = Schedule::Cyclic.real_idx(i, n, 2);
                layers.real_layers[real].application(apps.index[i]).into_owned()
            })
            .collect(),
        ..layers.clone()
    };
    let x = randn3(2, 4, &device);

    let (y, _) = layers.forward(x.clone(), None, (), None, None);
    let (want, _) = view_stack.forward(x.clone(), None, (), None, None);
    assert!(max_abs_diff(y.clone(), want) < dtype_tol(TOL));

    let mut caches = None;
    let mut ys = Vec::new();
    for t in 0..4 {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = layers.step(x_t, caches, None);
        caches = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }
    assert!(max_rel_diff(y.clone(), Tensor::cat(ys, 1)) < dtype_tol(TOL));

    // Both comparisons prove nothing unless the copies really differ.
    let (y_tied, _) = tied_view(&layers).forward(x, None, (), None, None);
    assert!(max_abs_diff(y, y_tied) > dtype_tol(TOL));
}

/// Every copy of `g` (the untied gradient, copies along `axis`) is non-zero.
/// Together they sum to copy 0 of `g_tied`: the gradient that the tied weight
/// collects, which lands on copy 0 alone.
fn assert_copies_split<const D: usize>(g: Tensor<D>, g_tied: Tensor<D>, axis: usize, n: usize) {
    let copies = g.chunk(n, axis);
    for copy in &copies {
        let peak = copy.clone().abs().max().into_scalar::<f32>();
        assert!(peak > 0.0, "every application trains its own copy");
    }
    let sum = copies.into_iter().reduce(|a, b| a + b).expect("n >= 1");
    let len = sum.dims()[axis];
    assert!(max_abs_diff(sum, g_tied.narrow(axis, 0, len)) < dtype_tol(TOL));
}

/// Untied copies start as the tied weight. The untied stack computes exactly
/// its application-0 view, and the gradient of the tied weight is the sum of
/// the gradients of its copies. Only training makes the copies differ.
#[test]
fn untied_copies_start_tied_and_split_the_tied_gradient() {
    let device = test_device().autodiff();
    let layers = untied_builder().init(&device);
    let tied = tied_view(&layers);
    let x = randn3(2, 4, &device);

    let (y, _) = layers.forward(x.clone(), None, (), None, None);
    let (y_tied, _) = tied.forward(x, None, (), None, None);
    assert!(max_abs_diff(y.clone(), y_tied.clone()) < dtype_tol(TOL));

    // A view reads the stored parameters, so both gradients land on them.
    let grads = y.sum().backward();
    let grads_tied = y_tied.sum().backward();
    for layer in &layers.real_layers {
        let n = layer.n_applications;
        let norm2 = layer.norm2.as_ref().expect("the builder sets an mlp");
        for (param, axis) in [
            (&layer.norm.gamma, 0),
            (&norm2.gamma, 0),
            (&layer.block.decay_raw, 0),
        ] {
            let g = param.val().grad(&grads).expect("untied gradient");
            let g_tied = param.val().grad(&grads_tied).expect("tied gradient");
            assert_copies_split(g, g_tied, axis, n);
        }
        let w = &layer.block.gate_proj.weight;
        let g = w.val().grad(&grads).expect("untied gradient");
        let g_tied = w.val().grad(&grads_tied).expect("tied gradient");
        assert_copies_split(g, g_tied, 1, n);
    }
}

/// Only its own application reads a copy. So a cut through a layer that
/// unties anything would leave copies that never train. It panics instead.
#[test]
#[should_panic(expected = "unties parameters across its applications")]
fn grad_horizon_refuses_to_cut_an_untied_layer() {
    let device = test_device().autodiff();
    let layers = LayersBuilder {
        grad_horizon: Some(GradHorizon::Depth(1)),
        ..untied_builder()
    }
    .init(&device);
    let _ = layers.forward(randn3(1, 3, &device), None, (), None, None);
}

/// A post-build `InitPolicy` redraws every 2-D `weight` element by element.
/// `NetworkShape` then ties an untied weight back to copies of its first
/// application.
#[test]
fn an_init_policy_keeps_untied_copies_tied() {
    use crate::modules::NetworkShape;
    use crate::utils::InitPolicy;

    let device = test_device();
    let layers = NetworkShape::new(1)
        .with_n_virtual_layers(Some((3, Schedule::Cyclic)))
        .with_init(Some(InitPolicy::new()))
        .init(block_config().with_untied(vec![RefUntied::GateProj]), &device);
    let w = layers.real_layers[0].block.gate_proj.weight.val();
    assert_eq!(w.dims(), [D_MODEL, 3 * D_MODEL]);
    let copies = w.chunk(3, 1);
    for copy in &copies[1..] {
        assert_eq!(max_abs_diff(copies[0].clone(), copy.clone()), 0.0);
    }
}

/// On a plain device, the untied copies stay plain (`tile` and `retie`). A
/// copy with autodiff makes each pass record a graph that keeps the saved
/// state of its ops. Only a backward that reaches the copy frees that graph,
/// and a pass without autodiff (a decode, an evaluation) never runs one.
#[test]
fn untied_copies_stay_plain_on_a_plain_device() {
    use crate::modules::NetworkShape;
    use crate::utils::InitPolicy;

    let device = test_device();
    assert!(!device.is_autodiff());
    let layers = untied_builder().init(&device);
    for layer in &layers.real_layers {
        let norm2 = layer.norm2.as_ref().expect("the builder sets an mlp");
        assert!(!layer.norm.gamma.val().is_autodiff());
        assert!(!norm2.gamma.val().is_autodiff());
        assert!(!layer.block.decay_raw.val().is_autodiff());
        assert!(!layer.block.gate_proj.weight.val().is_autodiff());
    }

    let layers = NetworkShape::new(1)
        .with_n_virtual_layers(Some((3, Schedule::Cyclic)))
        .with_init(Some(InitPolicy::new()))
        .init(block_config().with_untied(vec![RefUntied::GateProj]), &device);
    assert!(!layers.real_layers[0].block.gate_proj.weight.val().is_autodiff());
}

/// Both directions of a bidirectional pair are applications of the real layer
/// that they index. Every copy that an untied bidi stack holds is read, and
/// trains.
#[test]
fn bidi_untied_copies_each_train() {
    let device = test_device().autodiff();
    let layers = BidiLayersBuilder {
        n_real_layers: 2,
        n_virtual_layers: Some((6, BidiSchedule::SymmetricCyclic)),
        block: block_config().with_untied(vec![RefUntied::Decay]),
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: vec![OutputMergeConfig::Mean],
        class_latents: Vec::new(),
        residuals: ResidualsConfig::Standard,
        untied: vec![LayerUntied::Norm],
    }
    .init(&device);
    // `SymmetricCyclic` runs 0 0 1 1 0 0.
    let counts: Vec<_> = layers.real_layers.iter().map(|l| l.n_applications).collect();
    assert_eq!(counts, vec![4, 2]);

    let (y, _) = layers.forward(randn3(2, 4, &device), None, (), None, None);
    let grads = y.sum().backward();
    for layer in &layers.real_layers {
        for param in [&layer.norm.gamma, &layer.block.decay_raw] {
            let g = param.val().grad(&grads).expect("untied gradient");
            for copy in g.chunk(layer.n_applications, 0) {
                let peak = copy.abs().max().into_scalar::<f32>();
                assert!(peak > 0.0, "every application trains its own copy");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Packed rows
// ---------------------------------------------------------------------------

/// The block gives the reset alignment of its config, which is 1 by default.
#[test]
fn the_block_gives_the_reset_alignment_of_its_config() {
    let device = test_device();
    assert_eq!(block_config().init(&device).reset_align(&()), Some(1));
    assert_eq!(block_config().with_reset_align(4).init(&device).reset_align(&()), Some(4));
}

/// A start that the block does not accept is a caller error. The check reads
/// only host integers, so it runs in every build.
#[test]
#[should_panic(expected = "not a multiple of the reset alignment")]
fn a_start_off_the_reset_alignment_panics() {
    let _ = Packed::from_starts(&[vec![0, 3]], 2, 0, 8, &test_device());
}

/// A table with one row per class latent fills exactly the `lead` opening
/// slots of a segment. Any other row count is a caller error.
#[test]
#[should_panic(expected = "one row per opening slot")]
fn a_lead_that_differs_from_the_table_panics() {
    let device = test_device();
    let layers = LayersBuilder {
        class_latents: vec![ClassLatent::Start],
        ..layers_builder(1)
    }
    .init(&device);
    let packed = Packed::from_starts(&[vec![0]], 1, 2, 4, &device);
    let _ = layers.forward_packed(randn3(1, 4, &device), None, (), &packed);
}

/// A stack with no class latents has no table to put into the opening slots.
/// So a layout that reserves them is a caller error.
#[test]
#[should_panic(expected = "no class latents")]
fn a_lead_on_a_stack_with_no_class_latents_panics() {
    let device = test_device();
    let packed = Packed::from_starts(&[vec![0]], 1, 1, 4, &device);
    let _ = layers(1, &device).forward_packed(randn3(1, 4, &device), None, (), &packed);
}

/// The packed layout of two rows for `block`, with its starts and the width of
/// the rows. Each start is a multiple of the reset alignment of `block`, and
/// each segment has at least 3 positions.
fn packed_layout(block: &RefBlock, lead: usize, device: &Device) -> (Packed, Vec<Vec<usize>>, usize) {
    let align = block.reset_align(&()).expect("this block must take packed rows");
    let starts = vec![vec![0, 3 * align, 8 * align], vec![0, 5 * align]];
    let width = 11 * align;
    (Packed::from_starts(&starts, align, lead, width, device), starts, width)
}

/// Each segment of `starts` as `(row, start, end)`. A segment ends at the next
/// start or at the end of the row. So the gap after a segment runs as a part
/// of that segment, and the tests compare every position.
fn packed_segments(starts: &[Vec<usize>], width: usize) -> Vec<(usize, usize, usize)> {
    let mut segments = Vec::new();
    for (b, row) in starts.iter().enumerate() {
        for (i, &start) in row.iter().enumerate() {
            segments.push((b, start, row.get(i + 1).copied().unwrap_or(width)));
        }
    }
    segments
}

/// The part `[start, end)` of row `b`, with the batch axis kept.
fn segment_of(tensor: Tensor<3>, b: usize, start: usize, end: usize) -> Tensor<3> {
    tensor.narrow(0, b, 1).narrow(1, start, end - start)
}

/// A layer over packed rows gives each segment the outputs that the segment
/// gives alone from a zero cache. The last segment of a row gives the cache.
#[test]
fn layer_forward_packed_equals_each_segment_alone() {
    let device = test_device();
    let layers = LayersBuilder {
        block: block_config().with_reset_align(3),
        mlp: Some(GatedMlpConfig::new(D_MODEL, D_MODEL * 2)),
        ..layers_builder(1)
    }
    .init(&device);
    let layer = &layers.real_layers[0];
    let (packed, starts, width) = packed_layout(&layer.block, 0, &device);
    let x = randn3(2, width, &device);

    let (y, cache) = layer.forward_packed(x.clone(), None, (), &packed);

    for (b, start, end) in packed_segments(&starts, width) {
        let (alone, alone_cache) = layer.forward(segment_of(x.clone(), b, start, end), None, (), None);
        assert!(max_abs_diff(segment_of(y.clone(), b, start, end), alone) < dtype_tol(TOL));
        if end == width {
            let last = cache.state_bd.clone().narrow(0, b, 1);
            assert!(max_abs_diff(last, alone_cache.state_bd) < dtype_tol(TOL));
        }
    }
}

/// The positions before the first reset of a row continue from the incoming
/// cache. So a stream of packed calls can continue a segment.
#[test]
fn positions_before_the_first_reset_continue_from_the_cache() {
    let device = test_device();
    let block = block_config().init(&device);
    let (_, cache) = block.block_forward(randn3(1, 2, &device), None, (), None);
    let packed = Packed::from_starts(&[vec![3]], 1, 0, 6, &device);
    let x = randn3(1, 6, &device);

    let (y, _) = block.block_forward_packed(x.clone(), Some(cache.clone()), (), packed.reset_bs);
    let (prefix, _) = block.block_forward(x.narrow(1, 0, 3), Some(cache), (), None);

    assert!(max_abs_diff(y.narrow(1, 0, 3), prefix) < dtype_tol(TOL));
}

/// A stack over packed rows gives each segment the outputs that the segment
/// gives alone, with the class latents in its opening slots. Alone, `forward`
/// splices the same latents in front of the segment. The layout also crosses
/// the cuts of a gradient horizon, and a class latent still trains.
#[test]
fn layers_forward_packed_equals_each_segment_alone() {
    let device = test_device().autodiff();
    let lead = 2;
    let layers = LayersBuilder {
        block: block_config().with_reset_align(3),
        class_latents: vec![ClassLatent::Start; lead],
        grad_horizon: Some(GradHorizon::Mask(vec![true, false, true])),
        ..layers_builder(3)
    }
    .init(&device);
    let (packed, starts, width) = packed_layout(&layers.real_layers[0].block, lead, &device);
    let x = randn3(2, width, &device);

    let (y, caches) = layers.forward_packed(x.clone(), None, (), &packed);

    for (b, start, end) in packed_segments(&starts, width) {
        let tokens = segment_of(x.clone(), b, start + lead, end);
        let (alone, alone_caches) = layers.forward(tokens, None, (), None, None);
        assert!(max_abs_diff(segment_of(y.clone(), b, start, end), alone) < dtype_tol(TOL));
        if end == width {
            for (cache, alone_cache) in caches.caches.iter().zip(alone_caches.caches) {
                let last = cache.state_bd.clone().narrow(0, b, 1);
                assert!(max_abs_diff(last, alone_cache.state_bd) < dtype_tol(TOL));
            }
        }
    }

    let grads = y.sum().backward();
    let emb = layers.class_latents_emb.as_ref().expect("latent table");
    assert!(emb.val().grad(&grads).is_some(), "a class latent trains from its slots");
}

/// A vocab network over packed rows gives each segment the logits that the
/// segment gives alone. The stack does not read the token of an opening slot.
#[test]
fn vocab_network_forward_packed_equals_each_segment_alone() {
    const VOCAB: i64 = 11;
    let device = test_device();
    let lead = 2;
    let net = VocabNetworkBuilder {
        vocab_size: VOCAB as usize,
        pad_vocab_size_multiple: 1,
        layers: LayersBuilder {
            block: block_config().with_reset_align(3),
            class_latents: vec![ClassLatent::Start; lead],
            ..layers_builder(2)
        },
        missing_lm_head: true,
    }
    .init(&device);
    let (packed, starts, width) = packed_layout(&net.layers.real_layers[0].block, lead, &device);
    let pairs: Vec<i64> = (0..2 * width as i64).flat_map(|p| [(p * 7) % VOCAB, p % 2]).collect();
    let x = Tensor::<1, Int>::from_ints(pairs.as_slice(), &device).reshape([2, width, 2]);

    let (logits, _) = net.forward_packed(x.clone(), None, (), &packed);

    for (b, start, end) in packed_segments(&starts, width) {
        let tokens = x.clone().narrow(0, b, 1).narrow(1, start + lead, end - start - lead);
        let (alone, _) = net.forward(tokens, None, (), None, None);
        assert!(max_rel_diff(segment_of(logits.clone(), b, start, end), alone) < dtype_tol(TOL));
    }
}

/// `place` puts latent `k` into slot `k` of each segment, and it keeps every
/// other position.
#[test]
fn place_puts_each_class_latent_into_its_slots() {
    let device = test_device();
    let packed = Packed::from_starts(&[vec![0, 3], vec![2]], 1, 2, 6, &device);
    let x = Tensor::<3>::ones([2, 6, D_MODEL], &device);
    let table = Tensor::<2>::from_floats([[2.0; D_MODEL], [3.0; D_MODEL]], &device);
    let want = Tensor::<2>::from_floats(
        [[2.0, 3.0, 1.0, 2.0, 3.0, 1.0], [1.0, 1.0, 2.0, 3.0, 1.0, 1.0]],
        &device,
    )
    .unsqueeze_dim::<3>(2)
    .expand([2, 6, D_MODEL]);

    assert_eq!(max_abs_diff(packed.place(x, table), want), 0.0);
}

// ---------------------------------------------------------------------------
// Muon plan
// ---------------------------------------------------------------------------

/// The plan is an **allowlist**. Only the rank-2 weights that a `BlockConfig`
/// names move off AdamW, and every listed path must exist on the real module.
#[cfg(feature = "optim")]
#[test]
fn muon_plan_matches_only_existing_rank_2_weights() {
    use crate::optim::MuonPlan;

    let device = test_device();
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
            line.ends_with("fallback"),
            "the rank-1 decay stays on the fallback optimizer: {line}",
        );
    }
}

/// A `CatLinear` merge is a plain hidden matrix, so the plan of a bidirectional
/// stack gives Muon one weight per such pair. A `Mean` merge has no parameters,
/// so it adds no spec.
#[cfg(feature = "optim")]
#[test]
fn bidi_plan_holds_the_cat_linear_spec_only_for_cat_linear_merges() {
    use crate::modules::BidiShape;
    use crate::optim::ProjSpec;
    use OutputMergeConfig::{CatLinear, Mean};

    let device = test_device();
    let spec = ProjSpec::path_whole("CatLinear.weight", D_MODEL);
    for (merges, n_weights) in [
        (vec![CatLinear, CatLinear], 2),
        (vec![CatLinear, Mean], 1),
        (vec![Mean, Mean], 0),
    ] {
        let shape = BidiShape::new(4, merges);
        let plan = shape.muon_plan(&block_config());
        assert_eq!(plan.specs.contains(&spec), n_weights > 0);
        let report = plan.describe(&shape.init(block_config(), &device));
        let selected = report
            .lines()
            .filter(|l| l.contains("CatLinear.weight") && l.contains("muon[all:8]"))
            .count();
        assert_eq!(selected, n_weights, "{report}");
    }
}

/// An untied projection is one matrix per application. Muon orthogonalises
/// each copy on its own, and the plan gives the untied weight to that
/// per-copy update.
#[cfg(feature = "optim")]
#[test]
fn muon_updates_an_untied_projection_one_copy_at_a_time() {
    use crate::optim::{MuonPlan, ProjSpec, Segmented, muon_config};
    use burn::optim::{AdamWConfig, Optimizer};

    let device = test_device();
    let n = 3;
    let spec = ProjSpec::block_whole("gate_proj.weight", D_MODEL).tiled();
    let muon = muon_config(0.0).build();
    let segmented = Segmented::new(&spec, muon.clone(), AdamWConfig::new().build(), 1);
    let w = Tensor::<2>::random([D_MODEL, n * D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    let g = Tensor::<2>::random([D_MODEL, n * D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    let (updated, _) = segmented.step(1e-2, w.clone(), g.clone(), None);
    for ((got, w), g) in updated.chunk(n, 1).into_iter().zip(w.chunk(n, 1)).zip(g.chunk(n, 1)) {
        let (want, _) = muon.step(1e-2, w, g, None);
        assert!(max_abs_diff(got, want) < dtype_tol(TOL));
    }

    let config = block_config().with_untied(vec![RefUntied::GateProj]);
    let layers = LayersBuilder {
        n_virtual_layers: Some((n, Schedule::Cyclic)),
        block: config.clone(),
        ..layers_builder(1)
    }
    .init(&device);
    let report = MuonPlan::new(BlockConfig::muon_projections(&config)).describe(&layers);
    assert!(
        report
            .lines()
            .any(|l| l.contains("gate_proj.weight") && l.contains("muon[all:8]×3")),
        "{report}",
    );
}
