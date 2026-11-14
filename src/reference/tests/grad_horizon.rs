//! The gradient horizon: the cuts of a `grad_horizon`, the straight-through
//! carry across them, and the backend hops.
//!
//! A cut can let a tracked tensor into an untracked span and still give the
//! correct outputs and the correct gradients of the parameters. Then the
//! graph keeps the span, and the memory saving is lost. So these tests also
//! check the graph itself:
//!
//! - whether a loss on what an untracked layer produced reaches a tracked
//!   input. ADR-0003 tells why a test cannot use `is_require_grad` for this.
//! - whether each parameter of an untracked layer gets no gradient.
//!
//! `grad_horizon_memory_probe` measures the memory saving itself.

use super::*;
use crate::modules::Residuals;
use crate::utils::ClassCursors;
use burn::module::ModuleVisitor;
use burn::tensor::Gradients;

/// The gradient of each float parameter of `module` in `grads`, flattened,
/// with the path of the parameter. `None`: the parameter has no gradient.
///
/// `src/utils/padding/tests.rs` has a copy without the paths.
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

/// Panic unless each float parameter of real layer `i` has a gradient in
/// `grads` exactly when `tracked[i]`. The stack must run each real layer once,
/// so that real layer `i` is virtual layer `i`.
fn assert_only_tracked_layers_train(layers: &Layers<RefBlock>, grads: &Gradients, tracked: &[bool], label: &str) {
    assert!(layers.n_virtual_layers.is_none(), "the stack runs each real layer once");
    assert_eq!(layers.real_layers.len(), tracked.len(), "one flag for each real layer");
    for (i, (layer, &tracked)) in layers.real_layers.iter().zip(tracked).enumerate() {
        for (path, grad) in param_grads(layer, grads) {
            match tracked {
                true => assert!(grad.is_some(), "{label}: `{path}` of the tracked layer {i} gets no gradient"),
                false => assert!(grad.is_none(), "{label}: `{path}` of the untracked layer {i} gets a gradient"),
            }
        }
    }
}

/// Panic unless each float parameter of `module` has a gradient in `want`
/// and in `got`, and the two gradients match.
fn assert_same_grads(module: &impl Module, want: &Gradients, got: &Gradients, label: &str) {
    let want = param_grads(module, want);
    let got = param_grads(module, got);
    for ((path, want), (_, got)) in want.into_iter().zip(got) {
        let want = want.unwrap_or_else(|| panic!("{label}: `{path}` gets no gradient in the reference"));
        let got = got.unwrap_or_else(|| panic!("{label}: `{path}` gets no gradient"));
        let diff = max_rel_diff(got, want);
        assert!(diff < dtype_tol(TOL), "{label}: the gradients of `{path}` differ by {diff}");
    }
}

/// Panic unless each final cache in `got` matches its cache in `want`.
fn assert_same_caches(want: &RefCaches, got: &RefCaches, label: &str) {
    assert_eq!(want.caches.len(), got.caches.len(), "{label}: the counts of the cache slots differ");
    for (i, (want, got)) in want.caches.iter().zip(&got.caches).enumerate() {
        let diff = max_rel_diff(got.state_bd.clone(), want.state_bd.clone());
        assert!(diff < dtype_tol(TOL), "{label}: the final caches of virtual layer {i} differ by {diff}");
    }
}

/// A loss that weighs the outputs and each final cache of a stack by fixed
/// random weights, with the shapes of `y` and `caches`. In a plain sum, a sign
/// error or an order error can cancel. In a sum with random weights, it
/// stays.
fn weighted_loss(y: &Tensor<3>, caches: &RefCaches) -> impl Fn(Tensor<3>, RefCaches) -> Tensor<1> + use<> {
    let normal = Distribution::Normal(0.0, 1.0);
    let device = y.device();
    let y_weights = Tensor::<3>::random(y.dims(), normal, &device);
    let state_weights: Vec<Tensor<2>> =
        caches.caches.iter().map(|c| Tensor::random(c.state_bd.dims(), normal, &device)).collect();
    move |y, caches| {
        let states = caches.caches.into_iter().zip(&state_weights);
        states.fold((y * y_weights.clone()).sum(), |acc, (c, w)| acc + (c.state_bd * w.clone()).sum())
    }
}

/// A stack of 6 virtual layers over 2 real layers, with a cyclic layer
/// schedule.
fn shared_stack(device: &Device) -> Layers<RefBlock> {
    LayersBuilder {
        n_virtual_layers: Some((6, Schedule::Cyclic)),
        ..layers_builder(2)
    }
    .init(device)
}

/// A cyclic stack split by hand into two stacks with no horizon: its first
/// `n_lower` virtual layers, and the virtual layers above them. Both stacks
/// hold the real layers of `layers`. `n_lower` is a multiple of the real
/// layers, so the two layer schedules continue the schedule of `layers`.
fn split_by_hand(layers: &Layers<RefBlock>, n_lower: usize) -> (Layers<RefBlock>, Layers<RefBlock>) {
    let Some((n_virtual, Schedule::Cyclic)) = layers.n_virtual_layers.clone() else {
        panic!("the split needs a stack with a cyclic layer schedule");
    };
    assert!(n_lower.is_multiple_of(layers.n_real_layers), "the split breaks the layer schedule");
    let part = |n: usize| {
        let mut part = layers.clone();
        part.n_virtual_layers = Some((n, Schedule::Cyclic));
        part.grad_horizon = None;
        part
    };
    (part(n_lower), part(n_virtual - n_lower))
}

/// A `grad_horizon` that covers the whole stack must be a no-op on the values
/// and on the gradients. A shorter one must cut the gradient to the
/// parameters of the untracked layers, and keep the stack input reachable
/// (the straight-through re-attachment).
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
        max_abs_diff(y_full.clone(), y_uncut.clone()) < dtype_tol(TOL),
        "a horizon tracking everything must reproduce the untouched stack exactly",
    );
    let label = "a horizon that tracks every layer";
    assert_same_grads(&full, &y_full.sum().backward(), &y_uncut.sum().backward(), label);

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

/// No tracked input reaches what an untracked layer produced.
///
/// The check takes the backward of the sum of the final caches of the
/// untracked layers only, not of the stack output. A tracked layer reads the
/// incoming cache of its own slot, so a gradient to that cache is correct. A
/// separate tracked leaf keeps the graph of the sum not empty.
///
/// Two inputs must not reach the sum: the stack input, and the input of an
/// earlier tracked forward whose caches enter the cut forward. That second
/// input reaches the sum when an untracked layer reads its incoming cache on
/// the autodiff backend.
#[test]
fn nothing_tracked_reaches_the_untracked_caches() {
    let device = test_device().autodiff();
    let (n, k) = (4, 2);
    let mut layers = layers(n, &device);
    let grads_of_untracked_caches = |caches: RefCaches| {
        let anchor = Tensor::<1>::zeros([1], &device).require_grad();
        let untracked = caches.caches.into_iter().take(n - k);
        let grads = untracked.fold(anchor.clone().sum(), |acc, c| acc + c.state_bd.sum()).backward();
        assert!(anchor.grad(&grads).is_some(), "the anchor is not in the graph");
        grads
    };

    // With no horizon, the input reaches the sum. So the check can fail.
    let x = randn3(2, 4, &device).require_grad();
    let (_, caches) = layers.forward(x.clone(), None, (), None, None);
    assert!(
        x.grad(&grads_of_untracked_caches(caches)).is_some(),
        "with no horizon, the stack input does not reach the caches, so the check cannot fail",
    );

    // An earlier tracked forward. Its graph stays, because no backward walks it.
    let x_earlier = randn3(2, 4, &device).require_grad();
    let (_, earlier_caches) = layers.forward(x_earlier.clone(), None, (), None, None);

    layers.grad_horizon = Some(GradHorizon::last(k, n));
    let x = randn3(2, 4, &device).require_grad();
    let (_, caches) = layers.forward(x.clone(), None, (), None, None);
    assert!(
        x.grad(&grads_of_untracked_caches(caches)).is_none(),
        "the stack input reaches the caches of the untracked layers",
    );

    let x = randn3(2, 4, &device).require_grad();
    let (_, caches) = layers.forward(x.clone(), Some(earlier_caches), (), None, None);
    let grads = grads_of_untracked_caches(caches);
    assert!(
        x.grad(&grads).is_none(),
        "from incoming caches, the stack input reaches the caches of the untracked layers",
    );
    assert!(
        x_earlier.grad(&grads).is_none(),
        "the input of the earlier forward reaches the caches of the untracked layers. Their \
         incoming caches stayed on the autodiff backend",
    );
}

/// With each real layer applied once, each parameter of an untracked layer
/// gets no gradient, and each parameter of a tracked layer gets one. A
/// parameter that the cut misses is still in the graph, so it gets a
/// gradient. The check visits each float parameter: the pre-norm gain, the
/// three maps and the per-channel decay of the block.
#[test]
fn untracked_layers_get_no_gradient() {
    let device = test_device().autodiff();
    let mut layers = layers(4, &device);
    layers.grad_horizon = Some(GradHorizon::last(2, 4));

    let (y, _) = layers.forward(randn3(2, 4, &device), None, (), None, None);
    let grads = y.sum().backward();
    assert_only_tracked_layers_train(&layers, &grads, &[false, false, true, true], "the forward");
}

/// The step takes the cut of the forward through its cascade, which runs
/// each token through the depth. So each parameter of an untracked layer
/// gets no gradient, and each parameter of a tracked layer gets one. The
/// input keeps its gradient through the straight-through carry.
#[test]
fn stepped_untracked_layers_get_no_gradient() {
    let device = test_device().autodiff();
    let mut layers = layers(4, &device);
    layers.grad_horizon = Some(GradHorizon::last(2, 4));

    let x = Tensor::<2>::random([2, D_MODEL], Distribution::Normal(0.0, 1.0), &device).require_grad();
    let (y, _) = layers.step(x.clone(), None, None);
    let grads = y.sum().backward();
    assert_only_tracked_layers_train(&layers, &grads, &[false, false, true, true], "the step");
    assert!(x.grad(&grads).is_some(), "the step cuts the gradient of its input");
}

/// Under weight sharing, a real layer runs on both sides of a cut. Its
/// gradient is the gradient of its tracked applications alone.
///
/// The reference splits the stack by hand, over the same real layers. An
/// inner-backend stack runs the 4 untracked layers. Then a tracked stack runs
/// the 2 tracked layers. A clone keeps the parameter ids, so the gradients of
/// a clone are the gradients of the stack.
#[test]
fn shared_weight_grad_counts_tracked_applications_only() {
    let device = test_device().autodiff();
    let (n_virtual, k) = (6, 2);
    let mut layers = shared_stack(&device);
    layers.grad_horizon = Some(GradHorizon::last(k, n_virtual));
    let x = randn3(2, 4, &device);
    let (y, _) = layers.forward(x.clone(), None, (), None, None);

    let (untracked, tracked) = split_by_hand(&layers, n_virtual - k);
    let (h, _) = untracked.valid().forward(x.inner(), None, (), None, None);
    let (y_ref, _) = tracked.forward(Tensor::from_inner(h), None, (), None, None);

    assert!(
        max_rel_diff(y.clone(), y_ref.clone()) < dtype_tol(TOL),
        "the cut stack and the stack split by hand give different outputs",
    );
    let label = "the cut stack and the stack split by hand";
    assert_same_grads(&layers, &y_ref.sum().backward(), &y.sum().backward(), label);
}

/// Off the autodiff backend, a horizon does nothing: a stack with a horizon
/// gives the outputs of the same stack with none. So a decode can keep the
/// horizon of its training config.
#[test]
fn horizon_is_inert_without_autodiff() {
    let device = test_device();
    assert!(!device.is_autodiff(), "this test needs a plain device");
    let plain = layers(4, &device);
    let mut cut = plain.clone();
    cut.grad_horizon = Some(GradHorizon::last(2, 4));

    let x = randn3(2, 4, &device);
    let (want, _) = plain.forward(x.clone(), None, (), None, None);
    let (got, _) = cut.forward(x, None, (), None, None);
    assert!(max_abs_diff(got, want) < dtype_tol(TOL), "a horizon changes the outputs on a plain device");
}

/// A window split in two through the caches equals one window: on the
/// outputs, on each final cache and on the gradient of each parameter.
///
/// The gradients matter under a cut. The caches of the untracked layers
/// return from the inner backend with no gradient. The caches of the tracked
/// layers carry the gradient across the split. A cut that detached every
/// cache keeps the outputs and the caches, and changes only the gradients.
fn assert_split_matches_one_window(horizon: Option<GradHorizon>) {
    let device = test_device().autodiff();
    let mut layers = shared_stack(&device);
    layers.grad_horizon = horizon.clone();
    let x = randn3(2, 8, &device);

    let (y_all, caches_all) = layers.forward(x.clone(), None, (), None, None);
    let loss = weighted_loss(&y_all, &caches_all);
    let (y_first, mid) = layers.forward(x.clone().narrow(1, 0, 3), None, (), None, None);
    let (y_second, caches_split) = layers.forward(x.narrow(1, 3, 5), Some(mid), (), None, None);
    let y_split = Tensor::cat(vec![y_first, y_second], 1);

    let label = format!("the split and one window (horizon {horizon:?})");
    let diff = max_rel_diff(y_split.clone(), y_all.clone());
    assert!(diff < dtype_tol(TOL), "{label}: the outputs differ by {diff}");
    assert_same_caches(&caches_all, &caches_split, &label);
    assert_same_grads(&layers, &loss(y_all, caches_all).backward(), &loss(y_split, caches_split).backward(), &label);
}

/// The control of the test below: with no horizon, the split already equals
/// one window.
#[test]
fn split_forward_matches_single_forward() {
    assert_split_matches_one_window(None);
}

#[test]
fn split_forward_matches_single_forward_under_horizon() {
    assert_split_matches_one_window(Some(GradHorizon::last(2, 6)));
}

/// The step unrolled over a window equals one forward over it: on the
/// outputs, on each final cache and on the gradient of each parameter.
///
/// Under a cut, this pins the cut of the step to the cut of the forward. The
/// forward runs each layer over the sequence. The step runs each token
/// through the depth. A step that cuts at other layers, or does not cut,
/// still gives the values of the forward. Only its gradients differ.
fn assert_step_matches_forward(horizon: Option<GradHorizon>) {
    let device = test_device().autodiff();
    let mut layers = shared_stack(&device);
    layers.grad_horizon = horizon.clone();
    let x = randn3(2, 4, &device);

    let (y_fwd, caches_fwd) = layers.forward(x.clone(), None, (), None, None);
    let loss = weighted_loss(&y_fwd, &caches_fwd);
    let (y_step, caches_step) = step_unrolled_with_caches(&layers, x);

    let label = format!("the step and the forward (horizon {horizon:?})");
    let diff = max_rel_diff(y_step.clone(), y_fwd.clone());
    assert!(diff < dtype_tol(TOL), "{label}: the outputs differ by {diff}");
    assert_same_caches(&caches_fwd, &caches_step, &label);
    assert_same_grads(&layers, &loss(y_fwd, caches_fwd).backward(), &loss(y_step, caches_step).backward(), &label);
}

/// The control of the test below: with no horizon, the step already equals
/// the forward.
#[test]
fn step_matches_forward() {
    assert_step_matches_forward(None);
}

#[test]
fn step_matches_forward_under_horizon() {
    assert_step_matches_forward(Some(GradHorizon::last(2, 6)));
}

/// The prime under a cut, with a class latent of the stack. The prime has no
/// user token, so the latent is the whole stream. The stack splices the
/// latent below the first layer, so the stream crosses the cut. The cut must
/// lift the stream before a tracked layer reads it. Else the tracked layer
/// mixes two backends, and Burn panics.
///
/// Only the tracked layers are in the graph of the primed output. So it gives
/// a gradient to each of them and to no untracked layer. The latent trains
/// through the straight-through carry.
#[test]
fn prime_under_a_cut_with_a_latent_below_it() {
    let device = test_device().autodiff();
    let layers = LayersBuilder {
        class_latents: vec![ClassLatent::Start],
        grad_horizon: Some(GradHorizon::last(2, 4)),
        ..layers_builder(4)
    }
    .init(&device);

    let mut class = ClassCursors::new(4);
    let (primed, _) = layers.prime(2, None, Some(&mut class));
    let grads = primed.expect("a latent waits at the start").sum().backward();
    assert_only_tracked_layers_train(&layers, &grads, &[false, false, true, true], "the prime");
    let emb = layers.class_latents_emb.as_ref().expect("latent table");
    assert!(emb.val().grad(&grads).is_some(), "the class latent of the stack does not train");
}

/// A cut keeps the gradient of what produced the stack input. The input
/// enters the stack once, at the bottom, and rides the residual stream. A cut
/// at the bottom severs its only path, so the `in_proj` of a network would
/// never train. The straight-through carry keeps that path. The check runs a
/// whole `LatentNetwork`, from no horizon to a horizon that tracks no layer.
#[test]
fn the_io_weights_of_a_network_keep_their_gradient_under_a_cut() {
    let device = test_device().autodiff();
    let n = 4;
    for k in [None, Some(4), Some(2), Some(0)] {
        let horizon = k.map(|k| GradHorizon::last(k, n));
        let net = LatentNetworkBuilder {
            input_size: 3,
            layers: LayersBuilder { grad_horizon: horizon.clone(), ..layers_builder(n) },
            output_size: 3,
            final_norm: false,
            class_tokens: Vec::new(),
        }
        .init(&device);

        let x = Tensor::random([2, 4, 3], Distribution::Normal(0.0, 1.0), &device).require_grad();
        let (y, _) = net.forward(x.clone(), None, (), None, None);
        let grads = y.sum().backward();
        assert!(x.grad(&grads).is_some(), "the network input gets no gradient (horizon {horizon:?})");
        assert!(
            net.in_proj.weight.val().grad(&grads).is_some(),
            "`in_proj` gets no gradient (horizon {horizon:?})",
        );
        assert!(
            net.out_proj.weight.val().grad(&grads).is_some(),
            "`out_proj` gets no gradient (horizon {horizon:?})",
        );
    }
}

/// The straight-through carry adds a tensor of value zero. So a cut keeps the
/// outputs and the final caches. The horizons include one that tracks no
/// layer, which ends the stack inside an untracked span.
#[test]
fn a_cut_keeps_the_outputs_and_the_caches() {
    let device = test_device().autodiff();
    let n = 4;
    let plain = layers(n, &device);
    let x = randn3(2, 4, &device);
    let (want, want_caches) = plain.forward(x.clone(), None, (), None, None);

    for k in [4, 3, 1, 0] {
        let mut cut = plain.clone();
        cut.grad_horizon = Some(GradHorizon::last(k, n));
        let (got, got_caches) = cut.forward(x.clone(), None, (), None, None);
        let label = format!("a horizon of the last {k} layers");
        let diff = max_rel_diff(got, want.clone());
        assert!(diff < dtype_tol(TOL), "{label}: the outputs differ by {diff}");
        assert_same_caches(&want_caches, &got_caches, &label);
    }
}

/// When the untracked span is an identity, the straight-through carry is
/// exact. So the gradient of the input under the cut equals its gradient
/// with no cut.
///
/// A zero `out_proj` makes the output of the block of an untracked layer
/// zero. So the residual stream passes through the layer unchanged, and the
/// true gradient across the span is the identity that the straight-through
/// carry gives. The cut changes no value, so the gradient at the cut is the
/// same in both runs.
///
/// This pins the straight-through carry of the residual streams of
/// Multi-Gate. Under `Standard`, the token holds the residual. So the
/// straight-through carry of the token is the whole correction. Under
/// Multi-Gate, the residual streams hold the residual. With a straight-through
/// carry on the pooled token only, the gradient of the input omits the
/// gradient of the streams, and the two runs differ.
fn assert_identity_span_is_exact(residuals: ResidualsConfig) {
    let device = test_device().autodiff();
    let (n, k) = (4, 2);
    let mut base = LayersBuilder { residuals, ..layers_builder(n) }.init(&device);
    for layer in &mut base.real_layers[..n - k] {
        let w = &mut layer.block.out_proj.weight;
        *w = w.clone().map(|t| t.zeros_like());
    }
    // Under Multi-Gate, an untracked layer is an identity only when its gate
    // keeps the streams. A strong negative bias closes the gates of the
    // untracked layers only, so the tracked layers still mix.
    if let Residuals::MultiGate(mg) = &mut base.residuals {
        for gate in &mut mg.layers[..n - k] {
            let b = &mut gate.b_beta;
            *b = b.clone().map(|t| t.zeros_like() - 20.0);
        }
    }

    let x = randn3(2, 4, &device);
    let head = randn3(2, 4, &device);
    // The same weights in both runs. Only the horizon differs.
    let input_grad = |horizon: Option<GradHorizon>| {
        let mut layers = base.clone();
        layers.grad_horizon = horizon;
        let x = x.clone().require_grad();
        let (y, _) = layers.forward(x.clone(), None, (), None, None);
        x.grad(&(y * head.clone()).sum().backward()).expect("the input gets a gradient")
    };
    let want = input_grad(None);
    let got = input_grad(Some(GradHorizon::last(k, n)));

    // A stack that is an identity everywhere gives the gradient `head`. Then
    // the two runs can agree also with a wrong straight-through carry.
    let off_identity = max_abs_diff(want.clone(), head);
    assert!(off_identity > 1e-2, "the tracked layers are identities too, so the check cannot fail");
    let diff = max_rel_diff(got, want);
    assert!(
        diff < dtype_tol(TOL),
        "the cut changes the gradient of the input by {diff} across an identity span. The token \
         or a residual stream has no straight-through carry",
    );
}

/// The token holds the whole residual.
#[test]
fn an_identity_untracked_span_is_exact_under_standard_residuals() {
    assert_identity_span_is_exact(ResidualsConfig::Standard);
}

/// One residual stream with no accumulation phase. The closed gates keep the
/// untracked layers identities, and the open gates of the tracked layers mix.
#[test]
fn an_identity_untracked_span_is_exact_under_multi_gate_residuals() {
    assert_identity_span_is_exact(ResidualsConfig::MultiGate {
        n_stream: 1,
        init_bias: 0.0,
        init_bias_offset: 0.0,
        per_virtual_layer: false,
    });
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

/// A probe of the peak RSS of a training step under a horizon: the memory
/// that a cut saves. The `RefBlock` is wide, so the activations set the peak,
/// not the baseline of the process.
///
/// ADR-0003 gives the mechanism. A layer with detached parameters still
/// registers a step in the graph for each of its ops, and the graph keeps
/// these steps until a backward reaches them. A layer on the inner backend
/// registers nothing.
///
/// The peak RSS is for the whole process. So each configuration runs in its
/// own process:
///
/// ```bash
/// for nv in 8 32 64; do for k in none 2; do
///   BURN_STACK_N_VIRTUAL=$nv BURN_STACK_GRAD_HORIZON=$k \
///     cargo test --lib grad_horizon_memory_probe -- --ignored --nocapture
/// done; done
/// ```
///
/// Two more configurations give the reference curves:
///
/// - `BURN_STACK_PLAIN=1` runs the forward with no autodiff and no backward:
///   the working set of the forward alone.
/// - `BURN_STACK_DETACHED=1` runs the untracked layers with detached
///   parameters on the autodiff backend, the mechanism that ADR-0003 rejects.
///   It needs a horizon.
///
/// ```bash
/// BURN_STACK_N_VIRTUAL=64 BURN_STACK_PLAIN=1 \
///   cargo test --lib grad_horizon_memory_probe -- --ignored --nocapture
/// BURN_STACK_N_VIRTUAL=64 BURN_STACK_GRAD_HORIZON=2 BURN_STACK_DETACHED=1 \
///   cargo test --lib grad_horizon_memory_probe -- --ignored --nocapture
/// ```
#[test]
#[ignore = "peak-RSS probe: one process for each configuration, see the doc comment"]
fn grad_horizon_memory_probe() {
    let (n_real, d_model, batch, sequence) = (2, 256, 8, 128);
    let n_virtual: usize = match std::env::var("BURN_STACK_N_VIRTUAL") {
        Ok(v) => v.parse().expect("BURN_STACK_N_VIRTUAL: an integer"),
        Err(_) => 32,
    };
    let horizon_k: Option<usize> = match std::env::var("BURN_STACK_GRAD_HORIZON").as_deref() {
        Ok("none") | Err(_) => None,
        Ok(k) => Some(k.parse().expect("BURN_STACK_GRAD_HORIZON: `none` or an integer")),
    };
    let plain = std::env::var("BURN_STACK_PLAIN").is_ok();
    let detached = std::env::var("BURN_STACK_DETACHED").is_ok();

    let device = match plain {
        true => test_device(),
        false => test_device().autodiff(),
    };
    let mut layers = LayersBuilder {
        n_virtual_layers: Some((n_virtual, Schedule::Cyclic)),
        ..LayersBuilder::new(n_real, RefBlockConfig::new(d_model))
    }
    .init(&device);
    layers.grad_horizon = horizon_k.map(|k| GradHorizon::last(k, n_virtual));

    let x = Tensor::<3>::random([batch, sequence, d_model], Distribution::Normal(0.0, 1.0), &device);
    let x = if plain { x } else { x.require_grad() };
    let y = match detached {
        false => layers.forward(x, None, (), None, None).0,
        true => {
            let k = horizon_k.expect("BURN_STACK_DETACHED needs a horizon");
            let (untracked, tracked) = split_by_hand(&layers, n_virtual - k);
            let untracked = crate::utils::detach_params(untracked);
            let (h, _) = untracked.forward(x.detach(), None, (), None, None);
            tracked.forward(h, None, (), None, None).0
        }
    };
    match plain {
        true => drop(y.sum()),
        false => drop(y.sum().backward()),
    }

    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let peak = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .map_or("unavailable".to_string(), |v| v.trim().to_string());
    let mechanism = match (plain, detached, horizon_k) {
        (true, _, _) => "plain, no autodiff",
        (false, true, _) => "untracked layers with detached parameters",
        (false, false, None) => "every layer tracked",
        (false, false, Some(_)) => "untracked layers on the inner backend, as the horizon runs them",
    };
    let horizon = horizon_k.map_or("none".to_string(), |k| format!("the last {k} layers"));
    eprintln!(
        "horizon = {horizon} ({mechanism}) | n_virtual = {n_virtual}, d_model = {d_model}, batch = {batch}, \
         sequence = {sequence} | peak RSS = {peak}",
    );
}
