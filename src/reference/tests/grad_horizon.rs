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
use crate::utils::test_helpers::panic_text;
use burn::tensor::Gradients;

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

/// A loss that weighs the outputs and each final cache of a stack by fixed
/// random weights, with the shapes of `y` and `caches`. In a plain sum, a sign
/// error or an order error can cancel. In a sum with random weights, it
/// stays.
fn weighted_loss(y: &Tensor<3>, caches: &[RefCache]) -> impl Fn(Tensor<3>, Vec<RefCache>) -> Tensor<1> + use<> {
    let normal = Distribution::Normal(0.0, 1.0);
    let device = y.device();
    let y_weights = Tensor::<3>::random(y.dims(), normal, &device);
    let state_weights: Vec<Tensor<2>> =
        caches.iter().map(|c| Tensor::random(c.state_bd.dims(), normal, &device)).collect();
    move |y, caches| {
        let states = caches.into_iter().zip(&state_weights);
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
    let grads_of_untracked_caches = |caches: Vec<RefCache>| {
        let anchor = Tensor::<1>::zeros([1], &device).require_grad();
        let untracked = caches.into_iter().take(n - k);
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

/// The gates of the untracked layers in [`assert_zero_transform_span_is_exact`].
/// The tracked layers keep the init gates, so they still mix.
#[derive(Clone, Copy, Debug)]
enum UntrackedGates {
    /// `b = −20`: each gate keeps the streams, so each untracked layer is an
    /// identity.
    Closed,
    /// The init `b`. At `b = 0`, `β = 0.5`, so the carry path of each stream
    /// halves at each mixing layer.
    Open,
    /// The MGR modules of the untracked layers alternate from the bottom:
    /// closed, open, closed, …. So the modules of a span differ.
    Alternating,
}

/// How [`assert_zero_transform_span_is_exact`] calls the stack.
#[derive(Clone, Copy, Debug)]
enum Call {
    /// `Layers::forward` over a window. The window is the input.
    Forward,
    /// `Layers::step` over each token of the window. The window is the input.
    Step,
    /// `Layers::prime` of one `Start` class latent of the stack. The latent is
    /// the input.
    Prime,
}

/// Each call of a stack.
const CALLS: [Call; 3] = [Call::Forward, Call::Step, Call::Prime];

/// The class latents of a stack for `call`: one `Start` latent for the prime,
/// and none for the other calls.
fn class_latents_for(call: Call) -> Vec<ClassLatent> {
    match call {
        Call::Prime => vec![ClassLatent::Start],
        Call::Forward | Call::Step => Vec::new(),
    }
}

/// A stack and its cut for [`assert_zero_transform_span_is_exact`].
#[derive(Clone, Debug)]
struct ZeroTransformSpan {
    /// The number of real layers.
    n_real: usize,
    /// The layer schedule `(n_virtual_layers, schedule)`. `None`: each real
    /// layer runs once.
    n_virtual_layers: Option<(usize, Schedule)>,
    /// The residual scheme.
    residuals: ResidualsConfig,
    /// `ignore_first_residual` and `ignore_last_residual` of the stack.
    ignore: (bool, bool),
    /// The tracked flag of each virtual layer in the run with a cut. No real
    /// layer and no MGR module runs on both sides of a cut.
    tracked: Vec<bool>,
    /// The gates of the untracked layers.
    gates: UntrackedGates,
}

/// The real layer and the MGR module of virtual layer `i`. The module is
/// `None` under Standard.
fn parts_of(layers: &Layers<RefBlock>, i: usize) -> (usize, Option<usize>) {
    let real = layers.real_idx(i);
    let module = match &layers.residuals {
        Residuals::MultiGate(mg) => Some(mg.module_index(i, real)),
        Residuals::Standard(_) => None,
    };
    (real, module)
}

impl ZeroTransformSpan {
    /// A stack of 4 layers whose bottom 2 layers are untracked.
    fn bottom(residuals: ResidualsConfig, gates: UntrackedGates) -> Self {
        let tracked = vec![false, false, true, true];
        Self { n_real: 4, n_virtual_layers: None, residuals, ignore: (false, false), tracked, gates }
    }

    /// The stack for `call`, with no horizon. The untracked layers have a
    /// zero `out_proj`, so their transforms are zero. Their MGR modules have
    /// the gates of [`Self::gates`]. Returns the stack, the real layers of the
    /// tracked layers and their MGR modules.
    fn stack(&self, call: Call, device: &Device) -> (Layers<RefBlock>, Vec<usize>, Vec<usize>) {
        let mut layers = LayersBuilder {
            n_virtual_layers: self.n_virtual_layers.clone(),
            residuals: self.residuals.clone(),
            ignore_first_residual: self.ignore.0,
            ignore_last_residual: self.ignore.1,
            class_latents: class_latents_for(call),
            ..layers_builder(self.n_real)
        }
        .init(device);
        assert_eq!(self.tracked.len(), layers.n_virtual_count(), "one flag for each virtual layer");
        let side = |tracked: bool| -> Vec<(usize, Option<usize>)> {
            let virtual_layers = (0..self.tracked.len()).filter(|&i| self.tracked[i] == tracked);
            virtual_layers.map(|i| parts_of(&layers, i)).collect()
        };
        let (kept, untracked) = (side(true), side(false));
        for (real, module) in &untracked {
            assert!(
                kept.iter().all(|(r, m)| r != real && (m.is_none() || m != module)),
                "real layer {real} or MGR module {module:?} runs on both sides of the cut",
            );
        }

        for (real, _) in &untracked {
            let w = &mut layers.real_layers[*real].block.out_proj.weight;
            *w = w.clone().map(|t| t.zeros_like());
        }
        if let Residuals::MultiGate(mg) = &mut layers.residuals {
            for (j, module) in untracked.iter().filter_map(|(_, m)| *m).enumerate() {
                let closed = match self.gates {
                    UntrackedGates::Closed => true,
                    UntrackedGates::Open => false,
                    UntrackedGates::Alternating => j % 2 == 0,
                };
                if closed {
                    let b = &mut mg.layers[module].b_beta;
                    *b = b.clone().map(|t| t.zeros_like() - 20.0);
                }
            }
        }
        let (mut kept_layers, kept_modules): (Vec<usize>, Vec<Option<usize>>) = kept.into_iter().unzip();
        let mut kept_modules: Vec<usize> = kept_modules.into_iter().flatten().collect();
        for parts in [&mut kept_layers, &mut kept_modules] {
            parts.sort_unstable();
            parts.dedup();
        }
        (layers, kept_layers, kept_modules)
    }
}

/// The gradient of `x` in `grads`, flattened. It is zero when the graph does
/// not reach `x`.
fn flat_grad<const D: usize>(x: &Tensor<D>, grads: &Gradients) -> Tensor<1> {
    x.grad(grads).unwrap_or_else(|| x.zeros_like().inner()).reshape([-1])
}

/// The head of the prime: the first token of `head`. The prime gives one
/// token.
fn prime_head(head: &Tensor<3>) -> Tensor<2> {
    head.clone().narrow(1, 0, 1).squeeze_dim::<2>(1)
}

/// One run of `layers` in `call`, with the loss `Σ y · head`. In the forward
/// and in the step, `x` is the input. The prime reads only the shape of `x`.
/// Returns the gradient of the input ([`flat_grad`]) and the gradients of the
/// run.
fn run_call(layers: &Layers<RefBlock>, call: Call, x: &Tensor<3>, head: &Tensor<3>) -> (Tensor<1>, Gradients) {
    match call {
        Call::Forward => {
            let x = x.clone().require_grad();
            let (y, _) = layers.forward(x.clone(), None, (), None, None);
            let grads = (y * head.clone()).sum().backward();
            (flat_grad(&x, &grads), grads)
        }
        Call::Step => {
            let x = x.clone().require_grad();
            let y = step_unrolled(layers, x.clone());
            let grads = (y * head.clone()).sum().backward();
            (flat_grad(&x, &grads), grads)
        }
        Call::Prime => {
            // One latent row for each row of the batch.
            let [batch, sequence, _] = x.dims();
            let mut class = ClassCursors::new(sequence);
            let (y, _) = layers.prime(batch, None, Some(&mut class));
            let grads = (y.expect("a latent waits at the start") * prime_head(head)).sum().backward();
            let emb = layers.class_latents_emb.as_ref().expect("the latent table").val();
            (flat_grad(&emb, &grads), grads)
        }
    }
}

/// The gradient of the input of `call` in a stack that is an identity
/// everywhere, flattened.
fn identity_grad(call: Call, head: &Tensor<3>) -> Tensor<1> {
    match call {
        Call::Forward | Call::Step => head.clone().reshape([-1]),
        // Each row of the batch reads the same latent row.
        Call::Prime => prime_head(head).sum_dim(0).reshape([-1]),
    }
}

/// [`assert_zero_transform_span_is_exact`] with closed gates under the cut of
/// [`ZeroTransformSpan::bottom`]: the untracked span is an identity.
fn assert_identity_span_is_exact(residuals: ResidualsConfig) {
    assert_zero_transform_span_is_exact(&ZeroTransformSpan::bottom(residuals, UntrackedGates::Closed), Call::Forward);
}

/// When the untracked layers have zero transforms, the straight-through carry
/// is exact. So under the cut, two gradients equal their values with no cut:
/// that of the input, and that of each parameter of a tracked layer. The
/// tracked layers below a cut get their gradient through the straight-through
/// carry. The parameters of the tracked layers include their MGR modules.
///
/// A zero `out_proj` makes the output of the block of an untracked layer
/// zero. So only the carry path of the residual scheme crosses the span:
///
/// - Under `Standard`, the carry path is the skip, with gain 1. The token
///   holds the residual, so the straight-through carry of the token is the
///   whole correction.
/// - Under Multi-Gate, the residual streams hold the residual. The carry path
///   of stream `i` has gain `1 − βᵢ` at each mixing layer. A stream that the
///   accumulation phase appends holds zero.
///
/// The init `w = 0` makes the scores of the gates and of the aggregator
/// constant. So the gradient across the span is the gain of each carry path.
/// The cut changes no value, so the gradient at the cut is the same in both
/// runs. A straight-through carry of the pooled token, or a gain of 1 across
/// an open gate, gives a different gradient of the input.
fn assert_zero_transform_span_is_exact(span: &ZeroTransformSpan, call: Call) {
    let device = test_device().autodiff();
    let label = format!("{span:?} in {call:?}");
    let (base, kept_layers, kept_modules) = span.stack(call, &device);

    let x = randn3(2, 4, &device);
    let head = randn3(2, 4, &device);
    // The same weights in both runs. Only the horizon differs.
    let run = |horizon: Option<GradHorizon>| {
        let mut layers = base.clone();
        layers.grad_horizon = horizon;
        run_call(&layers, call, &x, &head)
    };
    let (want, want_grads) = run(None);
    let (got, got_grads) = run(Some(GradHorizon::Mask(span.tracked.clone())));

    // A stack that is an identity everywhere gives the gradient `identity`.
    // Then the two runs can agree also with a wrong straight-through carry.
    let off_identity = max_abs_diff(want.clone(), identity_grad(call, &head));
    assert!(off_identity > 1e-2, "{label}: the tracked layers are identities too, so the check cannot fail");
    let diff = max_rel_diff(got, want);
    assert!(
        diff < dtype_tol(TOL),
        "{label}: the cut changes the gradient of the input by {diff}. The straight-through carry \
         does not follow the carry path of the residual scheme",
    );
    for real in kept_layers {
        let label = format!("{label}, real layer {real}");
        assert_same_grads(&base.real_layers[real], &want_grads, &got_grads, &label);
    }
    if let Residuals::MultiGate(mg) = &base.residuals {
        for module in kept_modules {
            let label = format!("{label}, MGR module {module}");
            assert_matching_grads(&mg.layers[module], &want_grads, &got_grads, &label);
        }
    }
}

/// The token holds the whole residual.
#[test]
fn an_identity_untracked_span_is_exact_under_standard_residuals() {
    assert_identity_span_is_exact(ResidualsConfig::Standard);
}

/// Multi-Gate Residuals over `n_stream` streams. The gates are open at the
/// init: `b = 0`, so `β = 0.5`.
fn multi_gate_residuals(n_stream: usize) -> ResidualsConfig {
    ResidualsConfig::MultiGate { n_stream, init_bias: 0.0, init_bias_offset: 0.0, per_virtual_layer: false }
}

/// One residual stream with no accumulation phase. The closed gates keep the
/// untracked layers identities, and the open gates of the tracked layers mix.
#[test]
fn an_identity_untracked_span_is_exact_under_multi_gate_residuals() {
    assert_identity_span_is_exact(multi_gate_residuals(1));
}

/// Two residual streams. The first untracked layer appends the second stream.
/// That stream holds only the output of an untracked layer, so it gets no
/// straight-through carry.
#[test]
fn an_identity_untracked_span_is_exact_under_two_residual_streams() {
    assert_identity_span_is_exact(multi_gate_residuals(2));
}

/// Four residual streams. The two untracked layers append two streams, and
/// the seed stream gets the only straight-through carry.
#[test]
fn an_identity_untracked_span_is_exact_under_four_residual_streams() {
    assert_identity_span_is_exact(multi_gate_residuals(4));
}

/// Open gates in the untracked layers. With 1 and 2 streams, the untracked
/// layers mix, so the carry path of the seed stream has a gain below 1. With
/// 4 streams, the untracked layers only append.
#[test]
fn an_untracked_span_with_open_gates_is_exact_under_multi_gate_residuals() {
    for n_stream in [1, 2, 4] {
        let span = ZeroTransformSpan::bottom(multi_gate_residuals(n_stream), UntrackedGates::Open);
        assert_zero_transform_span_is_exact(&span, Call::Forward);
    }
}

/// Multi-Gate Residuals over `n_stream` streams with open gates. Stream `i`
/// starts at `b = −i`, so each carry path has a different gain.
fn ramped_multi_gate_residuals(n_stream: usize) -> ResidualsConfig {
    ResidualsConfig::MultiGate { n_stream, init_bias: 0.0, init_bias_offset: -1.0, per_virtual_layer: false }
}

/// [`ramped_multi_gate_residuals`] with one MGR module for each virtual
/// layer.
fn ramped_per_virtual_multi_gate_residuals(n_stream: usize) -> ResidualsConfig {
    ResidualsConfig::MultiGate { n_stream, init_bias: 0.0, init_bias_offset: -1.0, per_virtual_layer: true }
}

/// [`assert_zero_transform_span_is_exact`] in the forward, in the step and in
/// the prime.
fn assert_exact_in_each_call(span: &ZeroTransformSpan) {
    for call in CALLS {
        assert_zero_transform_span_is_exact(span, call);
    }
}

/// Two untracked spans under MGR: one inside the stack, and one at the top.
/// Two different residual streams enter each span, and the spans only mix.
/// The tracked layers below each cut train through the straight-through
/// carry. This holds in each call.
///
/// Real layer 0 runs each tracked layer, and real layer 1 runs each untracked
/// layer. So the MGR module of real layer 0 also pools below a tracked layer,
/// and its `w_alpha` trains (see
/// `grad_horizon_refuses_a_multi_gate_pool_that_feeds_only_untracked_layers`).
#[test]
fn an_untracked_span_inside_the_stack_is_exact_under_multi_gate_residuals() {
    assert_exact_in_each_call(&ZeroTransformSpan {
        n_real: 2,
        n_virtual_layers: Some((7, Schedule::Custom(vec![0, 0, 1, 1, 0, 0, 1]))),
        residuals: ramped_multi_gate_residuals(2),
        ignore: (false, false),
        tracked: vec![true, true, false, false, true, true, false],
        gates: UntrackedGates::Open,
    });
}

/// The accumulation phase ends inside an untracked span. Two residual streams
/// enter the span. Its first layer appends the third stream, and its second
/// layer mixes all three. The appended stream holds only the output of an
/// untracked layer, so it gets no straight-through carry. This holds in each
/// call.
///
/// Real layer 0 runs each tracked layer, as in
/// `an_untracked_span_inside_the_stack_is_exact_under_multi_gate_residuals`.
#[test]
fn an_untracked_span_that_ends_the_accumulation_phase_is_exact() {
    assert_exact_in_each_call(&ZeroTransformSpan {
        n_real: 2,
        n_virtual_layers: Some((6, Schedule::Custom(vec![0, 1, 1, 0, 0, 0]))),
        residuals: ramped_multi_gate_residuals(3),
        ignore: (false, false),
        tracked: vec![true, false, false, true, true, true],
        gates: UntrackedGates::Open,
    });
}

/// One MGR module for each virtual layer, across a cut at the bottom. Real
/// layer 1 runs the three virtual layers of the span, and each of them has
/// its own module. These modules alternate between closed and open gates. So
/// the gain of each carry path comes from the module of each virtual layer,
/// not from that of its real layer. This holds in each call.
///
/// With one module for each virtual layer, a span above a tracked layer
/// panics (see
/// `grad_horizon_refuses_a_multi_gate_pool_that_feeds_only_untracked_layers`).
/// So the span is at the bottom.
#[test]
fn an_untracked_span_is_exact_with_one_multi_gate_module_for_each_virtual_layer() {
    assert_exact_in_each_call(&ZeroTransformSpan {
        n_real: 2,
        n_virtual_layers: Some((6, Schedule::Custom(vec![1, 1, 1, 0, 0, 0]))),
        residuals: ramped_per_virtual_multi_gate_residuals(2),
        ignore: (false, false),
        tracked: vec![false, false, false, true, true, true],
        gates: UntrackedGates::Alternating,
    });
}

/// Panic unless `run` panics with a message that holds `expected`.
fn assert_panics_with(expected: &str, label: &str, run: impl FnOnce()) {
    let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) else {
        panic!("{label}: the run does not panic");
    };
    let message = panic_text(&*payload).unwrap_or_default();
    assert!(message.contains(expected), "{label}: the panic `{message}` does not hold `{expected}`");
}

/// Panic unless the stack that `stack` builds for each call panics in that
/// call with a message that holds `expected`.
fn assert_refused_in_each_call(label: &str, expected: &str, stack: impl Fn(Call, &Device) -> Layers<RefBlock>) {
    let device = test_device().autodiff();
    for call in CALLS {
        let layers = stack(call, &device);
        let (x, head) = (randn3(2, 4, &device), randn3(2, 4, &device));
        let label = format!("{label} in {call:?}");
        assert_panics_with(expected, &label, || drop(run_call(&layers, call, &x, &head)));
    }
}

/// [`assert_refused_in_each_call`] on the stack of `span` under its mask.
fn assert_horizon_is_refused(span: &ZeroTransformSpan, expected: &str) {
    assert_refused_in_each_call(&format!("{span:?}"), expected, |call, device| {
        let (mut layers, _, _) = span.stack(call, device);
        layers.grad_horizon = Some(GradHorizon::Mask(span.tracked.clone()));
        layers
    });
}

/// A suppressed skip at an untracked end layer panics, under each scheme and
/// in each call (`Residuals::assert_horizon_keeps_gradients` gives the
/// reason). Each case has a span of only the end layer, and a span of two
/// layers.
#[test]
fn grad_horizon_refuses_a_suppressed_skip_at_an_untracked_end_layer() {
    for (ignore, tracked, expected) in [
        ((true, false), vec![false, true, true, true], "ignore_first_residual suppresses its skip"),
        ((true, false), vec![false, false, true, true], "ignore_first_residual suppresses its skip"),
        ((false, true), vec![true, true, true, false], "ignore_last_residual suppresses its skip"),
        ((false, true), vec![true, true, false, false], "ignore_last_residual suppresses its skip"),
    ] {
        for residuals in [ResidualsConfig::Standard, ramped_multi_gate_residuals(2)] {
            let span = ZeroTransformSpan {
                n_real: 4,
                n_virtual_layers: None,
                residuals,
                ignore,
                tracked: tracked.clone(),
                gates: UntrackedGates::Open,
            };
            assert_horizon_is_refused(&span, expected);
        }
    }
}

/// Under MGR, an MGR module that pools in a tracked layer, but only directly
/// below untracked layers, panics (`Residuals::assert_horizon_keeps_gradients`
/// gives the reason). Each stack below has such a module, and each one panics
/// in each call:
///
/// - a span inside a stack with no weight sharing,
/// - one MGR module for each virtual layer, with a span inside the stack,
/// - a stretched layer schedule under `Depth(1)`. Each real layer except the
///   top one has one tracked application, directly below the run of the next
///   real layer.
#[test]
fn grad_horizon_refuses_a_multi_gate_pool_that_feeds_only_untracked_layers() {
    let expected = "with no tracked pool that feeds a tracked layer";
    let inside = ZeroTransformSpan {
        n_real: 4,
        n_virtual_layers: None,
        residuals: ramped_multi_gate_residuals(2),
        ignore: (false, false),
        tracked: vec![true, false, true, true],
        gates: UntrackedGates::Open,
    };
    assert_horizon_is_refused(&inside, expected);
    let per_virtual = ZeroTransformSpan {
        n_real: 2,
        n_virtual_layers: Some((6, Schedule::Custom(vec![0, 1, 1, 1, 0, 0]))),
        residuals: ramped_per_virtual_multi_gate_residuals(2),
        ignore: (false, false),
        tracked: vec![true, false, false, false, true, true],
        gates: UntrackedGates::Open,
    };
    assert_horizon_is_refused(&per_virtual, expected);
    assert_refused_in_each_call("Stretched under Depth(1)", expected, |call, device| {
        LayersBuilder {
            n_virtual_layers: Some((8, Schedule::Stretched)),
            residuals: ramped_multi_gate_residuals(2),
            grad_horizon: Some(GradHorizon::Depth(1)),
            class_latents: class_latents_for(call),
            ..layers_builder(3)
        }
        .init(device)
    });
}

/// A span next to a tracked suppressed skip stays exact. The suppressed first
/// skip below the span starts the residual streams again from `F_0`. The
/// suppressed last skip above the span reads the token at the cut. This holds
/// under each scheme and in each call.
#[test]
fn an_untracked_span_next_to_a_tracked_suppressed_skip_is_exact() {
    for residuals in [ResidualsConfig::Standard, ramped_multi_gate_residuals(2)] {
        assert_exact_in_each_call(&ZeroTransformSpan {
            n_real: 4,
            n_virtual_layers: None,
            residuals,
            ignore: (true, true),
            tracked: vec![true, false, false, true],
            gates: UntrackedGates::Open,
        });
    }
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
    for (a, b) in caches_uncut.into_iter().zip(caches_cut) {
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

/// The mask can alternate. Only the real layers that it tracks collect a
/// gradient, and `forward` still equals `step` unrolled through every one of
/// those backend hops. This holds under both residual schemes, and the
/// residual streams of Multi-Gate make the same hop as the tokens.
///
/// The stack shares its real layers, so the MGR module of each tracked real
/// layer also pools below a tracked layer. Real layer 2 runs only the
/// untracked layers.
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
            n_virtual_layers: Some((8, Schedule::Cyclic)),
            residuals: residuals.clone(),
            ..layers_builder(3)
        }
        .init(&device);
        let uncut = build();
        let (y_uncut, _) = uncut.forward(x.clone(), None, (), None, None);

        let mut cut = build();
        cut.real_layers = uncut.real_layers.clone();
        cut.residuals = uncut.residuals.clone();
        // Two untracked spans inside the stack: four backend hops.
        let mask = vec![true, true, false, true, true, false, true, true];
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
        for (real, layer) in cut.real_layers.iter().enumerate() {
            let tracked = (0..mask.len()).any(|i| mask[i] && parts_of(&cut, i).0 == real);
            let g = layer.block.in_proj.weight.val().grad(&grads);
            assert_eq!(
                g.is_some(),
                tracked,
                "real layer {real} must collect a gradient exactly when the mask tracks one of its applications",
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
    let (y_t, _) = layers.step(x_t, Some(cache_detach(caches)), None);
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
