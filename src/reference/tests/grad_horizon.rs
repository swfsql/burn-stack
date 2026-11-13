//! The gradient horizon: the cuts of a `grad_horizon`, the straight-through
//! carry across them, and the backend hops.

use super::*;

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
