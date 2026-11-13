//! Multi-Gate Residuals in a stack: in `Layers`, in the pairs of a
//! bidirectional stack, and with the class markers.

use super::*;
use crate::modules::{Layer, Residuals};
use crate::utils::ClassCursors;

/// Multi-Gate Residuals over `n_stream` streams, with a carry-biased init.
fn multi_gate(n_stream: usize, init_bias_offset: f64, per_virtual_layer: bool) -> ResidualsConfig {
    ResidualsConfig::MultiGate { n_stream, init_bias: -1.0, init_bias_offset, per_virtual_layer }
}

/// The Multi-Gate modules of a stack.
fn multi_gate_modules(residuals: &Residuals) -> &crate::modules::MultiGate {
    match residuals {
        Residuals::MultiGate(mg) => mg,
        Residuals::Standard(_) => panic!("expected MultiGate residuals"),
    }
}

/// A bidirectional stack of 2 real pairs with mean merges.
fn bidi_builder(class_latents: Vec<ClassLatent>, residuals: ResidualsConfig) -> BidiLayersBuilder<RefBlockConfig> {
    let n_real_layers = 4;
    BidiLayersBuilder {
        n_real_layers,
        n_virtual_layers: None,
        block: block_config(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::mean(n_real_layers),
        class_latents,
        residuals,
        untied: Vec::new(),
    }
}

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

    assert!(max_abs_diff(y_fwd, step_unrolled(&layers, x)) < dtype_tol(TOL));
}

/// 2 real layers stretched over 6 virtual layers. The virtual layers use the
/// Multi-Gate module of their real layer again. The forward equals the step
/// unrolled.
#[test]
fn layers_multi_gate_virtual_forward_step_parity() {
    let device = test_device();
    let layers = LayersBuilder {
        n_virtual_layers: Some((6, Schedule::Stretched)),
        residuals: multi_gate(4, 0.0, false),
        ..layers_builder(2)
    }
    .init(&device);
    let x = randn3(2, 4, &device);

    let (y_fwd, _) = layers.forward(x.clone(), None, (), None, None);

    assert!(max_abs_diff(y_fwd, step_unrolled(&layers, x)) < dtype_tol(TOL));
}

/// A Multi-Gate module for each virtual layer, with the residuals of the first
/// and of the last virtual layer ignored. The forward equals the step
/// unrolled.
#[test]
fn layers_multi_gate_per_virtual_ignore_residuals_parity() {
    let device = test_device();
    let layers = LayersBuilder {
        n_virtual_layers: Some((5, Schedule::Stretched)),
        residuals: multi_gate(3, 0.0, true),
        ignore_first_residual: true,
        ignore_last_residual: true,
        ..layers_builder(2)
    }
    .init(&device);
    let mg = multi_gate_modules(&layers.residuals);
    assert_eq!(mg.layers.len(), 5, "one Multi-Gate module for each virtual layer");
    assert!(mg.per_virtual);
    let x = randn3(2, 4, &device);

    let (y_fwd, _) = layers.forward(x.clone(), None, (), None, None);

    assert!(max_abs_diff(y_fwd, step_unrolled(&layers, x)) < dtype_tol(TOL));
}

/// A stack must keep its streams distinct.
///
/// A seed of all `n_stream` streams with copies of the input is a symmetry
/// that the model cannot break. Identical streams give identical scores, so
/// identical gates and gradients, and the stack collapses to one lerped
/// stream. The stack uses the two phases of the paper instead. The first
/// `n_stream−1` layers append their output as a new stream, and the other
/// layers mix.
///
/// The reference below runs the stack by hand, one layer at a time, with the
/// Multi-Gate modules. It equals `Layers::forward`. The wiring with all
/// copies does not.
#[test]
fn layers_multi_gate_streams_are_distinct() {
    let device = test_device();
    let (n_stream, n_virtual) = (3, 5);
    let layers = LayersBuilder {
        n_virtual_layers: Some((n_virtual, Schedule::Stretched)),
        residuals: multi_gate(n_stream, 0.0, true),
        ..layers_builder(1)
    }
    .init(&device);
    let mg = multi_gate_modules(&layers.residuals);
    let (batch, sequence) = (2, 4);
    let x = randn3(batch, sequence, &device);
    let (y, _) = layers.forward(x.clone(), None, (), None, None);

    // The reference: the input is stream 1. Each layer appends its output
    // until `n_stream` streams exist. Only then do the layers mix them.
    let layer = &layers.real_layers[0];
    let mut streams = x.clone().unsqueeze_dim::<4>(2);
    let mut h = x.clone();
    for mgr in &mg.layers {
        let (out, _) = layer.forward(h, None, (), None);
        (h, streams) = match streams.dims()[2] < n_stream {
            true => mgr.accumulate(out, streams),
            false => mgr.forward(out, streams),
        };
    }
    assert_eq!(streams.dims(), [batch, sequence, n_stream, D_MODEL]);
    let diff = max_abs_diff(y.clone(), h);
    assert!(diff < dtype_tol(TOL), "the stack must accumulate, then mix: {diff}");

    // The wiring with all copies: each stream is a copy of the input. The
    // streams stay identical, so the stack is one lerped stream.
    let mut h = x;
    for mgr in &mg.layers {
        let (out, _) = layer.forward(h.clone(), None, (), None);
        let copies = h.unsqueeze_dim::<4>(2).expand([batch, sequence, n_stream, D_MODEL]);
        let (new_h, new_streams) = mgr.forward(out, copies);
        let first = new_streams.clone().narrow(2, 0, 1);
        let last = new_streams.narrow(2, n_stream - 1, 1);
        assert!(max_abs_diff(first, last) < dtype_tol(1e-6), "the symmetry is exact");
        h = new_h;
    }
    let diff = max_abs_diff(y, h);
    assert!(diff > dtype_tol(1e-3), "the stack must not collapse to one lerped stream: {diff}");
}

/// A bidirectional stack puts its residuals between its pairs: the unit of
/// Multi-Gate is the pair, not the layer. So the stack has one Multi-Gate
/// module for each real pair, and the gradient reaches each layer and each
/// Multi-Gate module.
#[test]
fn bidi_multi_gate_forward_and_grads() {
    let device = test_device().autodiff();
    // Two streams over two pairs: the first pair accumulates (the input and its
    // own output), and the second pair mixes.
    let layers = bidi_builder(Vec::new(), multi_gate(2, 0.0, false)).init(&device);
    let mg = multi_gate_modules(&layers.residuals);
    assert_eq!(mg.layers.len(), layers.real_layers.len() / 2, "one Multi-Gate module for each pair");

    let (y, _) = layers.forward(randn3(2, 5, &device), None, (), None, None);
    assert_eq!(y.dims(), [2, 5, D_MODEL]);

    let grads = y.sum().backward();
    for (i, Layer { norm, .. }) in layers.real_layers.iter().enumerate() {
        assert!(norm.gamma.val().grad(&grads).is_some(), "no gradient reaches the norm of real layer {i}");
    }
    assert!(
        mg.layers[0].w_alpha.val().grad(&grads).is_some(),
        "no gradient reaches the pool query of the pair that accumulates",
    );
    assert!(
        mg.layers[1].w_beta.val().grad(&grads).is_some(),
        "no gradient reaches the gate query of the pair that mixes",
    );
}

/// In a bidirectional stack with Multi-Gate, the class latents enter below the
/// first pair. So they make the sequence longer, and seed the streams.
#[test]
fn bidi_multi_gate_class_latents_forward() {
    let device = test_device();
    let latents = vec![ClassLatent::Start, ClassLatent::End];
    let layers = bidi_builder(latents, multi_gate(3, 0.0, false)).init(&device);
    let sequence = 4;

    let (y, _) = layers.forward(randn3(2, sequence, &device), None, (), None, None);

    assert_eq!(y.dims(), [2, sequence + 2, D_MODEL]);
}

/// The class latents of the stack enter below the first layer, so they seed
/// the streams as any other token does. The forward equals the step unrolled.
/// Each latent falls on the step that its cursor reaches.
#[test]
fn layers_multi_gate_stack_class_latents_step_matches_forward() {
    let device = test_device();
    let layers = LayersBuilder {
        class_latents: vec![ClassLatent::Start, ClassLatent::Custom(2)],
        residuals: multi_gate(3, -0.5, false),
        ..layers_builder(3)
    }
    .init(&device);
    let sequence = 4;
    let x = randn3(2, sequence, &device);

    let (y_fwd, _) = layers.forward(x.clone(), None, (), None, None);
    assert_eq!(y_fwd.dims(), [2, sequence + 2, D_MODEL]);

    // Start, u0, u1, Custom, u2, u3.
    let user_positions = [1, 2, 4, 5];
    let mut class = ClassCursors::new(sequence);
    let mut caches = None;
    for (t, pos) in user_positions.into_iter().enumerate() {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = layers.step(x_t, caches, Some(&mut class));
        caches = Some(c);
        let want = y_fwd.clone().narrow(1, pos, 1).squeeze_dim::<2>(1);
        assert!(max_abs_diff(y_t, want) < dtype_tol(TOL), "the step and the forward differ at t={t}");
    }
    assert_eq!(class.stack, sequence + 2);
}

/// With class latents, a forward in two windows gives the one forward. The
/// cursors carry from the first window to the second window, so each latent
/// falls at its place, and the streams follow.
#[test]
fn layers_multi_gate_class_latents_split_forward_matches_single() {
    let device = test_device();
    let layers = LayersBuilder {
        class_latents: vec![ClassLatent::Start],
        residuals: multi_gate(3, 0.0, false),
        ..layers_builder(3)
    }
    .init(&device);
    let sequence = 6;
    let x = randn3(2, sequence, &device);
    let (y_one, _) = layers.forward(x.clone(), None, (), None, None);

    let mut class = ClassCursors::new(sequence);
    let mut caches = None;
    let mut outs = Vec::new();
    for (at, len) in [(0, 2), (2, 4)] {
        let (y, c) = layers.forward(x.clone().narrow(1, at, len), caches, (), Some(&mut class), None);
        caches = Some(c);
        outs.push(y);
    }

    assert!(max_abs_diff(Tensor::cat(outs, 1), y_one) < dtype_tol(TOL));
}

/// A prime of the class latents that wait for the next token, and then a step
/// of that token, run what the step alone runs. Multi-Gate does not change
/// this.
#[test]
fn layers_multi_gate_prime_then_step_matches_step() {
    let device = test_device();
    let layers = LayersBuilder {
        class_latents: vec![ClassLatent::Start],
        residuals: multi_gate(3, -0.5, false),
        ..layers_builder(3)
    }
    .init(&device);
    let (batch, sequence) = (2, 3);
    let x = randn3(batch, sequence, &device);
    let token = |t: usize| x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);

    let mut class = ClassCursors::new(sequence);
    let mut caches = None;
    let mut plain = Vec::new();
    for t in 0..sequence {
        let (y, c) = layers.step(token(t), caches, Some(&mut class));
        caches = Some(c);
        plain.push(y);
    }

    // A prime before each step gives the same outputs and cursors.
    let mut primed_class = ClassCursors::new(sequence);
    let mut caches = None;
    for (t, want) in plain.into_iter().enumerate() {
        let (primed, c) = layers.prime(batch, caches, Some(&mut primed_class));
        // The `Start` of the stack waits before the first token only.
        assert_eq!(primed.is_some(), t == 0, "primed latents at t={t}");
        let (y, c) = layers.step(token(t), c, Some(&mut primed_class));
        caches = Some(c);
        assert!(max_abs_diff(y, want) < dtype_tol(TOL), "the prime and step differ from the step at t={t}");
    }
    assert_eq!(class, primed_class);
}

/// Both marker levels over Multi-Gate: the class token of a network and the
/// class latent of its stack. A prime and then the steps give the one
/// forward, marker for marker, and the same final caches.
#[test]
fn latent_network_multi_gate_class_markers_prime_step_matches_forward() {
    let device = test_device();
    let net = LatentNetworkBuilder {
        input_size: 3,
        layers: LayersBuilder {
            class_latents: vec![ClassLatent::Start],
            residuals: multi_gate(2, -0.5, false),
            ..layers_builder(2)
        },
        output_size: 2,
        // Multi-Gate pools convexly, so a latent head wants the final norm
        // (see the header of the Multi-Gate module).
        final_norm: true,
        class_tokens: vec![ClassToken::Start],
    }
    .init(&device);
    let (batch, sequence) = (2, 3);
    let x = Tensor::random([batch, sequence, 3], Distribution::Normal(0.0, 1.0), &device);

    // The forward gives `L T u0 u1 u2`: `L` is the class latent of the stack,
    // and `T` is the class token of the network. The stack places its latent
    // before the first token that it gets, and that token is `T`.
    let (y_fwd, c_fwd) = net.forward(x.clone(), None, (), None, None);
    assert_eq!(y_fwd.dims(), [batch, sequence + 2, 2]);
    let output_at = |p: usize| y_fwd.clone().narrow(1, p, 1).squeeze_dim::<2>(1);

    // One prime gives both markers. The class token of the network is the last.
    let mut class = ClassCursors::new(sequence);
    let (y_prime, mut caches) = net.prime(batch, None, Some(&mut class));
    let y_prime = y_prime.expect("two markers wait");
    assert!(max_abs_diff(y_prime, output_at(1)) < dtype_tol(TOL), "the prime and the forward differ");
    assert_eq!(class.network, 1);
    assert_eq!(class.stack, 2);

    for t in 0..sequence {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = net.step(x_t, caches, Some(&mut class));
        caches = Some(c);
        assert!(max_abs_diff(y_t, output_at(t + 2)) < dtype_tol(TOL), "the step {t} and the forward differ");
    }
    let caches = caches.expect("the steps give caches");
    for (i, (f, s)) in c_fwd.caches.into_iter().zip(caches.caches).enumerate() {
        assert!(max_abs_diff(s.state_bd, f.state_bd) < dtype_tol(TOL), "the caches of layer {i} differ");
    }
}
