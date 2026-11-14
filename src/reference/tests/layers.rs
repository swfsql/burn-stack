//! `Layer` and `Layers`: forward/step parity, the residuals of a layer, and
//! the virtual layers.

use super::*;

/// `Layers::forward` equals `Layers::step` unrolled: the container threads
/// caches per virtual layer, and does not reorder or drop any.
#[test]
fn layers_forward_equals_step_unrolled() {
    let device = test_device();
    let layers = layers(3, &device);
    let x = randn3(2, 5, &device);

    let (y_fwd, _) = layers.forward(x.clone(), None, (), None, None);

    assert!(max_abs_diff(y_fwd, step_unrolled(&layers, x)) < dtype_tol(TOL));
}

/// With the plain additive skip, the stack can ignore the residual of its
/// first and of its last virtual layer. The forward still equals the step
/// unrolled, over virtual layers.
#[test]
fn layers_standard_ignore_residuals_parity() {
    let device = test_device();
    let layers = LayersBuilder {
        n_virtual_layers: Some((5, Schedule::Stretched)),
        ignore_first_residual: true,
        ignore_last_residual: true,
        ..layers_builder(2)
    }
    .init(&device);
    let x = randn3(2, 4, &device);

    let (y_fwd, _) = layers.forward(x.clone(), None, (), None, None);

    assert!(max_abs_diff(y_fwd, step_unrolled(&layers, x)) < dtype_tol(TOL));
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

/// With a feed-forward, the forward of the stack still equals its step
/// unrolled. The feed-forward acts on each position alone, so it couples no
/// tokens.
#[test]
fn a_layer_with_a_feed_forward_keeps_forward_step_parity() {
    let device = test_device();
    let layers = LayersBuilder {
        mlp: Some(GatedMlpConfig::new(D_MODEL, D_MODEL * 2)),
        ..layers_builder(2)
    }
    .init(&device);
    let x = randn3(2, 5, &device);

    let (y_fwd, _) = layers.forward(x.clone(), None, (), None, None);

    assert!(max_abs_diff(y_fwd, step_unrolled(&layers, x)) < dtype_tol(TOL));
}

/// Without a feed-forward, the delta of a layer is the output of its block
/// alone. So the stack gives `x + h₁`, with `h₁` the output of the block.
#[test]
fn a_layer_without_a_feed_forward_adds_only_its_block() {
    let device = test_device();
    let layers = layers(1, &device);
    let layer = &layers.real_layers[0];
    assert!(layer.mlp.is_none() && layer.norm2.is_none());
    let x = randn3(2, 4, &device);

    let (got, _) = layers.forward(x.clone(), None, (), None, None);

    let (h1, _) = layer.block.block_forward(layer.norm.forward(x.clone()), None, (), None);
    assert!(max_abs_diff(got, x + h1) < dtype_tol(TOL));
}

/// `norm2` exists exactly when `mlp` exists. A layer with an `mlp` and no
/// `norm2` comes from a wrong construction. Its forward panics, and does not
/// skip the feed-forward.
#[test]
#[should_panic(expected = "`norm2` is allocated alongside `mlp`")]
fn a_feed_forward_without_norm2_panics() {
    let device = test_device();
    let mut layers = LayersBuilder {
        mlp: Some(GatedMlpConfig::new(D_MODEL, D_MODEL * 2)),
        ..layers_builder(1)
    }
    .init(&device);
    layers.real_layers[0].norm2 = None;

    let _ = layers.forward(randn3(1, 3, &device), None, (), None, None);
}

/// The pre-norm of the block (`norm`) and the pre-norm of the feed-forward
/// (`norm2`) are two parameters. An optimizer holds the state of a parameter
/// by its id, so the two ids differ.
#[test]
fn norm_and_norm2_are_distinct_parameters() {
    let device = test_device();
    let layers = LayersBuilder {
        mlp: Some(GatedMlpConfig::new(D_MODEL, D_MODEL * 2)),
        ..layers_builder(1)
    }
    .init(&device);
    let layer = &layers.real_layers[0];
    let norm2 = layer.norm2.as_ref().expect("norm2 present with an mlp");

    assert_ne!(layer.norm.gamma.id, norm2.gamma.id, "`norm` and `norm2` are one parameter");
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
