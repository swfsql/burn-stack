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
