//! The class markers: the class tokens of a network and the class latents of
//! a stack.

use super::*;

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
