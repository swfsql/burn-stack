use super::*;
use crate::utils::test_helpers::{dtype_tol, max_abs_diff, test_device};
use burn::tensor::Distribution;

/// A [`MultiGateResidual`] with random queries and bias, so that the gates of
/// the streams differ.
fn random_mgr(d_model: usize, n_stream: usize, device: &Device) -> MultiGateResidual {
    MultiGateResidual {
        w_beta: Param::from_tensor(Tensor::random([d_model], Distribution::Default, device)),
        w_alpha: Param::from_tensor(Tensor::random([d_model], Distribution::Default, device)),
        b_beta: Param::from_tensor(Tensor::random([n_stream], Distribution::Default, device)),
        d_model,
        n_stream,
    }
}

/// MGR is point-wise over the sequence. So the step of each position gives
/// the forward at that position.
#[test]
fn forward_step_parity() {
    let device = test_device();
    let (b, s, n, d) = (2, 5, 4, 8);
    let m = random_mgr(d, n, &device);

    let layer_output = Tensor::<3>::random([b, s, d], Distribution::Default, &device);
    let streams = Tensor::<4>::random([b, s, n, d], Distribution::Default, &device);

    let (h_fwd, streams_fwd) = m.forward(layer_output.clone(), streams.clone());
    assert_eq!(h_fwd.dims(), [b, s, d]);
    assert_eq!(streams_fwd.dims(), [b, s, n, d]);

    for t in 0..s {
        let out_t = layer_output.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let streams_in_t = streams.clone().narrow(1, t, 1).squeeze_dim::<3>(1);
        let (h_t, streams_t) = m.step(out_t, streams_in_t);

        let h_want = h_fwd.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let streams_want = streams_fwd.clone().narrow(1, t, 1).squeeze_dim::<3>(1);
        assert!(max_abs_diff(h_t, h_want) < dtype_tol(1e-5), "h mismatch at t={t}");
        assert!(max_abs_diff(streams_t, streams_want) < dtype_tol(1e-5), "stream mismatch at t={t}");
    }
}

/// Zero queries and a zero bias give `β = σ(0) = 0.5` and a uniform `α = 1/n`.
/// So each new stream is the midpoint of its stream and the layer output, and
/// the pool is the mean of the new streams.
#[test]
fn init_is_convex_mean() {
    let device = test_device();
    let (b, n, d) = (2, 3, 6);
    let m = MultiGateResidualConfig::new(d, n).init(&device);

    let layer_output = Tensor::<2>::random([b, d], Distribution::Default, &device);
    let streams = Tensor::<3>::random([b, n, d], Distribution::Default, &device);
    let (h, new_streams) = m.step(layer_output.clone(), streams.clone());

    let expected_streams = (streams + layer_output.unsqueeze_dim::<3>(1)) * 0.5;
    assert!(max_abs_diff(new_streams.clone(), expected_streams) < dtype_tol(1e-5));

    let expected_h = new_streams.mean_dim(1).squeeze_dim::<2>(1);
    assert!(max_abs_diff(h, expected_h) < dtype_tol(1e-5));
}

/// The gradient reaches the streams and each parameter.
#[test]
fn gradients_flow() {
    let device = test_device().autodiff();
    let (b, s, n, d) = (2, 4, 3, 8);
    let m = MultiGateResidualConfig::new(d, n).with_init_bias(0.5).init(&device);

    let layer_output = Tensor::<3>::random([b, s, d], Distribution::Default, &device);
    let streams = Param::from_tensor(Tensor::<4>::random([b, s, n, d], Distribution::Default, &device));

    let (h, new_streams) = m.forward(layer_output, streams.val());
    let grads = (h.sum() + new_streams.sum()).backward();

    assert!(streams.val().grad(&grads).is_some(), "grad streams");
    assert!(m.w_beta.val().grad(&grads).is_some(), "grad w_beta");
    assert!(m.w_alpha.val().grad(&grads).is_some(), "grad w_alpha");
    assert!(m.b_beta.val().grad(&grads).is_some(), "grad b_beta");
}

/// The accumulation phase appends the layer output as one more stream, and
/// pools over the wider set: `k+1` streams out of `k`. At the zero init, the
/// pool is their mean.
#[test]
fn accumulate_appends_a_stream() {
    let device = test_device();
    let (b, s, n, d) = (2, 3, 4, 8);
    let m = MultiGateResidualConfig::new(d, n).init(&device);

    let layer_output = Tensor::<3>::random([b, s, d], Distribution::Default, &device);
    let streams = Tensor::<4>::random([b, s, 2, d], Distribution::Default, &device);

    let (h, new_streams) = m.accumulate(layer_output.clone(), streams.clone());
    assert_eq!(new_streams.dims(), [b, s, 3, d], "one stream wider");
    // The new stream is the layer output itself, with no gate and no mix. The
    // streams that it found do not change.
    let expected = Tensor::cat(vec![streams, layer_output.clone().unsqueeze_dim::<4>(2)], 2);
    assert!(max_abs_diff(new_streams.clone(), expected) < dtype_tol(1e-6));
    // Zero queries give a uniform `α`, so the pool is the mean over the wider set.
    let mean = new_streams.mean_dim(2).squeeze_dim::<3>(2);
    assert!(max_abs_diff(h, mean) < dtype_tol(1e-5));

    let (h_t, s_t) = m.accumulate_step(
        layer_output.narrow(1, 0, 1).squeeze_dim::<2>(1),
        Tensor::<3>::zeros([b, 2, d], &device),
    );
    assert_eq!(s_t.dims(), [b, 3, d]);
    assert_eq!(h_t.dims(), [b, d]);
}

/// `init_bias_offset` is the offset between the initial gate biases of two
/// consecutive streams: stream `i` starts at `init_bias + i · init_bias_offset`.
/// So each stream lerps by its own `β` before any training.
#[test]
fn init_bias_offset_ramps_the_gates() {
    let device = test_device();
    let (n, d) = (4, 8);
    let m = MultiGateResidualConfig::new(d, n)
        .with_init_bias(-2.7)
        .with_init_bias_offset(0.9)
        .init(&device);

    let expected = Tensor::<1>::from_floats([-2.7, -1.8, -0.9, 0.0], &device);
    assert!(max_abs_diff(m.b_beta.val(), expected) < dtype_tol(1e-5));

    // Zero streams and a layer output of ones: each new stream holds its `β`.
    // The first stream carries (`β ≈ 0.06`), the last one updates (`β = 0.5`).
    let streams = Tensor::<4>::zeros([1, 1, n, d], &device);
    let layer_output = Tensor::<3>::ones([1, 1, d], &device);
    let (_h, new_streams) = m.forward(layer_output, streams);
    let betas: Vec<f32> = new_streams.narrow(3, 0, 1).reshape([n]).into_data().try_into_vec_as().unwrap();
    for (i, w) in betas.windows(2).enumerate() {
        assert!(w[0] < w[1], "stream {i} must lerp less than stream {}", i + 1);
    }
    assert!((betas[n - 1] - 0.5).abs() < dtype_tol(1e-4), "last stream is σ(0) = 0.5");
}

/// A vector in each stream comes out of the pool unchanged. All streams then
/// score alike, so `α` is uniform and the convex pool gives the vector. So a
/// class latent in the streams gives the layer above the same embedding that
/// a plain additive skip gives it.
#[test]
fn attn_pool_reproduces_a_vector_present_in_every_stream() {
    let device = test_device();
    let (b, n, d) = (2, 4, 8);
    let m = random_mgr(d, n, &device);

    let vector = Tensor::<2>::random([b, d], Distribution::Default, &device);
    let streams = vector.clone().unsqueeze_dim::<3>(1).expand([b, n, d]);
    // `attn_pool` keeps the stream axis at size 1.
    let pooled = m.attn_pool(streams).squeeze_dim::<2>(1);
    assert!(max_abs_diff(pooled, vector) < dtype_tol(1e-5));
}
