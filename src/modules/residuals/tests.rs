use super::*;
use crate::modules::MultiGateResidual;
use crate::utils::test_helpers::{dtype_tol, max_abs_diff, test_device};
use burn::module::Param;
use burn::tensor::Distribution;

const D_MODEL: usize = 8;

/// The stand-in for a layer: a point-wise map, so it runs at each rank.
fn transform<const R: usize>(x: Tensor<R>) -> Tensor<R> {
    x.tanh() * 2.0 - 0.5
}

/// The closure of a join: the transform, and a marker as the extra value.
fn layer<const R: usize>(x: Tensor<R>) -> (Tensor<R>, &'static str) {
    (transform(x), "cache")
}

/// The join at virtual layer `i` of a stack with no layer schedule.
fn at(i: usize, first: bool, last: bool) -> JoinAt {
    JoinAt { virtual_idx: i, real_idx: i, first, last }
}

/// A window `[2, 5, D_MODEL]`.
fn window(device: &Device) -> Tensor<3> {
    Tensor::random([2, 5, D_MODEL], Distribution::Default, device)
}

/// Multi-Gate Residuals over `n_stream` streams, with one module for each of
/// `n_layers` real layers. The queries start at zero, so the aggregator weights
/// are uniform: the pooled token is the mean of the streams.
fn multi_gate(n_stream: usize, n_layers: usize, device: &Device) -> Residuals {
    ResidualsConfig::MultiGate { n_stream, init_bias: -1.0, init_bias_offset: 0.5, per_virtual_layer: false }
        .init(D_MODEL, n_layers, n_layers, device)
}

/// The residual streams of an MGR value.
fn streams<const R: usize, const S: usize>(value: &ResidualStreams<R, S>) -> Tensor<S> {
    value.streams.clone().expect("an MGR value has residual streams")
}

/// The number of residual streams of an MGR window.
fn stream_count(value: &ResidualStreams<3, 4>) -> usize {
    streams(value).dims()[2]
}

/// The accumulation phase appends the output of each layer as a new stream,
/// up to `n_stream` streams. After that, the mixing phase keeps the count.
/// The token is the pool of the new streams.
#[test]
fn multi_gate_join_grows_the_streams_to_n_stream() {
    let device = test_device();
    let residuals = multi_gate(3, 6, &device);
    let mut value = residuals.seed::<3, 4>(window(&device));
    assert_eq!(stream_count(&value), 1, "the input is the one seed stream");

    for (i, want) in [2, 3, 3, 3, 3].into_iter().enumerate() {
        let (input, before) = (value.token.clone(), streams(&value));
        let accumulates = before.dims()[2] < 3;
        (value, _) = residuals.join(value, at(i + 1, false, false), layer);
        let after = streams(&value);
        assert_eq!(stream_count(&value), want, "the stream count after layer {}", i + 1);
        if accumulates {
            let k = before.dims()[2];
            let kept = max_abs_diff(after.clone().narrow(2, 0, k), before);
            assert!(kept < dtype_tol(1e-6), "the accumulation phase keeps the streams that it finds");
            let appended = after.clone().narrow(2, k, 1).squeeze_dim::<3>(2);
            assert!(max_abs_diff(appended, transform(input)) < dtype_tol(1e-6), "the new stream is `F_l`");
        }
        let mean = after.mean_dim(2).squeeze_dim::<3>(2);
        assert!(max_abs_diff(value.token.clone(), mean) < dtype_tol(1e-5), "the token is the pool");
    }
}

/// A suppressed first skip starts the streams again from the output alone. A
/// suppressed last skip gives the output, and keeps the streams.
#[test]
fn multi_gate_join_under_a_suppressed_skip() {
    let device = test_device();
    let residuals = multi_gate(3, 4, &device);
    let x = window(&device);

    let (value, _) = residuals.join(residuals.seed::<3, 4>(x.clone()), at(0, true, false), layer);
    assert_eq!(stream_count(&value), 1, "the streams start again from one stream");
    let restarted = streams(&value).squeeze_dim::<3>(2);
    assert!(max_abs_diff(restarted, transform(x.clone())) < dtype_tol(1e-6), "the stream is `F_0`, not the input");
    assert!(max_abs_diff(value.token.clone(), transform(x)) < dtype_tol(1e-6), "the token is `F_0`");

    let (value, _) = residuals.join(value, at(1, false, false), layer);
    let (value, _) = residuals.join(value, at(2, false, false), layer);
    let (input, before) = (value.token.clone(), streams(&value));
    let (value, extra) = residuals.join(value, at(3, false, true), layer);
    assert_eq!(extra, "cache", "the join gives back the extra value of the layer");
    assert!(max_abs_diff(streams(&value), before) < dtype_tol(1e-6), "the streams do not change");
    assert!(max_abs_diff(value.into_token(), transform(input)) < dtype_tol(1e-6), "the output is `F_l` alone");
}

/// Multi-Gate Residuals with random queries and biases, so that the gates and
/// the aggregator weights of the streams differ.
fn random_multi_gate(n_stream: usize, n_layers: usize, device: &Device) -> Residuals {
    let random = |n: usize| Param::from_tensor(Tensor::<1>::random([n], Distribution::Default, device));
    let layers = (0..n_layers)
        .map(|_| MultiGateResidual {
            w_beta: random(D_MODEL),
            w_alpha: random(D_MODEL),
            b_beta: random(n_stream),
            d_model: D_MODEL,
            n_stream,
        })
        .collect();
    Residuals::MultiGate(MultiGate { layers, n_stream, per_virtual: false })
}

/// The join of a window of one token (ranks 3 and 4) equals the join of that
/// token (ranks 2 and 3). The chain of layers has each kind of join: a
/// suppressed first skip, the accumulation phase, the mixing phase and a
/// suppressed last skip.
#[test]
fn join_at_each_rank_gives_the_same_residual_streams() {
    let device = test_device();
    let n = 5;
    for (scheme, residuals) in [
        ("Standard", ResidualsConfig::Standard.init(D_MODEL, n, n, &device)),
        ("MultiGate", random_multi_gate(2, n, &device)),
    ] {
        let x = Tensor::<2>::random([3, D_MODEL], Distribution::Default, &device);
        let mut window = residuals.seed::<3, 4>(x.clone().unsqueeze_dim(1));
        let mut token = residuals.seed::<2, 3>(x);
        for i in 0..n {
            let layer_i = at(i, i == 0, i + 1 == n);
            (window, _) = residuals.join(window, layer_i, layer);
            (token, _) = residuals.join(token, layer_i, layer);
            let diff = max_abs_diff(window.token.clone().squeeze_dim::<2>(1), token.token.clone());
            assert!(diff < dtype_tol(1e-5), "{scheme}: the token differs after layer {i}");
            match (&window.streams, &token.streams) {
                (None, None) => {}
                (Some(w), Some(t)) => {
                    let diff = max_abs_diff(w.clone().squeeze_dim::<3>(1), t.clone());
                    assert!(diff < dtype_tol(1e-5), "{scheme}: the streams differ after layer {i}");
                }
                _ => panic!("{scheme}: one rank has residual streams and the other has none"),
            }
        }
    }
}

/// The closure of a join with a zero transform: an untracked layer of the
/// test of the straight-through carry.
fn zero_layer<const R: usize>(x: Tensor<R>) -> (Tensor<R>, ()) {
    (x.zeros_like(), ())
}

/// How [`run_joins`] runs its two middle joins.
#[derive(Clone, Copy)]
enum MiddleJoins {
    /// In the graph, as the other joins.
    Tracked,
    /// As an untracked span: on the inner backend, between
    /// [`Residuals::open_span`] and [`Residuals::close_span`].
    Untracked,
}

/// A stack of 4 joins. Joins 1 and 2 have zero transforms. The loss is on the
/// token and on the residual streams at the top. Returns the residual streams
/// at the top, and the gradient of the input.
fn run_joins(
    residuals: &Residuals,
    x: &Tensor<3>,
    heads: &(Tensor<3>, Tensor<4>),
    middle: MiddleJoins,
) -> (ResidualStreams<3, 4>, Tensor<3>) {
    let device = x.device();
    let x = x.clone().require_grad();
    let (mut value, _) = residuals.join(residuals.seed::<3, 4>(x.clone()), at(0, false, false), layer);
    match middle {
        MiddleJoins::Tracked => {
            for i in [1, 2] {
                (value, _) = residuals.join(value, at(i, false, false), zero_layer);
            }
        }
        MiddleJoins::Untracked => {
            let inner = residuals.valid();
            let (mut inside, carry) = residuals.open_span(value);
            assert!(!inside.token.device().is_autodiff(), "the span runs on the inner backend");
            for i in [1, 2] {
                (inside, _) = inner.join(inside, at(i, false, false), zero_layer);
            }
            value = residuals.close_span(inside, carry, &device);
        }
    }
    let (value, _) = residuals.join(value, at(3, false, false), layer);
    let mut loss = (value.token.clone() * heads.0.clone()).sum();
    if let Some(streams) = &value.streams {
        loss = loss + (streams.clone() * heads.1.clone()).sum();
    }
    let grad = x.grad(&loss.backward()).expect("the input gets a gradient");
    (value, grad)
}

/// An untracked span of two joins with zero transforms, between two tracked
/// joins. The cut keeps each value. The straight-through carry gives the
/// gradient of the same joins with no span. The rule is exact here, because
/// the init `w = 0` makes the scores of the gates and of the aggregator
/// constant.
///
/// Under MGR, the first join appends a stream, so 2 different streams enter
/// the span. The span appends a third stream, and then it mixes. The gate bias
/// of each stream is different, so each carry path has a different gain.
#[test]
fn straight_through_carry_follows_the_carry_path_of_each_scheme() {
    let device = test_device().autodiff();
    let n_stream = 3;
    for (scheme, residuals) in [
        ("Standard", ResidualsConfig::Standard.init(D_MODEL, 4, 4, &device)),
        ("MultiGate", multi_gate(n_stream, 4, &device)),
    ] {
        let x = window(&device);
        let heads = (window(&device), Tensor::random([2, 5, n_stream, D_MODEL], Distribution::Default, &device));
        let (want, want_grad) = run_joins(&residuals, &x, &heads, MiddleJoins::Tracked);
        let (got, got_grad) = run_joins(&residuals, &x, &heads, MiddleJoins::Untracked);

        let diff = max_abs_diff(got.token.clone(), want.token.clone());
        assert!(diff < dtype_tol(1e-6), "{scheme}: the cut changes the token by {diff}");
        match (&got.streams, &want.streams) {
            (None, None) => {}
            (Some(got), Some(want)) => {
                let diff = max_abs_diff(got.clone(), want.clone());
                assert!(diff < dtype_tol(1e-6), "{scheme}: the cut changes the streams by {diff}");
            }
            _ => panic!("{scheme}: one run has residual streams and the other has none"),
        }
        let diff = max_abs_diff(got_grad, want_grad);
        assert!(diff < dtype_tol(1e-5), "{scheme}: the cut changes the gradient of the input by {diff}");
    }
}

/// A middle layer adds the skip: the output plus the input. A suppressed first
/// or last skip gives the output alone.
#[test]
fn standard_join_adds_the_skip_unless_it_is_suppressed() {
    let device = test_device();
    let residuals = ResidualsConfig::Standard.init(D_MODEL, 3, 3, &device);
    let x = window(&device);

    let (value, extra) = residuals.join(residuals.seed::<3, 4>(x.clone()), at(1, false, false), layer);
    assert_eq!(extra, "cache", "the join gives back the extra value of the layer");
    assert!(value.streams.is_none(), "the token is the one residual stream of Standard");
    let want = transform(x.clone()) + x.clone();
    assert!(max_abs_diff(value.into_token(), want) < dtype_tol(1e-6), "a middle layer adds the skip");

    for (skip, layer_i) in [("first", at(0, true, false)), ("last", at(2, false, true))] {
        let (value, _) = residuals.join(residuals.seed::<3, 4>(x.clone()), layer_i, layer);
        let diff = max_abs_diff(value.into_token(), transform(x.clone()));
        assert!(diff < dtype_tol(1e-6), "a suppressed {skip} skip gives the output alone");
    }
}
