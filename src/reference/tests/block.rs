//! The block itself: its forward/step parity and its windows through the
//! cache, which the containers assume.

use super::*;

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
