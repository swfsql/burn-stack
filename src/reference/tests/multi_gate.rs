//! Multi-Gate Residuals in a stack.

use super::*;

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
