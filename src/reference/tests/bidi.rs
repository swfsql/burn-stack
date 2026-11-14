//! The bidirectional stacks: the pairs, their merges and their ignored
//! residuals.

use super::*;

/// A bidi stack pairs a straight and a reversed pass. With more virtual than
/// real pairs, the per-pair merge must be indexed by the **real** pair.
#[test]
fn bidi_virtual_pairs_share_the_real_pair_merge() {
    let device = test_device().autodiff();
    let layers = BidiLayersBuilder {
        n_real_layers: 4,
        n_virtual_layers: Some((10, Default::default())),
        block: block_config(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: vec![OutputMergeConfig::CatLinear; 2],
        class_latents: Vec::new(),
        residuals: ResidualsConfig::Standard,
        untied: Vec::new(),
    }
    .init(&device);
    assert_eq!(layers.outputs_merge.len(), 2, "one merge for each real pair");

    let x = Tensor::random([2, 6, D_MODEL], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();
    let (y, _) = layers.forward(x, None, (), None, None);
    assert_eq!(y.dims(), [2, 6, D_MODEL]);

    let grads = y.sum().backward();
    for pair in &layers.outputs_merge {
        if let crate::modules::OutputMerge::CatLinear(lin) = pair {
            assert!(
                lin.weight.val().grad(&grads).is_some(),
                "both real pairs' merges must be exercised, not dangling",
            );
        }
    }
}

/// Two forwards of a bidirectional stack over the same input agree. So the
/// straight passes, the reversed passes and the merges hold no hidden
/// randomness. The stack runs 6 virtual layers over 2 real layers with the
/// `StridedStretched` layer schedule.
#[test]
fn bidi_virtual_forward_is_deterministic() {
    let device = test_device();
    let layers = BidiLayersBuilder {
        n_real_layers: 2,
        n_virtual_layers: Some((6, BidiSchedule::StridedStretched)),
        block: block_config(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::mean(2),
        class_latents: Vec::new(),
        residuals: ResidualsConfig::Standard,
        untied: Vec::new(),
    }
    .init(&device);
    let x = randn3(2, 5, &device);

    let (y0, _) = layers.forward(x.clone(), None, (), None, None);
    let (y1, _) = layers.forward(x, None, (), None, None);

    assert!(max_abs_diff(y0, y1) < dtype_tol(TOL), "two forwards of the same input differ");
}

/// A bidirectional stack can ignore the residual of its first pair and of its
/// last pair. A pair with an ignored residual drops the input skip, and gives
/// its bare merge. With one pair, the stack with the skip gives the stack
/// with no skip plus the input.
///
/// The one pair is the first pair and also the last pair. So a swap of the
/// two flags does not show.
///
/// Each forward below runs the same weights. So the check also fails when a
/// forward draws new weights.
#[test]
fn bidi_ignored_residuals_drop_the_input_skip() {
    let device = test_device();
    let mut layers = BidiLayersBuilder {
        n_real_layers: 2,
        n_virtual_layers: None,
        block: block_config(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::cat_linear(2),
        class_latents: Vec::new(),
        residuals: ResidualsConfig::Standard,
        untied: Vec::new(),
    }
    .init(&device);
    let x = randn3(2, 5, &device);
    let (y_skip, _) = layers.forward(x.clone(), None, (), None, None);

    for (first, last) in [(true, false), (false, true), (true, true)] {
        layers.ignore_first_residual = first;
        layers.ignore_last_residual = last;
        let (y_bare, _) = layers.forward(x.clone(), None, (), None, None);
        let label = format!("ignore first {first}, ignore last {last}: the bare merge plus the input");
        assert_same_outputs(y_skip.clone(), y_bare + x.clone(), &label);
    }
}
