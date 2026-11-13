use super::*;
use crate::optim::{SgdConfig, SubProjection, muon_config};
use crate::utils::test_helpers::{dtype_tol, max_abs_diff, max_rel_diff, test_device};
use burn::optim::{AdamWConfig, MuonConfig};
use burn::tensor::Distribution;

/// The widths of [`mixed_spec`].
const MIXED_WIDTHS: [usize; 3] = [6, 4, 2];

fn muon() -> MuonConfig {
    muon_config(1e-4)
}

fn rand_2d(shape: [usize; 2], device: &Device) -> Tensor<2> {
    Tensor::random(shape, Distribution::Default, device)
}

/// A fused projection of 3 sub-projections: Muon, the fallback, and Muon.
fn mixed_spec() -> ProjSpec {
    let [a, b, c] = MIXED_WIDTHS;
    ProjSpec::path("w", vec![SubProjection::muon("a", a), SubProjection::adamw("b", b), SubProjection::muon("c", c)])
}

/// The largest difference between the weights that `got` and `want` give
/// after 3 optimizer updates. Both start from one weight and take the same
/// gradients.
fn diff_after_three_updates(got: &impl Optimizer, want: &impl Optimizer, device: &Device) -> f32 {
    let mut w_got = rand_2d([6, 12], device);
    let mut w_want = w_got.clone();
    let (mut s_got, mut s_want) = (None, None);
    for _ in 0..3 {
        let grad = rand_2d([6, 12], device);
        (w_got, s_got) = got.step(1e-2, w_got, grad.clone(), s_got);
        (w_want, s_want) = want.step(1e-2, w_want, grad, s_want);
    }
    max_abs_diff(w_got, w_want)
}

/// AdamW is elementwise. So AdamW on each sub-projection of a weight gives
/// plain AdamW on the whole weight. The channels of a fused projection that
/// Muon does not own keep the update that they had.
#[test]
fn all_adamw_sub_projections_equal_plain_adamw() {
    let device = test_device();
    let spec = ProjSpec::path(
        "w",
        vec![SubProjection::adamw("a", 3), SubProjection::adamw("b", 5), SubProjection::adamw("c", 4)],
    );
    let segmented = Segmented::new(&spec, muon().build(), AdamWConfig::new().build(), 1);

    let diff = diff_after_three_updates(&segmented, &AdamWConfig::new().build(), &device);
    assert!(diff < dtype_tol(1e-6), "segmented AdamW diverged from plain AdamW: {diff}");
}

/// One sub-projection that Muon owns in full gives plain Muon.
#[test]
fn one_muon_sub_projection_equals_plain_muon() {
    let device = test_device();
    let spec = ProjSpec::path_whole("w", 12);
    let segmented = Segmented::new(&spec, muon().build(), AdamWConfig::new().build(), 1);

    let diff = diff_after_three_updates(&segmented, &muon().build(), &device);
    assert!(diff < dtype_tol(1e-6), "Segmented with one sub-projection diverged from Muon: {diff}");
}

/// Muon on each sub-projection alone differs from Muon on the fused matrix.
/// This is why the plan holds the column seams.
#[test]
fn per_sub_projection_muon_differs_from_fused_muon() {
    let device = test_device();
    let spec = ProjSpec::path("w", vec![SubProjection::muon("a", 6), SubProjection::muon("b", 6)]);
    let segmented = Segmented::new(&spec, muon().build(), AdamWConfig::new().build(), 1);
    let fused = muon().build();

    let lr = 1e-2;
    let w = rand_2d([6, 12], &device);
    let grad = rand_2d([6, 12], &device);
    let (w_seg, _) = segmented.step(lr, w.clone(), grad.clone(), None);
    let (w_fused, _) = fused.step(lr, w.clone(), grad, None);

    // The two optimizer updates, in units of the learning rate.
    let diff = max_rel_diff((w_seg - w.clone()) / lr, (w_fused - w) / lr);
    assert!(diff > dtype_tol(1e-3), "the two updates differ only by {diff}");
}

/// The hand-written record of a [`SegmentedState`] makes a round trip. Each
/// sub-projection keeps the state of its own optimizer.
#[test]
fn segmented_state_round_trips() {
    let device = test_device();
    let segmented = Segmented::new(&mixed_spec(), muon().build(), AdamWConfig::new().build(), 1);

    let w = rand_2d([6, 12], &device);
    let grad = rand_2d([6, 12], &device);
    let (w, state) = segmented.step(1e-2, w, grad.clone(), None);
    let state = state.expect("state");

    let mut sink = StateSink::default();
    state.state_flatten("p", &mut sink);
    let mut source = StateSource::new(sink.scalars.into_iter().collect());
    for (name, data) in sink.tensors {
        source.insert_tensor(name, data);
    }
    let reloaded = SegmentedState::<2>::state_unflatten("p", &mut source, &device).expect("reloaded state");
    assert_eq!(reloaded.states.len(), 3);
    assert!(matches!(reloaded.states[0], SubProjectionState::Muon(_)));
    assert!(matches!(reloaded.states[1], SubProjectionState::AdamW(_)));
    assert!(matches!(reloaded.states[2], SubProjectionState::Muon(_)));

    // An optimizer update from the reloaded state gives the update from the
    // live state.
    let (a, _) = segmented.step(1e-2, w.clone(), grad.clone(), Some(state));
    let (b, _) = segmented.step(1e-2, w, grad, Some(reloaded));
    assert!(max_abs_diff(a, b) < dtype_tol(1e-6));
}

/// Under an SGD fallback, the sub-projections that Muon does not own take
/// plain SGD updates and hold no state. The state holds only the Muon
/// sub-projections, and each one pairs back with its sub-projection.
#[test]
fn sgd_fallback_sub_projections_are_stateless_sgd() {
    let device = test_device();
    let sgd = SgdConfig::new().build();
    let segmented = Segmented::new(&mixed_spec(), muon().build(), sgd.clone(), 1);
    let plain = muon().build();

    let mut w = rand_2d([6, 12], &device);
    let mut w_ref = w.clone().split_with_sizes(MIXED_WIDTHS.to_vec(), 1);
    let (mut state, mut s_a, mut s_c) = (None, None, None);
    for _ in 0..3 {
        let grad = rand_2d([6, 12], &device);
        let (next, s) = segmented.step(1e-2, w, grad.clone(), state);
        let s = s.expect("state");
        assert_eq!(s.states.len(), 2, "no entry for the SGD sub-projection");
        assert!(matches!(s.states[1], SubProjectionState::Muon(_)));
        (w, state) = (next, Some(s));

        let g = grad.split_with_sizes(MIXED_WIDTHS.to_vec(), 1);
        (w_ref[0], s_a) = plain.step(1e-2, w_ref[0].clone(), g[0].clone(), s_a);
        w_ref[1] = sgd.step(1e-2, w_ref[1].clone(), g[1].clone(), None).0;
        (w_ref[2], s_c) = plain.step(1e-2, w_ref[2].clone(), g[2].clone(), s_c);
    }

    let diff = max_abs_diff(w, Tensor::cat(w_ref, 1));
    assert!(diff < dtype_tol(1e-6), "segmented Muon + SGD diverged from its sub-projections: {diff}");
}
