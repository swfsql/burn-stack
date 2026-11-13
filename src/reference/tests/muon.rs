//! The Muon plan: its allowlist, and the update of an untied projection.

use super::*;

/// The plan is an **allowlist**. Only the rank-2 weights that a `BlockConfig`
/// names move off AdamW, and every listed path must exist on the real module.
#[cfg(feature = "optim")]
#[test]
fn muon_plan_matches_only_existing_rank_2_weights() {
    use crate::optim::MuonPlan;

    let device = test_device();
    let layers = layers(2, &device);
    let plan = MuonPlan::new(BlockConfig::muon_projections(&block_config()));

    let report = plan.describe(&layers);
    let mut on_muon = 0;
    for line in report.lines().filter(|l| l.contains("muon[")) {
        on_muon += 1;
        assert!(
            line.trim_start().starts_with("2D"),
            "Muon only ever owns rank-2 weights: {line}",
        );
    }
    assert_eq!(on_muon, 6, "3 listed weights x 2 real layers");
    for line in report.lines().filter(|l| l.contains("decay_raw")) {
        assert!(
            line.ends_with("fallback"),
            "the rank-1 decay stays on the fallback optimizer: {line}",
        );
    }
}

/// A `CatLinear` merge is a plain hidden matrix, so the plan of a bidirectional
/// stack gives Muon one weight per such pair. A `Mean` merge has no parameters,
/// so it adds no spec.
#[cfg(feature = "optim")]
#[test]
fn bidi_plan_holds_the_cat_linear_spec_only_for_cat_linear_merges() {
    use crate::modules::BidiShape;
    use crate::optim::ProjSpec;
    use OutputMergeConfig::{CatLinear, Mean};

    let device = test_device();
    let spec = ProjSpec::path_whole("CatLinear.weight", D_MODEL);
    for (merges, n_weights) in [
        (vec![CatLinear, CatLinear], 2),
        (vec![CatLinear, Mean], 1),
        (vec![Mean, Mean], 0),
    ] {
        let shape = BidiShape::new(4, merges);
        let plan = shape.muon_plan(&block_config());
        assert_eq!(plan.specs.contains(&spec), n_weights > 0);
        let report = plan.describe(&shape.init(block_config(), &device));
        let selected = report
            .lines()
            .filter(|l| l.contains("CatLinear.weight") && l.contains("muon[all:8]"))
            .count();
        assert_eq!(selected, n_weights, "{report}");
    }
}

/// An untied projection is one matrix per application. Muon orthogonalises
/// each copy on its own, and the plan gives the untied weight to that
/// per-copy update.
#[cfg(feature = "optim")]
#[test]
fn muon_updates_an_untied_projection_one_copy_at_a_time() {
    use crate::optim::{MuonPlan, ProjSpec, Segmented, muon_config};
    use burn::optim::{AdamWConfig, Optimizer};

    let device = test_device();
    let n = 3;
    let spec = ProjSpec::block_whole("gate_proj.weight", D_MODEL).tiled();
    let muon = muon_config(0.0).build();
    let segmented = Segmented::new(&spec, muon.clone(), AdamWConfig::new().build(), 1);
    let w = Tensor::<2>::random([D_MODEL, n * D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    let g = Tensor::<2>::random([D_MODEL, n * D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    let (updated, _) = segmented.step(1e-2, w.clone(), g.clone(), None);
    for ((got, w), g) in updated.chunk(n, 1).into_iter().zip(w.chunk(n, 1)).zip(g.chunk(n, 1)) {
        let (want, _) = muon.step(1e-2, w, g, None);
        assert!(max_abs_diff(got, want) < dtype_tol(TOL));
    }

    let config = block_config().with_untied(vec![RefUntied::GateProj]);
    let layers = LayersBuilder {
        n_virtual_layers: Some((n, Schedule::Cyclic)),
        block: config.clone(),
        ..layers_builder(1)
    }
    .init(&device);
    let report = MuonPlan::new(BlockConfig::muon_projections(&config)).describe(&layers);
    assert!(
        report
            .lines()
            .any(|l| l.contains("gate_proj.weight") && l.contains("muon[all:8]×3")),
        "{report}",
    );
}
