//! The Muon plan: its allowlist, and the plan of each container (a network
//! and a bidirectional stack). Also its module optimizer, its report, and the
//! update of an untied projection.

use super::*;
use crate::modules::{LatentNetwork, LatentShape, NetworkShape, VocabShape};
use crate::optim::report::{FloatParam, float_params, selected};
use crate::optim::{MuonPlan, ProjScope, ProjSpec, SubProjection, muon_config};
use burn::optim::{AdamWConfig, GradientsParams, ModuleOptimizer};

/// Each spec of `plan` selects at least one parameter. Each parameter that it
/// selects has rank 2, and its columns are the width of the spec.
fn assert_plan_fits(plan: &MuonPlan, params: &[FloatParam]) {
    for spec in &plan.specs {
        let hits = selected(spec, params);
        assert!(!hits.is_empty(), "the spec `{}` selects no parameter", spec.path);
        for param in hits {
            assert_eq!(param.dims.len(), 2, "the spec `{}` selects `{}`, which is not rank 2", spec.path, param.path);
            assert_eq!(param.dims[1], spec.width(), "the spec `{}` and `{}` differ in width", spec.path, param.path);
        }
    }
}

/// No spec of `plan` selects a weight outside the layers of a network, for
/// example an I/O weight.
fn assert_plan_selects_no_io_weight(plan: &MuonPlan, params: &[FloatParam]) {
    for spec in &plan.specs {
        for param in selected(spec, params) {
            assert!(
                param.path.starts_with("layers.real_layers."),
                "the spec `{}` selects the weight `{}`, which is outside the layers",
                spec.path,
                param.path,
            );
        }
    }
}

/// A feed-forward of each layer, with a hidden width of `2 · D_MODEL`.
fn mlp() -> GatedMlpConfig {
    GatedMlpConfig::new(D_MODEL, 2 * D_MODEL).with_multiple_of(D_MODEL)
}

/// A plan with a fused spec. Muon owns 6 columns of `in_proj`. The fallback
/// optimizer owns its last 2 columns, as it owns the scalar channels of the
/// block of a family. So the plan builds a `Segmented` group. `RefBlock` has
/// no fused projection, so no container gives this plan.
fn fused_plan() -> MuonPlan {
    MuonPlan::new(vec![
        ProjSpec::block(
            "in_proj.weight",
            vec![SubProjection::muon("value", D_MODEL - 2), SubProjection::adamw("scale", 2)],
        ),
        ProjSpec::block_whole("out_proj.weight", D_MODEL),
    ])
}

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

/// Each pair of a bidirectional stack holds one block for each direction. Real
/// layer 0 holds the block of the straight pass, and real layer 1 holds the
/// block of the reversed pass. Each block spec of the plan selects each of the
/// two blocks once. A bidirectional stack holds no I/O weight.
#[test]
fn bidi_plan_fits_the_model() {
    use crate::modules::BidiShape;

    let device = test_device();
    let shape = BidiShape::new(2, OutputMergeConfig::cat_linear(2));
    let plan = shape.muon_plan(&block_config());
    let params = float_params(&shape.init(block_config(), &device));

    for spec in plan.specs.iter().filter(|spec| spec.scope == ProjScope::Block) {
        let hits = selected(spec, &params);
        assert_eq!(hits.len(), 2, "the spec `{}` selects one block for each direction", spec.path);
        for direction in ["real_layers.0.block.", "real_layers.1.block."] {
            assert!(
                hits.iter().any(|param| param.path.starts_with(direction)),
                "the spec `{}` does not select the block under `{direction}`",
                spec.path,
            );
        }
    }
    assert_plan_fits(&plan, &params);
}

/// The plan of a vocab network holds the feed-forward of each of its layers.
/// It leaves the embedding and the LM head on the fallback optimizer.
#[test]
fn vocab_plan_holds_the_mlp() {
    let device = test_device();
    let shape = VocabShape::new(20, NetworkShape::new(2).with_mlp(Some(mlp()))).with_missing_lm_head(false);
    let plan = shape.stack.muon_plan(&block_config());
    let params = float_params(&shape.init(block_config(), &device));

    let n_block_specs = BlockConfig::muon_projections(&block_config()).len();
    assert_eq!(plan.specs.len(), n_block_specs + 2, "the feed-forward adds 2 specs");
    for path in ["mlp.fc1.weight", "mlp.fc2.weight"] {
        let spec = plan.specs.iter().find(|spec| spec.path == path).unwrap_or_else(|| panic!("the plan has no {path} spec"));
        assert_eq!(selected(spec, &params).len(), 2, "{path}");
    }
    assert_plan_fits(&plan, &params);
    assert_plan_selects_no_io_weight(&plan, &params);
}

/// The module optimizer of a plan has one group for each spec that Muon owns,
/// over the fallback optimizer. A spec that selects a weight of rank 1 or 3
/// gives that weight to Muon, and Muon panics in the training step.
#[test]
fn build_assembles_groups_without_panicking() {
    let device = test_device().autodiff();
    let model = LatentShape::new(2, 2, NetworkShape::new(1)).init(block_config(), &device);
    let mut optim = fused_plan().build(&AdamWConfig::new().into(), &muon_config(1e-4));

    // One training step over the whole model: each parameter goes through the
    // optimizer of its group.
    let x = Tensor::random([1, 8, 2], Distribution::Default, &device);
    let (y, _) = model.forward(x, None, (), None, None);
    let grads = GradientsParams::from_grads(y.sum().backward(), &model);
    let _model = optim.step(1e-3, model, grads);
}

/// The module optimizer makes a round trip through its bytes. On the reload,
/// each parameter goes back to its group, and each sub-projection gets the
/// state of its own optimizer back. So the training continues as it does
/// without the round trip.
#[test]
fn module_optimizer_state_round_trips_through_a_record() {
    let device = test_device().autodiff();
    let lr = 1e-3;
    let (plan, fallback, muon) = (fused_plan(), AdamWConfig::new().into(), muon_config(1e-4));
    let model = LatentShape::new(2, 2, NetworkShape::new(1)).init(block_config(), &device);
    // Two inputs, so that the second training step has a gradient of its own.
    let [first_x, second_x] = [0, 1].map(|_| Tensor::<3>::random([1, 8, 2], Distribution::Default, &device));

    let train_step = |model: LatentNetwork<RefBlock>, optim: &mut ModuleOptimizer, x: &Tensor<3>| {
        let (y, _) = model.forward(x.clone(), None, (), None, None);
        let grads = GradientsParams::from_grads(y.sum().backward(), &model);
        optim.step(lr, model, grads)
    };
    let in_proj = |model: &LatentNetwork<RefBlock>| model.layers.real_layers[0].block.in_proj.weight.val();

    // Two optimizers take the same first training step.
    let (mut live, mut twin) = (plan.build(&fallback, &muon), plan.build(&fallback, &muon));
    let stepped = train_step(model.clone(), &mut live, &first_x);
    let _ = train_step(model, &mut twin, &first_x);

    let bytes = live.into_bytes().expect("serialize");
    let mut reloaded = plan.build(&fallback, &muon).from_bytes(bytes).expect("deserialize");
    let mut fresh = plan.build(&fallback, &muon);

    // The optimizer update of the second training step, in units of the
    // learning rate.
    let before = in_proj(&stepped);
    let update = |optim: &mut ModuleOptimizer| (in_proj(&train_step(stepped.clone(), optim, &second_x)) - before.clone()) / lr;
    let (from_record, from_twin, from_scratch) = (update(&mut reloaded), update(&mut twin), update(&mut fresh));

    let diff = max_rel_diff(from_record.clone(), from_twin);
    assert!(diff < dtype_tol(1e-5), "the reloaded optimizer and the live one differ: {diff}");
    let diff = max_rel_diff(from_record, from_scratch);
    assert!(diff > dtype_tol(1e-3), "the record holds no state: its update equals a first update ({diff})");
}

/// The report has one line for each parameter, and a last line with the
/// share of Muon. The line of a fused projection lists its sub-projections,
/// with `*` on each one that the fallback optimizer owns.
#[test]
fn describe_reports_every_parameter() {
    let device = test_device();
    let model = LatentShape::new(1, 10, NetworkShape::new(2)).init(block_config(), &device);

    let report = fused_plan().describe(&model);

    assert_eq!(report.lines().count(), float_params(&model).len() + 1, "{report}");
    assert_eq!(report.matches("muon[value:6 scale:2*]").count(), 2, "{report}");
    assert!(report.contains("on muon:"), "{report}");
}
