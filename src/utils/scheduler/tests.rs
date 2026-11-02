use super::*;

#[test]
fn test_cosine_annealing_warmup() {
    let lr_schedule = CosineAnnealingLr::new(1000)
        .with_max_lr(1.)
        .with_min_lr(0.01)
        .with_warmup_training_steps(100);

    // At training step 0, LR should be 0
    assert_eq!(lr_schedule.get_lr(0), 0.);

    // At half warmup, LR should be max_lr / 2
    assert_eq!(lr_schedule.get_lr(50), 0.5);

    // At end of warmup, LR should be max_lr
    assert_eq!(lr_schedule.get_lr(100), 1.0);

    // After total_training_steps, LR should be min_lr
    assert_eq!(lr_schedule.get_lr(1000), 0.01);
}

#[test]
fn test_constant_lr() {
    let lr_schedule = ConstantLr::new().with_lr(0.001);
    assert_eq!(lr_schedule.get_lr(0), 0.001);
    assert_eq!(lr_schedule.get_lr(1000), 0.001);
    assert_eq!(lr_schedule.get_lr(10000), 0.001);
}

#[test]
fn a_linear_ramp_holds_its_end() {
    let ramp = LinearLr::new(0.0, 4.0, 8);
    assert_eq!(ramp.get_lr(0), 0.0);
    assert_eq!(ramp.get_lr(2), 1.0);
    assert_eq!(ramp.get_lr(8), 4.0);
    assert_eq!(ramp.get_lr(100), 4.0);
    assert_eq!(LinearLr::new(1.0, 2.0, 0).get_lr(0), 2.0, "a ramp of 0 training steps is its end");
}

/// A warmup, a drop, and a cosine cooldown: the LR schedule of a whole run in
/// one chain.
fn run_chain() -> Lr {
    Lr::Chain(vec![
        LrLink::new(0, Lr::Linear(LinearLr::new(0.0, 4.0, 4))),
        LrLink::new(10, Lr::Constant(ConstantLr::new().with_lr(1.0))),
        LrLink::new(
            20,
            Lr::CosineAnnealing(CosineAnnealingLr::new(10).with_max_lr(1.0).with_min_lr(0.01)),
        ),
    ])
}

#[test]
fn a_chain_counts_each_schedule_from_its_start() {
    let lr = run_chain();
    lr.validate();
    assert_eq!(lr.get_lr(0), 0.0);
    assert_eq!(lr.get_lr(2), 2.0, "the warmup ramp");
    assert_eq!(lr.get_lr(9), 4.0, "the end of the ramp holds until the next schedule of the chain");
    assert_eq!(lr.get_lr(10), 1.0, "the drop");
    assert_eq!(
        lr.get_lr(20),
        1.0,
        "the cosine starts at its max: its training step 0 is the global training step 20"
    );
    let mid = lr.get_lr(25);
    assert!((mid - 0.505).abs() < 1e-12, "half of the cosine: {mid}");
    assert_eq!(lr.get_lr(30), 0.01);
    assert_eq!(lr.get_lr(1000), 0.01, "the last schedule of the chain runs to the end");
}

#[test]
fn a_chain_survives_a_json_round_trip() {
    let lr = run_chain();
    let json = lr.to_string();
    assert!(json.contains("\"Chain\"") && json.contains("\"from\""), "{json}");
    let back = Lr::load_binary(json.as_bytes()).expect("the JSON of a chain loads");
    for training_step in [0, 2, 9, 10, 19, 20, 25, 30, 100] {
        assert_eq!(back.get_lr(training_step), lr.get_lr(training_step), "training step {training_step}");
    }
}

#[test]
#[should_panic(expected = "at least one schedule")]
fn an_empty_chain_is_invalid() {
    Lr::Chain(vec![]).validate();
}

#[test]
#[should_panic(expected = "must start at training step 0")]
fn a_chain_starts_at_training_step_zero() {
    Lr::Chain(vec![LrLink::new(5, Lr::Constant(ConstantLr::new()))]).validate();
}

#[test]
#[should_panic(expected = "increasing training steps")]
fn the_schedules_of_a_chain_start_at_increasing_training_steps() {
    let constant = || Lr::Constant(ConstantLr::new());
    Lr::Chain(vec![
        LrLink::new(0, constant()),
        LrLink::new(10, constant()),
        LrLink::new(10, constant()),
    ])
    .validate();
}

#[test]
#[should_panic(expected = "increasing training steps")]
fn validate_checks_the_inner_chains() {
    let constant = || Lr::Constant(ConstantLr::new());
    let inner = Lr::Chain(vec![LrLink::new(0, constant()), LrLink::new(0, constant())]);
    Lr::Chain(vec![LrLink::new(0, inner)]).validate();
}

#[test]
fn set_peak_scales_a_chain() {
    let mut lr = run_chain();
    assert_eq!(lr.peak(), 4.0);
    lr.set_peak(2.0);
    assert_eq!(lr.peak(), 2.0);
    assert_eq!(lr.get_lr(2), 1.0, "the ramp keeps its shape");
    assert_eq!(lr.get_lr(10), 0.5);
    assert_eq!(lr.get_lr(1000), 0.005, "the min rate of the cosine scales too");
}

#[test]
fn set_peak_keeps_the_min_rate_of_a_cosine() {
    let mut lr = Lr::CosineAnnealing(CosineAnnealingLr::new(10).with_max_lr(1.0).with_min_lr(0.01));
    lr.set_peak(2.0);
    let Lr::CosineAnnealing(cosine) = &lr else { panic!() };
    assert_eq!((cosine.max_lr, cosine.min_lr), (2.0, 0.01));
}

#[test]
fn scale_steps_moves_every_training_step_of_a_chain() {
    let mut lr = run_chain();
    lr.scale_steps(2, 1, false);
    lr.validate();
    assert_eq!(lr.get_lr(4), 2.0, "the ramp is twice as long");
    assert_eq!(lr.get_lr(19), 4.0);
    assert_eq!(lr.get_lr(20), 1.0, "the drop moves to training step 20");
    assert_eq!(lr.get_lr(40), 1.0);
    assert_eq!(lr.get_lr(60), 0.01, "the cooldown is twice as long");
    let mut cosine = Lr::CosineAnnealing(CosineAnnealingLr::new(100).with_warmup_training_steps(10));
    cosine.scale_steps(1, 2, false);
    let Lr::CosineAnnealing(inner) = &cosine else { panic!() };
    assert_eq!(
        (inner.total_training_steps, inner.warmup_training_steps),
        (50, 10),
        "a cosine outside a chain keeps its warmup"
    );
}
