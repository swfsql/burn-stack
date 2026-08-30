use super::*;

#[test]
fn schedule() {
    use Schedule::*;
    assert_eq!(
        (0..8)
            .map(|i| Schedule::real_idx(&Cyclic, i, 8, 3))
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 0, 1, 2, 0, 1]
    );
    assert_eq!(
        (0..8)
            .map(|i| Schedule::real_idx(&Stretched, i, 8, 3))
            .collect::<Vec<_>>(),
        vec![0, 0, 0, 1, 1, 1, 2, 2]
    );
    let custom = vec![0, 1, 2, 2, 1, 0, 0, 0];
    assert_eq!(
        (0..8)
            .map(|i| Schedule::real_idx(&Custom(custom.clone()), i, 8, 3))
            .collect::<Vec<_>>(),
        custom
    );
}

#[test]
fn bidi_schedule() {
    use BidiSchedule::*;
    assert_eq!(
        (0..10)
            .map(|i| BidiSchedule::real_idx(&StridedCyclic, i, 10, 4))
            .collect::<Vec<_>>(),
        vec![
            0, 1, /**/ 2, 3, /**/ 0, 1, /**/ 2, 3, /**/ 0, 1
        ]
    );
    assert_eq!(
        (0..10)
            .map(|i| BidiSchedule::real_idx(&StridedStretched, i, 10, 4))
            .collect::<Vec<_>>(),
        vec![
            0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 2, 3, /**/ 2, 3
        ]
    );
    assert_eq!(
        (0..10)
            .map(|i| BidiSchedule::real_idx(&SymmetricCyclic, i, 10, 4))
            .collect::<Vec<_>>(),
        vec![
            0, 0, /**/ 1, 1, /**/ 2, 2, /**/ 3, 3, /**/ 0, 0
        ]
    );
    assert_eq!(
        (0..10)
            .map(|i| BidiSchedule::real_idx(&SymmetricStretched, i, 10, 4))
            .collect::<Vec<_>>(),
        vec![
            0, 0, /**/ 0, 0, /**/ 1, 1, /**/ 2, 2, /**/ 3, 3
        ]
    );
    let custom = vec![
        0, 1, /**/ 2, 2, /**/ 1, 0, /**/ 0, 0, /**/ 3, 2,
    ];
    assert_eq!(
        (0..10)
            .map(|i| BidiSchedule::real_idx(&Custom(custom.clone()), i, 10, 4))
            .collect::<Vec<_>>(),
        custom
    );
}

#[test]
fn grad_horizon_depth_keeps_every_real_layer() {
    let t = |schedule: Option<&Schedule>, k: usize| {
        GradHorizon::Depth(k)
            .tracked(schedule, 8, 3)
            .into_iter()
            .map(|b| if b { 'T' } else { '.' })
            .collect::<String>()
    };

    // Cyclic (0 1 2 0 1 2 0 1) spreads each real layer's applications, so the
    // last `k` of each are a single top suffix.
    assert_eq!(t(Some(&Schedule::Cyclic), 0), "........");
    assert_eq!(t(Some(&Schedule::Cyclic), 1), ".....TTT");
    assert_eq!(t(Some(&Schedule::Cyclic), 2), "..TTTTTT");
    assert_eq!(t(Some(&Schedule::Cyclic), 3), "TTTTTTTT");

    // Stretched (0 0 0 1 1 1 2 2) runs each real layer once, contiguously, so
    // the tail of *every* run is tracked — one cut per real layer.
    assert_eq!(t(Some(&Schedule::Stretched), 1), "..T..T.T");
    assert_eq!(t(Some(&Schedule::Stretched), 2), ".TT.TTTT");
    assert_eq!(t(Some(&Schedule::Stretched), 3), "TTTTTTTT");

    // No virtual scheduling: one application per real layer, so any k >= 1
    // tracks the whole stack (`GradHorizon::last` is the suffix cut there).
    assert_eq!(
        GradHorizon::Depth(1).tracked(None, 3, 3),
        vec![true, true, true]
    );
    assert_eq!(
        GradHorizon::Depth(0).tracked(None, 3, 3),
        vec![false, false, false]
    );
}

#[test]
fn grad_horizon_last_is_a_plain_suffix() {
    let mask = |h: GradHorizon| match h {
        GradHorizon::Mask(m) => m,
        _ => unreachable!("last() builds a mask"),
    };
    assert_eq!(mask(GradHorizon::last(0, 3)), vec![false, false, false]);
    assert_eq!(mask(GradHorizon::last(2, 3)), vec![false, true, true]);
    assert_eq!(mask(GradHorizon::last(9, 3)), vec![true, true, true]);
    // A mask is taken as given, whatever the schedule.
    let m = vec![true, false, true, false, false, true, false, true];
    assert_eq!(
        GradHorizon::Mask(m.clone()).tracked(Some(&Schedule::Stretched), 8, 3),
        m
    );
}

#[test]
#[should_panic(expected = "GradHorizon::Depth is undefined for Schedule::Custom")]
fn grad_horizon_depth_rejects_a_custom_schedule() {
    let custom = Schedule::Custom(vec![0, 1, 2, 2, 1, 0, 0, 0]);
    GradHorizon::Depth(1).tracked(Some(&custom), 8, 3);
}

#[test]
#[should_panic(expected = "one flag per virtual layer")]
fn grad_horizon_mask_must_match_the_stack() {
    GradHorizon::Mask(vec![true, false]).tracked(Some(&Schedule::Cyclic), 8, 3);
}
