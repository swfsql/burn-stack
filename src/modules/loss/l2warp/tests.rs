//! That the penalty is invisible in the value and exact in the gradient.

use super::*;
use burn::module::Param;
use crate::utils::test_helpers::{dtype_tol, test_device};

type Device = burn::prelude::Device;

const FACTOR: f64 = 1e-2;

fn logits(device: &Device) -> Tensor<3> {
    // [batch = 1, sequence = 2, vocab = 3], one clear winner per position.
    Tensor::from_data(
        burn::tensor::TensorData::new(vec![1.0f32, 4.0, 2.0, 5.0, 0.0, -1.0], [1, 2, 3]),
        device,
    )
}

/// The wrapped loss reports the number that it got. The penalty is in the
/// gradient alone, so a training curve stays comparable to an unpenalised run
/// (and to the reference, whose hand-written backward does the same).
#[test]
fn the_reported_loss_is_unchanged() {
    let device = test_device();
    let loss = Tensor::<1>::from_data(burn::tensor::TensorData::new(vec![3.5f32], [1]), &device);

    let wrapped = l2_warp(loss.clone(), logits(&device), FACTOR);

    let before = loss.into_data().try_into_vec_as::<f32>().unwrap();
    let after = wrapped.into_data().try_into_vec_as::<f32>().unwrap();
    assert!((before[0] - after[0]).abs() < dtype_tol(1e-6), "{before:?} vs {after:?}");
}

/// With f16 logits, the penalty computes in f32:
///
/// - A max logit of 300 (its square overflows f16) leaves the reported loss
///   unchanged, and the gradient is `c/(B·T) · z_max` to one f16 rounding.
/// - At `B·T = 8192` and the default factor, the gradient
///   `c/(B·T) · 10 ≈ 1.2·10⁻⁷` is within half a subnormal step of its value.
///   In f16, the backward chain (`c/2`, then `/(B·T)`) would give 0.
#[test]
#[cfg_attr(
    not(feature = "dev-f16"),
    ignore = "f16 build only: a half-precision check (f16 logits)"
)]
fn an_f16_penalty_keeps_the_loss_and_reaches_the_logits() {
    use burn::tensor::{FloatDType, TensorData};
    let device = test_device().autodiff();
    let c = DEFAULT_L2_PENALTY;
    let host = |t: Tensor<3>| -> Vec<f64> {
        let v: Vec<f32> = t.into_data().try_into_vec_as().unwrap();
        v.into_iter().map(f64::from).collect()
    };

    let values = vec![300.0f32, 1.0, 2.0, 5.0, 0.0, -1.0];
    let z = Param::from_tensor(
        Tensor::<3>::from_data(TensorData::new(values, [1, 2, 3]), &device).cast(FloatDType::F16),
    );
    let loss = Tensor::<1>::full([1], 3.5, &device).cast(FloatDType::F16);
    let wrapped = l2_warp(loss, z.val(), c);
    let reported = wrapped.clone().into_data().try_into_vec_as::<f32>().unwrap()[0];
    assert_eq!(reported, 3.5, "the reported loss");
    let g = host(z.val().grad(&wrapped.backward()).expect("the penalty reaches the logits"));
    let expected = [300.0 * c / 2.0, 0.0, 0.0, 5.0 * c / 2.0, 0.0, 0.0];
    let step = FloatDType::F16.finfo().epsilon;
    for (i, (got, want)) in g.iter().zip(expected).enumerate() {
        assert!((got - want).abs() <= step * want, "max 300, position {i}: {got} vs {want}");
    }

    let (batch, seq, vocab) = (8, 1024, 4);
    let winners = Tensor::<3>::full([batch, seq, 1], 10.0, &device);
    let others = Tensor::<3>::zeros([batch, seq, vocab - 1], &device);
    let z = Param::from_tensor(Tensor::cat(vec![winners, others], 2).cast(FloatDType::F16));
    let loss = Tensor::<1>::zeros([1], &device).cast(FloatDType::F16);
    let grads = l2_warp(loss, z.val(), c).backward();
    let g = host(z.val().grad(&grads).expect("the penalty reaches the logits"));
    let want = c / (batch * seq) as f64 * 10.0;
    let half_step = 2f64.powi(-25);
    for (i, got) in g.iter().enumerate() {
        let want = if i % vocab == 0 { want } else { 0.0 };
        assert!((got - want).abs() <= half_step, "B·T = 8192, position {i}: {got:e} vs {want:e}");
    }
}

/// Only the winning logit is pulled, and by exactly `factor/(B·T) · z_max`:
/// the derivative of `½·factor·mean(max²)`, which the custom backward of the
/// reference scatters.
#[test]
fn only_the_max_logit_is_pulled_and_by_the_right_amount() {
    let device = test_device().autodiff();
    let z = Param::from_tensor(logits(&device));
    let loss = Tensor::<1>::zeros([1], &device);

    let grads = l2_warp(loss, z.val(), FACTOR).backward();
    let g = z
        .val()
        .grad(&grads)
        .expect("the penalty reaches the logits")
        .into_data()
        .try_into_vec_as::<f32>()
        .unwrap();

    // Two positions, so the mean divides by 2. The winners are 4.0 and 5.0.
    let scale = FACTOR as f32 / 2.0;
    let expected = [0.0, 4.0 * scale, 0.0, 5.0 * scale, 0.0, 0.0];
    for (i, (got, want)) in g.iter().zip(expected).enumerate() {
        assert!((got - want).abs() < dtype_tol(1e-7), "position {i}: {got} vs {want}");
    }
}
