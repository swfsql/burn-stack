//! That the penalty is invisible in the value and exact in the gradient.

use super::*;
use burn::module::Param;

type Device = burn::prelude::Device;

const FACTOR: f64 = 1e-2;

fn logits(device: &Device) -> Tensor<3> {
    // [batch = 1, sequence = 2, vocab = 3], one clear winner per position.
    Tensor::from_data(
        burn::tensor::TensorData::new(vec![1.0f32, 4.0, 2.0, 5.0, 0.0, -1.0], [1, 2, 3]),
        device,
    )
}

/// The wrapped loss reports the number it was given: the penalty rides in the
/// gradient alone, so a training curve stays comparable to an unpenalised run
/// (and to the reference, whose hand-written backward does the same).
#[test]
fn the_reported_loss_is_unchanged() {
    let device: Device = Default::default();
    let loss = Tensor::<1>::from_data(burn::tensor::TensorData::new(vec![3.5f32], [1]), &device);

    let wrapped = l2_warp(loss.clone(), logits(&device), FACTOR);

    let before = loss.into_data().try_to_vec::<f32>().unwrap();
    let after = wrapped.into_data().try_to_vec::<f32>().unwrap();
    assert!((before[0] - after[0]).abs() < 1e-6, "{before:?} vs {after:?}");
}

/// Only the winning logit is pulled, and by exactly `factor/(B·T) · z_max` —
/// the derivative of `½·factor·mean(max²)`, which is what the reference's
/// custom backward scatters.
#[test]
fn only_the_max_logit_is_pulled_and_by_the_right_amount() {
    let device = Device::default().autodiff();
    let z = Param::from_tensor(logits(&device));
    let loss = Tensor::<1>::zeros([1], &device);

    let grads = l2_warp(loss, z.val(), FACTOR).backward();
    let g = z
        .val()
        .grad(&grads)
        .expect("the penalty reaches the logits")
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();

    // Two positions, so the mean divides by 2; the winners are 4.0 and 5.0.
    let scale = FACTOR as f32 / 2.0;
    let expected = [0.0, 4.0 * scale, 0.0, 5.0 * scale, 0.0, 0.0];
    for (i, (got, want)) in g.iter().zip(expected).enumerate() {
        assert!((got - want).abs() < 1e-7, "position {i}: {got} vs {want}");
    }
}
