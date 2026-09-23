//! A trainer's captured SGD steps are its eager ones: under plain SGD the whole
//! step is captured (a hardware graph under `backend-cuda`, which
//! [`expects_graph`] asserts), a batch of another shape steps eagerly into the
//! same weights, and every loss, output and final weight is what Burn's `Sgd`
//! gets stepping the same [`LossFn`] eagerly.

use super::Trainer;
use crate::examples::training::OptimizerConfig;
use crate::modules::{CacheTensors, Layers, LayersBuilder, TensorZip};
use crate::optim::SgdConfig;
use crate::reference::{RefBlock, RefBlockConfig};
use crate::utils::Weights;
use crate::utils::test_helpers::max_abs_diff;
use burn::grad_clipping::GradientClippingConfig;
use burn::prelude::*;
use burn::tensor::Distribution;

const D_MODEL: usize = 8;
const LEN: usize = 5;
const STEPS: usize = 8;

/// Whether this build and device give a hardware graph (cubecl without
/// fusion, on a CUDA/HIP device).
fn expects_graph(device: &Device) -> bool {
    let name = format!("{device:?}");
    cfg!(all(feature = "cubecl", not(feature = "fusion")))
        && (name.contains("Cuda") || name.contains("Hip"))
}

/// The mean squared error of the layers' output against `y`, and the output.
fn loss(layers: &Layers<RefBlock>, (x, y): (Tensor<3>, Tensor<3>)) -> (Tensor<1>, Tensor<3>) {
    let out = layers.forward(x.autodiff(), None, (), None, None).0;
    let loss = (out.clone() - y.autodiff()).square().mean();
    (loss, out.inner())
}

#[test]
fn captured_sgd_steps_are_the_eager_steps() {
    let device = Device::default();
    let autodiff = device.clone().autodiff();
    let sgd = SgdConfig::new()
        .with_weight_decay(Some(1e-2))
        .with_grad_clipping(Some(GradientClippingConfig::Value(0.05)));
    let optimizer = OptimizerConfig::new(sgd.clone().into());
    let normal = |b| Tensor::<3>::random([b, LEN, D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    // Batches of 3, but one of 2: the shape the graph was not captured at.
    let data: Vec<(Tensor<3>, Tensor<3>)> = (0..STEPS)
        .map(|k| if k == STEPS / 2 { 2 } else { 3 })
        .map(|b| (normal(b), normal(b)))
        .collect();
    let lr = |k: usize| 0.3 / (k + 1) as f64;
    let layers: Layers<RefBlock> = LayersBuilder::new(2, RefBlockConfig::new(D_MODEL)).init(&autodiff);

    let run = |graphs: bool| {
        let layers = Weights(layers.clone()).into_owned_buffers().0;
        let mut trainer = Trainer::new(layers, loss, &optimizer, graphs);
        let mut optim = sgd.init();
        let outputs: Vec<_> = data
            .iter()
            .enumerate()
            .map(|(k, batch)| trainer.step(batch.clone(), &mut optim, lr(k)))
            .collect();
        (outputs, trainer)
    };
    let (eager, eager_trainer) = run(false);
    let (captured, captured_trainer) = run(true);
    assert!(!eager_trainer.is_captured());
    assert_eq!(captured_trainer.is_captured(), expects_graph(&device));
    for (k, ((l0, y0), (l1, y1))) in eager.into_iter().zip(captured).enumerate() {
        let d = max_abs_diff(l0, l1);
        assert_eq!(d, 0.0, "loss {k} differs by {d}");
        let d = max_abs_diff(y0, y1);
        assert_eq!(d, 0.0, "outputs {k} differ by {d}");
    }

    struct MaxDiff(f32);
    impl TensorZip for MaxDiff {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            self.0 = self.0.max(max_abs_diff(a.clone(), b));
            a
        }
    }
    let mut diff = MaxDiff(0.0);
    let _ = Weights(captured_trainer.module()).zip_tensors(Weights(eager_trainer.module()), &mut diff);
    assert_eq!(diff.0, 0.0, "the final weights differ by {}", diff.0);
}
