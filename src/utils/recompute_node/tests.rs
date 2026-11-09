//! The recompute node gives the outputs and the gradients of plain autodiff
//! of the same formula. The test op has four differentiable inputs, one input
//! that can return no gradient, and a tensor in `Extra`:
//!
//! `y_bs = keep_bs · a_bs · x_bs`, plus `origin_bs` if `adds_origin`.
//! `final_state_b = scale_b · Σₛ x_bs`.
//!
//! The call goes through the `Dispatch` backend, so the test device selects
//! the checkpoint strategy of the `Autodiff<B, C>` impl.

use super::{Prim, RecomputeOp, recompute_pair};
use crate::utils::fprim::F;
use crate::utils::test_helpers::{dtype_tol, max_rel_diff, test_device};
use burn::backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::prelude::*;

const TOL: f32 = 1e-5;

crate::recompute_inputs!(TestInputs { a_bs, x_bs, scale_b, origin_bs });

#[derive(Clone, Debug)]
struct TestExtra<B: Backend> {
    keep_bs: Prim<B>,
    adds_origin: bool,
}

struct TestOp;

#[backend_extension(
    Cube: cfg(any(
        feature = "backend-cpu",
        feature = "backend-cuda",
        feature = "backend-rocm",
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-wgpu",
        feature = "backend-webgpu",
    )),
    Flex: cfg(feature = "backend-flex"),
    NdArray: cfg(feature = "backend-ndarray"),
    LibTorch: cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu")),
    Autodiff: cfg(feature = "autodiff"),
)]
trait TestOpExt: Backend {
    /// Returns `y_bs` and `final_state_b`.
    fn test_op(
        a_bs: FloatTensor<Self>,
        x_bs: FloatTensor<Self>,
        scale_b: FloatTensor<Self>,
        origin_bs: FloatTensor<Self>,
        keep_bs: FloatTensor<Self>,
        adds_origin: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let x_bs = F::<Self, 2>::new(x_bs);
        let mut y_bs = F::<Self, 2>::new(keep_bs) * F::new(a_bs) * x_bs.clone();
        if adds_origin {
            y_bs = y_bs + F::new(origin_bs);
        }
        let final_state_b = F::<Self, 1>::new(scale_b) * x_bs.sum_dim(1).squeeze_dim::<1>(1);
        (y_bs.inner(), final_state_b.inner())
    }
}

crate::impl_backend_ext_for_burn_backends!(TestOpExt);

impl<B: Backend + TestOpExt> RecomputeOp<B, 4> for TestOp {
    type Inputs<T> = TestInputs<T>;
    type Extra = TestExtra<B>;

    fn forward(inputs: TestInputs<Prim<B>>, extra: &TestExtra<B>) -> (Prim<B>, Prim<B>) {
        B::test_op(
            inputs.a_bs,
            inputs.x_bs,
            inputs.scale_b,
            inputs.origin_bs,
            extra.keep_bs.clone(),
            extra.adds_origin,
        )
    }

    fn backward(
        d_y: Prim<B>,
        d_final_state: Prim<B>,
        inputs: TestInputs<Prim<B>>,
        extra: TestExtra<B>,
    ) -> TestInputs<Option<Prim<B>>> {
        let a_bs = F::<B, 2>::new(inputs.a_bs);
        let x_bs = F::<B, 2>::new(inputs.x_bs);
        let scale_b = F::<B, 1>::new(inputs.scale_b);
        let keep_bs = F::<B, 2>::new(extra.keep_bs);
        let d_y_bs = F::<B, 2>::new(d_y);
        let d_final_state_b = F::<B, 1>::new(d_final_state);
        let shape_bs = x_bs.dims();

        let d_kept_bs = d_y_bs.clone() * keep_bs;
        let d_a_bs = d_kept_bs.clone() * x_bs.clone();
        let d_scale_b = d_final_state_b.clone() * x_bs.sum_dim(1).squeeze_dim::<1>(1);
        let d_state_x_bs = (d_final_state_b * scale_b).unsqueeze_dim::<2>(1).expand(shape_bs);
        let d_x_bs = d_kept_bs * a_bs + d_state_x_bs;

        TestInputs {
            a_bs: Some(d_a_bs.inner()),
            x_bs: Some(d_x_bs.inner()),
            scale_b: Some(d_scale_b.inner()),
            origin_bs: extra.adds_origin.then(|| d_y_bs.inner()),
        }
    }
}

impl<B: Backend + TestOpExt, C: CheckpointStrategy> TestOpExt for Autodiff<B, C> {
    fn test_op(
        a_bs: FloatTensor<Self>,
        x_bs: FloatTensor<Self>,
        scale_b: FloatTensor<Self>,
        origin_bs: FloatTensor<Self>,
        keep_bs: FloatTensor<Self>,
        adds_origin: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        recompute_pair::<B, C, TestOp, 4>(
            TestInputs { a_bs, x_bs, scale_b, origin_bs },
            TestExtra { keep_bs: keep_bs.into_primitive(), adds_origin },
        )
    }
}

type Forward =
    fn(Tensor<2>, Tensor<2>, Tensor<1>, Tensor<2>, Tensor<2>, bool) -> (Tensor<2>, Tensor<1>);

/// The test op through the recompute node.
fn through_node(
    a_bs: Tensor<2>,
    x_bs: Tensor<2>,
    scale_b: Tensor<1>,
    origin_bs: Tensor<2>,
    keep_bs: Tensor<2>,
    adds_origin: bool,
) -> (Tensor<2>, Tensor<1>) {
    let (y_bs, final_state_b) = <Dispatch as TestOpExt>::test_op(
        a_bs.into_dispatch(),
        x_bs.into_dispatch(),
        scale_b.into_dispatch(),
        origin_bs.into_dispatch(),
        keep_bs.into_dispatch(),
        adds_origin,
    );
    (Tensor::from_dispatch(y_bs), Tensor::from_dispatch(final_state_b))
}

/// The same formula as plain autodiff, with no node.
fn plain_autodiff(
    a_bs: Tensor<2>,
    x_bs: Tensor<2>,
    scale_b: Tensor<1>,
    origin_bs: Tensor<2>,
    keep_bs: Tensor<2>,
    adds_origin: bool,
) -> (Tensor<2>, Tensor<1>) {
    let mut y_bs = keep_bs * a_bs * x_bs.clone();
    if adds_origin {
        y_bs = y_bs + origin_bs;
    }
    let final_state_b = scale_b * x_bs.sum_dim(1).squeeze_dim::<1>(1);
    (y_bs, final_state_b)
}

/// The outputs on the inner backend, and the gradient of each input.
struct Run {
    y_bs: Tensor<2>,
    final_state_b: Tensor<1>,
    d_a_bs: Option<Tensor<2>>,
    d_x_bs: Option<Tensor<2>>,
    d_scale_b: Option<Tensor<1>>,
    d_origin_bs: Option<Tensor<2>>,
}

/// `tracked` holds one flag for each of `a_bs`, `x_bs`, `scale_b` and `origin_bs`.
/// With no flag, the run has no backward and no gradient.
fn run(forward: Forward, device: &Device, tracked: [bool; 4], adds_origin: bool) -> Run {
    fn leaf<const D: usize>(tensor: Tensor<D>, tracked: bool) -> Tensor<D> {
        if tracked { tensor.require_grad() } else { tensor }
    }

    let a_bs = leaf(
        Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]], device),
        tracked[0],
    );
    let x_bs = leaf(
        Tensor::<2>::from_floats([[0.5, -1.0, 2.0], [0.25, 3.0, -2.0]], device),
        tracked[1],
    );
    let scale_b = leaf(Tensor::<1>::from_floats([-2.0, 1.5], device), tracked[2]);
    let origin_bs = leaf(
        Tensor::<2>::from_floats([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], device),
        tracked[3],
    );
    let keep_bs = Tensor::<2>::from_floats([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], device);

    let (y_bs, final_state_b) = forward(
        a_bs.clone(),
        x_bs.clone(),
        scale_b.clone(),
        origin_bs.clone(),
        keep_bs,
        adds_origin,
    );
    assert_eq!(y_bs.gradient_checkpointing_strategy(), device.gradient_checkpointing_strategy());
    let values = Run {
        y_bs: y_bs.clone().inner(),
        final_state_b: final_state_b.clone().inner(),
        d_a_bs: None,
        d_x_bs: None,
        d_scale_b: None,
        d_origin_bs: None,
    };
    if !tracked.contains(&true) {
        return values;
    }

    let weight_bs = Tensor::<2>::from_floats([[1.0, -1.0, 2.0], [0.5, 3.0, -2.0]], device);
    let weight_b = Tensor::<1>::from_floats([0.25, 2.0], device);
    let loss = (y_bs * weight_bs).sum() + (final_state_b * weight_b).sum();
    let grads = loss.backward();
    Run {
        d_a_bs: a_bs.grad(&grads),
        d_x_bs: x_bs.grad(&grads),
        d_scale_b: scale_b.grad(&grads),
        d_origin_bs: origin_bs.grad(&grads),
        ..values
    }
}

fn assert_close<const D: usize>(label: &str, node: Tensor<D>, plain: Tensor<D>) {
    assert_eq!(node.dims(), plain.dims(), "{label}: shape");
    let diff = max_rel_diff(node, plain);
    let tol = dtype_tol(TOL);
    assert!(diff < tol, "{label}: max rel diff {diff}, tol {tol}");
}

fn assert_grad<const D: usize>(label: &str, node: Option<Tensor<D>>, plain: Option<Tensor<D>>) {
    match (node, plain) {
        (Some(node), Some(plain)) => assert_close(label, node, plain),
        (None, None) => {}
        (node, plain) => panic!(
            "{label}: the node has a gradient: {}, plain autodiff has a gradient: {}",
            node.is_some(),
            plain.is_some(),
        ),
    }
}

fn assert_node_matches_plain_autodiff(device: &Device, tracked: [bool; 4], adds_origin: bool) {
    let label = format!(
        "{:?}, tracked {tracked:?}, adds_origin {adds_origin}",
        device.gradient_checkpointing_strategy(),
    );
    let node = run(through_node, device, tracked, adds_origin);
    let plain = run(plain_autodiff, device, tracked, adds_origin);
    assert_close(&format!("{label}: y_bs"), node.y_bs, plain.y_bs);
    assert_close(&format!("{label}: final_state_b"), node.final_state_b, plain.final_state_b);
    assert_grad(&format!("{label}: d_a_bs"), node.d_a_bs, plain.d_a_bs);
    assert_grad(&format!("{label}: d_x_bs"), node.d_x_bs, plain.d_x_bs);
    assert_grad(&format!("{label}: d_scale_b"), node.d_scale_b, plain.d_scale_b);
    assert_grad(&format!("{label}: d_origin_bs"), node.d_origin_bs, plain.d_origin_bs);
}

/// One device for each checkpoint strategy.
fn autodiff_devices() -> [Device; 2] {
    [test_device().autodiff(), test_device().autodiff().gradient_checkpointing()]
}

#[test]
fn node_matches_plain_autodiff_with_every_input_tracked() {
    for device in autodiff_devices() {
        assert_node_matches_plain_autodiff(&device, [true; 4], true);
        assert_node_matches_plain_autodiff(&device, [true; 4], false);
    }
}

/// Each input alone receives its gradient. A swap of two inputs in the wiring
/// gives a gradient to the wrong input, or the gradient of another input.
#[test]
fn node_matches_plain_autodiff_with_each_input_tracked_alone() {
    for device in autodiff_devices() {
        for alone in 0..4 {
            let tracked = core::array::from_fn(|input| input == alone);
            assert_node_matches_plain_autodiff(&device, tracked, true);
        }
    }
}

/// The untracked arm moves the inputs into the forward. It packs and splits
/// the outputs as the tracked arm does.
#[test]
fn node_matches_plain_autodiff_with_no_input_tracked() {
    for device in autodiff_devices() {
        assert_node_matches_plain_autodiff(&device, [false; 4], true);
        assert_node_matches_plain_autodiff(&device, [false; 4], false);
    }
}

/// When `adds_origin` is `false`, the test op returns no gradient for
/// `origin_bs`, so the gradients hold no entry for it. A registered zero
/// gradient fails this test.
#[test]
fn a_none_gradient_registers_nothing() {
    for device in autodiff_devices() {
        let every_input = run(through_node, &device, [true; 4], false);
        assert!(every_input.d_a_bs.is_some());
        assert!(every_input.d_x_bs.is_some());
        assert!(every_input.d_scale_b.is_some());
        assert!(every_input.d_origin_bs.is_none());

        let origin_alone = run(through_node, &device, [false, false, false, true], false);
        assert!(origin_alone.d_origin_bs.is_none());
    }
}
