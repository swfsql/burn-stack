//! That the loss and its gradient are exact to the rounding of the dtype, in
//! f16 too.

use super::*;
use crate::utils::test_helpers::test_device;
use burn::module::Param;
use burn::tensor::{DType, Distribution};

fn host(t: Tensor<2>) -> Vec<f64> {
    t.into_data().try_into_vec_as::<f32>().unwrap().into_iter().map(f64::from).collect()
}

/// The loss is `mean(d²)` and its gradient is `2·d/N`, in f32 and in f16:
///
/// - with small residuals (≈ 1e-2) and residuals ≈ 1,
/// - with a residual of 300, whose square alone overflows f16.
#[test]
fn the_loss_and_its_gradient_are_exact() {
    let ad = test_device().autodiff();
    let n = 32;
    let mut big = vec![0.5f32; n];
    big[0] = 300.0;
    let cases = [
        ("residuals ≈ 1e-2", Tensor::<2>::random([4, 8], Distribution::Normal(0.0, 1e-2), &ad)),
        ("residuals ≈ 1", Tensor::<2>::random([4, 8], Distribution::Normal(0.0, 1.0), &ad)),
        ("a residual of 300", Tensor::<1>::from_floats(big.as_slice(), &ad).reshape([4, 8])),
    ];
    for dtype in [DType::F32, DType::F16] {
        // The tolerance is that of `dtype`, not that of the test device.
        let tol = if dtype == DType::F16 { 1e-2 } else { 1e-5 };
        for (name, x) in &cases {
            let x = Param::from_tensor(x.clone().cast(dtype));
            let d = host(x.val());
            let loss = MseLoss::new().forward(x.val(), x.val().zeros_like(), Reduction::Mean);
            let got = f64::from(loss.clone().into_scalar::<f32>());
            let want = d.iter().map(|v| v * v).sum::<f64>() / n as f64;
            assert!(((got - want) / want).abs() < tol, "{dtype:?}, {name}: loss {got} vs {want}");

            let grad = host(x.val().grad(&loss.backward()).expect("grad"));
            let peak = d.iter().fold(0.0f64, |m, v| m.max(2.0 * v.abs() / n as f64));
            for (g, v) in grad.iter().zip(&d) {
                let want = 2.0 * v / n as f64;
                assert!((g - want).abs() <= tol * peak, "{dtype:?}, {name}: gradient {g} vs {want}");
            }
        }
    }
}

/// Each row (sample) keeps the precision of its gradient, relative to the
/// residuals of that row, beside a row with a large residual: 900 beside rows
/// of ≈ 1e-3, and 1000 beside rows of ≈ 1e-4. A scale that all rows share
/// (the largest residual) would make `d/s` subnormal in f16 for the small
/// rows: `≈ 10⁻⁷` in the second case, which puts about 10% of error into
/// their gradient.
#[test]
fn each_row_keeps_its_own_precision() {
    for (big, small) in [(900.0f32, 1e-3f32), (1000.0, 1e-4)] {
        check_rows(big, small);
    }
}

fn check_rows(big: f32, small: f32) {
    let ad = test_device().autodiff();
    let (rows, cols) = (8, 2);
    let n = rows * cols;
    let mut values = vec![big, 0.5];
    for b in 1..rows {
        let b = b as f32;
        values.extend([(1.0 + 0.13 * b) * small, -(1.5 + 0.11 * b) * small]);
    }
    for dtype in [DType::F32, DType::F16] {
        let tol = if dtype == DType::F16 { 1e-2 } else { 1e-5 };
        let x = Param::from_tensor(
            Tensor::<1>::from_floats(values.as_slice(), &ad).reshape([rows, cols]).cast(dtype),
        );
        let d = host(x.val());
        let loss = MseLoss::new().forward(x.val(), x.val().zeros_like(), Reduction::Mean);
        let got = f64::from(loss.clone().into_scalar::<f32>());
        let want = d.iter().map(|v| v * v).sum::<f64>() / n as f64;
        assert!(((got - want) / want).abs() < tol, "{dtype:?}, big {big}: loss {got} vs {want}");

        let grad = host(x.val().grad(&loss.backward()).expect("grad"));
        for b in 0..rows {
            let row = b * cols..(b + 1) * cols;
            let peak = d[row.clone()].iter().fold(0.0f64, |m, v| m.max(2.0 * v.abs() / n as f64));
            for (g, v) in grad[row.clone()].iter().zip(&d[row]) {
                let want = 2.0 * v / n as f64;
                assert!(
                    (g - want).abs() <= tol * peak,
                    "{dtype:?}, big {big}, row {b}: gradient {g} vs {want}"
                );
            }
        }
    }
}
