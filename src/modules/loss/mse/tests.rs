//! That the fp16 rescale is exact, in the value and in the gradient.

use super::*;
use crate::utils::test_helpers::test_device;
use burn::module::Param;
use burn::tensor::Distribution;

fn host(t: Tensor<2>) -> Vec<f64> {
    t.into_data().try_into_vec_as::<f32>().unwrap().into_iter().map(f64::from).collect()
}

/// The loss is `mean(d²)` and its gradient is `2·d/N`, in f32 and in f16:
///
/// - with small residuals (≈ 1e-2), where a rescale that divides by
///   `max + eps` but multiplies back by `max` alone is ≈ 5% low,
/// - with a residual of 300, whose square alone overflows f16.
#[test]
fn the_f16_rescale_is_exact() {
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
/// residuals of that row, beside a row with a residual of 900. The scale is
/// global, so `sub/s ≈ 1e-6` is subnormal in f16 for the rows of residuals
/// ≈ 1e-3. Only the half of the gradient that reads `sub/s` carries that
/// rounding. The other half, `(g·sub)/s`, stays a normal number.
#[test]
fn each_row_keeps_its_own_precision() {
    let ad = test_device().autodiff();
    let (rows, cols) = (8, 2);
    let n = rows * cols;
    let mut values = vec![900.0f32, 0.5];
    for b in 1..rows {
        let b = b as f32;
        values.extend([(1.0 + 0.13 * b) * 1e-3, -(1.5 + 0.11 * b) * 1e-3]);
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
        assert!(((got - want) / want).abs() < tol, "{dtype:?}: loss {got} vs {want}");

        let grad = host(x.val().grad(&loss.backward()).expect("grad"));
        for b in 0..rows {
            let row = b * cols..(b + 1) * cols;
            let peak = d[row.clone()].iter().fold(0.0f64, |m, v| m.max(2.0 * v.abs() / n as f64));
            for (g, v) in grad[row.clone()].iter().zip(&d[row]) {
                let want = 2.0 * v / n as f64;
                assert!((g - want).abs() <= tol * peak, "{dtype:?}, row {b}: gradient {g} vs {want}");
            }
        }
    }
}
