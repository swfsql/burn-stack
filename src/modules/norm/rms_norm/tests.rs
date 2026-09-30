use super::*;
use burn::module::Param;
use burn::tensor::Distribution;
use crate::utils::test_helpers::test_device;

/// A half-precision input computes the formula `x / √(mean(x²) + ε)` with the
/// `ε` of f32, the dtype that it computes in, at each scale of a row: zero,
/// the subnormals of f16, and up to the top of its range. The forward and the
/// gradient are finite, and each is within two steps of the dtype of the f64
/// formula.
///
/// A row near zero is the limit case of the backward. Its gradient is about
/// `|h|/√ε ≈ 3500·|h|`, and f16 holds that for the `|h| ≤ 5` here. The gradient
/// is compared with the size of its two terms (`|h|/rms`), because at width 1
/// the terms cancel.
#[test]
#[cfg_attr(
    not(feature = "dev-f16"),
    ignore = "f16 build only: a half-precision check (f16 and bf16 inputs)"
)]
fn a_half_input_computes_the_formula_at_each_scale() {
    let device = test_device();
    let eps = f64::from(crate::utils::div_eps(burn::tensor::DType::F32));
    let rows = 8;
    let host = |t: Tensor<2>| -> Vec<f64> {
        let v: Vec<f32> = t.into_data().try_into_vec_as().unwrap();
        v.into_iter().map(f64::from).collect()
    };
    for dtype in [FloatDType::F16, FloatDType::BF16] {
        let step = dtype.finfo().epsilon;
        for scale in [0.0f32, 1e-7, 1e-5, 1e-3, 1e-1, 1.0, 1e2, 3e3] {
            for width in [1usize, 19, 152, 1024] {
                let norm = RmsNormConfig::new(width).init(&device.clone().autodiff());
                let draw = || Tensor::<2>::random([rows, width], Distribution::Normal(0.0, 1.0), &device);
                let x = (draw() * scale).cast(dtype);
                let h = draw().clamp(-5.0, 5.0).cast(dtype);
                let (xs, hs) = (host(x.clone()), host(h.clone()));
                let x = Param::from_tensor(Tensor::from_inner(x));
                let y = norm.forward(x.val());
                let grads = (y.clone() * Tensor::from_inner(h)).sum().backward();
                let ys = host(y.inner());
                let gs = host(x.val().grad(&grads).expect("grad exists"));
                let (mut err_y, mut size_y, mut err_g, mut size_g) = (0f64, 0f64, 0f64, 0f64);
                for r in 0..rows {
                    let row = r * width..(r + 1) * width;
                    let (x, h) = (&xs[row.clone()], &hs[row.clone()]);
                    let rms = (x.iter().map(|v| v * v).sum::<f64>() / width as f64 + eps).sqrt();
                    let xh: f64 = x.iter().zip(h).map(|(a, b)| a * b).sum();
                    for (c, i) in row.enumerate() {
                        let (want_y, want_g) = (x[c] / rms, h[c] / rms - x[c] * xh / (width as f64 * rms.powi(3)));
                        assert!(ys[i].is_finite() && gs[i].is_finite(), "{dtype:?}, scale {scale}, width {width}: not finite");
                        err_y = err_y.max((ys[i] - want_y).abs());
                        size_y = size_y.max(want_y.abs());
                        err_g = err_g.max((gs[i] - want_g).abs());
                        size_g = size_g.max(h[c].abs() / rms);
                    }
                }
                assert!(
                    err_y <= 2.0 * step * size_y,
                    "{dtype:?}, scale {scale}, width {width}: the output is off by {err_y:.2e} (size {size_y:.2e})"
                );
                assert!(
                    err_g <= 2.0 * step * size_g,
                    "{dtype:?}, scale {scale}, width {width}: the gradient is off by {err_g:.2e} (size {size_g:.2e})"
                );
            }
        }
    }
}

/// An f64 input computes in f64, with the `ε` of f64, in [`RmsNorm`] and in
/// [`rms_denom`](crate::modules::rms_denom). The rows have `mean(x²) ≈ 10⁻¹⁰`.
/// There, the `ε` of f32 (`8.2·10⁻⁸`) would make the output about 29× too
/// small. Each result is within `10⁻¹²` (relative) of the f64 formula.
#[test]
fn an_f64_input_computes_in_f64() {
    use burn::tensor::DType;
    let device = test_device();
    let eps = f64::from(crate::utils::div_eps(DType::F64));
    let (rows, width) = (4, 16);
    let x = (Tensor::<2>::random([rows, width], Distribution::Normal(0.0, 1.0), &device) * 1e-5)
        .cast(FloatDType::F64);
    let xs: Vec<f64> = x.to_data().try_into_vec_as().unwrap();
    let mut norm = RmsNormConfig::new(width).init(&device);
    norm.gamma = Param::from_tensor(norm.gamma.val().cast(FloatDType::F64));

    let y = norm.forward(x.clone());
    let denom = crate::modules::rms_denom(x);
    assert_eq!(y.dtype(), DType::F64, "the output of RmsNorm");
    assert_eq!(denom.dtype(), DType::F64, "the output of rms_denom");
    let ys: Vec<f64> = y.into_data().try_into_vec_as().unwrap();
    let ds: Vec<f64> = denom.into_data().try_into_vec_as().unwrap();
    for r in 0..rows {
        let row = &xs[r * width..(r + 1) * width];
        let rms = (row.iter().map(|v| v * v).sum::<f64>() / width as f64 + eps).sqrt();
        assert!((ds[r] - rms).abs() <= 1e-12 * rms, "row {r}: rms_denom {} vs {rms}", ds[r]);
        for (c, x) in row.iter().enumerate() {
            let (got, want) = (ys[r * width + c], x / rms);
            assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0), "row {r}: {got} vs {want}");
        }
    }
}

/// The backward of RMSNorm must stay finite when a normalised slice collapses
/// to zero norm (`mean(x²) = 0`): a dead token/channel, or a subnormal flushed
/// to zero on CUDA. The forward guards the *division* (`rms + div_eps`). But
/// the backward of the `sqrt` node is `1/(2·√(mean x²))`, singular at zero
/// unless the epsilon is *inside* the root. Regression guard for a
/// training-run NaN, localised to this op (the incoming `d_y` of the mixer
/// backward).
#[test]
fn rms_norm_gradient_finite_on_collapsed_slice() {
    let device = test_device();
    let (batch, seq, d_model) = (2, 3, 8);
    let norm = RmsNormConfig::new(d_model).init(&device.clone().autodiff());

    // A normal batch with one token collapsed to exactly zero norm.
    let normal = Tensor::<3>::random(
        [batch, seq - 1, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let collapsed = Tensor::<3>::zeros([batch, 1, d_model], &device);
    let base = Tensor::cat(vec![collapsed, normal], 1);

    let x = Param::from_tensor(Tensor::from_inner(base));
    let grads = norm.forward(x.val()).sum().backward();
    let g = x.val().grad(&grads).expect("grad exists");
    let gvec = g.into_data().try_into_vec_as::<f32>().unwrap();
    assert!(
        gvec.iter().all(|v| v.is_finite()),
        "RMSNorm gradient must stay finite for a zero-norm slice"
    );
}
