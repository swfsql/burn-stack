//! The users of `rms` at each scale of a finite input: [`RmsNorm`],
//! [`RmsNormGated`] in both orders of the gate, [`normed_score`] and
//! [`rms_denom`]. The forward and the gradient must be finite, and each must
//! give the value of the formula `√(max(mean(x²), ε))` (ADR-0010).
//!
//! The scales go from zero and the subnormals to the overflow of `x²`, and
//! they include scales on both sides of `√ε`. The overflow of `x²` at the top
//! of the range is out of scope. One more input of each dtype holds one row of
//! each scale, so a row that reads another row fails.
//!
//! The reference computes the formula in f64. A row that the clamp holds at
//! `ε` gives no gradient to `mean(x²)`.
//!
//! [`RmsNorm`]: super::rms_norm::RmsNorm
//! [`RmsNormGated`]: super::rms_norm_gated::RmsNormGated

use super::rms_norm::RmsNormConfig;
use super::rms_norm_gated::RmsNormGatedConfig;
use crate::modules::{normed_score, rms_denom, score_scale};
use crate::utils::div_eps;
use crate::utils::test_helpers::test_device;
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{DType, Distribution, FloatDType};

/// The rows of each input at one scale.
const ROWS: usize = 4;

/// The widths of each input. Width 1 is the limit case of the gradient: there,
/// its two terms cancel.
const WIDTHS: [usize; 3] = [1, 8, 1024];

/// One input dtype of the check.
struct Case {
    dtype: FloatDType,
    /// The scales of the inputs, from zero to below the overflow of `x²`.
    scales: &'static [f64],
    /// The relative error that the dtype allows.
    tol: f64,
    /// The smallest subnormal of the dtype (see [`Case::floor`]).
    tiny: f64,
}

/// The cases of the check. `√ε` is about `2.9·10⁻⁴` for an f16, bf16 or f32
/// input, and about `2·10⁻⁸` for an f64 input.
fn cases() -> [Case; 4] {
    [
        Case {
            dtype: FloatDType::F32,
            scales: &[0.0, 1e-45, 1e-40, 1e-30, 1e-20, 1e-10, 1e-4, 3e-4, 1e-3, 1.0, 1e10, 1e16],
            tol: 1e-5,
            tiny: f64::from(f32::from_bits(1)),
        },
        Case {
            dtype: FloatDType::F64,
            scales: &[0.0, 5e-324, 1e-310, 1e-200, 1e-100, 1e-20, 1e-8, 2e-8, 1e-7, 1.0, 1e100, 1e150],
            tol: 1e-12,
            tiny: f64::from_bits(1),
        },
        Case {
            dtype: FloatDType::F16,
            scales: &[0.0, 6e-8, 1e-6, 1e-4, 3e-4, 1e-3, 1e-2, 1.0, 1e2, 1e4],
            tol: 2.0 * FloatDType::F16.finfo().epsilon,
            tiny: 2f64.powi(-24),
        },
        Case {
            dtype: FloatDType::BF16,
            scales: &[0.0, 1e-40, 1e-30, 1e-10, 1e-4, 3e-4, 1e-3, 1.0, 1e10, 1e16],
            tol: 2.0 * FloatDType::BF16.finfo().epsilon,
            tiny: 2f64.powi(-133),
        },
    ]
}

impl Case {
    /// The dtype that `rms` computes in for this input dtype.
    fn compute(&self) -> DType {
        match self.dtype {
            FloatDType::F64 => DType::F64,
            _ => DType::F32,
        }
    }

    /// The `ε` of `rms` for this dtype: that of the dtype that it computes in.
    fn eps(&self) -> f64 {
        f64::from(div_eps(self.compute()))
    }

    /// The absolute error that the check also allows. A subnormal output
    /// rounds by about `tiny`. With `product`, a product before the norm
    /// (`x·w` or `x·SiLU(z)`) also rounds, by about the smallest subnormal of
    /// the compute dtype. `1/rms ≤ 1/√ε` multiplies that rounding, and a sum
    /// over the widest row (1024 values) adds a factor of `√1024`.
    fn floor(&self, product: bool) -> f64 {
        let output = 4.0 * self.tiny;
        if !product {
            return output;
        }
        let compute_tiny = match self.compute() {
            DType::F64 => f64::from_bits(1),
            _ => f64::from(f32::from_bits(1)),
        };
        output + 4.0 * compute_tiny * 32.0 / self.eps().sqrt()
    }

    /// A standard normal draw `[rows, width]` times `scale`, in this dtype.
    fn draw(&self, device: &Device, rows: usize, width: usize, scale: f64) -> Tensor<2> {
        Tensor::<2>::random([rows, width], Distribution::Normal(0.0, 1.0), device).cast(self.dtype) * scale
    }

    /// The inputs `x` of width `width`, each with its name in a report: one
    /// input for each scale, and one input with a row of each scale.
    fn inputs(&self, device: &Device, width: usize) -> Vec<(String, Tensor<2>)> {
        let name = |what: String| format!("{:?}, {what}, width {width}", self.dtype);
        let mut inputs: Vec<_> = self
            .scales
            .iter()
            .map(|&scale| (name(format!("scale {scale:e}")), self.draw(device, ROWS, width, scale)))
            .collect();
        let rows = self.scales.iter().map(|&scale| self.draw(device, 1, width, scale)).collect();
        inputs.push((name("mixed scales".to_string()), Tensor::cat(rows, 0)));
        inputs
    }

    /// `γ = 1` of width `width` in this dtype, as a leaf on the autodiff
    /// backend.
    fn gamma(&self, device: &Device, width: usize) -> Param<Tensor<1>> {
        Param::from_tensor(Tensor::from_inner(Tensor::<1>::ones([width], device).cast(self.dtype)))
    }
}

/// The host values of `t`, as f64.
fn host<const D: usize>(t: Tensor<D>) -> Vec<f64> {
    t.into_data().try_into_vec_as::<f64>().unwrap()
}

/// `f(x)` and the gradient of `Σ f(x)·h` with respect to `x`, as host values.
/// `x` enters `f` as a leaf on the autodiff backend.
fn forward_and_gradient(x: Tensor<2>, h: Tensor<2>, f: impl FnOnce(Tensor<2>) -> Tensor<2>) -> (Vec<f64>, Vec<f64>) {
    let x = Param::from_tensor(Tensor::from_inner(x));
    let y = f(x.val());
    let grads = (y.clone() * Tensor::from_inner(h)).sum().backward();
    let gradient = x.val().grad(&grads).expect("the input has a gradient");
    (host(y.inner()), host(gradient))
}

/// The largest magnitude of `v`.
fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0f64, |m, a| m.max(a.abs()))
}

/// `mean(x²)` of one row.
fn mean_sq(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64
}

/// Whether the clamp holds the row `x` at `ε`.
fn is_clamped(x: &[f64], eps: f64) -> bool {
    mean_sq(x) < eps
}

/// The size of the gradient check of the row `x`: `size`, or no limit when
/// `mean(x²)` is within `10⁻³` of `ε`. There, the rounding of the code can
/// select the other side of the clamp, and the gradient jumps. The finite
/// check of the gradient stays.
fn gradient_size(x: &[f64], eps: f64, size: f64) -> f64 {
    if (mean_sq(x) / eps - 1.0).abs() < 1e-3 { f64::INFINITY } else { size }
}

/// `√(max(mean(x²), ε))` of one row.
fn rms_ref(x: &[f64], eps: f64) -> f64 {
    mean_sq(x).max(eps).sqrt()
}

/// `Jᵀv`, with `J` the Jacobian of `x ↦ x / rms(x)` at the row `x`. It is
/// `(v − u·(u·v)/n) / rms` with `u = x / rms`. Under the clamp, `rms` is
/// constant, and `Jᵀv` is `v / √ε`.
fn norm_vjp(x: &[f64], v: &[f64], eps: f64) -> Vec<f64> {
    let (r, n) = (rms_ref(x, eps), x.len() as f64);
    if is_clamped(x, eps) {
        return v.iter().map(|b| b / r).collect();
    }
    let u: Vec<f64> = x.iter().map(|a| a / r).collect();
    let uv: f64 = u.iter().zip(v).map(|(a, b)| a * b).sum();
    u.iter().zip(v).map(|(a, b)| (b - a * uv / n) / r).collect()
}

/// `SiLU(z)`.
fn silu(z: f64) -> f64 {
    z / (1.0 + (-z).exp())
}

/// The expected values of one check, each with its size.
#[derive(Default)]
struct Want {
    values: Vec<f64>,
    sizes: Vec<f64>,
}

impl Want {
    /// Add `values`, each with the size `size`.
    fn extend(&mut self, values: impl IntoIterator<Item = f64>, size: f64) {
        self.values.extend(values);
        self.sizes.resize(self.values.len(), size);
    }
}

/// The checks that fail, one line for each.
#[derive(Default)]
struct Failures(Vec<String>);

impl Failures {
    /// Record a failure when a value of `got` is not finite, or when it
    /// differs from the expected value by more than `tol·size + floor`.
    fn check(&mut self, what: String, got: &[f64], want: &Want, floor: f64, case: &Case) {
        assert_eq!(got.len(), want.values.len(), "{what}: the code and the reference differ in length");
        assert!(want.values.iter().all(|w| w.is_finite()), "{what}: the reference is not finite");
        let bad = (0..got.len()).find(|&i| {
            let allowed = case.tol * want.sizes[i] + floor;
            !got[i].is_finite() || !((got[i] - want.values[i]).abs() <= allowed)
        });
        if let Some(i) = bad {
            self.0.push(format!("{what}: value {i} is {:e}, the formula gives {:e}.", got[i], want.values[i]));
        }
    }

    /// Panic with the list of the failures, if there is one.
    fn assert_none(self) {
        assert!(self.0.is_empty(), "{} checks fail:\n{}", self.0.len(), self.0.join("\n"));
    }
}

/// [`RmsNorm`](super::rms_norm::RmsNorm) at each scale: `y = x / rms(x)` with
/// `γ = 1`, and the gradient of `Σ y·h`.
#[test]
fn rms_norm_is_the_formula_at_each_scale() {
    let device = test_device();
    let mut failures = Failures::default();
    for case in cases() {
        let eps = case.eps();
        for width in WIDTHS {
            let mut norm = RmsNormConfig::new(width).init(&device.clone().autodiff());
            norm.gamma = case.gamma(&device, width);
            for (name, x) in case.inputs(&device, width) {
                let h = case.draw(&device, x.dims()[0], width, 1.0);
                let (xs, hs) = (host(x.clone()), host(h.clone()));
                let (ys, gs) = forward_and_gradient(x, h, |x| norm.forward(x));
                let (mut want_y, mut want_g) = (Want::default(), Want::default());
                for (x, h) in xs.chunks(width).zip(hs.chunks(width)) {
                    let r = rms_ref(x, eps);
                    let y: Vec<f64> = x.iter().map(|a| a / r).collect();
                    let size_y = max_abs(&y);
                    want_y.extend(y, size_y);
                    want_g.extend(norm_vjp(x, h, eps), gradient_size(x, eps, max_abs(h) / r));
                }
                failures.check(format!("{name}, the output"), &ys, &want_y, case.floor(false), &case);
                failures.check(format!("{name}, the gradient"), &gs, &want_g, case.floor(false), &case);
            }
        }
    }
    failures.assert_none();
}

/// [`RmsNormGated`](super::rms_norm_gated::RmsNormGated) at each scale of `x`,
/// in both orders of the gate, with `γ = 1` and a standard normal `z`.
///
/// - Norm first: `y = x / rms(x) · SiLU(z)`.
/// - Gate first: `y = w / rms(w)`, with `w = x · SiLU(z)`.
#[test]
fn rms_norm_gated_is_the_formula_at_each_scale() {
    let device = test_device();
    let mut failures = Failures::default();
    for case in cases() {
        let eps = case.eps();
        for norm_before_gate in [true, false] {
            for width in WIDTHS {
                let mut norm = RmsNormGatedConfig::new(width)
                    .with_norm_before_gate(norm_before_gate)
                    .init(&device.clone().autodiff());
                norm.gamma = case.gamma(&device, width);
                for (name, x) in case.inputs(&device, width) {
                    let rows = x.dims()[0];
                    let (z, h) = (case.draw(&device, rows, width, 1.0), case.draw(&device, rows, width, 1.0));
                    let (xs, zs, hs) = (host(x.clone()), host(z.clone()), host(h.clone()));
                    let z = Tensor::from_inner(z);
                    let (ys, gs) = forward_and_gradient(x, h, |x| norm.forward(x, z));
                    let (mut want_y, mut want_g) = (Want::default(), Want::default());
                    for ((x, z), h) in xs.chunks(width).zip(zs.chunks(width)).zip(hs.chunks(width)) {
                        let s: Vec<f64> = z.iter().map(|&z| silu(z)).collect();
                        if norm_before_gate {
                            let r = rms_ref(x, eps);
                            let y: Vec<f64> = x.iter().zip(&s).map(|(a, s)| a / r * s).collect();
                            let size_y = max_abs(&y);
                            want_y.extend(y, size_y);
                            let hs: Vec<f64> = h.iter().zip(&s).map(|(h, s)| h * s).collect();
                            let size_g = gradient_size(x, eps, max_abs(&s) * max_abs(h) / r);
                            want_g.extend(norm_vjp(x, &hs, eps), size_g);
                        } else {
                            let w: Vec<f64> = x.iter().zip(&s).map(|(a, s)| a * s).collect();
                            let r = rms_ref(&w, eps);
                            let y: Vec<f64> = w.iter().map(|a| a / r).collect();
                            let size_y = max_abs(&y);
                            want_y.extend(y, size_y);
                            let g = norm_vjp(&w, h, eps).into_iter().zip(&s).map(|(g, s)| g * s);
                            want_g.extend(g, gradient_size(&w, eps, max_abs(&s) * max_abs(h) / r));
                        }
                    }
                    let name = format!("{name}, norm first {norm_before_gate}");
                    let floor_y = case.floor(!norm_before_gate);
                    failures.check(format!("{name}, the output"), &ys, &want_y, floor_y, &case);
                    failures.check(format!("{name}, the gradient"), &gs, &want_g, case.floor(false), &case);
                }
            }
        }
    }
    failures.assert_none();
}

/// [`normed_score`] at each scale of `x`: `c · Σ_feat(x · w) / rms(x)`, with
/// `c = 1/√width` and a standard normal query `w`, and the gradient of
/// `Σ score·h` with respect to `x`.
#[test]
fn normed_score_is_the_formula_at_each_scale() {
    let device = test_device();
    let mut failures = Failures::default();
    for case in cases() {
        let eps = case.eps();
        for width in WIDTHS {
            let c = score_scale(width);
            for (name, x) in case.inputs(&device, width) {
                let (w, h) = (case.draw(&device, 1, width, 1.0), case.draw(&device, x.dims()[0], 1, 1.0));
                let (xs, ws, hs) = (host(x.clone()), host(w.clone()), host(h.clone()));
                let w = Tensor::from_inner(w);
                let (ys, gs) = forward_and_gradient(x, h, |x| normed_score(x, w, c));
                let (mut want_y, mut want_g) = (Want::default(), Want::default());
                for (x, h) in xs.chunks(width).zip(&hs) {
                    let r = rms_ref(x, eps);
                    let terms: Vec<f64> = x.iter().zip(&ws).map(|(a, w)| c * a / r * w).collect();
                    want_y.extend([terms.iter().sum::<f64>()], terms.iter().map(|t| t.abs()).sum());
                    let g = norm_vjp(x, &ws, eps).into_iter().map(|g| c * h * g);
                    want_g.extend(g, gradient_size(x, eps, c * h.abs() * max_abs(&ws) / r));
                }
                failures.check(format!("{name}, the score"), &ys, &want_y, case.floor(true), &case);
                failures.check(format!("{name}, the gradient"), &gs, &want_g, case.floor(false), &case);
            }
        }
    }
    failures.assert_none();
}

/// [`rms_denom`] at each scale of `x`: `rms(x)`, and the gradient of
/// `Σ rms·h`. That gradient is `h · x / (n · rms)`, or 0 under the clamp.
#[test]
fn rms_denom_is_the_formula_at_each_scale() {
    let device = test_device();
    let mut failures = Failures::default();
    for case in cases() {
        let eps = case.eps();
        for width in WIDTHS {
            for (name, x) in case.inputs(&device, width) {
                let h = case.draw(&device, x.dims()[0], 1, 1.0);
                let (xs, hs) = (host(x.clone()), host(h.clone()));
                let (ys, gs) = forward_and_gradient(x, h, rms_denom);
                let (mut want_y, mut want_g) = (Want::default(), Want::default());
                for (x, h) in xs.chunks(width).zip(&hs) {
                    let (r, n) = (rms_ref(x, eps), width as f64);
                    want_y.extend([r], r);
                    let clamped = is_clamped(x, eps);
                    let g = x.iter().map(|a| if clamped { 0.0 } else { h * a / (n * r) });
                    want_g.extend(g, gradient_size(x, eps, h.abs() * max_abs(x) / (n * r)));
                }
                failures.check(format!("{name}, the rms"), &ys, &want_y, case.floor(false), &case);
                failures.check(format!("{name}, the gradient"), &gs, &want_g, case.floor(false), &case);
            }
        }
    }
    failures.assert_none();
}
