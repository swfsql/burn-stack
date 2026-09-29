//! Shared test fixtures for "several implementations of one kernel must agree"
//! suites.
//!
//! A block family often has more than one implementation of the same math (a
//! readable one, a fast one, a recompute-backward one). They must agree on
//! both the forward outputs and the input gradients. The tests follow the same
//! pattern: pick a baseline, run two alternatives, and compare per field. The
//! per-input gradient struct differs per family, so only the small generic
//! primitives are shared here.
//!
//! Every test takes its device from
//! [`test_device`](crate::utils::test_helpers::test_device), so `dev-f16` runs
//! the whole suite in fp16. A check takes its tolerance from
//! [`dtype_tol`](crate::utils::test_helpers::dtype_tol). A
//! host read-back converts to its element type (`try_into_vec_as`), because
//! `try_to_vec` fails on a different dtype.
//!
//! The `test-helpers` feature enables it (the tests of this crate also have
//! it), so a downstream crate can reuse the macro from its dev-dependencies.

use burn::prelude::*;

/// The device of the tests: [`Device::default`], with fp16 (and i32) as its
/// dtype defaults under `dev-f16`. Without the feature, the dtype defaults of
/// the backend apply.
///
/// The dtype defaults of a device are global to the process, and the first
/// tensor on the device fixes them. So a test that takes its device from
/// [`Device::default`] can fix fp32 for all tests. Under `dev-f16`, this
/// function panics if that occurred.
pub fn test_device() -> Device {
    #[allow(unused_mut)]
    let mut device = Device::default();
    #[cfg(feature = "dev-f16")]
    {
        use burn::tensor::{FloatDType, IntDType};
        // Only the first call can install the defaults. Each later call gets
        // `AlreadyInitialized`, and the assert below checks the result.
        let _ = device.configure((FloatDType::F16, IntDType::I32));
        assert_eq!(
            device.settings().float_dtype,
            FloatDType::F16,
            "the test device is not fp16: a tensor was on it before `test_device`",
        );
    }
    device
}

/// The tolerance on the float dtype of [`test_device`] for a check whose
/// tolerance in fp32 is `f32_tol`. Under fp32, it is `f32_tol`.
///
/// A tolerance of `2^-k` in fp32 keeps `k` of the 23 fraction bits. On a
/// dtype with `f` fraction bits, the tolerance keeps the same share of them,
/// `2^-(k·f/23)`. So in fp16, `1e-5` gives `≈ 6.7e-3`, `1e-4` gives
/// `≈ 1.8e-2` and `1e-3` gives `≈ 5.0e-2`. The result is never smaller than
/// `f32_tol`, and `0` stays exact.
pub fn dtype_tol(f32_tol: f32) -> f32 {
    use burn::tensor::FloatDType;
    let dtype = test_device().settings().float_dtype;
    if dtype == FloatDType::F32 {
        return f32_tol;
    }
    let share = dtype.finfo().epsilon.ln() / f64::from(f32::EPSILON).ln();
    let tol = f64::from(f32_tol);
    tol.powf(share).max(tol) as f32
}

/// Element-wise max absolute difference between two same-shape tensors,
/// returned as `f32` (already pulled to host via `into_scalar()`).
pub fn max_abs_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
    (a - b).abs().max().into_scalar::<f32>()
}

/// [`max_abs_diff`] divided by the largest magnitude in `b` (the reference),
/// or by `1` if that magnitude is smaller.
///
/// A float format rounds relative to the value, so the rounding of a check on
/// large values grows with them. A check on values that can be much larger
/// than `1` (states, gradients, logits) uses this function in every dtype.
/// Near zero, the difference stays absolute.
pub fn max_rel_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
    let scale = b.clone().abs().max().into_scalar::<f32>().max(1.0);
    max_abs_diff(a, b) / scale
}

/// Compare two `PathRun`-style structs field-by-field against a baseline,
/// asserting every named field is within `tol` of the baseline, relative to
/// the magnitude of the baseline ([`max_rel_diff`]). `tol` is the fp32
/// tolerance, and [`dtype_tol`] adapts it.
///
/// The field set differs per family: pass exactly the fields that your
/// `PathRun` has.
///
/// # Example
/// ```ignore
/// check_grads_match_two_paths!(
///     baseline: r_base,
///     alt1: ("Fast", r_fast),
///     alt2: ("Recompute", r_rec),
///     tol: 1e-3,
///     fields: [d_x => "x", d_w => "w", /* ... */],
/// );
/// ```
#[macro_export]
macro_rules! check_grads_match_two_paths {
    (
        baseline: $r_min:ident,
        alt1: ($alt1_label:literal, $r_alt1:ident),
        alt2: ($alt2_label:literal, $r_alt2:ident),
        tol: $tol:expr,
        fields: [ $($field:ident => $name:literal),* $(,)? ] $(,)?
    ) => {{
        let tol: f32 = $crate::utils::test_helpers::dtype_tol($tol);
        let mut failures: Vec<String> = Vec::new();
        $(
            let d1 = $crate::utils::test_helpers::max_rel_diff(
                $r_alt1.$field.clone(), $r_min.$field.clone(),
            );
            let d2 = $crate::utils::test_helpers::max_rel_diff(
                $r_alt2.$field.clone(), $r_min.$field.clone(),
            );
            eprintln!(
                "grad {:>14} | base↔{} = {:>10.6} | base↔{} = {:>10.6}",
                $name, $alt1_label, d1, $alt2_label, d2,
            );
            if d1 >= tol {
                failures.push(format!(
                    "baseline vs {}: grad of {} max rel diff = {:.6} (tol {})",
                    $alt1_label, $name, d1, tol,
                ));
            }
            if d2 >= tol {
                failures.push(format!(
                    "baseline vs {}: grad of {} max rel diff = {:.6} (tol {})",
                    $alt2_label, $name, d2, tol,
                ));
            }
        )*
        assert!(
            failures.is_empty(),
            "gradient mismatches:\n  {}",
            failures.join("\n  "),
        );
    }};
}
