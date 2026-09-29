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
//! Every test takes its device from [`test_device`], so `dev-f16` runs the
//! whole suite in fp16.
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

/// Element-wise max absolute difference between two same-shape tensors,
/// returned as `f32` (already pulled to host via `into_scalar()`).
pub fn max_abs_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
    (a - b).abs().max().into_scalar::<f32>()
}

/// Compare two `PathRun`-style structs field-by-field against a baseline,
/// asserting every named field is within `tol` of the baseline.
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
        let tol: f32 = $tol;
        let mut failures: Vec<String> = Vec::new();
        $(
            let d1 = $crate::utils::test_helpers::max_abs_diff(
                $r_min.$field.clone(), $r_alt1.$field.clone(),
            );
            let d2 = $crate::utils::test_helpers::max_abs_diff(
                $r_min.$field.clone(), $r_alt2.$field.clone(),
            );
            eprintln!(
                "grad {:>14} | base↔{} = {:>10.6} | base↔{} = {:>10.6}",
                $name, $alt1_label, d1, $alt2_label, d2,
            );
            if d1 >= tol {
                failures.push(format!(
                    "baseline vs {}: grad of {} max abs diff = {:.6} (tol {})",
                    $alt1_label, $name, d1, tol,
                ));
            }
            if d2 >= tol {
                failures.push(format!(
                    "baseline vs {}: grad of {} max abs diff = {:.6} (tol {})",
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
