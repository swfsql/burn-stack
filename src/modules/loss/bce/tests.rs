//! The loss on probabilities, with the `ε` of the dtype that it computes in.

use super::*;
use crate::utils::test_helpers::test_device;
use burn::tensor::{DType, FloatDType};

/// On probabilities, the loss of a positive target with `p = 10⁻⁴` is
/// `−ln(p + ε)`, with the `ε` of the dtype that the loss computes in: f32 for
/// an f16 or bf16 input, and f64 for an f64 input. The loss has the dtype of
/// the input, and it is within two steps of that dtype. The `ε` of f16
/// (`7.1·10⁻⁴`) would give `7.1`, not `9.2`.
#[test]
fn a_small_probability_keeps_its_loss() {
    let device = test_device();
    let loss = BinaryCrossEntropyLossConfig::new().init();
    for (dtype, want_dtype, eps_dtype) in [
        (FloatDType::F16, DType::F16, DType::F32),
        (FloatDType::BF16, DType::BF16, DType::F32),
        (FloatDType::F64, DType::F64, DType::F64),
    ] {
        let eps = f64::from(div_eps(eps_dtype));
        let p = Tensor::<1>::from_floats([1e-4], &device).cast(dtype);
        let targets = Tensor::<1>::from_floats([1.0], &device).cast(dtype);
        let p0 = p.to_data().try_into_vec_as::<f64>().unwrap()[0];
        let got = loss.forward(p, targets);
        assert_eq!(got.dtype(), want_dtype, "{dtype:?}: the dtype of the loss");
        let got = got.into_data().try_into_vec_as::<f64>().unwrap()[0];
        let want = -(p0 + eps).ln();
        let step = dtype.finfo().epsilon;
        assert!((got - want).abs() <= 2.0 * step * want, "{dtype:?}: loss {got} vs {want}");
    }
}
