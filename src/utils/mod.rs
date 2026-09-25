//! # Shared utilities
//!
//! The lower-level plumbing under the composition modules:
//!
//! - virtual-layer scheduling (`schedule`) and untied parameters (`untied`),
//! - class tokens (`class`), right-padded batches (`padding`) and packed rows
//!   (`packing`),
//! - the custom-backward helpers that a block family needs to register its own
//!   memory-efficient kernels (`backend_macros` / `combined_grad` / `fprim`),
//! - no-grad regions (`detach`) and graph capture of a recurrent step
//!   (`graph`),
//! - the whole-model init policy (`init`) and the LR `scheduler`s,
//! - the per-dtype numerical constants below.

use burn::prelude::ToElement;
use burn::tensor::DType;

/// Macros emitting per-backend `BackendExt` impls + autodiff marker traits.
#[macro_use]
pub mod backend_macros;
/// Learnable `[CLS]`-style class tokens/latents spliced into the sequence.
pub mod class;
/// Flatten/unflatten `(y, final_state)` into one tracked tensor for the custom
/// backward.
#[cfg(feature = "autodiff")]
pub mod combined_grad;
/// Detach a module's parameters, so a no-grad prefix builds no autodiff graph.
pub mod detach;
/// Rank-tagged `FloatTensor` primitive wrapper mirroring the `Tensor` method
/// API, used by custom-backward gradient math.
pub mod fprim;
/// Graph capture/replay of a recurrent step, its caches written back in place.
pub mod graph;
/// A whole-model init policy, applied to a module after its build.
pub mod init;
/// Packed rows: several sequences one after another in each slot, each from a
/// reset, with reserved opening slots for the class latents.
pub mod packing;
/// Right-padded batches: which rows are padding, and each row's place in its
/// own slot's sequence once class markers are spliced.
pub mod padding;
/// Virtual-layer → real-weight index scheduling shared by all block families.
pub mod schedule;
/// Learning-rate schedulers (cosine-annealing + warmup, constant).
pub mod scheduler;
/// `max_abs_diff` + gradient-comparison macros used across the test suites.
#[cfg(any(test, feature = "test-helpers"))]
pub mod test_helpers;
/// Parameters held once per application of a real layer instead of tied.
pub mod untied;

pub use class::{ClassCursor, ClassCursors, ClassLatent, ClassToken};
pub use detach::detach_params;
pub use graph::{CapturedStep, StepInput};
#[cfg(feature = "autodiff")]
pub use graph::Weights;
pub use init::InitPolicy;
pub use packing::Packed;
pub use padding::Padding;
pub use schedule::{Applications, BidiSchedule, GradHorizon, Schedule};
pub use untied::UntiedParam;
pub use scheduler::{ConstantLr, CosineAnnealingLr, Lr};

/// A small `dtype`-specific epsilon for safe division (`x / (y + eps)`),
/// returned as `f32`.
///
/// The value is chosen per float format as the geometric mean (average in
/// log10 space) of two reference magnitudes: a scaled function of the minimum
/// exponent of the format, and its machine epsilon. For each of
/// f64/f32/f16/bf16, this places `eps` well above the denormal/underflow
/// floor, and keeps it negligible relative to typical activations. The
/// comments in the code note the resulting constants. `dtype` is the runtime
/// float dtype of the tensor to divide (e.g. `x.dtype()`). Panics on non-float
/// dtypes.
pub fn div_eps(dtype: DType) -> f32 {
    match dtype {
        // 4.0693917e-16
        DType::F64 => {
            let raw_exp = -(-f64::MIN_EXP as f32 * 2.3f32).powf(0.35f32);
            let eps_exp = (f64::EPSILON as f32).log10();
            let avg = (raw_exp + eps_exp) / 2f32;
            10f32.powf(avg)
        }
        // 8.1584695e-8
        DType::F32 | DType::Flex32 => {
            let raw_exp = -(-f32::MIN_EXP as f32 * 2.3f32).powf(0.35f32);
            let eps_exp = f32::EPSILON.log10();
            let avg = (raw_exp + eps_exp) / 2f32;
            10f32.powf(avg)
        }
        // 7.1209995e-4
        DType::F16 => {
            let raw_exp = -(-burn::tensor::f16::MIN_EXP.to_f32() * 2.3f32).powf(0.35f32);
            let eps_exp = burn::tensor::f16::EPSILON.to_f32().log10();
            let avg = (raw_exp + eps_exp) / 2f32;
            10f32.powf(avg)
        }
        // 2.0885676e-5
        DType::BF16 => {
            let raw_exp = -(-burn::tensor::bf16::MIN_EXP.to_f32() * 2.3f32).powf(0.35f32);
            let eps_exp = burn::tensor::bf16::EPSILON.to_f32().log10();
            let avg = (raw_exp + eps_exp) / 2f32;
            10f32.powf(avg)
        }
        DType::I64
        | DType::I32
        | DType::I16
        | DType::I8
        | DType::U64
        | DType::U32
        | DType::U16
        | DType::U8
        | DType::Bool(_) => {
            unreachable!()
        }
        DType::QFloat(_) => {
            unimplemented!()
        }
    }
}
