//! Runtime dtype selection for the examples.
//!
//! With the Dispatch architecture, the [`Device`] chooses the backend at
//! runtime. So the examples use [`Device::default`]: it resolves to the
//! enabled `backend-*` feature (each one enables the matching
//! `burn/<backend>`), with the `BURN_DEVICE` env override and a built-in
//! priority list when several are compiled in. [`configure_dtype`] optionally
//! installs a non-default dtype (`dev-f16` uses it to switch the device to
//! fp16/i32). Otherwise the backend defaults do not change.
//!
//! Model and optimizer state is saved in the burnpack [`store`](burn::store)
//! format. The on-disk dtype is the dtype that the module holds (fp16 under
//! `dev-f16`, fp32 otherwise), so there is no separate recorder precision to
//! configure.

use burn::prelude::*;

/// The host-side scalar type matching the device's default float dtype.
///
/// Used to read tensor values back to the host (`try_to_vec`/`into_data`), so
/// that the element type matches the runtime dtype: fp16 under `dev-f16`,
/// fp32 otherwise.
#[cfg(feature = "dev-f16")]
pub type FloatElement = burn::tensor::f16;
/// The host-side scalar type matching the device's default float dtype.
#[cfg(not(feature = "dev-f16"))]
pub type FloatElement = f32;

/// The device on which the workers of a dataloader build batches: the host
/// (flex), so no worker ever touches the accelerator. The loop moves each
/// batch to the device of the model, on its own thread ([`batch_float`],
/// [`batch_int`]).
///
/// A worker that builds batches directly on a GPU (`set_device(gpu)`) issues
/// device work from its own thread, concurrently with the loop. On CUDA,
/// cubecl gives every thread its own stream over one memory pool. A worker
/// upload that lands during the capture of a step invalidates the recording,
/// and the [`CapturedStep`](crate::utils::CapturedStep) then silently steps
/// eagerly. Whether it lands is a race against the prefetch of the loader.
/// PyTorch avoids it in the same way: its workers produce only CPU tensors.
///
/// Without `backend-flex`, there is no host device to name, and this is the
/// inner device of `device` (then the race can happen).
pub fn loader_device(device: &Device) -> Device {
    #[cfg(feature = "backend-flex")]
    {
        let _ = device;
        Device::flex()
    }
    #[cfg(not(feature = "backend-flex"))]
    device.clone().inner()
}

/// A batch tensor built on [`loader_device`], moved to `device` (on the
/// calling thread) in its default float dtype.
pub fn batch_float<const D: usize>(t: Tensor<D>, device: &Device) -> Tensor<D> {
    t.to_device(device).cast(device.settings().float_dtype)
}

/// A batch tensor built on [`loader_device`], moved to `device` (on the
/// calling thread) in its default int dtype.
pub fn batch_int<const D: usize>(t: Tensor<D, Int>, device: &Device) -> Tensor<D, Int> {
    t.to_device(device).cast(device.settings().int_dtype)
}

/// When `dev-f16` is enabled, install fp16 (and i32) as the device defaults.
///
/// Call it before any tensor is created on `device`. No-op when the feature
/// is off: the dtype defaults of the backend apply.
pub fn configure_dtype(device: &mut Device) {
    #[cfg(feature = "dev-f16")]
    {
        use burn::tensor::{FloatDType, IntDType};
        device
            .configure((FloatDType::F16, IntDType::I32))
            .expect("Failed to install fp16/i32 device defaults");
    }
    #[cfg(not(feature = "dev-f16"))]
    {
        let _ = device;
    }
}
