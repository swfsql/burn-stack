//! A macro that removes the per-backend boilerplate of a `*BackendExt` trait.
//!
//! A block family that ships a custom (memory-efficient) backward declares a
//! backend-extension trait. Its default body is correct for each Burn backend
//! except `Autodiff`, which gets the custom backward. The macro below emits
//! one impl with the default body for each other backend that this crate
//! supports.

/// Emit `impl $trait_name for <backend> {}` blocks for every burn backend that
/// this crate supports, with the default body of the trait.
///
/// The matching `backend-*` / `cubecl` / `fusion` feature gates each impl.
/// The `cfg`s are evaluated in the crate that expands the macro, so that crate
/// must declare the same features.
#[macro_export]
macro_rules! impl_backend_ext_for_burn_backends {
    ($trait_name:path) => {
        #[cfg(feature = "backend-ndarray")]
        impl $trait_name for burn::backend::NdArray {}

        #[cfg(feature = "backend-flex")]
        impl $trait_name for burn::backend::Flex {}

        #[cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu"))]
        impl $trait_name for burn::backend::libtorch::LibTorch {}

        // Every cubecl runtime is this one type. The device says which one.
        #[cfg(feature = "cubecl")]
        impl $trait_name for burn_cubecl::CubeBackend {}

        // Fusion delegates to the default impl of the inner backend.
        #[cfg(feature = "fusion")]
        impl<B: burn_fusion::FusionBackend + $trait_name> $trait_name for burn_fusion::Fusion<B> {}
    };
}
