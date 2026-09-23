//! [`ModelConfigExt`]: the config → module interface that a generic training
//! loop needs.
//!
//! A whole-model config (for example the runtime-selectable network config of
//! a family) knows two things that a model-agnostic driver cannot find by
//! itself:
//!
//! - how to allocate the module on a device,
//! - which of its weights Muon can own.
//!
//! `ModelConfigExt` names exactly those two. So the artifact load and the
//! optimizer construction can be written once, against `C: ModelConfigExt`.
//!
//! Consumers implement it on their own network configs. The trait is tiny on
//! purpose, because both methods normally forward to an inherent method.

use burn::prelude::*;

#[cfg(feature = "optim")]
use crate::optim::MuonPlan;

/// A model config that can build its module on a device.
pub trait ModelConfigExt: Config {
    /// The module type this config builds.
    type Model: Module;

    /// Allocate and initialise the model on `device`.
    fn init(&self, device: &Device) -> Self::Model;

    /// Which weights of the model Muon can own, and where the fused
    /// projections split (see [`crate::optim`]). Not used when the training
    /// config leaves Muon unset.
    #[cfg(feature = "optim")]
    fn muon_plan(&self) -> MuonPlan;
}
