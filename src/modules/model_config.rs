//! [`ModelConfigExt`]: the config → module seam a generic training loop needs.
//!
//! A whole-model config (this crate's [`LatentNetworkConfig`](super::network) or
//! a family's runtime-selectable wrapper around it) knows two things a
//! model-agnostic driver cannot look up for itself: how to allocate the module
//! on a device, and which of its weights Muon may own. `ModelConfigExt` names
//! exactly those two, so artifact loading and optimizer construction can be
//! written once against `C: ModelConfigExt`.
//!
//! Consumers implement it on their own network configs; the trait is
//! deliberately tiny, since both methods normally forward to an inherent one.

use burn::prelude::*;

#[cfg(feature = "optim")]
use crate::optim::MuonPlan;

/// A model config that can build its module on a device.
pub trait ModelConfigExt: Config {
    /// The module type this config builds.
    type Model: Module;

    /// Allocate and initialise the model on `device`.
    fn init(&self, device: &Device) -> Self::Model;

    /// Which of the model's weights Muon may own, and where the fused
    /// projections split (see [`crate::optim`]). Irrelevant when the training
    /// config leaves Muon unset.
    #[cfg(feature = "optim")]
    fn muon_plan(&self) -> MuonPlan;
}
