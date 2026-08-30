//! # burn-stack — block-generic layer/network composition on Burn
//!
//! Everything that sits *around* a sequence-mixing block: the Pre-LN residual
//! [`Layer`](modules::Layer), the (virtual-)layer [`Layers`](modules::Layers)
//! stack, bidirectional pairs, latent/vocabulary networks, multi-stream gated
//! residuals, class tokens, virtual-layer scheduling, and the Muon parameter
//! groups over fused projections.
//!
//! The crate knows nothing about any particular mixer. A family plugs in by
//! implementing two traits:
//!
//! - [`Block`](modules::Block) — `block_forward` (chunked, for training and
//!   prefill), `block_step` (recurrent, for decoding), and the zero-cache
//!   constructors, plus the associated `Cache` / `Caches` / `Options` types.
//! - [`BlockConfig`](modules::BlockConfig) — `d_model`, `init_block`, and the
//!   block's Muon-eligible projections.
//!
//! Its cache collection additionally implements
//! [`CacheStack`](modules::CacheStack), which is all the generic loops need:
//! one slot per (virtual) layer, plus the backend hop
//! [`Layers::grad_horizon`](modules::Layers::grad_horizon) performs.
//!
//! ## Composition hierarchy
//!
//! ```text
//! VocabNetwork<M>   embedding → Layers<M> → final RMSNorm → LM head → logits
//! LatentNetwork<M>  in_proj → Layers<M> → [norm_f] → out_proj (continuous I/O)
//! Layers<M>         a stack of N (virtual) layers over R real weight sets
//! Layer<M>          Pre-LN sub-blocks: Block(RMSNorm(x)), then the optional
//!                   SwiGLU MLP over norm2; Layers owns the outer residual
//! M: Block          the mixer core, supplied by the caller
//! ```
//!
//! ## Three execution modes
//!
//! Every layer and network exposes `forward()` (parallel/chunked: training and
//! prefill), `step()` (recurrent: token-by-token decode, O(state)/token), and
//! `prime()` (`step()` without a user token — it emits the class
//! tokens/latents waiting for the next one). `forward()` from any cache equals
//! `step()` unrolled from that same cache; a block family is expected to hold
//! up its half of that contract.
//!
//! ## Where the pieces live
//!
//! - [`modules`] — the composition types and the shared NN modules
//!   (activations, norms, losses, small tensor helpers).
//! - [`utils`] — virtual-layer/LR scheduling, class tokens, and the
//!   custom-backward plumbing a family needs to register its own kernels.
//! - [`optim`] — Muon parameter groups over fused projection weights.
//! - `examples` (feature `examples-common`) — the CLI/training/dataset
//!   scaffolding the consumer crates' `examples/` directories share.

#![warn(missing_docs)]
#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

#[cfg(feature = "examples-common")]
pub mod examples;
pub mod modules;
#[cfg(feature = "optim")]
pub mod optim;
#[cfg(any(test, feature = "test-helpers"))]
pub mod reference;
pub mod utils;

/// Convenience re-exports: `use burn_stack::prelude::*;` brings the composition
/// types and the two block traits into scope.
pub mod prelude {
    pub use crate::modules::{
        BidiLayerPair, BidiLayers, Block, BlockConfig, CacheStack, GatedMlp, GatedMlpConfig,
        LatentNetwork, Layer, Layers, LayersBuilder, MultiGateResidualConfig, OutputMerge,
        OutputMergeConfig, Residuals, ResidualsConfig, RmsNorm, RmsNormConfig, VocabNetwork,
    };
    pub use crate::utils::{
        BidiSchedule, ClassLatent, ClassToken, GradHorizon, InitPolicy, Schedule,
    };

    #[cfg(feature = "optim")]
    pub use crate::optim::{MuonPlan, ProjSegment, ProjSpec, muon_config};
}

/// When `true`, [`modules::sanity`] panics if it observes a `NaN`.
///
/// Compiled-in guard (off by default) for debugging numerical issues; leaving
/// it `false` removes the check entirely.
#[cfg(feature = "check-nan")]
pub const DENY_NAN: bool = true;
/// When `true`, [`modules::sanity`] panics if it observes a `NaN`.
#[cfg(not(feature = "check-nan"))]
pub const DENY_NAN: bool = false;

/// When `true`, [`modules::sanity`] panics if it observes an `Inf`.
///
/// Compiled-in guard (off by default), companion to [`DENY_NAN`].
#[cfg(feature = "check-inf")]
pub const DENY_INF: bool = true;
/// When `true`, [`modules::sanity`] panics if it observes an `Inf`.
#[cfg(not(feature = "check-inf"))]
pub const DENY_INF: bool = false;
