//! # Composition modules
//!
//! The block-generic composition types ([`Layer`], [`Layers`],
//! [`LatentNetwork`]/[`VocabNetwork`], [`BidiLayers`], [`Residuals`]) plus the
//! shared neural pieces they are built from (activations, norms, losses, small
//! tensor helpers).
//!
//! Everything here is parameterised by the mixer block `M: `[`Block`] — the one
//! trait a family of sequence-mixing blocks implements to drop into this stack.
//!
//! Their *serializable* counterparts are the [`shape`] types
//! ([`NetworkShape`]/[`LatentShape`]/[`VocabShape`]/[`BidiShape`]): the same
//! knobs with the block generic split off, so a consumer's model config is the
//! pair `{ shape, block }` and no architecture is restated per family.

use burn::config::Config;
use burn::prelude::*;

/// Custom activations (fp16-stable `silu` / `softplus` / `log_sigmoid`).
pub mod activation;
/// Bidirectional layer stacks (straight + reversed passes, merged per pair).
pub mod bidi;
/// The per-network cache collection trait ([`CacheStack`]).
pub mod cache;
/// A single Pre-LN residual layer wrapping one mixer block ([`Layer`]).
pub mod layer;
/// The (virtual-)layer stack over real weight sets ([`Layers`]).
pub mod layers;
/// Loss functions (binary cross-entropy, cross-entropy, mean squared error).
pub mod loss;
/// Tensor helpers: `segsum`, `gqa`, typed `split`, and `sanity` guards.
pub mod misc;
/// The SwiGLU feed-forward block interleaved with the mixer ([`GatedMlp`]).
pub mod mlp;
pub mod model_config;
/// Multi-Gate Residuals: multi-stream gated depth-wise residuals ([`Residuals`]).
pub mod multi_gate;
/// Block-generic networks ([`LatentNetwork`] / [`VocabNetwork`]).
pub mod network;
/// RMS norms ([`RmsNorm`] and [`RmsNormGated`]), fp16-safe.
pub mod norm;
pub mod shape;

pub use activation::log_sigmoid::log_sigmoid;
pub use activation::silu::Silu;
pub use activation::softplus::softplus;
pub use misc::gqa::gqa_expand_to_heads;
pub use misc::sanity::sanity;
pub use misc::segsum::segsum;
pub use misc::split::split_into;
pub use mlp::{GatedMlp, GatedMlpConfig};
pub use model_config::ModelConfigExt;
pub use norm::rms_norm::{RmsNorm, RmsNormConfig};
pub use norm::rms_norm_gated::{RmsNormGated, RmsNormGatedConfig};
pub use norm::rms_score::{normed_score, rms_denom, score_scale};

pub use bidi::{BidiLayerPair, BidiLayers, BidiLayersBuilder, OutputMerge, OutputMergeConfig};
pub use cache::CacheStack;
pub use layer::Layer;
pub use layers::{Layers, LayersBuilder};
pub use multi_gate::{
    MultiGate, MultiGateResidual, MultiGateResidualConfig, Residuals, ResidualsConfig,
};
pub use shape::{BidiShape, LatentShape, NetworkShape, VocabShape};
pub use network::{
    LatentNetwork, LatentNetworkBuilder, VocabNetwork, VocabNetworkBuilder,
};

/// The mixer-block interface the generic [`Layer`]/[`Layers`] delegate to.
///
/// Implement it once per block family (a selective SSM, an attention variant, a
/// gated convolution, …) and every container in this crate — layers, virtual
/// stacks, bidirectional pairs, latent/vocab networks, class tokens, the Muon
/// plan — applies unchanged.
///
/// `ModuleDisplay` and `AutodiffModule` are supertraits so that the generic
/// containers are themselves `Module`/`AutodiffModule` (Burn's derive requires
/// both of every module-typed generic), which is what lets
/// [`Layers::grad_horizon`](crate::modules::Layers::grad_horizon) move the stack
/// to the inner backend for its no-grad prefix. A `#[derive(Module)]` block
/// satisfies them.
pub trait Block: Module + burn::module::ModuleDisplay + burn::module::AutodiffModule {
    /// Per-block streaming cache (one layer's worth of state).
    type Cache;
    /// The per-network cache collection for this family.
    type Caches: CacheStack<Cache = Self::Cache>;
    /// Per-call algorithm/chunking options threaded down to
    /// [`Self::block_forward`]. `()` for a block with nothing to select.
    type Options;

    /// Full-sequence (chunked) pass — training / prefill.
    fn block_forward(
        &self,
        x: Tensor<3>,
        cache: Option<Self::Cache>,
        options: Self::Options,
    ) -> (Tensor<3>, Self::Cache);

    /// Single-token recurrent step — decoding.
    fn block_step(&self, x: Tensor<2>, cache: Option<Self::Cache>) -> (Tensor<2>, Self::Cache);

    /// Build `n_virtual` zero caches sized for a `[batch, sequence, d_model]` input.
    fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> Self::Caches;
    /// Build `n_virtual` zero caches sized for a `[batch, d_model]` input.
    fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> Self::Caches;
}

/// A block *config* that knows its `d_model` and how to build its [`Block`].
/// Lets the generic builders construct `Layers<M>` without knowing the family.
pub trait BlockConfig: Config {
    /// The block this config builds.
    type Block: Block;
    /// Model width, used to size each layer's pre-norm.
    fn d_model(&self) -> usize;
    /// Allocate and initialise the block on `device`.
    fn init_block(&self, device: &Device) -> Self::Block;

    /// The block's 2-D weights Muon may own, and where their fused columns
    /// split. See [`crate::optim`] for what is (and is not) listed.
    #[cfg(feature = "optim")]
    fn muon_projections(&self) -> Vec<crate::optim::ProjSpec>;
}
