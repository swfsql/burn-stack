//! # Composition modules
//!
//! The block-generic composition types ([`Layer`], [`Layers`],
//! [`LatentNetwork`]/[`VocabNetwork`], [`BidiLayers`], [`Residuals`]) plus the
//! shared neural pieces they are built from (activations, norms, losses, small
//! tensor helpers).
//!
//! Everything here is parameterised by the mixer block `M: `[`Block`]. A family
//! of sequence-mixing blocks implements this one trait to join this stack.
//!
//! The [`shape`] types are their *serializable* counterparts
//! ([`NetworkShape`]/[`LatentShape`]/[`VocabShape`]/[`BidiShape`]): the same
//! knobs, with the block generic split off. So the model config of a consumer
//! is the pair `{ shape, block }`, and no family restates the architecture.

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
/// Loss functions (binary cross-entropy, cross-entropy, mean squared error,
/// the L2-warp penalty).
pub mod loss;
/// Tensor helpers: `segsum`, `gqa`, typed `split`, and `sanity` guards.
pub mod misc;
/// The SwiGLU feed-forward block interleaved with the mixer ([`GatedMlp`]).
pub mod mlp;
/// The config → module interface of a generic training loop
/// ([`ModelConfigExt`]).
pub mod model_config;
/// Multi-Gate Residuals: multi-stream gated depth-wise residuals ([`Residuals`]).
pub mod multi_gate;
/// Block-generic networks ([`LatentNetwork`] / [`VocabNetwork`]).
pub mod network;
/// RMS norms ([`RmsNorm`] and [`RmsNormGated`]), fp16-safe.
pub mod norm;
/// Serializable network shapes: a stack without its block ([`NetworkShape`]).
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
pub use cache::{CacheStack, CacheTensors, TensorZip};
pub use layer::{Layer, LayerUntied};
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
/// Implement it once per block family (a selective SSM, an attention variant,
/// a gated convolution, …). Then every container in this crate applies
/// unchanged: layers, virtual stacks, bidirectional pairs, latent/vocab
/// networks, class tokens, the Muon plan.
///
/// `ModuleDisplay` is a supertrait, so that the generic containers are
/// themselves `Module`s (the derive of Burn requires it of every module-typed
/// generic). The `valid` of `Module` lets
/// [`Layers::grad_horizon`](crate::modules::Layers::grad_horizon) move the
/// stack to the inner backend for its untracked segments. A
/// `#[derive(Module)]` block satisfies both.
pub trait Block: Module + burn::module::ModuleDisplay {
    /// Per-block streaming cache (one layer's worth of state).
    type Cache;
    /// The per-network cache collection for this family.
    type Caches: CacheStack<Cache = Self::Cache>;
    /// Per-call algorithm/chunking options threaded down to
    /// [`Self::block_forward`]. `()` for a block with nothing to select.
    type Options;

    /// Full-sequence (chunked) pass — training / prefill.
    ///
    /// `pad` (`[batch, sequence]`, `true` at padding, `None` ⇒ every row real)
    /// is **right** padding: in every slot, it follows all of the real rows. A
    /// padded row is absent. The outputs of the real rows and the returned
    /// cache are exactly those of the real rows of the slot run alone, which
    /// is `step` unrolled over them. The output of a padded row is
    /// unspecified. The containers keep the mask right-padded, whatever class
    /// markers they splice (see [`crate::utils::padding`]).
    fn block_forward(
        &self,
        x: Tensor<3>,
        cache: Option<Self::Cache>,
        options: Self::Options,
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, Self::Cache);

    /// Single-token recurrent step — decoding.
    fn block_step(&self, x: Tensor<2>, cache: Option<Self::Cache>) -> (Tensor<2>, Self::Cache);

    /// Build `n_virtual` zero caches sized for a `[batch, sequence, d_model]` input.
    fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> Self::Caches;
    /// Build `n_virtual` zero caches sized for a `[batch, d_model]` input.
    fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> Self::Caches;

    /// The parameters that this block stores **once per application** of its
    /// real layer (not once), each with the axis of its copies. Empty when its
    /// config unties nothing. The [`BlockConfig`] decides which they are and
    /// tiles them. The containers run the block only through
    /// [`Layer::application`], which narrows each one to the copy of the
    /// running application (and also sizes the zero caches from that view).
    /// See [`crate::utils::untied`].
    fn untied_params(&self) -> Vec<crate::utils::UntiedParam>;
}

/// A block *config* that knows its `d_model` and how to build its [`Block`].
/// Lets the generic builders construct `Layers<M>` without knowing the family.
pub trait BlockConfig: Config {
    /// The block this config builds.
    type Block: Block;
    /// Model width, used to size the pre-norm of each layer.
    fn d_model(&self) -> usize;
    /// Allocate and initialise the block on `device`, for a real layer applied
    /// `n_applications` times. Every parameter that the config unties is
    /// [tiled](crate::utils::untied::tile) that many times. Every other
    /// parameter is built once. A config that unties nothing ignores the count.
    fn init_block(&self, n_applications: usize, device: &Device) -> Self::Block;

    /// The 2-D weights of the block that Muon can own, and where their fused
    /// columns split. See [`crate::optim`] for what is (and is not) listed.
    #[cfg(feature = "optim")]
    fn muon_projections(&self) -> Vec<crate::optim::ProjSpec>;
}
