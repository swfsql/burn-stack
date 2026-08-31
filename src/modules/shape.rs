//! Serializable **network shapes**: everything about a stack *except* the block
//! it is made of.
//!
//! The builders next door ([`LayersBuilder`], [`LatentNetworkBuilder`],
//! [`VocabNetworkBuilder`], [`BidiLayersBuilder`]) carry the block config `C`,
//! so they cannot be `#[derive(Config)]` without dragging that generic through
//! serde. Splitting the block off leaves a plain, block-*independent* struct
//! that can: [`NetworkShape`] is the whole stack — depth, virtual scheduling,
//! residuals, the feed-forward, the init policy — and [`LatentShape`] /
//! [`VocabShape`] / [`BidiShape`] add only their own I/O boundary.
//!
//! A family's serializable model config is then the pair `{ shape, block }`,
//! with the family chosen in `block` and *nothing* about the surrounding
//! architecture restated per family. Every knob an example's `model_config()`
//! can turn is declared here, once, so one file describes what any stack in any
//! consumer crate does:
//!
//! ```text
//!   NetworkShape   depth, virtual layers, grad horizon, residuals, class
//!                  latents, the SwiGLU MLP, the global init policy
//!   LatentShape    + input_size / output_size / final_norm / class tokens
//!   VocabShape     + vocab_size / vocab padding / tied LM head
//!   BidiShape      the bidirectional counterpart (pairs, per-pair merges)
//! ```
//!
//! **The reference architecture.** The delta-rule and Mamba language models are
//! Llama's macro design with a recurrent mixer in place of self-attention, so a
//! faithful stack sets [`NetworkShape::mlp`]
//! (`GatedMlpConfig::from_hidden_ratio(d_model, 4)`) and
//! [`NetworkShape::init`] ([`InitPolicy`], the reference `initializer_range`);
//! a mixer-only stack with per-module Burn defaults is the ablation, not the
//! default. The knobs with no reference counterpart —
//! [`NetworkShape::n_virtual_layers`], [`NetworkShape::grad_horizon`],
//! [`Residuals::MultiGate`](crate::modules::Residuals) — buy depth or memory at
//! a parameter budget the reference never has to work under; each is off unless
//! asked for.

use crate::modules::{
    BidiLayers, BidiLayersBuilder, BlockConfig, GatedMlpConfig, LatentNetwork,
    LatentNetworkBuilder, Layers, LayersBuilder, OutputMergeConfig, ResidualsConfig, VocabNetwork,
    VocabNetworkBuilder,
};
use crate::utils::{BidiSchedule, ClassLatent, ClassToken, GradHorizon, InitPolicy, Schedule};
use burn::prelude::*;

// ===========================================================================
// NetworkShape
// ===========================================================================

/// The block-independent half of a network config: everything about the
/// *stack* rather than the block.
#[derive(Config, Debug)]
pub struct NetworkShape {
    /// Number of real weight sets.
    pub n_real_layers: usize,

    /// Optional virtual-layer scheduling: run `n` logical layers over the real
    /// weight sets, mapped by a [`Schedule`]. Depth at no parameter cost, and
    /// not something the reference architectures do.
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, Schedule)>,

    /// Which virtual layers back-propagate; everything else runs on the inner
    /// backend (truncated BPTT for deep recursion). `None` ⇒ track the whole
    /// stack. See [`Layers::grad_horizon`].
    #[config(default = "None")]
    pub grad_horizon: Option<GradHorizon>,

    /// Stack-level class latents, spliced into the sequence before the first
    /// layer (width `d_model`).
    #[config(default = "Vec::new()")]
    pub class_latents: Vec<ClassLatent>,

    /// Suppress the first virtual layer's residual.
    #[config(default = false)]
    pub ignore_first_residual: bool,

    /// Suppress the last virtual layer's residual (the output is then the last
    /// layer's transform alone).
    #[config(default = false)]
    pub ignore_last_residual: bool,

    /// Inter-layer residual scheme (plain additive vs Multi-Gate). The
    /// reference architectures are `Standard`.
    #[config(default = "ResidualsConfig::Standard")]
    pub residuals: ResidualsConfig,

    /// Optional per-layer SwiGLU feed-forward sub-block, with its own pre-norm
    /// and inner residual. `None` ⇒ mixer-only layers.
    ///
    /// Every reference language model in this family has one: the architecture
    /// is Llama's macro design with a recurrent mixer in place of
    /// self-attention, so a token mixer is followed by a SwiGLU MLP of
    /// [`GatedMlpConfig::from_hidden_ratio`] width.
    #[config(default = "None")]
    pub mlp: Option<GatedMlpConfig>,

    /// Optional post-build re-initialisation of the whole network (the
    /// reference `initializer_range` + residual rescale). `None` ⇒ keep Burn's
    /// per-module defaults. See [`InitPolicy`].
    #[config(default = "None")]
    pub init: Option<InitPolicy>,
}

impl NetworkShape {
    /// The number of residual sub-blocks per layer this stack has, which is
    /// what an [`InitPolicy`] rescale is counted over: the mixer, plus the
    /// feed-forward when there is one.
    pub fn residuals_per_layer(&self) -> usize {
        if self.mlp.is_some() { 2 } else { 1 }
    }

    /// The stack's depth in *applied* layers — the virtual count when there is
    /// one, else the real count.
    pub fn n_applied_layers(&self) -> usize {
        self.n_virtual_layers
            .as_ref()
            .map(|(l, _)| *l)
            .unwrap_or(self.n_real_layers)
    }

    /// The [`InitPolicy`] to apply after building, with its rescale resolved
    /// against this stack's depth.
    pub fn init_policy(&self) -> Option<InitPolicy> {
        self.init
            .clone()
            .map(|init| {
                init.with_default_residual_depth(
                    self.residuals_per_layer() * self.n_applied_layers(),
                )
            })
    }

    /// Apply [`Self::init_policy`] to a built module (a no-op when unset).
    pub fn apply_init<M: Module>(&self, module: M) -> M {
        match self.init_policy() {
            Some(init) => init.apply(module),
            None => module,
        }
    }

    /// The layer-stack builder for a given block config.
    pub fn layers<C: BlockConfig>(&self, block: C) -> LayersBuilder<C> {
        LayersBuilder::new(self.n_real_layers, block)
            .with_n_virtual_layers(self.n_virtual_layers.clone())
            .with_grad_horizon(self.grad_horizon.clone())
            .with_residuals(self.residuals.clone())
            .with_ignore_first_residual(self.ignore_first_residual)
            .with_ignore_last_residual(self.ignore_last_residual)
            .with_class_latents(self.class_latents.clone())
            .with_mlp(self.mlp.clone())
    }

    /// Allocate the bare layer stack on `device`, with the init policy applied.
    pub fn init<C: BlockConfig>(&self, block: C, device: &Device) -> Layers<C::Block> {
        self.apply_init(self.layers(block).init(device))
    }

    /// The [`MuonPlan`](crate::optim::MuonPlan) for a stack of this shape at
    /// `block`: the block's fused projections plus the optional MLP's.
    ///
    /// A network's own boundary weights — `in_proj`/`out_proj`, the embedding
    /// and LM head, class-marker tables — are deliberately left out; see
    /// [`crate::optim`].
    #[cfg(feature = "optim")]
    pub fn muon_plan<C: BlockConfig>(&self, block: &C) -> crate::optim::MuonPlan {
        crate::optim::MuonPlan::new(block.muon_projections()).with_mlp(self.mlp.as_ref())
    }
}

// ===========================================================================
// LatentShape
// ===========================================================================

/// A [`LatentNetwork`]'s own knobs, on top of [`NetworkShape`].
#[derive(Config, Debug)]
pub struct LatentShape {
    /// Input feature width, fed to `in_proj`.
    pub input_size: usize,
    /// Output feature width, produced by `out_proj`.
    pub output_size: usize,
    /// The stack's knobs.
    pub stack: NetworkShape,
    /// Insert a final RMSNorm before `out_proj` — the counterpart of the
    /// unconditional `norm_f` a [`VocabNetwork`] puts before its LM head.
    #[config(default = false)]
    pub final_norm: bool,
    /// Network-level class tokens, spliced into the input before `in_proj`
    /// (width `input_size`, unlike the stack's class latents).
    #[config(default = "Vec::new()")]
    pub class_tokens: Vec<ClassToken>,
}

impl LatentShape {
    /// The builder for this shape around a given block config.
    pub fn build<C: BlockConfig>(&self, block: C) -> LatentNetworkBuilder<C> {
        LatentNetworkBuilder {
            input_size: self.input_size,
            layers: self.stack.layers(block),
            output_size: self.output_size,
            final_norm: self.final_norm,
            class_tokens: self.class_tokens.clone(),
        }
    }

    /// Allocate the network on `device`, with the stack's init policy applied.
    pub fn init<C: BlockConfig>(&self, block: C, device: &Device) -> LatentNetwork<C::Block> {
        self.stack.apply_init(self.build(block).init(device))
    }
}

// ===========================================================================
// VocabShape
// ===========================================================================

/// A [`VocabNetwork`]'s own knobs, on top of [`NetworkShape`].
#[derive(Config, Debug)]
pub struct VocabShape {
    /// Unpadded vocabulary size.
    pub vocab_size: usize,
    /// The stack's knobs.
    pub stack: NetworkShape,
    /// Round `vocab_size` up to a multiple of this (1 disables rounding).
    #[config(default = 1)]
    pub pad_vocab_size_multiple: usize,
    /// Tie the LM head to the (transposed) embedding weights.
    #[config(default = true)]
    pub missing_lm_head: bool,
}

impl VocabShape {
    /// The builder for this shape around a given block config.
    pub fn build<C: BlockConfig>(&self, block: C) -> VocabNetworkBuilder<C> {
        VocabNetworkBuilder {
            vocab_size: self.vocab_size,
            pad_vocab_size_multiple: self.pad_vocab_size_multiple,
            layers: self.stack.layers(block),
            missing_lm_head: self.missing_lm_head,
        }
    }

    /// Allocate the model on `device`, with the stack's init policy applied.
    pub fn init<C: BlockConfig>(&self, block: C, device: &Device) -> VocabNetwork<C::Block> {
        self.stack.apply_init(self.build(block).init(device))
    }
}

// ===========================================================================
// BidiShape
// ===========================================================================

/// A [`BidiLayers`] stack's block-independent knobs.
///
/// Its own shape rather than a [`NetworkShape`]: the pairs carry per-pair merge
/// configs and a [`BidiSchedule`], and there is no feed-forward or init policy
/// on this path.
#[derive(Config, Debug)]
pub struct BidiShape {
    /// Number of real (weight-bearing) layers. Must be even — they pair up.
    pub n_real_layers: usize,
    /// One merge config per pair; length `n_real_layers / 2`.
    pub outputs_merge: Vec<OutputMergeConfig>,
    /// Optional virtual-layer scheduling over the real pairs.
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, BidiSchedule)>,
    /// Zero the first virtual pair's residual.
    #[config(default = false)]
    pub ignore_first_residual: bool,
    /// Zero the last virtual pair's residual.
    #[config(default = false)]
    pub ignore_last_residual: bool,
    /// Stack-level class latents, spliced once before the first pair.
    #[config(default = "Vec::new()")]
    pub class_latents: Vec<ClassLatent>,
    /// Inter-pair residual scheme.
    #[config(default = "ResidualsConfig::Standard")]
    pub residuals: ResidualsConfig,
}

impl BidiShape {
    /// The builder for this shape around a given block config.
    pub fn build<C: BlockConfig>(&self, block: C) -> BidiLayersBuilder<C> {
        BidiLayersBuilder {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            block,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
            outputs_merge: self.outputs_merge.clone(),
            class_latents: self.class_latents.clone(),
            residuals: self.residuals.clone(),
        }
    }

    /// Allocate the bidirectional stack on `device`.
    pub fn init<C: BlockConfig>(&self, block: C, device: &Device) -> BidiLayers<C::Block> {
        self.build(block).init(device)
    }

    /// The [`MuonPlan`](crate::optim::MuonPlan) for a stack of this shape at
    /// `block` (no feed-forward on this path).
    #[cfg(feature = "optim")]
    pub fn muon_plan<C: BlockConfig>(&self, block: &C) -> crate::optim::MuonPlan {
        crate::optim::MuonPlan::new(block.muon_projections())
    }
}
