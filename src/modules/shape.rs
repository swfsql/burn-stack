//! Serializable **network shapes**: everything about a stack *except* the
//! block that it is made of.
//!
//! The builders next door ([`LayersBuilder`], [`LatentNetworkBuilder`],
//! [`VocabNetworkBuilder`], [`BidiLayersBuilder`]) carry the block config `C`.
//! So they cannot be `#[derive(Config)]` unless serde also handles that
//! generic. A split of the block leaves plain, block-*independent* structs
//! that can:
//!
//! - [`NetworkShape`] is the whole stack: depth, virtual scheduling,
//!   residuals, the feed-forward, the init policy.
//! - [`LatentShape`] / [`VocabShape`] add only their own I/O boundary to it.
//! - [`BidiShape`] is the bidirectional counterpart.
//!
//! The serializable model config of a family is then the pair
//! `{ shape, block }`. `block` chooses the family, and no family restates
//! *anything* about the surrounding architecture. Every knob that the
//! `model_config()` of an example can turn is declared here, once. So one file
//! describes what any stack in any consumer crate does:
//!
//! ```text
//!   NetworkShape   depth, virtual layers, grad horizon, residuals, class
//!                  latents, the SwiGLU MLP, untied norms, the global init policy
//!   LatentShape    + input_size / output_size / final_norm / class tokens
//!   VocabShape     + vocab_size / vocab padding / tied LM head
//!   BidiShape      the bidirectional counterpart (pairs, per-pair merges)
//! ```
//!
//! **The reference architecture.** The reference language models of the
//! recurrent families follow the macro design of Llama, with a recurrent
//! mixer in place of self-attention. So a faithful stack sets
//! [`NetworkShape::mlp`] (`GatedMlpConfig::from_hidden_ratio(d_model, 4)`) and
//! [`NetworkShape::init`] ([`InitPolicy`](crate::utils::InitPolicy), the reference
//! `initializer_range`). A mixer-only stack with the per-module defaults of
//! Burn is the ablation, not the default. The knobs with no reference
//! counterpart ([`NetworkShape::n_virtual_layers`],
//! [`NetworkShape::grad_horizon`],
//! [`Residuals::MultiGate`](crate::modules::Residuals)) buy depth or memory at
//! a parameter budget that the reference never has to work under. Each one is
//! off unless you ask for it.

use crate::modules::{
    BidiLayers, BidiLayersBuilder, Block, BlockConfig, GatedMlpConfig, LatentNetwork,
    LatentNetworkBuilder, LayerUntied, Layers, LayersBuilder, OutputMergeConfig, ResidualsConfig,
    VocabNetwork,
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
    /// weight sets, mapped by a [`Schedule`]. Depth at no parameter cost. The
    /// reference architectures do not do this.
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, Schedule)>,

    /// Which virtual layers back-propagate. The others run on the inner
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

    /// Suppress the last virtual layer's residual (the output is then the
    /// transform of the last layer alone).
    #[config(default = false)]
    pub ignore_last_residual: bool,

    /// Inter-layer residual scheme (plain additive vs Multi-Gate). The
    /// reference architectures are `Standard`.
    #[config(default = "ResidualsConfig::Standard")]
    pub residuals: ResidualsConfig,

    /// Optional per-layer SwiGLU feed-forward sub-block, with its own pre-norm
    /// and inner residual. `None` ⇒ mixer-only layers.
    ///
    /// Every reference language model of the recurrent families has one. The
    /// architecture is the macro design of Llama, with a recurrent mixer in
    /// place of self-attention. So a SwiGLU MLP of
    /// [`GatedMlpConfig::from_hidden_ratio`] width follows each token mixer.
    #[config(default = "None")]
    pub mlp: Option<GatedMlpConfig>,

    /// The parameters of the layer itself that are held once per application
    /// of its real layer, not tied across them. The block config names those
    /// of the block. Only virtual layers give a real layer more than one
    /// application. See [`crate::utils::untied`].
    #[config(default = "Vec::new()")]
    pub untied: Vec<LayerUntied>,

    /// Optional post-build re-initialisation of the whole network (the
    /// reference `initializer_range` + residual rescale). `None` ⇒ keep the
    /// per-module defaults of Burn. See [`InitPolicy`].
    #[config(default = "None")]
    pub init: Option<InitPolicy>,
}

impl NetworkShape {
    /// The number of residual sub-blocks per layer of this stack: the mixer,
    /// plus the feed-forward when there is one. An [`InitPolicy`] rescale
    /// counts over this number.
    pub fn residuals_per_layer(&self) -> usize {
        if self.mlp.is_some() { 2 } else { 1 }
    }

    /// The depth of the stack in *applied* layers: the virtual count when
    /// there is one, else the real count.
    pub fn n_applied_layers(&self) -> usize {
        self.n_virtual_layers
            .as_ref()
            .map(|(l, _)| *l)
            .unwrap_or(self.n_real_layers)
    }

    /// The [`InitPolicy`] to apply after the build, with its rescale resolved
    /// against the depth of this stack.
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
            .with_untied(self.untied.clone())
    }

    /// Allocate the bare layer stack on `device`, with the init policy applied.
    pub fn init<C: BlockConfig>(&self, block: C, device: &Device) -> Layers<C::Block> {
        self.retie(self.apply_init(self.layers(block).init(device)))
    }

    /// Tie the untied copies of a stack back together after
    /// [`Self::apply_init`]. Its redraw of every 2-D `weight` would start each
    /// application from a draw of its own (see [`Layers::retie`]). A no-op
    /// without an init policy.
    pub fn retie<M: Block>(&self, layers: Layers<M>) -> Layers<M> {
        match self.init {
            Some(_) => layers.retie(),
            None => layers,
        }
    }

    /// The [`MuonPlan`](crate::optim::MuonPlan) for a stack of this shape at
    /// `block`: the fused projections of the block, plus those of the optional
    /// MLP.
    ///
    /// The boundary weights of a network (`in_proj`/`out_proj`, the embedding
    /// and LM head, class-marker tables) stay out on purpose (see
    /// [`crate::optim`]).
    #[cfg(feature = "optim")]
    pub fn muon_plan<C: BlockConfig>(&self, block: &C) -> crate::optim::MuonPlan {
        crate::optim::MuonPlan::new(block.muon_projections()).with_mlp(self.mlp.as_ref())
    }
}

// ===========================================================================
// LatentShape
// ===========================================================================

/// The knobs that only a [`LatentNetwork`] has, on top of [`NetworkShape`].
#[derive(Config, Debug)]
pub struct LatentShape {
    /// Input feature width, fed to `in_proj`.
    pub input_size: usize,
    /// Output feature width, produced by `out_proj`.
    pub output_size: usize,
    /// The knobs of the stack.
    pub stack: NetworkShape,
    /// Insert a final RMSNorm before `out_proj`: the counterpart of the
    /// unconditional `norm_f` that a [`VocabNetwork`] puts before its LM head.
    #[config(default = false)]
    pub final_norm: bool,
    /// Network-level class tokens, spliced into the input before `in_proj`
    /// (width `input_size`, unlike the class latents of the stack).
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

    /// Allocate the network on `device`, with the init policy of the stack
    /// applied.
    pub fn init<C: BlockConfig>(&self, block: C, device: &Device) -> LatentNetwork<C::Block> {
        let mut net = self.stack.apply_init(self.build(block).init(device));
        net.layers = self.stack.retie(net.layers);
        net
    }
}

// ===========================================================================
// VocabShape
// ===========================================================================

/// The knobs that only a [`VocabNetwork`] has, on top of [`NetworkShape`].
#[derive(Config, Debug)]
pub struct VocabShape {
    /// Unpadded vocabulary size.
    pub vocab_size: usize,
    /// The knobs of the stack.
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

    /// Allocate the model on `device`, with the init policy of the stack
    /// applied.
    pub fn init<C: BlockConfig>(&self, block: C, device: &Device) -> VocabNetwork<C::Block> {
        let mut net = self.stack.apply_init(self.build(block).init(device));
        net.layers = self.stack.retie(net.layers);
        net
    }
}

// ===========================================================================
// BidiShape
// ===========================================================================

/// The block-independent knobs of a [`BidiLayers`] stack.
///
/// It is its own shape, not a [`NetworkShape`]: the pairs carry per-pair
/// merge configs and a [`BidiSchedule`], and this path has no feed-forward or
/// init policy.
#[derive(Config, Debug)]
pub struct BidiShape {
    /// Number of real (weight-bearing) layers. Must be even: they pair up.
    pub n_real_layers: usize,
    /// One merge config per real pair, length `n_real_layers / 2`.
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
    /// The parameters of the layers that are held once per application, not
    /// tied (see [`NetworkShape::untied`]).
    #[config(default = "Vec::new()")]
    pub untied: Vec<LayerUntied>,
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
            untied: self.untied.clone(),
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
