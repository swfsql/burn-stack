//! The serializable shapes of an H-Net: everything except the block configs.
//!
//! As for the other networks of this crate (see [`crate::modules::shape`]),
//! the model config of a consumer is a shape plus its block configs:
//! `{ shape: HNetShape, stage_blocks: Vec<CE>, main_block: CM }`. The block
//! config of stage `s` sets the width of that stage, and it builds both its
//! encoder and its decoder. Each of the three stacks of a stage has its own
//! [`NetworkShape`](crate::modules::NetworkShape).
//!
//! **The init.** Each stack takes the
//! [`InitPolicy`](crate::utils::InitPolicy) of its own shape. Its
//! residual rescale counts the residual branches of every stack from the
//! outermost encoder down to that stack, as the reference does. Stage `s`
//! counts the encoders and decoders of stages `0..=s`. The main network
//! counts those and its own. The router (identity), the residual of the
//! dechunking layer (zero) and the width vectors (zero) keep their own init.

use crate::modules::hnet::{HNet, HNetMain, HNetStage, HNetVocabNetwork, Router};
use crate::modules::network::padded_vocab;
use crate::modules::{BlockConfig, Layers, NetworkShape, RmsNormConfig};
use crate::utils::InitPolicy;
use crate::utils::class::init_class_emb;
use burn::module::Param;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;

/// The two stacks of one stage.
#[derive(Config, Debug)]
pub struct HNetStageShape {
    /// The encoder stack.
    pub encoder: NetworkShape,
    /// The decoder stack.
    pub decoder: NetworkShape,
}

/// The block-independent half of an [`HNet`] config.
#[derive(Config, Debug)]
pub struct HNetShape {
    /// The stages, outermost first. Empty ⇒ the main network alone.
    pub stages: Vec<HNetStageShape>,
    /// The main network.
    pub main: NetworkShape,
    /// See [`HNet::chunk_multiple`].
    #[config(default = 1)]
    pub chunk_multiple: usize,
    /// See [`HNet::smooth_block`].
    #[config(default = 64)]
    pub smooth_block: usize,
}

/// The residual branches of a stack (for the init rescale).
fn branches(shape: &NetworkShape) -> usize {
    shape.residuals_per_layer() * shape.n_applied_layers()
}

/// Build one stack of an H-Net, with its init policy over `depth` residual
/// branches.
fn init_stack<C: BlockConfig + Clone>(
    shape: &NetworkShape,
    block: &C,
    depth: usize,
    what: &str,
    device: &Device,
) -> Layers<C::Block> {
    assert!(
        shape.class_latents.is_empty(),
        "the {what} of an H-Net takes no class latents: they would move the rows that the chunks point at"
    );
    let layers = shape.layers(block.clone()).init(device);
    match &shape.init {
        Some(init) => shape.retie(init.clone().with_default_residual_depth(depth).apply(layers)),
        None => layers,
    }
}

impl HNetShape {
    /// The init depth of each stack: `(per stage, main)`. See the module
    /// header.
    pub fn residual_depths(&self) -> (Vec<usize>, usize) {
        let mut depth = 0;
        let stages = self
            .stages
            .iter()
            .map(|stage| {
                depth += branches(&stage.encoder) + branches(&stage.decoder);
                depth
            })
            .collect();
        (stages, depth + branches(&self.main))
    }

    /// Allocate and initialise the H-Net on `device`. `stage_blocks` holds one
    /// block config per stage (outermost first). The widths must not
    /// decrease inward.
    pub fn init<CE: BlockConfig + Clone, CM: BlockConfig + Clone>(
        &self,
        stage_blocks: &[CE],
        main_block: &CM,
        device: &Device,
    ) -> HNet<CE::Block, CM::Block> {
        assert_eq!(stage_blocks.len(), self.stages.len(), "one block config per stage");
        let (depths, main_depth) = self.residual_depths();
        let pad_dimension = |d_in: Option<usize>, d: usize| {
            let d_in = d_in?;
            assert!(d >= d_in, "the H-Net widths must not decrease inward: {d_in} → {d}");
            (d > d_in).then(|| Param::from_tensor(Tensor::zeros([d - d_in], device)))
        };
        let mut d_in = None;
        let stages = self
            .stages
            .iter()
            .zip(stage_blocks)
            .zip(depths)
            .enumerate()
            .map(|(s, ((shape, block), depth))| {
                let d = block.d_model();
                let mut residual_proj = LinearConfig::new(d, d).init(device);
                // `Param::map` reads the flag again from the tensor that it gets.
                residual_proj.weight =
                    residual_proj.weight.map(|w| w.zeros_like().set_require_grad(w.is_require_grad()));
                let stage = HNetStage {
                    pad_dimension: pad_dimension(d_in, d),
                    encoder: init_stack(&shape.encoder, block, depth, &format!("encoder {s}"), device),
                    encoder_norm: RmsNormConfig::new(d).init(device),
                    router: Router::init(d, device),
                    residual_proj,
                    decoder: init_stack(&shape.decoder, block, depth, &format!("decoder {s}"), device),
                    decoder_norm: RmsNormConfig::new(d).init(device),
                };
                d_in = Some(d);
                stage
            })
            .collect();
        let d = main_block.d_model();
        HNet {
            stages,
            main: HNetMain {
                pad_dimension: pad_dimension(d_in, d),
                layers: init_stack(&self.main, main_block, main_depth, "main network", device),
                norm: RmsNormConfig::new(d).init(device),
            },
            chunk_multiple: self.chunk_multiple,
            smooth_block: self.smooth_block,
        }
    }

    /// The [`MuonPlan`](crate::optim::MuonPlan) of the H-Net: the plan of each
    /// stack at its own block config, scoped to that stack
    /// ([`MuonPlan::within`](crate::optim::MuonPlan::within)). The router, the
    /// residual of the dechunking layer and the width vectors stay on the
    /// fallback optimizer.
    #[cfg(feature = "optim")]
    pub fn muon_plan<CE: BlockConfig, CM: BlockConfig>(
        &self,
        stage_blocks: &[CE],
        main_block: &CM,
    ) -> crate::optim::MuonPlan {
        assert_eq!(stage_blocks.len(), self.stages.len(), "one block config per stage");
        let mut plan = crate::optim::MuonPlan::empty();
        for (s, (shape, block)) in self.stages.iter().zip(stage_blocks).enumerate() {
            plan = plan
                .extend(shape.encoder.muon_plan(block).within(&format!("stages.{s}.encoder.")))
                .extend(shape.decoder.muon_plan(block).within(&format!("stages.{s}.decoder.")));
        }
        plan.extend(self.main.muon_plan(main_block).within("main.layers."))
    }
}

/// The block-independent half of an [`HNetVocabNetwork`] config.
#[derive(Config, Debug)]
pub struct HNetVocabShape {
    /// The backbone.
    pub hnet: HNetShape,
    /// The unpadded vocabulary size.
    pub vocab_size: usize,
    /// Round the vocabulary up to a multiple of this (1 disables it).
    #[config(default = 1)]
    pub pad_vocab_size_multiple: usize,
    /// Tie the LM head to the transposed embedding.
    #[config(default = false)]
    pub missing_lm_head: bool,
    /// The init of the LM head and the flag head (`None` ⇒ the defaults of
    /// Burn). The embedding keeps `N(0, 1)`, as in the reference.
    #[config(default = "None")]
    pub head_init: Option<InitPolicy>,
    /// The number of learnable start rows that open a sequence (see
    /// [`HNetVocabNetwork`]). `0` ⇒ none.
    #[config(default = 0)]
    pub n_start_tokens: usize,
}

impl HNetVocabShape {
    /// Allocate and initialise the network on `device` (see
    /// [`HNetShape::init`]).
    pub fn init<CE: BlockConfig + Clone, CM: BlockConfig + Clone>(
        &self,
        stage_blocks: &[CE],
        main_block: &CM,
        device: &Device,
    ) -> HNetVocabNetwork<CE::Block, CM::Block> {
        let hnet = self.hnet.init(stage_blocks, main_block, device);
        let d = hnet.width();
        let vocab = padded_vocab(self.vocab_size, self.pad_vocab_size_multiple);
        let head = |linear: burn::nn::Linear| match &self.head_init {
            Some(init) => init.apply(linear),
            None => linear,
        };
        HNetVocabNetwork {
            embedding: EmbeddingConfig::new(vocab, d).init(device),
            flag_embedding: Param::from_tensor(Tensor::zeros([d], device)),
            hnet,
            lm_head: (!self.missing_lm_head)
                .then(|| head(LinearConfig::new(d, vocab).with_bias(false).init(device))),
            flag_head: head(LinearConfig::new(d, 1).init(device)),
            start_emb: init_class_emb(self.n_start_tokens, d, device),
        }
    }

    /// The [`MuonPlan`](crate::optim::MuonPlan) of the network: that of the
    /// backbone. The embedding and the heads stay on the fallback optimizer.
    #[cfg(feature = "optim")]
    pub fn muon_plan<CE: BlockConfig, CM: BlockConfig>(
        &self,
        stage_blocks: &[CE],
        main_block: &CM,
    ) -> crate::optim::MuonPlan {
        self.hnet.muon_plan(stage_blocks, main_block)
    }
}
