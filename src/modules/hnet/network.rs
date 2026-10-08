//! A language model over an [`HNet`]: the token boundary of
//! [`VocabNetwork`](crate::modules::VocabNetwork) around an H-Net backbone.

use crate::modules::hnet::{HNet, HNetCaches, Routing, StepMode};
use crate::modules::network::{apply_vocab_heads, embed_pairs};
use crate::modules::{Block, CacheTensors};
use burn::module::Param;
use burn::nn::{Embedding, Linear};
use burn::prelude::*;

/// `Embedding (vocab → d₀) → HNet → LM head (d₀ → vocab)`, with the flag
/// channel of [`VocabNetwork`](crate::modules::VocabNetwork). A token is a
/// pair `(id, flag)`. The logits are `[…, padded_vocab + 1]`, and the last
/// column is the flag logit of the next token.
///
/// There is no norm before the head. The final RMSNorm of the outermost
/// decoder is that norm, as in the reference (with no stage, that of the main
/// network).
///
/// **Start rows.** A stack of an H-Net takes no class markers (see
/// [`HNet`]). The network can still open a sequence with learnable rows of its
/// own ([`Self::start_emb`]). They go in front of the H-Net, as its first
/// rows, so the H-Net routes and chunks them like any other row.
/// [`Self::forward_opened`] puts them in front of a new sequence, and
/// [`Self::prime`] steps them alone. Then the output of the last start row
/// predicts the first token, and a sampler needs no seed token.
#[derive(Module, Debug)]
pub struct HNetVocabNetwork<E: Module, M: Module> {
    /// Token embedding table, weight shape `[padded_vocab, d₀]`.
    pub embedding: Embedding,
    /// The flag input, `[d₀]`: added to the embedding of a token whose flag
    /// is 1. Zero at init.
    pub flag_embedding: Param<Tensor<1>>,
    /// The backbone.
    pub hnet: HNet<E, M>,
    /// Optional dedicated LM head. `None` ⇒ weight-tied (the transposed
    /// embedding).
    pub lm_head: Option<Linear>,
    /// The flag output: one logit per position (`d₀ → 1`).
    pub flag_head: Linear,
    /// The start rows, `[n_start, d₀]` (`None` ⇒ none).
    pub start_emb: Option<Param<Tensor<2>>>,
}

impl<E: Block, M: Block> HNetVocabNetwork<E, M>
where
    E::Options: Clone,
    M::Options: Clone,
{
    /// Full-sequence pass: token pairs `[batch, sequence, 2]` → logits
    /// `[batch, sequence, padded_vocab + 1]`, the caches, and one [`Routing`]
    /// per stage. See [`HNet::forward`].
    pub fn forward(
        &self,
        x: Tensor<3, Int>,
        caches: Option<HNetCaches<E, M>>,
        options: (E::Options, M::Options),
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, HNetCaches<E, M>, Vec<Routing>) {
        let x = embed_pairs(&self.embedding, &self.flag_embedding, x);
        let (y, caches, routing) = self.hnet.forward(x, caches, options, pad);
        (self.heads(y), caches, routing)
    }

    /// The number of start rows.
    pub fn n_start(&self) -> usize {
        self.start_emb.as_ref().map_or(0, |emb| emb.dims()[0])
    }

    /// [`Self::forward`] over a **new** sequence, with the start rows in front
    /// of it: logits `[batch, n_start + sequence, padded_vocab + 1]`. The logits
    /// of the last start row predict the first token. `pad` marks the padding
    /// of `x` (the start rows are always real).
    pub fn forward_opened(
        &self,
        x: Tensor<3, Int>,
        options: (E::Options, M::Options),
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, HNetCaches<E, M>, Vec<Routing>) {
        let x = embed_pairs(&self.embedding, &self.flag_embedding, x);
        let [batch, _sequence, d] = x.dims();
        let (x, pad) = match &self.start_emb {
            None => (x, pad),
            Some(emb) => {
                let n = self.n_start();
                let start = emb.val().unsqueeze_dim::<3>(0).expand([batch, n, d]);
                let pad = pad.map(|pad| {
                    let real = Tensor::<2, Int>::zeros([batch, n], &pad.device()).equal_elem(1);
                    Tensor::cat(vec![real, pad], 1)
                });
                (Tensor::cat(vec![start, x], 1), pad)
            }
        };
        let (y, caches, routing) = self.hnet.forward(x, None, options, pad);
        (self.heads(y), caches, routing)
    }

    /// Single-token step: token pairs `[batch, 2]` → logits `[batch,
    /// padded_vocab + 1]`, the caches, and one [`Routing`] per stage. See
    /// [`HNet::step`].
    pub fn step(
        &self,
        x: Tensor<2, Int>,
        caches: Option<HNetCaches<E, M>>,
        mode: StepMode,
    ) -> (Tensor<2>, HNetCaches<E, M>, Vec<Routing>)
    where
        E::Caches: CacheTensors,
        M::Caches: CacheTensors,
    {
        let x = embed_pairs(&self.embedding, &self.flag_embedding, x.unsqueeze_dim(1)).squeeze_dim(1);
        let (y, caches, routing) = self.hnet.step(x, caches, mode);
        (self.heads(y.unsqueeze_dim(1)).squeeze_dim(1), caches, routing)
    }

    /// Step the start rows alone, from a new sequence, for `batch` slots.
    /// Returns the logits of the last start row (which predict the first
    /// token) and the caches, or `None` when there are no start rows. `prime`
    /// then `step` runs what [`Self::forward_opened`] runs.
    pub fn prime(&self, batch: usize, mode: StepMode) -> Option<(Tensor<2>, HNetCaches<E, M>)>
    where
        E::Caches: CacheTensors,
        M::Caches: CacheTensors,
    {
        let emb = self.start_emb.as_ref()?.val();
        let [n, d] = emb.dims();
        let mut caches = None;
        let mut last = None;
        for i in 0..n {
            let row = emb.clone().narrow(0, i, 1).expand([batch, d]);
            let (y, c, _) = self.hnet.step(row, caches, mode);
            caches = Some(c);
            last = Some(y);
        }
        let logits = self.heads(last?.unsqueeze_dim(1)).squeeze_dim(1);
        Some((logits, caches?))
    }

    fn heads(&self, x: Tensor<3>) -> Tensor<3> {
        apply_vocab_heads(&self.embedding, self.lm_head.as_ref(), &self.flag_head, x)
    }
}
