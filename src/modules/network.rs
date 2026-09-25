use crate::modules::LayersBuilder;
use crate::modules::{RmsNorm, RmsNormConfig};
use crate::prelude::*;
use crate::utils::class::{
    assert_full_len_known, class_chunk_plan, class_emb_width, class_marker_output_indices,
    class_prime_plan, class_row, init_class_emb, insert_class_markers_padded,
};
use crate::utils::{ClassCursor, ClassCursors, Packed, Padding};
use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

// ===========================================================================
// LatentNetwork<M>
// ===========================================================================

/// A feature/regression network on latents:
/// `in_proj (input_size → d_model) → Layers<M> → [norm_f] → out_proj (d_model →
/// output_size)`.
#[derive(Module, Debug)]
pub struct LatentNetwork<M: Module> {
    /// Linear projection `input_size → d_model`.
    pub in_proj: Linear,
    /// The shared layer stack.
    pub layers: Layers<M>,
    /// Optional final RMSNorm before [`Self::out_proj`]: the counterpart of
    /// [`VocabNetwork::norm_f`], which is unconditional there.
    ///
    /// It makes the input of the head scale-free. This matters whenever the
    /// output magnitude of the stack is not `O(1)`. A plain additive residual
    /// grows it with depth. [`Residuals::MultiGate`](crate::modules::Residuals)
    /// is a convex mixture (mean-pooled over `n` streams) that *shrinks* it.
    /// Without the norm, the two schemes give `out_proj` signals of very
    /// different scale.
    pub norm_f: Option<RmsNorm>,
    /// Linear projection `d_model → output_size`.
    pub out_proj: Linear,
    /// Positions of the network's class tokens, spliced into the input sequence
    /// (at `input_size` width) **before** `in_proj`. Empty ⇒ none.
    #[module(skip)]
    pub class_tokens: Vec<ClassToken>,
    /// The class-token embeddings, `[num_class_tokens, input_size]`.
    pub class_tokens_emb: Option<Param<Tensor<2>>>,
}

impl<M: Block> LatentNetwork<M>
where
    M::Options: Clone,
{
    /// Output positions of the class tokens for an `orig_len` input.
    ///
    /// A marker that never lands (a `Custom` at or past the end) reports a
    /// position past the emitted sequence. Compare it against the sequence
    /// length.
    pub fn class_token_output_indices(&self, orig_len: usize) -> Vec<usize> {
        class_marker_output_indices(&self.class_tokens, orig_len)
    }

    /// Splice the class tokens of this network into the chunk `x` (no-op when
    /// there are none), and advance the network-level cursor.
    fn insert_tokens(
        &self,
        x: Tensor<3>,
        padding: Option<Padding>,
        class: &mut ClassCursors,
    ) -> (Tensor<3>, Option<Padding>) {
        let mut cursor = ClassCursor::at(class.network, class.full_len);
        let out = insert_class_markers_padded(
            x,
            padding,
            &self.class_tokens,
            self.class_tokens_emb.as_ref(),
            &mut cursor,
            "LatentNetwork",
        );
        class.network = cursor.offset;
        out
    }

    /// `in_proj → layers → out_proj` over a full sequence
    /// (`[batch, sequence, input_size]` → `[batch, sequence (+ class tokens),
    /// output_size]`).
    ///
    /// `class` places the class tokens of this network *and* the class latents
    /// of the inner stack. `None` takes `x` as the whole sequence. The same
    /// [`ClassCursors`] given to consecutive chunks places every marker exactly
    /// where a single call over the concatenated sequence would place it.
    ///
    /// `pad` marks a right-padded batch (`None` ⇒ none), as in
    /// [`Layers::forward`]. The class tokens are also placed against the length
    /// of each slot.
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<M::Caches>,
        options: M::Options,
        class: Option<&mut ClassCursors>,
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, M::Caches) {
        // No cursors ⇒ this one call covers the whole sequence.
        let mut whole = ClassCursors::new(x.dims()[1]);
        let class = class.unwrap_or(&mut whole);
        let (x, padding) = self.insert_tokens(x, pad.map(Padding::new), class);
        let x = self.in_proj.forward(x);
        // The sequence of the stack is this one, made longer by the class
        // tokens.
        let saved = class.enter(&self.class_tokens);
        let (x, caches) = self
            .layers
            .forward_padded(x, caches, options, Some(&mut *class), padding);
        class.leave(saved);
        let x = self.head(x);
        (x, caches)
    }

    /// Single-token step (`[batch, input_size]` → `[batch, output_size]`).
    ///
    /// `class` drives all three class levels at once: the
    /// [`Self::class_tokens`] of this network (`class.network`), plus the
    /// cursors of the inner [`Layers::step`] (`class.stack`, `class.per_layer`).
    ///
    /// As in `forward`, the class tokens of the network are part of the
    /// sequence that enters the layers. So each one runs through a full network
    /// pass. The pass carries the inner cursors, so the layers splice their own
    /// latents around it exactly as in `forward`. The call returns the output
    /// of the **last** token that the step emitted. This is the user token,
    /// unless an `End` marker (at either level) follows it. That marker is then
    /// the true last token of the sequence. `None` injects nothing anywhere.
    /// `Middle`/`End` markers then panic, as they do without a
    /// [`ClassCursors::full_len`] hint.
    pub fn step(
        &self,
        x: Tensor<2>,
        caches: Option<M::Caches>,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, M::Caches) {
        let Some(class) = class else {
            assert_full_len_known(&self.class_tokens, None, "LatentNetwork");
            return self.step_one(x, caches, None);
        };
        let mut cursor = ClassCursor::at(class.network, class.full_len);
        let plan = class_chunk_plan(&self.class_tokens, 1, &mut cursor, "LatentNetwork");
        class.network = cursor.offset;
        if plan.is_empty() {
            return self.step_one(x, caches, Some(&mut *class));
        }
        // `at == 0` ⇒ the class token precedes the user token. `at == 1` ⇒ it is
        // an `End` that closes the sequence, and follows the token.
        let [batch, input_size] = x.dims();
        let row = |i: usize| class_row(self.class_tokens_emb.as_ref(), i, batch, input_size);
        let (before, after): (Vec<_>, Vec<_>) = plan.into_iter().partition(|&(at, _)| at == 0);
        let mut caches = caches;
        for (_, i) in before {
            let (_discard, c) = self.step_one(row(i), caches, Some(&mut *class));
            caches = Some(c);
        }
        let (mut out, mut caches) = self.step_one(x, caches, Some(&mut *class));
        for (_, i) in after {
            // A closing `End` token *is* the last token of the sequence. This
            // step produced its output, not that of the user token.
            let (o, c) = self.step_one(row(i), Some(caches), Some(&mut *class));
            out = o;
            caches = c;
        }
        (out, caches)
    }

    /// Step the class tokens/latents that this network has waiting for its next
    /// user token, with **no** user token. So the call needs no input data.
    ///
    /// This is the opening half of [`Self::step`] on its own, at all three
    /// class levels. Each class token of the network that is due now runs a
    /// full pass (so the layers splice their latents around it exactly as in
    /// `step`). After them, the call flushes what the stack still has waiting
    /// for its next token ([`Layers::prime`]). So a `prime` followed by a
    /// `step` runs the same sequence that the `step` alone would run. `prime`
    /// → sample → `step` → sample → … is the seedless-generation loop. `End`
    /// markers are never primed. They close the sequence, so they belong to
    /// the step that carries its last user token.
    ///
    /// Returns the output of the **last** marker emitted, or `None` when none
    /// were waiting (the caches then come back untouched, `None` included).
    /// `batch` sizes the marker rows, which are the only inputs.
    pub fn prime(
        &self,
        batch: usize,
        caches: Option<M::Caches>,
        class: Option<&mut ClassCursors>,
    ) -> (Option<Tensor<2>>, Option<M::Caches>) {
        let Some(class) = class else {
            // No cursors ⇒ nothing is injected, exactly as in a `None` step.
            assert_full_len_known(&self.class_tokens, None, "LatentNetwork");
            return self.layers.prime(batch, caches, None);
        };
        let mut cursor = ClassCursor::at(class.network, class.full_len);
        let plan = class_prime_plan(&self.class_tokens, 0, &mut cursor, "LatentNetwork");
        class.network = cursor.offset;

        let mut caches = caches;
        let mut out = None;
        if !plan.is_empty() {
            let width = class_emb_width(self.class_tokens_emb.as_ref());
            for (_at, i) in plan {
                let row = class_row(self.class_tokens_emb.as_ref(), i, batch, width);
                let (y, c) = self.step_one(row, caches, Some(&mut *class));
                out = Some(y);
                caches = Some(c);
            }
        }
        // The levels of the stack can still hold latents that wait for the
        // next token to reach them. The class tokens above just went past.
        let saved = class.enter(&self.class_tokens);
        let (y, caches) = self.layers.prime(batch, caches, Some(&mut *class));
        class.leave(saved);
        if let Some(y) = y {
            out = Some(self.head(y));
        }
        (out, caches)
    }

    /// One token through `in_proj → layers → out_proj`. [`Self::step`] places
    /// the class tokens of the network. The inner cursors are forwarded.
    fn step_one(
        &self,
        x: Tensor<2>,
        caches: Option<M::Caches>,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, M::Caches) {
        let x = self.in_proj.forward(x);
        let (x, caches) = match class {
            // The sequence of the stack is this one, made longer by the class
            // tokens.
            Some(class) => {
                let saved = class.enter(&self.class_tokens);
                let out = self.layers.step(x, caches, Some(&mut *class));
                class.leave(saved);
                out
            }
            None => self.layers.step(x, caches, None),
        };
        (self.head(x), caches)
    }

    /// The output head: the optional [`Self::norm_f`], then [`Self::out_proj`].
    /// Rank-generic, so every call path (sequence and single-token) shares it.
    fn head<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let x = match &self.norm_f {
            Some(norm) => norm.forward(x),
            None => x,
        };
        self.out_proj.forward(x)
    }
}

/// Plain factory for [`LatentNetwork`].
pub struct LatentNetworkBuilder<C> {
    /// Width of the input features fed to `in_proj`.
    pub input_size: usize,
    /// Builder for the layer stack.
    pub layers: LayersBuilder<C>,
    /// Width of the output features produced by `out_proj`.
    pub output_size: usize,
    /// Insert a final RMSNorm before `out_proj` (see [`LatentNetwork::norm_f`]).
    pub final_norm: bool,
    /// Network-level class tokens (spliced into the input before `in_proj`).
    pub class_tokens: Vec<ClassToken>,
}

impl<C: BlockConfig> LatentNetworkBuilder<C> {
    /// Allocate and initialise the network on `device`.
    pub fn init(&self, device: &Device) -> LatentNetwork<C::Block> {
        let d_model = self.layers.block.d_model();
        LatentNetwork {
            in_proj: LinearConfig::new(self.input_size, d_model)
                .with_bias(true)
                .init(device),
            layers: self.layers.init(device),
            norm_f: self
                .final_norm
                .then(|| RmsNormConfig::new(d_model).init(device)),
            out_proj: LinearConfig::new(d_model, self.output_size)
                .with_bias(true)
                .init(device),
            class_tokens_emb: init_class_emb(self.class_tokens.len(), self.input_size, device),
            class_tokens: self.class_tokens.clone(),
        }
    }
}

// ===========================================================================
// VocabNetwork<M>
// ===========================================================================

/// A complete autoregressive language model over a token vocabulary:
/// `Embedding (vocab → d_model) → Layers<M> → norm_f → LM head (d_model →
/// vocab)`.
///
/// This is the token-LM counterpart of [`LatentNetwork`]. Both are built on
/// the shared [`Layers`] core. They differ in three things:
///
/// - the I/O boundary: a token `Embedding` and a vocab logit head, not two
///   latent `Linear`s,
/// - the final pre-head [`RmsNorm`]: always present here, optional there,
/// - the class tokens: only [`LatentNetwork`] has network-level ones.
///
/// The LM head is **tied** (`lm_head = None`, the transposed embedding weight
/// is reused) or **untied** (a dedicated `Linear`). The vocabulary is rounded
/// up to a multiple for GPU alignment (see [`VocabNetworkBuilder`]).
#[derive(Module, Debug)]
pub struct VocabNetwork<M: Module> {
    /// Token embedding table, weight shape `[padded_vocab, d_model]`.
    pub embedding: Embedding,
    /// The shared layer stack.
    pub layers: Layers<M>,
    /// Final RMSNorm applied before the LM head (`norm_f`).
    pub norm_f: RmsNorm,
    /// Optional dedicated LM head. `None` ⇒ weight-tied (reuse embedding`ᵀ`).
    pub lm_head: Option<Linear>,
}

impl<M: Block> VocabNetwork<M>
where
    M::Options: Clone,
{
    /// Full-sequence pass: token IDs `[batch, sequence]` → logits
    /// `[batch, sequence, padded_vocab]`. `class` places the class latents of
    /// the inner stack (`None` ⇒ `x` is the whole sequence), and `pad` marks a
    /// right-padded batch (see [`Layers::forward`]).
    pub fn forward(
        &self,
        x: Tensor<2, Int>,
        caches: Option<M::Caches>,
        options: M::Options,
        class: Option<&mut ClassCursors>,
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, M::Caches) {
        let x = self.embedding.forward(x);
        let (x, caches) = self.layers.forward(x, caches, options, class, pad);
        let x = self.norm_f.forward(x);
        (self.apply_lm_head(x), caches)
    }

    /// [`Self::forward`] over packed rows: token IDs `[batch, rows]` → logits
    /// `[batch, rows, padded_vocab]`, with every sequence of a slot restarted
    /// at its reset (see [`Layers::forward_packed`]). The token of an opening
    /// slot is not read: the stack puts its class latent there.
    pub fn forward_packed(
        &self,
        x: Tensor<2, Int>,
        caches: Option<M::Caches>,
        options: M::Options,
        packed: &Packed,
    ) -> (Tensor<3>, M::Caches) {
        let x = self.embedding.forward(x);
        let (x, caches) = self.layers.forward_packed(x, caches, options, packed);
        let x = self.norm_f.forward(x);
        (self.apply_lm_head(x), caches)
    }

    /// Single-token step: token IDs `[batch]` → logits `[batch, padded_vocab]`.
    ///
    /// The vocab network has no class tokens of its own (they would duplicate
    /// the class latents of the layers). It forwards `class` (the stack-level
    /// and per-virtual-layer cursors) to [`Layers::step`].
    pub fn step(
        &self,
        x: Tensor<1, Int>,
        caches: Option<M::Caches>,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, M::Caches) {
        // Embed the single token via a temporary unit sequence axis.
        let x = self
            .embedding
            .forward(x.unsqueeze_dim::<2>(1))
            .squeeze_dim(1);
        let (x, caches) = self.layers.step(x, caches, class);
        let x = self.norm_f.forward(x);
        // Reuse the 3-D head by lifting/lowering the sequence axis.
        let logits = self.apply_lm_head(x.unsqueeze_dim(1)).squeeze_dim(1);
        (logits, caches)
    }

    /// Step the class latents that the stack has waiting for its next token,
    /// with no token of its own. Returns the logits `[batch, padded_vocab]` of
    /// the **last** latent emitted, or `None` when none were waiting. This is
    /// the entry point of seedless generation (`prime` → sample → `step` → …).
    /// The vocab network has no class tokens of its own, so it forwards `class`
    /// to [`Layers::prime`], whose docs state the placement rules.
    pub fn prime(
        &self,
        batch: usize,
        caches: Option<M::Caches>,
        class: Option<&mut ClassCursors>,
    ) -> (Option<Tensor<2>>, Option<M::Caches>) {
        let (x, caches) = self.layers.prime(batch, caches, class);
        let logits = x.map(|x| {
            let x = self.norm_f.forward(x);
            // Reuse the 3-D head by lifting/lowering the sequence axis.
            self.apply_lm_head(x.unsqueeze_dim(1)).squeeze_dim(1)
        });
        (logits, caches)
    }

    /// Project `[batch, sequence, d_model]` → `[batch, sequence, padded_vocab]`
    /// using the dedicated head, or the tied (transposed embedding) weight.
    fn apply_lm_head(&self, x: Tensor<3>) -> Tensor<3> {
        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(x)
        } else {
            // Weight tying: reuse embedding.weight^T ([d_model, padded_vocab]).
            let weight = self.embedding.weight.clone().map(|w| w.transpose());
            Linear { weight, bias: None }.forward(x)
        }
    }
}

/// Plain factory for [`VocabNetwork`]. Mirrors [`LatentNetworkBuilder`] but adds
/// vocab padding and the tied/untied LM-head choice.
pub struct VocabNetworkBuilder<C> {
    /// Unpadded vocabulary size (rounded up at init).
    pub vocab_size: usize,
    /// Round `vocab_size` up to a multiple of this (1 disables rounding).
    pub pad_vocab_size_multiple: usize,
    /// Builder for the layer stack.
    pub layers: LayersBuilder<C>,
    /// When `true`, tie the LM head to the (transposed) embedding weights.
    pub missing_lm_head: bool,
}

impl<C: BlockConfig> VocabNetworkBuilder<C> {
    /// Round `vocab_size` up to the next multiple of `multiple`.
    fn padded_vocab(vocab_size: usize, multiple: usize) -> usize {
        if vocab_size.is_multiple_of(multiple) {
            vocab_size
        } else {
            ((vocab_size / multiple) + 1) * multiple
        }
    }

    /// Allocate and initialise the network on `device`.
    pub fn init(&self, device: &Device) -> VocabNetwork<C::Block> {
        let d_model = self.layers.block.d_model();
        let padded_vocab = Self::padded_vocab(self.vocab_size, self.pad_vocab_size_multiple);
        let lm_head = if self.missing_lm_head {
            None
        } else {
            Some(
                LinearConfig::new(d_model, padded_vocab)
                    .with_bias(false)
                    .init(device),
            )
        };
        VocabNetwork {
            embedding: EmbeddingConfig::new(padded_vocab, d_model).init(device),
            layers: self.layers.init(device),
            norm_f: RmsNormConfig::new(d_model).init(device),
            lm_head,
        }
    }
}
