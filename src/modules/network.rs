use crate::modules::LayersBuilder;
use crate::modules::{RmsNorm, RmsNormConfig};
use crate::prelude::*;
use crate::utils::class::{
    assert_full_len_known, class_chunk_plan, class_emb_width, class_marker_output_indices,
    class_prime_plan, class_row, init_class_emb, insert_class_markers,
};
use crate::utils::{ClassCursor, ClassCursors};
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
    /// Optional final RMSNorm before [`Self::out_proj`] — the counterpart of
    /// [`VocabNetwork::norm_f`], which is unconditional there.
    ///
    /// It makes the head's input scale-free, which matters whenever the stack's
    /// output magnitude is not `O(1)`: a plain additive residual grows it with
    /// depth, while [`Residuals::MultiGate`](crate::modules::Residuals) is a
    /// convex mixture (mean-pooled over `n` streams) that *shrinks* it, so the
    /// two schemes otherwise hand `out_proj` signals of very different scale.
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
    /// position past the emitted sequence — compare against its length.
    pub fn class_token_output_indices(&self, orig_len: usize) -> Vec<usize> {
        class_marker_output_indices(&self.class_tokens, orig_len)
    }

    /// Splice this network's class tokens into the chunk `x` (no-op when there
    /// are none), advancing the network-level cursor.
    fn insert_tokens(&self, x: Tensor<3>, class: &mut ClassCursors) -> Tensor<3> {
        let mut cursor = ClassCursor::at(class.network, class.full_len);
        let x = insert_class_markers(
            x,
            &self.class_tokens,
            self.class_tokens_emb.as_ref(),
            &mut cursor,
            "LatentNetwork",
        );
        class.network = cursor.offset;
        x
    }

    /// `in_proj → layers → out_proj` over a full sequence
    /// (`[batch, sequence, input_size]` → `[batch, sequence (+ class tokens),
    /// output_size]`).
    ///
    /// `class` places this network's class tokens *and* the inner stack's class
    /// latents; `None` takes `x` for the whole sequence. Handing the same
    /// [`ClassCursors`] to consecutive chunks places every marker exactly where
    /// a single call over the concatenated sequence would.
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<M::Caches>,
        options: M::Options,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<3>, M::Caches) {
        // No cursors ⇒ this one call covers the whole sequence.
        let mut whole = ClassCursors::new(x.dims()[1]);
        let class = class.unwrap_or(&mut whole);
        let x = self.insert_tokens(x, class);
        let x = self.in_proj.forward(x);
        // The stack's sequence is this one, lengthened by the class tokens.
        let saved = class.enter(self.class_tokens.len());
        let (x, caches) = self.layers.forward(x, caches, options, Some(&mut *class));
        class.leave(saved);
        let x = self.head(x);
        (x, caches)
    }

    /// Single-token step (`[batch, input_size]` → `[batch, output_size]`).
    ///
    /// `class` drives all three class levels at once: this network's own
    /// [`Self::class_tokens`] (`class.network`) plus the inner [`Layers::step`]
    /// cursors (`class.stack`, `class.per_layer`).
    ///
    /// As in `forward`, the network's class tokens are part of the sequence that
    /// enters the layers, so each is run through a full network pass (carrying
    /// the inner cursors, so the layers splice their own latents around it
    /// exactly as in `forward`). What comes back is the output of the **last**
    /// token the step emitted: the user token, unless an `End` marker (at either
    /// level) follows it, that marker being then the sequence's true last token.
    /// `None` injects nothing anywhere; `Middle`/`End` markers then panic, as
    /// they do without a [`ClassCursors::full_len`] hint.
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
        // `at == 0` ⇒ the class token precedes the user token, `at == 1` ⇒ it is
        // an `End` closing the sequence, and follows it.
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
            // A closing `End` token *is* the sequence's last token — its output,
            // not the user token's, is what this step produced.
            let (o, c) = self.step_one(row(i), Some(caches), Some(&mut *class));
            out = o;
            caches = c;
        }
        (out, caches)
    }

    /// Step the class tokens/latents this network has waiting for its next user
    /// token — with **no** user token, so no input data is needed.
    ///
    /// This is [`Self::step`]'s opening half on its own, at all three class
    /// levels: the network's own class tokens due now each run a full pass (so
    /// the layers splice their latents around them exactly as in `step`), and
    /// whatever the stack still has waiting for its next token is flushed after
    /// them ([`Layers::prime`]). A `prime` followed by a `step` therefore runs
    /// the very sequence that `step` alone would have — `prime` → sample →
    /// `step` → sample → … is the seedless-generation loop. `End` markers are
    /// never primed: they close the sequence, so they belong to the step
    /// carrying its last user token.
    ///
    /// Returns the output of the **last** marker emitted, or `None` when none
    /// were waiting (the caches then come back untouched, `None` included).
    /// `batch` sizes the marker rows, the only inputs there are.
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
        // The stack's own levels may still hold latents waiting for the next
        // token to reach them — the class tokens above just went past.
        let saved = class.enter(self.class_tokens.len());
        let (y, caches) = self.layers.prime(batch, caches, Some(&mut *class));
        class.leave(saved);
        if let Some(y) = y {
            out = Some(self.head(y));
        }
        (out, caches)
    }

    /// One token through `in_proj → layers → out_proj`; the network's own class
    /// tokens are placed by [`Self::step`], the inner cursors are forwarded.
    fn step_one(
        &self,
        x: Tensor<2>,
        caches: Option<M::Caches>,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, M::Caches) {
        let x = self.in_proj.forward(x);
        let (x, caches) = match class {
            // The stack's sequence is this one, lengthened by the class tokens.
            Some(class) => {
                let saved = class.enter(self.class_tokens.len());
                let out = self.layers.step(x, caches, Some(&mut *class));
                class.leave(saved);
                out
            }
            None => self.layers.step(x, caches, None),
        };
        (self.head(x), caches)
    }

    /// Stationary fixed point of the network under a constant input token:
    /// `in_proj → `[`Layers::step_infinite`]` → out_proj`, no caches.
    /// Cursorless (class tokens are not injected).
    pub fn step_infinite(&self, x: Tensor<2>) -> Tensor<2> {
        assert_full_len_known(&self.class_tokens, None, "LatentNetwork");
        let x = self.in_proj.forward(x);
        let x = self.layers.step_infinite(x);
        self.head(x)
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
/// This is the token-LM counterpart of [`LatentNetwork`]; both are built on the
/// shared [`Layers`] core. The only differences are the I/O boundary (a token
/// `Embedding` and a vocab logit head, instead of two latent `Linear`s) and a
/// final pre-head [`RmsNorm`].
///
/// The LM head is **tied** (`lm_head = None`, the transposed embedding weight is
/// reused) or **untied** (a dedicated `Linear`); the vocabulary is rounded up to
/// a multiple for GPU alignment (see [`VocabNetworkBuilder`]).
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
    /// `[batch, sequence, padded_vocab]`. `class` places the inner stack's class
    /// latents (`None` ⇒ `x` is the whole sequence) — see [`Layers::forward`].
    pub fn forward(
        &self,
        x: Tensor<2, Int>,
        caches: Option<M::Caches>,
        options: M::Options,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<3>, M::Caches) {
        let x = self.embedding.forward(x);
        let (x, caches) = self.layers.forward(x, caches, options, class);
        let x = self.norm_f.forward(x);
        (self.apply_lm_head(x), caches)
    }

    /// Single-token step: token IDs `[batch]` → logits `[batch, padded_vocab]`.
    ///
    /// The vocab network has no class tokens of its own (those would duplicate
    /// the layers' class latents); it simply forwards `class` — the stack-level
    /// and per-virtual-layer cursors — to [`Layers::step`].
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

    /// Step the class latents the stack has waiting for its next token, with no
    /// token of its own: logits `[batch, padded_vocab]` for the **last** latent
    /// emitted, or `None` when none were waiting — the seedless-generation entry
    /// point (`prime` → sample → `step` → …). Having no class tokens of its own,
    /// the vocab network just forwards `class` to [`Layers::prime`], whose docs
    /// carry the placement rules.
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

    /// Stationary fixed point of the LM under a constant token: logits
    /// `[batch, padded_vocab]` after infinitely many repeats of `x`, no caches
    /// (see [`Layers::step_infinite`]).
    pub fn step_infinite(&self, x: Tensor<1, Int>) -> Tensor<2> {
        let x = self
            .embedding
            .forward(x.unsqueeze_dim::<2>(1))
            .squeeze_dim(1);
        let x = self.layers.step_infinite(x);
        let x = self.norm_f.forward(x);
        self.apply_lm_head(x.unsqueeze_dim(1)).squeeze_dim(1)
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

