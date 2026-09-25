use crate::modules::{GatedMlp, RmsNorm};
use crate::prelude::*;
use crate::utils::class::{
    assert_full_len_known, class_chunk_plan, class_emb_width, class_prime_plan, class_row,
    insert_class_markers,
};
use crate::utils::{ClassCursor, ClassLatent, Packed, Padding, UntiedParam};
use burn::module::Param;
use burn::prelude::*;
use std::borrow::Cow;

/// A single Pre-LN block wrapper that computes `M(RMSNorm(x))`. It does
/// **not** apply the residual. The enclosing [`Layers`] owns that decision:
/// add the input back, suppress it on the first/last layer, or thread it
/// through Multi-Gate streams. So no input clone or zero-add is wasted when no
/// residual is wanted.
///
/// With [`Self::mlp`] set, the layer also runs a second Pre-LN sub-block, a
/// SwiGLU feed-forward (see [`GatedMlp`]). It has a residual of its own,
/// *inside* the layer. This is why the methods below return the **total
/// delta** of the layer, not the mixer output:
///
/// ```text
///   h₁ = M(norm(x))                     the mixer sub-block
///   h₂ = mlp(norm2(x + h₁))             the feed-forward sub-block
///   return h₁ + h₂                      so that Layers' `x + delta` is
///                                       (x + h₁) + h₂ — both residuals
/// ```
///
/// This fold keeps [`Layers`] the single owner of the *outer* residual, and of
/// the `ignore_first/last_residual` ablations. So these ablations govern only
/// the outer add. The inner residual of the feed-forward is part of the
/// sub-block and always applies. Without an `mlp`, the delta is just `h₁`, and
/// nothing changes for a block family that has no feed-forward.
///
/// A layer can have its own [`ClassLatent`]s, placed from a [`ClassCursor`]:
///
/// - `step` splices them around the token that it gets.
/// - In `forward`, the caller splices them first (with
///   [`Self::insert_latents`]), so the residual that it adds sees the same
///   longer sequence.
/// - [`Self::prime`] steps the latents that wait for the next token, *without*
///   that token.
///
/// They are independent of the class latents of the enclosing [`Layers`].
///
/// A layer built for a real layer that is applied several times can **untie**
/// parameters: its [`LayerUntied`] pre-norms and those of its block
/// ([`Block::untied_params`]). It then holds one copy per application. A
/// container runs application `k` as [`Self::application`]`(k)`, the layer
/// that this application sees. Every other method acts on the layer that it is
/// called on.
#[derive(Module, Debug)]
pub struct Layer<M: Module> {
    /// Pre-norm applied before the inner block.
    pub norm: RmsNorm,
    /// The inner mixer block.
    pub block: M,
    /// Pre-norm of the feed-forward sub-block. `Some` exactly when [`Self::mlp`]
    /// is `Some` (`norm2` in the reference checkpoints).
    pub norm2: Option<RmsNorm>,
    /// Optional SwiGLU feed-forward sub-block run after the mixer, with its own
    /// residual. `None` ⇒ the layer is mixer-only.
    pub mlp: Option<GatedMlp>,
    /// Positions of this layer's class latents (empty ⇒ none).
    #[module(skip)]
    pub class_latents: Vec<ClassLatent>,
    /// The class-latent embeddings, `[num_class_latents, d_model]` (`None` ⇒ none).
    pub class_latents_emb: Option<Param<Tensor<2>>>,
    /// The pre-norms held once per application ([`LayerUntied`]). Empty ⇒ both
    /// tied.
    #[module(skip)]
    pub untied: Vec<LayerUntied>,
    /// The number of applications for which the untied parameters (of the
    /// layer and of its block) hold a copy. It is `1` for a layer applied once,
    /// and for every [`Self::application`] view.
    #[module(skip)]
    pub n_applications: usize,
}

/// A [`Layer`] parameter that can be **untied**: held once per application of
/// its real layer (see [`crate::utils::untied`]). The block config names the
/// untiable parameters of the block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LayerUntied {
    /// The pre-norm gain of the mixer ([`Layer::norm`]).
    Norm,
    /// The pre-norm gain of the feed-forward ([`Layer::norm2`]). Needs
    /// [`Layer::mlp`].
    Norm2,
}

impl<M: Block> Layer<M> {
    /// A new layer around `block`, for a real layer applied `n_applications`
    /// times. The pre-norms that `untied` names get one gain per application.
    pub(crate) fn init(
        block: M,
        d_model: usize,
        mlp: Option<&GatedMlpConfig>,
        untied: &[LayerUntied],
        n_applications: usize,
        device: &Device,
    ) -> Self {
        assert!(
            mlp.is_some() || !untied.contains(&LayerUntied::Norm2),
            "LayerUntied::Norm2 unties the pre-norm of the feed-forward, but this layer has no \
             feed-forward",
        );
        let norm = |part: LayerUntied| {
            let RmsNorm { gamma } = RmsNormConfig::new(d_model).init(device);
            let gamma = match untied.contains(&part) {
                true => crate::utils::untied::tile(gamma, 0, n_applications),
                false => gamma,
            };
            RmsNorm { gamma }
        };
        Layer {
            norm: norm(LayerUntied::Norm),
            block,
            // `norm2` exists exactly when `mlp` exists. `Layer` relies on this.
            norm2: mlp.map(|_| norm(LayerUntied::Norm2)),
            mlp: mlp.map(|mlp| mlp.init(device)),
            class_latents: Vec::new(),
            class_latents_emb: None,
            untied: untied.to_vec(),
            n_applications,
        }
    }

    /// Every parameter of this layer that is held once per application: its
    /// untied pre-norms, then those of its block ([`Block::untied_params`]).
    pub fn untied_params(&self) -> Vec<UntiedParam> {
        let own = self.untied.iter().map(|part| match part {
            LayerUntied::Norm => UntiedParam::new(&self.norm.gamma, 0),
            LayerUntied::Norm2 => {
                let norm2 = self.norm2.as_ref().expect("`Norm2` is untied only alongside an `mlp`");
                UntiedParam::new(&norm2.gamma, 0)
            }
        });
        own.chain(self.block.untied_params()).collect()
    }

    /// Whether any weight of this layer differs between its applications.
    pub fn has_untied(&self) -> bool {
        self.n_applications > 1 && !self.untied_params().is_empty()
    }

    /// This layer as its `application`-th application sees it. Each untied
    /// parameter is narrowed to the copy of that application, and everything
    /// else is shared (see [`crate::utils::untied`]). The view is a plain tied
    /// layer (`n_applications = 1`).
    ///
    /// When nothing differs between the applications, this borrows the layer
    /// itself. So a layer built for a single application reads that copy at
    /// any application.
    ///
    /// # Panics
    /// If the layer unties anything and `application` is past the count that
    /// the layer was built for.
    pub fn application(&self, application: usize) -> Cow<'_, Self> {
        if self.n_applications == 1 {
            return Cow::Borrowed(self);
        }
        let params = self.untied_params();
        if params.is_empty() {
            return Cow::Borrowed(self);
        }
        let mut view = crate::utils::untied::view(self, &params, application, self.n_applications);
        view.n_applications = 1;
        Cow::Owned(view)
    }

    /// Reset every untied parameter to copies of its first application.
    /// [`BlockConfig::init_block`] starts from such copies. Use this on a
    /// layer that something redrew after the build (see
    /// [`crate::utils::untied::retie`]).
    pub fn retie(self) -> Self {
        let params = self.untied_params();
        let n_applications = self.n_applications;
        crate::utils::untied::retie(self, &params, n_applications)
    }

    /// Splice the class latents of this layer into the chunk `x` (no-op when
    /// there are none), and advance `class` past the chunk.
    ///
    /// This is public, so a caller that drives a bare [`Layer`] can make the
    /// sequence longer itself (and add the matching residual) before it calls
    /// [`Self::forward`]. `None` cursors ⇒ this chunk is the whole sequence.
    /// [`Layers`] splices the latents of its layers itself, because under
    /// [`MultiGate`](crate::modules::MultiGate) residuals the same rows must
    /// also enter the carried streams.
    pub fn insert_latents(&self, x: Tensor<3>, class: Option<&mut ClassCursor>) -> Tensor<3> {
        let mut whole = ClassCursor::whole(x.dims()[1]);
        let cursor = class.unwrap_or(&mut whole);
        insert_class_markers(
            x,
            &self.class_latents,
            self.class_latents_emb.as_ref(),
            cursor,
            "Layer",
        )
    }

    /// The layer input, kept only when the feed-forward sub-block needs it for
    /// its inner residual. Otherwise `None`, so the mixer-only path still moves
    /// `x` straight into the pre-norm with no clone.
    fn mlp_residual<const D: usize>(&self, x: &Tensor<D>) -> Option<Tensor<D>> {
        self.mlp.as_ref().map(|_| x.clone())
    }

    /// Completes the total delta of the layer: `h₁ ↦ h₁ + mlp(norm2(x + h₁))`.
    ///
    /// `residual` is what [`Self::mlp_residual`] captured. So `None` here means
    /// that there is no feed-forward, and the delta is the mixer output alone.
    fn add_mlp_delta<const D: usize>(
        &self,
        residual: Option<Tensor<D>>,
        h1: Tensor<D>,
    ) -> Tensor<D> {
        let Some(mlp) = self.mlp.as_ref() else {
            return h1;
        };
        let x = residual.expect("`mlp_residual` captures the input whenever `mlp` is present");
        let norm2 = self
            .norm2
            .as_ref()
            .expect("`norm2` is allocated alongside `mlp`");
        let h2 = mlp.forward(norm2.forward(x + h1.clone()));
        h1 + h2
    }

    /// Full-sequence Pre-LN block **without** the outer residual. Returns the
    /// total delta of the layer: `M(RMSNorm(x))`, plus the contribution of the
    /// feed-forward sub-block when [`Self::mlp`] is set (see the type docs).
    ///
    /// The caller owns any class-latent insertion ([`Self::insert_latents`]) and
    /// the outer residual.
    ///
    /// `pad` marks the padded rows of a right-padded batch (`None` ⇒ none). The
    /// block runs on the rows of each slot in the order of that slot (see
    /// [`Padding::in_slot_order`]). Everything else here is per row.
    pub fn forward(
        &self,
        x: Tensor<3>,
        cache: Option<M::Cache>,
        options: M::Options,
        pad: Option<&Padding>,
    ) -> (Tensor<3>, M::Cache) {
        let residual = self.mlp_residual(&x);
        let normed = self.norm.forward(x);
        let (h1, cache) = match pad {
            None => self.block.block_forward(normed, cache, options, None),
            Some(pad) => pad.in_slot_order(normed, |normed, pad_bs| {
                self.block.block_forward(normed, cache, options, Some(pad_bs))
            }),
        };
        (self.add_mlp_delta(residual, h1), cache)
    }

    /// [`Self::forward`] over packed rows (see [`crate::utils::packing`]). The
    /// block restarts at each reset of `packed`. Everything else here is per
    /// row.
    pub fn forward_packed(
        &self,
        x: Tensor<3>,
        cache: Option<M::Cache>,
        options: M::Options,
        packed: &Packed,
    ) -> (Tensor<3>, M::Cache) {
        let residual = self.mlp_residual(&x);
        let normed = self.norm.forward(x);
        let (h1, cache) =
            self.block
                .block_forward_packed(normed, cache, options, packed.reset_bs.clone());
        (self.add_mlp_delta(residual, h1), cache)
    }

    /// Single-token Pre-LN block step **without** the residual.
    ///
    /// `class` is the class-latent cursor of this layer. With `Some`, every
    /// latent whose position falls on this token gets a step of its own,
    /// around the token:
    ///
    /// - before it: `Start`/`Middle`/`Custom`, which precede a token,
    /// - after it: `End`, which closes the sequence.
    ///
    /// The call returns the **last** token that the step emitted (see
    /// [`ClassCursors`](crate::utils::ClassCursors)). This is the user token,
    /// unless an `End` latent follows it. That latent is then the true last
    /// token of the sequence. With `None`, no class latents are injected, and
    /// `Middle`/`End` latents panic (their positions need the full sequence
    /// length). The caller owns the residual.
    pub fn step(
        &self,
        x: Tensor<2>,
        cache: Option<M::Cache>,
        class: Option<&mut ClassCursor>,
    ) -> (Tensor<2>, M::Cache) {
        let Some(cursor) = class else {
            assert_full_len_known(&self.class_latents, None, "Layer");
            return self.step_one(x, cache);
        };
        let plan = class_chunk_plan(&self.class_latents, 1, cursor, "Layer");
        if plan.is_empty() {
            return self.step_one(x, cache);
        }
        // `at == 0` ⇒ the latent precedes the user token. `at == 1` ⇒ it is an
        // `End` that closes the sequence, and follows the token.
        let [batch, d_model] = x.dims();
        let row = |i: usize| class_row(self.class_latents_emb.as_ref(), i, batch, d_model);
        let (before, after): (Vec<_>, Vec<_>) = plan.into_iter().partition(|&(at, _)| at == 0);
        let mut cache = cache;
        for (_, i) in before {
            let (_discard, c) = self.step_one(row(i), cache);
            cache = Some(c);
        }
        let (mut out, mut cache) = self.step_one(x, cache);
        for (_, i) in after {
            // A closing `End` *is* the last token of the sequence. This step
            // produced its output, not that of the user token.
            let (o, c) = self.step_one(row(i), Some(cache));
            out = o;
            cache = c;
        }
        (out, cache)
    }

    /// Step the class latents that this layer has waiting for its next token,
    /// with **no** token of its own. So the call consumes only class data.
    ///
    /// This is the opening half of [`Self::step`] on its own (see
    /// [`ClassCursors`](crate::utils::ClassCursors)). The latents that would
    /// precede the next token get their steps now, in the same order. So a
    /// `prime` followed by a `step` runs exactly the sequence that the `step`
    /// alone would run. `End` latents are never primed. They close the
    /// sequence, so they belong to the step that carries its last token.
    ///
    /// Returns the **last** latent stepped, as the pair `(delta, latent)`: the
    /// embedding row of this layer with the delta that it produced. The caller
    /// has no other way to complete the residual (`delta + latent`, as it does
    /// with the token that it gives to [`Self::step`]). `None` ⇒ nothing was
    /// waiting, and the cache comes back exactly as it went in. This includes
    /// `None`: a layer that stepped nothing has the state that it already had.
    pub fn prime(
        &self,
        batch: usize,
        cache: Option<M::Cache>,
        class: Option<&mut ClassCursor>,
    ) -> (Option<(Tensor<2>, Tensor<2>)>, Option<M::Cache>) {
        let Some(cursor) = class else {
            // No cursor ⇒ nothing is injected, exactly as in a `None` step.
            assert_full_len_known(&self.class_latents, None, "Layer");
            return (None, cache);
        };
        let plan = class_prime_plan(&self.class_latents, 0, cursor, "Layer");
        if plan.is_empty() {
            return (None, cache);
        }
        let width = class_emb_width(self.class_latents_emb.as_ref());
        let mut cache = cache;
        let mut last = None;
        for (_at, i) in plan {
            let row = class_row(self.class_latents_emb.as_ref(), i, batch, width);
            let (out, c) = self.step_one(row.clone(), cache);
            last = Some((out, row));
            cache = Some(c);
        }
        (last, cache)
    }

    /// The actual one-token work: no class injection, no outer residual.
    ///
    /// The cascade of [`Layers`] uses it to place the class latents of this
    /// layer from the stack-wide [`ClassCursors`](crate::utils::ClassCursors)
    /// itself. This bypasses the cursorless guard of [`Self::step`], which
    /// rejects `Middle`/`End` (the cascade has already resolved them). It is
    /// public for an external container that owns the residual and threads
    /// its own state between layers, not a per-layer cache. Such a container
    /// needs exactly this: the delta of the layer and the cache that it
    /// produced, with nothing added.
    pub fn step_one(&self, x: Tensor<2>, cache: Option<M::Cache>) -> (Tensor<2>, M::Cache) {
        let residual = self.mlp_residual(&x);
        let normed = self.norm.forward(x);
        let (h1, cache) = self.block.block_step(normed, cache);
        (self.add_mlp_delta(residual, h1), cache)
    }
}