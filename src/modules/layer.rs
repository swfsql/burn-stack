use crate::modules::{GatedMlp, RmsNorm};
use crate::prelude::*;
use crate::utils::class::{
    assert_full_len_known, class_chunk_plan, class_emb_width, class_prime_plan, class_row,
    insert_class_markers,
};
use crate::utils::{ClassCursor, ClassLatent, UntiedParam};
use burn::module::Param;
use burn::prelude::*;
use std::borrow::Cow;

/// A single Pre-LN block wrapper computing `M(RMSNorm(x))` — the residual is
/// **not** applied here. The enclosing [`Layers`] owns
/// that decision (add the input back, suppress it on the first/last layer, or
/// thread it through Multi-Gate streams), so no input clone / zero-add is wasted
/// when no residual is wanted.
///
/// With [`Self::mlp`] set the layer additionally runs a second Pre-LN sub-block,
/// a SwiGLU feed-forward (see [`GatedMlp`]). It
/// has a residual of its own, *inside* the layer, which is the reason the
/// methods below return the layer's **total delta** rather than the mixer output:
///
/// ```text
///   h₁ = M(norm(x))                     the mixer sub-block
///   h₂ = mlp(norm2(x + h₁))             the feed-forward sub-block
///   return h₁ + h₂                      so that Layers' `x + delta` is
///                                       (x + h₁) + h₂ — both residuals
/// ```
///
/// Folding it this way keeps [`Layers`] the single owner of the *outer* residual
/// (and of the `ignore_first/last_residual` ablations, which therefore govern
/// only that outer add — the feed-forward's inner residual is intrinsic to the
/// sub-block and always applies). Without an `mlp` the delta is just `h₁` and
/// nothing changes for a block family that carries no feed-forward.
///
/// May carry its own [`ClassLatent`]s, placed from a [`ClassCursor`]: `step`
/// splices them around the token it is given, while in `forward` the caller
/// splices them first (via [`Self::insert_latents`]) so the residual it adds
/// sees the same lengthened sequence; [`Self::prime`] steps the ones waiting for
/// the next token *without* that token. They are independent of any class
/// latents on the enclosing [`Layers`].
///
/// Built for a real layer applied several times, it may **untie** parameters —
/// its [`LayerUntied`] pre-norms and its block's ([`Block::untied_params`]) —
/// holding one copy per application. A container runs application `k` as
/// [`Self::application`]`(k)`, the layer that application sees; every other
/// method acts on the layer it is called on.
#[derive(Module, Debug)]
pub struct Layer<M: Module> {
    /// Pre-norm applied before the inner block.
    pub norm: RmsNorm,
    /// The inner mixer block.
    pub block: M,
    /// Pre-norm of the feed-forward sub-block. `Some` exactly when [`Self::mlp`]
    /// is (`norm2` in the reference checkpoints).
    pub norm2: Option<RmsNorm>,
    /// Optional SwiGLU feed-forward sub-block run after the mixer, with its own
    /// residual. `None` ⇒ the layer is mixer-only.
    pub mlp: Option<GatedMlp>,
    /// Positions of this layer's class latents (empty ⇒ none).
    #[module(skip)]
    pub class_latents: Vec<ClassLatent>,
    /// The class-latent embeddings, `[num_class_latents, d_model]` (`None` ⇒ none).
    pub class_latents_emb: Option<Param<Tensor<2>>>,
    /// The pre-norms held once per application ([`LayerUntied`]); empty ⇒ both
    /// tied.
    #[module(skip)]
    pub untied: Vec<LayerUntied>,
    /// How many applications the untied parameters — the layer's and its
    /// block's — hold a copy for: `1` for a layer applied once, and for every
    /// [`Self::application`] view.
    #[module(skip)]
    pub n_applications: usize,
}

/// A [`Layer`] parameter that may be **untied**: held once per application of
/// its real layer (see [`crate::utils::untied`]). The block's own are its
/// config's to name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LayerUntied {
    /// The mixer's pre-norm gain ([`Layer::norm`]).
    Norm,
    /// The feed-forward's pre-norm gain ([`Layer::norm2`]); needs [`Layer::mlp`].
    Norm2,
}

impl<M: Block> Layer<M> {
    /// A fresh layer around `block`, for a real layer applied `n_applications`
    /// times: the pre-norms `untied` names get one gain per application.
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
            "LayerUntied::Norm2 unties the feed-forward's pre-norm, and this layer has no feed-forward",
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
            // `norm2` exists exactly when `mlp` does — `Layer` relies on it.
            norm2: mlp.map(|_| norm(LayerUntied::Norm2)),
            mlp: mlp.map(|mlp| mlp.init(device)),
            class_latents: Vec::new(),
            class_latents_emb: None,
            untied: untied.to_vec(),
            n_applications,
        }
    }

    /// Every parameter of this layer held once per application: its untied
    /// pre-norms, then its block's ([`Block::untied_params`]).
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

    /// Whether any of this layer's weights differ between its applications.
    pub fn has_untied(&self) -> bool {
        self.n_applications > 1 && !self.untied_params().is_empty()
    }

    /// This layer as its `application`-th application sees it: each untied
    /// parameter narrowed to that application's copy, everything else shared
    /// (see [`crate::utils::untied`]). The view is a plain tied layer
    /// (`n_applications = 1`).
    ///
    /// Borrows the layer itself when nothing differs between its applications —
    /// so a layer built for a single one reads that copy at any application.
    ///
    /// # Panics
    /// An `application` past the count the layer was built for, if it unties
    /// anything.
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

    /// Reset every untied parameter to copies of its first application: the
    /// copies [`BlockConfig::init_block`] starts from, over a layer something has
    /// since redrawn (see [`crate::utils::untied::retie`]).
    pub fn retie(self) -> Self {
        let params = self.untied_params();
        let n_applications = self.n_applications;
        crate::utils::untied::retie(self, &params, n_applications)
    }

    /// Splice this layer's class latents into the chunk `x` (no-op when there
    /// are none), advancing `class` past it.
    ///
    /// Public so a caller driving a bare [`Layer`] can lengthen the sequence
    /// itself (and add the matching residual) before calling [`Self::forward`].
    /// `None` cursors ⇒ this chunk is the whole sequence. [`Layers`] splices its
    /// layers' latents itself, since under
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
    /// its inner residual — otherwise `None`, so the mixer-only path still moves
    /// `x` straight into the pre-norm with no clone.
    fn mlp_residual<const D: usize>(&self, x: &Tensor<D>) -> Option<Tensor<D>> {
        self.mlp.as_ref().map(|_| x.clone())
    }

    /// Completes the layer's total delta: `h₁ ↦ h₁ + mlp(norm2(x + h₁))`.
    ///
    /// `residual` is whatever [`Self::mlp_residual`] captured, so a `None` here
    /// means there is no feed-forward and the delta is the mixer output alone.
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

    /// Full-sequence Pre-LN block **without** the outer residual: the layer's
    /// total delta `M(RMSNorm(x))`, plus the feed-forward sub-block's own
    /// contribution when [`Self::mlp`] is set (see the type docs).
    ///
    /// The caller owns any class-latent insertion ([`Self::insert_latents`]) and
    /// the outer residual.
    pub fn forward(
        &self,
        x: Tensor<3>,
        cache: Option<M::Cache>,
        options: M::Options,
    ) -> (Tensor<3>, M::Cache) {
        let residual = self.mlp_residual(&x);
        let normed = self.norm.forward(x);
        let (h1, cache) = self.block.block_forward(normed, cache, options);
        (self.add_mlp_delta(residual, h1), cache)
    }

    /// Single-token Pre-LN block step **without** the residual.
    ///
    /// `class` is this layer's own class-latent cursor. With `Some`, every
    /// latent whose position falls on this token is stepped around it — before
    /// it (`Start`/`Middle`/`Custom`, which precede a token) or after it (`End`,
    /// which closes the sequence) — each a step of its own. What comes back is
    /// the **last** token the step emitted (see
    /// [`ClassCursors`](crate::utils::ClassCursors)): the user token, unless an
    /// `End` latent follows it, that latent being then the sequence's true last
    /// token. With `None` no class latents are injected — and `Middle`/`End`
    /// latents panic (their positions need the full sequence length). The
    /// residual is the caller's responsibility.
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
        // `at == 0` ⇒ the latent precedes the user token, `at == 1` ⇒ it is an
        // `End` closing the sequence, and follows it.
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
            // A closing `End` *is* the sequence's last token — its output, not
            // the user token's, is what this step produced.
            let (o, c) = self.step_one(row(i), Some(cache));
            out = o;
            cache = c;
        }
        (out, cache)
    }

    /// Step the class latents this layer has waiting for its next token — with
    /// **no** token of its own, so nothing but class data is consumed.
    ///
    /// This is [`Self::step`]'s opening half on its own (see
    /// [`ClassCursors`](crate::utils::ClassCursors)): the latents that would
    /// have preceded the next token are stepped now, in the same order, so a
    /// `prime` followed by a `step` runs exactly the sequence that `step` alone
    /// would have. `End` latents are never primed — closing the sequence, they
    /// belong to the step carrying its last token.
    ///
    /// Returns the **last** latent stepped, as the pair `(delta, latent)` — this
    /// layer's own embedding row alongside the delta it produced, since the
    /// caller has no other way to complete the residual (`delta + latent`, as it
    /// does with the token it hands to [`Self::step`]). `None` ⇒ nothing was
    /// waiting, and the cache comes back exactly as it went in (`None` included:
    /// a layer that stepped nothing has the state it already had).
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
    /// [`Layers`]'s cascade uses it to place this layer's class latents from the
    /// stack-wide [`ClassCursors`](crate::utils::ClassCursors) itself, bypassing
    /// [`Self::step`]'s cursorless guard (that guard rejects `Middle`/`End`,
    /// which the cascade has already resolved). It is public because an external
    /// container that owns the residual — one threading its own state between
    /// layers rather than a per-layer cache — needs exactly this: the layer's
    /// delta and the cache it produced, with nothing added.
    pub fn step_one(&self, x: Tensor<2>, cache: Option<M::Cache>) -> (Tensor<2>, M::Cache) {
        let residual = self.mlp_residual(&x);
        let normed = self.norm.forward(x);
        let (h1, cache) = self.block.block_step(normed, cache);
        (self.add_mlp_delta(residual, h1), cache)
    }
}

