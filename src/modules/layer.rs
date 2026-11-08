use crate::modules::{GatedMlp, RmsNorm};
use crate::prelude::*;
use crate::utils::{Packed, Padding, UntiedParam};
use burn::prelude::*;
use std::borrow::Cow;

/// A single Pre-LN wrapper around one block. It computes `M(RMSNorm(x))`. It
/// does **not** apply the residual. The enclosing [`Layers`] owns that
/// decision: add the input back, suppress it on the first/last layer, or
/// thread it through the residual streams of MGR. So no input clone or
/// zero-add is wasted when no residual is wanted.
///
/// With [`Self::mlp`] set, the layer also runs a second Pre-LN sub-block, a
/// SwiGLU feed-forward (see [`GatedMlp`]). It has a residual of its own,
/// *inside* the layer. This is why the methods below return the **total
/// delta** of the layer, not the block output:
///
/// ```text
///   h₁ = M(norm(x))                     the block
///   h₂ = mlp(norm2(x + h₁))             the feed-forward sub-block
///   return h₁ + h₂                      so that Layers' `x + delta` is
///                                       (x + h₁) + h₂ — both residuals
/// ```
///
/// This fold keeps [`Layers`] the single owner of the *outer* residual, and of
/// the `ignore_first/last_residual` ablations. So these ablations govern only
/// the outer add. The inner residual of the feed-forward is part of the
/// sub-block and always applies. Without an `mlp`, the delta is just `h₁`, and
/// nothing changes for a family that has no feed-forward.
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
    /// The block of this layer.
    pub block: M,
    /// Pre-norm of the feed-forward sub-block. `Some` exactly when [`Self::mlp`]
    /// is `Some` (`norm2` in the reference checkpoints).
    pub norm2: Option<RmsNorm>,
    /// Optional SwiGLU feed-forward sub-block run after the block, with its own
    /// residual. `None` ⇒ the layer has only its block.
    pub mlp: Option<GatedMlp>,
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
    /// The pre-norm gain of the block ([`Layer::norm`]).
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

    /// The layer input, kept only when the feed-forward sub-block needs it for
    /// its inner residual. Otherwise `None`, so a layer with no feed-forward
    /// still moves `x` straight into the pre-norm with no clone.
    fn mlp_residual<const D: usize>(&self, x: &Tensor<D>) -> Option<Tensor<D>> {
        self.mlp.as_ref().map(|_| x.clone())
    }

    /// Completes the total delta of the layer: `h₁ ↦ h₁ + mlp(norm2(x + h₁))`.
    ///
    /// `residual` is what [`Self::mlp_residual`] captured. So `None` here means
    /// that there is no feed-forward, and the delta is the block output alone.
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

    /// The Pre-LN forward of a window, **without** the outer residual. Returns
    /// the total delta of the layer: `M(RMSNorm(x))`, plus the contribution of
    /// the feed-forward sub-block when [`Self::mlp`] is set (see the type docs).
    ///
    /// The caller owns the outer residual.
    ///
    /// `pad` marks the padded positions of a right-padded batch (`None` ⇒ no
    /// padding). The block runs on the positions of each row in the order of
    /// that row (see [`Padding::in_row_order`]). Everything else here is per
    /// position.
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
            Some(pad) => pad.in_row_order(normed, |normed, pad_bs| {
                self.block.block_forward(normed, cache, options, Some(pad_bs))
            }),
        };
        (self.add_mlp_delta(residual, h1), cache)
    }

    /// [`Self::forward`] over packed rows (see [`crate::utils::packing`]). The
    /// block starts again from a zero cache at each reset of `packed`.
    /// Everything else here is per position.
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

    /// The Pre-LN step of one token, **without** the outer residual. Returns
    /// the total delta of the layer and the cache that the step produced. The
    /// caller owns the outer residual.
    ///
    /// [`Layers`] calls it for each token of its stream. An external container
    /// that owns the residual and threads its own state between layers can
    /// also call it.
    pub fn step(&self, x: Tensor<2>, cache: Option<M::Cache>) -> (Tensor<2>, M::Cache) {
        let residual = self.mlp_residual(&x);
        let normed = self.norm.forward(x);
        let (h1, cache) = self.block.block_step(normed, cache);
        (self.add_mlp_delta(residual, h1), cache)
    }
}