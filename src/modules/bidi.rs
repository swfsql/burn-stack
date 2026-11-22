//! Bidirectional support: family-generic, forward-only, non-autoregressive.
//!
//! A pair runs a straight (→) and a reversed (← through `flip`) Pre-LN pass,
//! and merges them with an [`OutputMerge`]. [`BidiLayers`] stacks such pairs
//! of its own [`Layer`]s with a
//! [`BidiSchedule`](crate::utils::BidiSchedule). [`BidiLayerPair`] is one pair
//! as a standalone module. The block itself does not change: only the layer
//! schedule and the combination of its two passes are bidirectional. This is
//! written once for all families. The merge is family-agnostic
//! (`RmsNorm`/`Linear` over `Tensor<3>`).

use crate::modules::layers::zero_caches_for;
use crate::modules::residuals::JoinAt;
use crate::modules::{LayerUntied, Residuals, ResidualsConfig, RmsNorm};
use crate::prelude::*;
use crate::utils::{Applications, BidiSchedule};
use crate::utils::class::{
    class_marker_output_indices, init_class_emb, insert_class_markers_padded,
};
use crate::utils::{ClassCursor, ClassCursors, ClassLatent, Padding};
use burn::config::Config;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// A zero-parameter placeholder: the parameterless `Mean` merge, and the
/// `Standard` [`Residuals`].
#[derive(Module, Debug)]
pub struct NoOp;

/// How a bidirectional pair merges the outputs of its straight pass and its
/// reversed pass.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum OutputMerge {
    /// Element-wise average of the two passes (no parameters).
    Mean(NoOp),
    /// Concatenate along the feature axis and project back down with a learnable
    /// `[2 · d_model, d_model]` linear layer.
    CatLinear(Linear),
}

impl OutputMerge {
    /// Merge the outputs of the two passes (each `[batch, sequence, d_model]`).
    pub fn forward(&self, straight: Tensor<3>, reverse: Tensor<3>) -> Tensor<3> {
        let [batch, sequence, d_model] = straight.dims();
        assert_eq!(straight.dims(), reverse.dims());
        match self {
            OutputMerge::Mean(_) => (straight + reverse) * 0.5,
            OutputMerge::CatLinear(proj) => {
                let cat = Tensor::cat([straight, reverse].to_vec(), 2);
                assert_eq!([batch, sequence, 2 * d_model], cat.dims());
                let merged = proj.forward(cat);
                assert_eq!([batch, sequence, d_model], merged.dims());
                merged
            }
        }
    }
}

/// Configuration / factory for [`OutputMerge`].
#[derive(Config, Debug)]
pub enum OutputMergeConfig {
    /// Build an [`OutputMerge::Mean`].
    Mean,
    /// Build an [`OutputMerge::CatLinear`].
    CatLinear,
}

impl OutputMergeConfig {
    /// A vector of `n_real_layers / 2` [`Self::Mean`] configs (one per pair).
    pub fn mean(n_real_layers: usize) -> Vec<Self> {
        vec![Self::Mean; n_real_layers / 2]
    }
    /// A vector of `n_real_layers / 2` [`Self::CatLinear`] configs (one per pair).
    pub fn cat_linear(n_real_layers: usize) -> Vec<Self> {
        vec![Self::CatLinear; n_real_layers / 2]
    }
    /// Allocate the merge module on `device` for the given `d_model`.
    pub fn init(&self, d_model: usize, device: &Device) -> OutputMerge {
        match self {
            OutputMergeConfig::Mean => OutputMerge::Mean(NoOp),
            OutputMergeConfig::CatLinear => {
                OutputMerge::CatLinear(LinearConfig::new(d_model * 2, d_model).init(device))
            }
        }
    }
}

/// A single bidirectional pair as a standalone module: a straight (→) and a
/// reversed (←) Pre-LN pass, with merged outputs. It does **not** apply the
/// residual. The caller adds it (or suppresses it), as [`BidiLayers`] does
/// for its own pairs. This mirrors the [`Layer`] / [`Layers`] split.
#[derive(Module, Debug)]
pub struct BidiLayerPair<M: Module> {
    /// Pre-norm for the straight pass.
    pub straight_norm: RmsNorm,
    /// Pre-norm for the reversed pass.
    pub reverse_norm: RmsNorm,
    /// The block run left-to-right.
    pub straight_block: M,
    /// The block run right-to-left (over the flipped sequence).
    pub reverse_block: M,
    /// The merge of the two passes.
    pub output_merge: OutputMerge,
}

impl<M: Block> BidiLayerPair<M>
where
    M::Options: Clone,
{
    /// `[batch, sequence, d_model]` → `[batch, sequence, d_model]`, plus the
    /// updated caches of the two passes. Returns the merged passes **without**
    /// the residual: the caller adds it.
    ///
    /// `pad` marks a right-padded batch (`None` ⇒ no padding), as in
    /// [`BidiLayers::forward`].
    pub fn forward(
        &self,
        x: Tensor<3>,
        straight_cache: M::Cache,
        reverse_cache: M::Cache,
        options: M::Options,
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, M::Cache, M::Cache) {
        let padding = pad.map(Padding::new);
        bidi_pair_forward(
            &self.straight_norm,
            &self.reverse_norm,
            &self.straight_block,
            &self.reverse_block,
            &self.output_merge,
            x,
            straight_cache,
            reverse_cache,
            options,
            padding.as_ref(),
        )
    }
}

/// The straight pass, the reversed pass and the merge of a bidirectional pair,
/// over **borrowed** sub-modules.
///
/// [`BidiLayers`] calls this directly on its real layers (or their application
/// views, when they untie anything). It does not build a transient
/// [`BidiLayerPair`].
///
/// Under `padding`, the reversed pass reads the **real** tokens of each row in
/// reverse order ([`Padding::reversed`]), not the flipped batch, whose padding
/// would then lead.
#[allow(clippy::too_many_arguments)]
fn bidi_pair_forward<M: Block>(
    straight_norm: &RmsNorm,
    reverse_norm: &RmsNorm,
    straight_block: &M,
    reverse_block: &M,
    output_merge: &OutputMerge,
    x: Tensor<3>,
    straight_cache: M::Cache,
    reverse_cache: M::Cache,
    options: M::Options,
    padding: Option<&Padding>,
) -> (Tensor<3>, M::Cache, M::Cache)
where
    M::Options: Clone,
{
    let [batch, sequence, d_model] = x.dims();

    let (x, straight_cache, x_rev, reverse_cache) = match padding {
        None => {
            // x reads >x₀>x₁>…, and x_rev (flipped) reads the sequence in
            // reverse order.
            let x_rev = x.clone().flip([1]);
            let x = straight_norm.forward(x);
            let x_rev = reverse_norm.forward(x_rev);
            let (x, straight_cache) =
                straight_block.block_forward(x, straight_cache, options.clone(), None);
            let (x_rev, reverse_cache) = reverse_block.block_forward(x_rev, reverse_cache, options, None);
            // Re-align the output of the reversed pass.
            (x, straight_cache, x_rev.flip([1]), reverse_cache)
        }
        Some(padding) => {
            let x_rev = reverse_norm.forward(x.clone());
            let x = straight_norm.forward(x);
            let (x, straight_cache) = padding.in_row_order(x, |x, pad_bs| {
                straight_block.block_forward(x, straight_cache, options.clone(), Some(pad_bs))
            });
            let (x_rev, reverse_cache) = padding.reversed().in_row_order(x_rev, |x_rev, pad_bs| {
                reverse_block.block_forward(x_rev, reverse_cache, options, Some(pad_bs))
            });
            (x, straight_cache, x_rev, reverse_cache)
        }
    };
    assert_eq!([batch, sequence, d_model], x.dims());
    assert_eq!([batch, sequence, d_model], x_rev.dims());

    let merged = output_merge.forward(x, x_rev);
    (merged, straight_cache, reverse_cache)
}

/// A stack of bidirectional [`Layer`] pairs with an optional layer schedule.
/// One struct for every family. After each pair, the residual join of
/// [`Self::residuals`] combines the merged output with the residual streams.
/// The stack has no gradient horizon, so no pair runs in an untracked span.
#[derive(Module, Debug)]
pub struct BidiLayers<M: Module> {
    /// Number of real (weight-bearing) layers. Must be even (used in pairs).
    pub n_real_layers: usize,
    /// Optional layer schedule `(n_virtual_layers, schedule)`, for
    /// weight-sharing.
    #[module(skip)]
    pub n_virtual_layers: Option<(usize, BidiSchedule)>,
    /// The weight-bearing layers, length `n_real_layers`.
    pub real_layers: Vec<Layer<M>>,
    /// Zero the residual of the first virtual pair when `true`.
    pub ignore_first_residual: bool,
    /// Zero the residual of the last virtual pair when `true`.
    pub ignore_last_residual: bool,
    /// One merge per real pair, length `n_real_layers / 2`.
    pub outputs_merge: Vec<OutputMerge>,
    /// The residual scheme (plain additive or Multi-Gate). Its residual join
    /// runs after each **pair**. The MGR unit is the pair: one module per
    /// real/virtual pair.
    pub residuals: Residuals,
    /// Positions of the stack-level class latents, spliced into the sequence
    /// once before the first pair.
    #[module(skip)]
    pub class_latents: Vec<ClassLatent>,
    /// The stack-level class-latent embeddings, `[num_class_latents, d_model]`.
    pub class_latents_emb: Option<Param<Tensor<2>>>,
}

impl<M: Block + Clone> BidiLayers<M>
where
    M::Options: Clone,
{
    /// Output positions of the stack-level class latents for an `orig_len` input.
    ///
    /// A marker that never lands (a `Custom` at or past the end) reports a
    /// position past the emitted sequence. Compare it against the sequence
    /// length.
    pub fn class_latent_output_indices(&self, orig_len: usize) -> Vec<usize> {
        class_marker_output_indices(&self.class_latents, orig_len)
    }

    /// Splice the class latents of this stack into the window `x` (no-op when
    /// there are none), and advance the stack-level cursor.
    fn insert_latents(
        &self,
        x: Tensor<3>,
        padding: Option<Padding>,
        class: &mut ClassCursors,
    ) -> (Tensor<3>, Option<Padding>) {
        let mut cursor = ClassCursor::at(class.stack, class.full_len);
        let out = insert_class_markers_padded(
            x,
            padding,
            &self.class_latents,
            self.class_latents_emb.as_ref(),
            &mut cursor,
            "BidiLayers",
        );
        class.stack = cursor.offset;
        out
    }

    /// `[batch, sequence, d_model]` → `[batch, sequence, d_model]`
    /// (`sequence` grows by the stack-level class-latent count).
    ///
    /// Each pair returns its merged transform `F_l` (no residual). Then the
    /// residual join of [`Self::residuals`] runs, with the pair as its unit
    /// (see [`crate::modules::residuals`]):
    ///
    /// - [`Residuals::Standard`]: the join adds the input skip per pair (unless
    ///   it is suppressed).
    /// - [`Residuals::MultiGate`]: no skip. Up to `n_stream` residual streams
    ///   carry the residual between pairs, and `x` seeds the first one. Each
    ///   pair reads their attention-pooled aggregate as input. Its merged
    ///   output either *becomes* a new stream (while fewer than `n_stream`
    ///   exist) or is gated into every stream (see [`MultiGate`]).
    ///
    /// `pad` (`[batch, sequence]`, `true` at padding, `None` ⇒ no padding)
    /// marks a right-padded batch. The output of each row equals the output of
    /// its own sequence alone. Both passes read only its real tokens. The
    /// reversed pass starts from the last real token of the row, not from the
    /// last position of the batch (see [`crate::utils::padding`]).
    ///
    /// [`MultiGate`]: crate::modules::MultiGate
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<Vec<M::Cache>>,
        options: M::Options,
        class: Option<&mut ClassCursors>,
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, Vec<M::Cache>) {
        // No cursors ⇒ this one call covers the whole sequence.
        let mut whole = ClassCursors::new(x.dims()[1]);
        let (x, padding) =
            self.insert_latents(x, pad.map(Padding::new), class.unwrap_or(&mut whole));
        let n = self
            .n_virtual_layers
            .as_ref()
            .map(|(l, _)| {
                assert!(l.is_multiple_of(2), "Bidi uses its virtual layers in pairs");
                *l
            })
            .unwrap_or_else(|| {
                assert!(
                    self.n_real_layers.is_multiple_of(2),
                    "Bidi uses its layers in pairs"
                );
                self.n_real_layers
            });

        let caches = caches
            .unwrap_or_else(|| zero_caches_for(&self.real_layers[0], n, x.dims()[0], &x.device()));
        assert_eq!(
            caches.len(),
            n,
            "straight and reverse layers cannot share caches"
        );
        let apps = bidi_applications(&self.n_virtual_layers, self.n_real_layers);

        // The caches of pair `i` are those of the virtual layers `2i` and
        // `2i + 1`, so the pairs take the caches in order.
        let mut caches = caches.into_iter();
        let mut advanced = Vec::with_capacity(n);
        let mut value = self.residuals.seed::<3, 4>(x);
        for i in 0..n / 2 {
            let (straight_i, reverse_i) = (i * 2, i * 2 + 1);
            let (straight_idx, reverse_idx) =
                if let Some((n_virtual, schedule)) = &self.n_virtual_layers {
                    (
                        schedule.real_idx(straight_i, *n_virtual, self.n_real_layers),
                        schedule.real_idx(reverse_i, *n_virtual, self.n_real_layers),
                    )
                } else {
                    (straight_i, reverse_i)
                };
            // Each pass as its own application of its real layer sees it, if
            // that layer unties anything.
            let straight_layer = self.real_layers[straight_idx].application(apps.index[straight_i]);
            let reverse_layer = self.real_layers[reverse_idx].application(apps.index[reverse_i]);

            let straight_cache = caches.next().expect("one cache per virtual layer");
            let reverse_cache = caches.next().expect("one cache per virtual layer");

            // The MGR unit is the pair: virtual pair `i`, real pair
            // `straight_idx / 2` (the straight index of a pair is even).
            let at = JoinAt {
                virtual_idx: i,
                real_idx: straight_idx / 2,
                first: self.ignore_first_residual && i == 0,
                last: self.ignore_last_residual && i + 1 == n / 2,
            };

            // Run the pair directly on the layers above. The stack-level class
            // latents are already spliced.
            //
            // The pair returns its merged transform `F_l` without the residual.
            // Each real pair has its own merge (`n_real_layers / 2` of them).
            // So the *real* pair `straight_idx / 2` indexes it, not the virtual
            // pair `i`. It shares weights under a layer schedule, like the
            // blocks (and like the MGR module of a real pair). In the
            // non-virtual case `straight_idx == i * 2`, so this is `i`.
            let (next, (sc, rc)) = self.residuals.join(value, at, |x| {
                let (merged, sc, rc) = bidi_pair_forward(
                    &straight_layer.norm,
                    &reverse_layer.norm,
                    &straight_layer.block,
                    &reverse_layer.block,
                    &self.outputs_merge[straight_idx / 2],
                    x,
                    straight_cache,
                    reverse_cache,
                    options.clone(),
                    padding.as_ref(),
                );
                (merged, (sc, rc))
            });
            value = next;
            advanced.push(sc);
            advanced.push(rc);
        }

        (value.into_token(), advanced)
    }
}

/// Plain (non-serde) factory for [`BidiLayers`].
pub struct BidiLayersBuilder<C> {
    /// Number of real (weight-bearing) layers (must be even).
    pub n_real_layers: usize,
    /// Optional layer schedule `(n_virtual_layers, schedule)`.
    pub n_virtual_layers: Option<(usize, BidiSchedule)>,
    /// Shared block config.
    pub block: C,
    /// Zero the residual of the first virtual pair.
    pub ignore_first_residual: bool,
    /// Zero the residual of the last virtual pair.
    pub ignore_last_residual: bool,
    /// One merge config per real pair, length `n_real_layers / 2`.
    pub outputs_merge: Vec<OutputMergeConfig>,
    /// Stack-level class latents (spliced once before the first pair).
    pub class_latents: Vec<ClassLatent>,
    /// Inter-pair residual scheme (defaults to plain additive).
    pub residuals: ResidualsConfig,
    /// The parameters of the layers themselves that they hold once per
    /// application, not tied (see [`Layer::application`]). The block config
    /// names those of the block.
    pub untied: Vec<LayerUntied>,
}

impl<C: BlockConfig> BidiLayersBuilder<C> {
    /// Allocate and initialise the bidirectional stack on `device`.
    pub fn init(&self, device: &Device) -> BidiLayers<C::Block> {
        let d_model = self.block.d_model();
        let real_layers = bidi_applications(&self.n_virtual_layers, self.n_real_layers)
            .count
            .into_iter()
            .map(|n_applications| {
                // The bidirectional stack has no feed-forward interleave.
                Layer::init(
                    self.block.init_block(device),
                    d_model,
                    None,
                    &self.untied,
                    n_applications,
                    device,
                )
            })
            .collect();
        let outputs_merge = (0..self.n_real_layers / 2)
            .map(|i| self.outputs_merge[i].init(d_model, device))
            .collect();
        // The MGR unit is the pair, so size the modules by *pairs* (half the
        // real and virtual layer counts).
        let n_virtual = self
            .n_virtual_layers
            .as_ref()
            .map(|(l, _)| *l)
            .unwrap_or(self.n_real_layers);
        let residuals = self
            .residuals
            .init(d_model, self.n_real_layers / 2, n_virtual / 2, device);
        BidiLayers {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            real_layers,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
            outputs_merge,
            residuals,
            class_latents_emb: init_class_emb(self.class_latents.len(), d_model, device),
            class_latents: self.class_latents.clone(),
        }
    }
}

/// The [`Applications`] of a bidirectional stack of `n_real_layers` under this
/// optional layer schedule (none ⇒ each real layer applied once).
fn bidi_applications(
    n_virtual_layers: &Option<(usize, BidiSchedule)>,
    n_real_layers: usize,
) -> Applications {
    match n_virtual_layers {
        Some((n, schedule)) => schedule.applications(*n, n_real_layers),
        None => Applications::new(0..n_real_layers, n_real_layers),
    }
}
