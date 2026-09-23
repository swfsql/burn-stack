//! Bidirectional support: family-generic, forward-only, non-autoregressive.
//!
//! A pair runs a straight (→) and a reversed (← through `flip`) Pre-LN pass,
//! and merges them with an [`OutputMerge`]. [`BidiLayers`] stacks such pairs
//! of its own [`Layer`]s with a
//! [`BidiSchedule`](crate::utils::BidiSchedule). [`BidiLayerPair`] is one pair
//! as a standalone module. The block itself does not change: only the
//! schedule and the combination of its two passes are bidirectional. This is
//! written once for all families. The merge is family-agnostic
//! (`RmsNorm`/`Linear` over `Tensor<3>`).

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

/// How the two directions of a bidirectional pair are combined.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum OutputMerge {
    /// Element-wise average of the two directions (no parameters).
    Mean(NoOp),
    /// Concatenate along the feature axis and project back down with a learnable
    /// `[2 · d_model, d_model]` linear layer.
    CatLinear(Linear),
}

impl OutputMerge {
    /// Merge the two directional outputs (each `[batch, sequence, d_model]`).
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
/// reversed (←) Pre-LN block, with merged outputs. It does **not** apply the
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
    /// Merge strategy combining the two directions.
    pub output_merge: OutputMerge,
    /// Positions of the class latents of this pair, spliced in before either
    /// direction runs (both directions, and the residual, see the longer
    /// sequence). Empty ⇒ none.
    #[module(skip)]
    pub class_latents: Vec<ClassLatent>,
    /// The class-latent embeddings of this pair, `[num_class_latents, d_model]`.
    pub class_latents_emb: Option<Param<Tensor<2>>>,
}

impl<M: Block> BidiLayerPair<M>
where
    M::Options: Clone,
{
    /// Splice the class latents of this pair into the chunk `x` (no-op when
    /// there are none), and advance `class` past the chunk. `padding` follows
    /// the splice. `None` ⇒ this chunk is the whole sequence.
    fn insert_latents(
        &self,
        x: Tensor<3>,
        padding: Option<Padding>,
        class: Option<&mut ClassCursor>,
    ) -> (Tensor<3>, Option<Padding>) {
        let mut whole = ClassCursor::whole(x.dims()[1]);
        let cursor = class.unwrap_or(&mut whole);
        insert_class_markers_padded(
            x,
            padding,
            &self.class_latents,
            self.class_latents_emb.as_ref(),
            cursor,
            "BidiLayerPair",
        )
    }

    /// `[batch, sequence, d_model]` → `[batch, sequence, d_model]`, plus the two
    /// updated direction caches. (`sequence` grows by the class-latent count.)
    /// Returns the merged directions **without** the residual: the caller adds
    /// it.
    ///
    /// `pad` marks a right-padded batch (`None` ⇒ none), as in
    /// [`BidiLayers::forward`].
    pub fn forward(
        &self,
        x: Tensor<3>,
        straight_cache: Option<M::Cache>,
        reverse_cache: Option<M::Cache>,
        options: M::Options,
        class: Option<&mut ClassCursor>,
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, M::Cache, M::Cache) {
        let (x, padding) = self.insert_latents(x, pad.map(Padding::new), class);
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

/// The straight + reverse + merge computation of a bidirectional pair, over
/// **borrowed** sub-modules.
///
/// [`BidiLayers`] calls this directly on its real layers (or their application
/// views, when they untie anything). It does not build a transient
/// [`BidiLayerPair`].
///
/// Under `padding`, the reversed read is the **real** rows of each slot
/// backwards ([`Padding::reversed`]), not the flipped batch, whose padding
/// would then lead.
#[allow(clippy::too_many_arguments)]
fn bidi_pair_forward<M: Block>(
    straight_norm: &RmsNorm,
    reverse_norm: &RmsNorm,
    straight_block: &M,
    reverse_block: &M,
    output_merge: &OutputMerge,
    x: Tensor<3>,
    straight_cache: Option<M::Cache>,
    reverse_cache: Option<M::Cache>,
    options: M::Options,
    padding: Option<&Padding>,
) -> (Tensor<3>, M::Cache, M::Cache)
where
    M::Options: Clone,
{
    let [batch, sequence, d_model] = x.dims();

    let (x, straight_cache, x_rev, reverse_cache) = match padding {
        None => {
            // x reads >x₀>x₁>…, and x_rev (flipped) reads the sequence
            // backwards.
            let x_rev = x.clone().flip([1]);
            let x = straight_norm.forward(x);
            let x_rev = reverse_norm.forward(x_rev);
            let (x, straight_cache) =
                straight_block.block_forward(x, straight_cache, options.clone(), None);
            let (x_rev, reverse_cache) = reverse_block.block_forward(x_rev, reverse_cache, options, None);
            // Re-align the reversed read.
            (x, straight_cache, x_rev.flip([1]), reverse_cache)
        }
        Some(padding) => {
            let x_rev = reverse_norm.forward(x.clone());
            let x = straight_norm.forward(x);
            let (x, straight_cache) = padding.in_slot_order(x, |x, pad_bs| {
                straight_block.block_forward(x, straight_cache, options.clone(), Some(pad_bs))
            });
            let (x_rev, reverse_cache) = padding.reversed().in_slot_order(x_rev, |x_rev, pad_bs| {
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

/// A stack of bidirectional [`Layer`] pairs with optional virtual-layer
/// scheduling: one struct for every [`Block`] family.
#[derive(Module, Debug)]
pub struct BidiLayers<M: Module> {
    /// Number of real (weight-bearing) layers. Must be even (used in pairs).
    pub n_real_layers: usize,
    /// Optional `(n_virtual_layers, schedule)` for weight-sharing.
    #[module(skip)]
    pub n_virtual_layers: Option<(usize, BidiSchedule)>,
    /// The weight-bearing layers, length `n_real_layers`.
    pub real_layers: Vec<Layer<M>>,
    /// Zero the residual of the first virtual pair when `true`.
    pub ignore_first_residual: bool,
    /// Zero the residual of the last virtual pair when `true`.
    pub ignore_last_residual: bool,
    /// One direction-merge per real pair, length `n_real_layers / 2`.
    pub outputs_merge: Vec<OutputMerge>,
    /// How residuals are threaded between **pairs** (plain additive vs
    /// Multi-Gate). The MGR unit is the pair: one module per real/virtual pair.
    pub residuals: Residuals,
    /// Positions of the stack-level class latents, spliced into the sequence
    /// once before the first pair (independent of any per-pair class latents).
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

    /// Splice the class latents of this stack into the chunk `x` (no-op when
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

    /// Seed the MultiGate streams from a full-sequence input: the **single**
    /// stream `x` as `[batch, sequence, 1, d_model]`. The first pairs widen it
    /// to `n_stream` (see [`MultiGate`](crate::modules::MultiGate)). Returns
    /// `None` for the Standard path. `x` already carries the stack-level class
    /// latents (spliced before the seed), so they seed the streams like any
    /// other token.
    fn multi_gate_streams_seed(&self, x: &Tensor<3>) -> Option<Tensor<4>> {
        matches!(&self.residuals, Residuals::MultiGate(_)).then(|| x.clone().unsqueeze_dim::<4>(2))
    }

    /// `[batch, sequence, d_model]` → `[batch, sequence, d_model]`
    /// (`sequence` grows by the stack-level class-latent count).
    ///
    /// Each pair returns its merged transform `F_l` (no residual). Then:
    ///
    /// - [`Residuals::Standard`]: the input skip is added per pair (unless it
    ///   is suppressed).
    /// - [`Residuals::MultiGate`]: no skip. Up to `n_stream` parallel streams
    ///   carry the residual between pairs, and `x` seeds the first one. Each
    ///   pair reads their attention-pooled aggregate as input. Its merged
    ///   output either *becomes* a new stream (while fewer than `n_stream`
    ///   exist) or is gated into every stream (see [`MultiGate`]).
    ///
    /// `pad` (`[batch, sequence]`, `true` at padding, `None` ⇒ none) marks a
    /// right-padded batch. Each slot comes out as its own sequence would come
    /// out alone. Both directions read only its real rows, and the reversed
    /// direction starts from the last row of the slot, not of the batch (see
    /// [`crate::utils::padding`]).
    ///
    /// [`MultiGate`]: crate::modules::MultiGate
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
        let (mut x, padding) =
            self.insert_latents(x, pad.map(Padding::new), class.unwrap_or(&mut whole));
        let n = self
            .n_virtual_layers
            .as_ref()
            .map(|(l, _)| {
                assert!(l.is_multiple_of(2), "Bidi virtual layers are used in pairs");
                *l
            })
            .unwrap_or_else(|| {
                assert!(
                    self.n_real_layers.is_multiple_of(2),
                    "Bidi layers are used in pairs"
                );
                self.n_real_layers
            });

        // Sized from a view, because the tensors of an untied block hold the
        // copy of every application.
        let caches = caches
            .unwrap_or_else(|| self.real_layers[0].application(0).block.zero_caches_3d(&x, n));
        assert_eq!(
            caches.slot_count(),
            n,
            "straight and reverse layers cannot share caches"
        );
        let apps = bidi_applications(&self.n_virtual_layers, self.n_real_layers);

        let mut slots = caches.into_slots();
        // MultiGate carries up to `n_stream` parallel streams: the input is the
        // first, and the early pairs append the rest. Standard threads the
        // single tensor `x` directly, and `streams` stays `None`.
        let mut streams = self.multi_gate_streams_seed(&x);
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
            // Each direction as its own application of its real layer sees it,
            // if that layer unties anything.
            let straight_layer = self.real_layers[straight_idx].application(apps.index[straight_i]);
            let reverse_layer = self.real_layers[reverse_idx].application(apps.index[reverse_i]);

            let straight_cache = slots[straight_i].take().unwrap();
            let reverse_cache = slots[reverse_i].take().unwrap();

            let first = self.ignore_first_residual && i == 0;
            let last = self.ignore_last_residual && i + 1 == n / 2;

            // For the Standard path, the residual is the (pre-pair) input skip.
            // Clone it before the pair consumes `x`, and only when it is used.
            // MultiGate carries the residual in its streams, so it clones
            // nothing.
            let residual = match &self.residuals {
                Residuals::Standard(_) if !(first || last) => Some(x.clone()),
                _ => None,
            };

            // Run the pair directly on the layers above. The stack-level class
            // latents are already spliced. The pairs carry none of their own.
            //
            // The pair returns its merged transform `F_l` without the residual.
            // The merge is a per-real-pair weight set (`n_real_layers / 2` of
            // them). So the *real* pair `straight_idx / 2` indexes it, not the
            // virtual pair `i`. It shares weights under virtual scheduling,
            // like the blocks (and like the MGR real-pair index below). In the
            // non-virtual case `straight_idx == i * 2`, so this is `i`.
            let (merged, sc, rc) = bidi_pair_forward(
                &straight_layer.norm,
                &reverse_layer.norm,
                &straight_layer.block,
                &reverse_layer.block,
                &self.outputs_merge[straight_idx / 2],
                x,
                Some(straight_cache),
                Some(reverse_cache),
                options.clone(),
                padding.as_ref(),
            );
            slots[straight_i] = Some(sc);
            slots[reverse_i] = Some(rc);

            match &self.residuals {
                Residuals::Standard(_noop) => {
                    // Add the input skip here (the pair already consumed `x`).
                    // When the residual is suppressed, output the bare
                    // transform.
                    x = match residual {
                        Some(r) => merged + r,
                        None => merged,
                    };
                }
                Residuals::MultiGate(mg) => {
                    let s = streams.take().unwrap();
                    // A skipped residual is β ≡ 1 in the mixer (`new_streams =
                    // F_l`), and the aggregator then collapses to `F_l`. Both
                    // branches take a shortcut to that result (as in
                    // `Layers::forward`). The MGR unit is the pair: virtual pair
                    // `i`, real pair `straight_idx / 2` (the straight index of a
                    // pair is even).
                    if last {
                        x = merged;
                        streams = Some(s);
                    } else if first {
                        // Drop the input seed: restart from `F_0` alone (the
                        // accumulation phase refills the streams).
                        streams = Some(merged.clone().unsqueeze_dim::<4>(2));
                        x = merged;
                    } else {
                        let mgr = &mg.layers[mg.module_index(i, straight_idx / 2)];
                        // Accumulate `F_l` as a new stream while there is room,
                        // then switch to gated mixing.
                        let (new_h, new_streams) = if s.dims()[2] < mg.n_stream {
                            mgr.accumulate(merged, s)
                        } else {
                            mgr.forward(merged, s)
                        };
                        x = new_h;
                        streams = Some(new_streams);
                    }
                }
            }
        }

        (x, M::Caches::from_slots(slots))
    }
}

/// Plain (non-serde) factory for [`BidiLayers`].
pub struct BidiLayersBuilder<C> {
    /// Number of real (weight-bearing) layers (must be even).
    pub n_real_layers: usize,
    /// Optional virtual-layer scheduling.
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
    /// The parameters of the layers themselves that are held once per
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
                    self.block.init_block(n_applications, device),
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
/// optional virtual scheduling (none ⇒ each real layer applied once).
fn bidi_applications(
    n_virtual_layers: &Option<(usize, BidiSchedule)>,
    n_real_layers: usize,
) -> Applications {
    match n_virtual_layers {
        Some((n, schedule)) => schedule.applications(*n, n_real_layers),
        None => Applications::new(0..n_real_layers, n_real_layers),
    }
}
