use crate::modules::{GatedMlpConfig, LayerUntied, Residuals, ResidualsConfig};
use crate::prelude::*;
use crate::utils::{Applications, GradHorizon, Schedule};
use crate::utils::class::{
    assert_full_len_known, class_chunk_plan, class_emb_table, class_emb_width,
    class_marker_output_indices, class_prime_plan, class_row, init_class_emb,
    insert_class_markers_padded, landing_count, splice_class_rows,
};
use crate::utils::{ClassCursor, ClassCursors, ClassLatent, Packed, Padding};
use burn::module::Param;
use burn::prelude::*;

/// A stack of [`Layer`]s with optional virtual-layer scheduling — one struct for
/// every [`Block`] family.
#[derive(Module, Debug)]
pub struct Layers<M: Module> {
    /// Number of real (weight-bearing) layers.
    pub n_real_layers: usize,
    /// Optional `(n_virtual_layers, schedule)` for weight-sharing.
    #[module(skip)]
    pub n_virtual_layers: Option<(usize, Schedule)>,
    /// The weight-bearing layers, length `n_real_layers`.
    pub real_layers: Vec<Layer<M>>,
    /// Zero the first virtual layer's residual when `true`.
    pub ignore_first_residual: bool,
    /// Zero the last virtual layer's residual when `true`.
    pub ignore_last_residual: bool,
    /// How residuals are threaded between layers (plain additive vs Multi-Gate).
    pub residuals: Residuals,
    /// Positions of the stack-level class latents, spliced into the sequence
    /// once before the first virtual layer (independent of any per-[`Layer`]
    /// class latents). Empty ⇒ none.
    #[module(skip)]
    pub class_latents: Vec<ClassLatent>,
    /// The stack-level class-latent embeddings, `[num_class_latents, d_model]`.
    pub class_latents_emb: Option<Param<Tensor<2>>>,
    /// Back-propagate only **some** of the (virtual) layers. The other layers
    /// run without an autodiff graph. `None` (the default) tracks the whole
    /// stack.
    ///
    /// This is the truncated-BPTT knob of TRM/HRM-style deep recursion. When
    /// `n_virtual_layers` is far above `n_real_layers`, the tracking of every
    /// pass is what runs out of memory. Both papers back-propagate only a
    /// suffix: TRM one full recursion, HRM-Text a horizon `K` warmed from 2 to
    /// 5. The horizon counts **from the top**, so it keeps its meaning when the
    /// stack depth changes, and a training loop can move it per step.
    ///
    /// [`GradHorizon::Depth`]`(K)` counts `K` from the top of every **real**
    /// layer (its last `K` applications), not from the top of the stack. The
    /// [`Schedule`] decides which virtual layers these are:
    ///
    /// - [`Schedule::Cyclic`] spreads the applications of a real layer evenly.
    ///   They are the top `K · n_real_layers` virtual layers, and the stack
    ///   cuts **once**.
    /// - [`Schedule::Stretched`] gives each real layer one contiguous run. They
    ///   are the tail of each run, and the stack cuts and lifts back once **per
    ///   real layer**. A plain suffix would leave the lower real layers of a
    ///   stretched stack with no tracked application, so they would silently
    ///   not train.
    ///
    /// [`GradHorizon::Mask`] states the tracked layers directly, one flag per
    /// virtual layer. [`Schedule::Custom`] takes it, because it has no
    /// canonical run of its own. [`GradHorizon::last`] builds the plain suffix.
    ///
    /// A stack **without** weight sharing applies each real layer exactly
    /// once, so any `Depth(K >= 1)` tracks all of it. To cut such a stack, use
    /// `GradHorizon::last(K, n)`.
    ///
    /// Under weight sharing, the same real layer serves both sides of a cut.
    /// The untracked segments run an inner-backend copy, so each weight still
    /// receives gradient, but only from its tracked applications.
    ///
    /// The stack **input** is the exception, on purpose. It enters at the
    /// bottom and rides the residual stream upward. A cut would sever its only
    /// path, and the `in_proj` of a network (or the embedding of a vocab net)
    /// would silently never train. TRM and HRM do not have this problem,
    /// because they re-inject the input at every recursion. This stack reads
    /// it once.
    ///
    /// So every boundary re-attaches the input *straight-through*: a
    /// value-zero term restores an identity gradient path. Under
    /// [`Residuals::Standard`] this is not a guess. It is the exact leading
    /// term of `∂(x + Σ F_l)/∂x`, and the rest is the segment that the cut
    /// chose not to differentiate. The carry holds what enters an untracked
    /// segment, not only the stack input. So the tracked layers *below* a cut
    /// keep their gradient path to the layers above it.
    ///
    /// Under [`MultiGate`](crate::modules::MultiGate), the residual lives in
    /// the depth-streams, not in the token. So **every** carrier gets the
    /// identity path:
    ///
    /// - At the bottom of the stack, this is exact. The seed stream is the
    ///   input and the pool is convex, so an identity segment leaves all `k`
    ///   streams equal to the input. A correction of only the pooled token
    ///   would leave the contribution of the streams out of the gradient of
    ///   the input. Under the carry-biased gate init that MGR is built for,
    ///   that contribution is most of the gradient.
    /// - A cut that opens *inside* the stack (a stretched schedule takes one
    ///   per real layer) has streams that already differ. It carries the
    ///   pooled token into each of them. A carry per stream would be the exact
    ///   identity. But a segment also *widens* the streams as it accumulates,
    ///   so no stream is there to pair a carry with. Every stream still
    ///   receives gradient, routed by the convex aggregation that produced the
    ///   token, not one-to-one.
    ///
    /// In every case, the values do not change.
    ///
    /// **Every class embedding trains**, at all three levels and on both sides
    /// of a cut:
    ///
    /// - The [`ClassToken`]s of a network and the [`ClassLatent`]s of this
    ///   stack ride the carry, because the carry is taken *after* they are
    ///   spliced.
    /// - A per-[`Layer`] latent inside an untracked segment gets a **ghost**
    ///   row in the carry (value zero, taken from the tracked table).
    ///
    /// They are learnable *input rows*, not part of the transform of a layer
    /// (the transform stays undifferentiated below the cut). Anything else
    /// would leave a silently dead parameter.
    ///
    /// [`ClassToken`]: crate::utils::ClassToken
    /// [`ClassLatent`]: crate::utils::ClassLatent
    ///
    /// A horizon that tracks every layer behaves exactly like `None`. So does
    /// any horizon off the autodiff backend. [`Self::forward`], [`Self::step`]
    /// and [`Self::prime`] all obey it, so a cut stack decodes under the same
    /// truncation that it trains under.
    #[module(skip)]
    pub grad_horizon: Option<GradHorizon>,
}

impl<M: Block> Layers<M>
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

    /// Whether every class latent that the stack splices (its own and those
    /// of each layer) is a `Start`. Then, after the opening of a sequence has
    /// run, no later [`step`](Self::step) emits one. So every step runs the
    /// same launches (a [`CapturedStep`](crate::utils::graph::CapturedStep)
    /// needs this), and a step with no cursors equals the step with them.
    pub fn only_start_latents(&self) -> bool {
        let start = |m: &ClassLatent| matches!(m, ClassLatent::Start);
        self.class_latents.iter().all(start)
            && self.real_layers.iter().all(|l| l.class_latents.iter().all(start))
    }

    /// Splice the class latents of this stack into the chunk `x` (no-op when
    /// there are none), and advance the stack-level cursor. `padding` follows
    /// the splice.
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
            "Layers",
        );
        class.stack = cursor.offset;
        out
    }

    /// Number of (virtual) layers this stack runs.
    pub fn n_virtual_count(&self) -> usize {
        self.n_virtual_layers
            .as_ref()
            .map(|(l, _)| *l)
            .unwrap_or(self.n_real_layers)
    }

    fn real_idx(&self, virtual_idx: usize) -> usize {
        if let Some((n, schedule)) = &self.n_virtual_layers {
            schedule.real_idx(virtual_idx, *n, self.n_real_layers)
        } else {
            virtual_idx
        }
    }

    /// For each virtual layer, which application of its real layer it is. An
    /// untied parameter reads the copy of that application (see
    /// [`Layer::application`]).
    fn applications(&self) -> Applications {
        stack_applications(&self.n_virtual_layers, self.n_real_layers)
    }

    /// Which of the `n` virtual layers back-propagate, per
    /// [`Self::grad_horizon`]. A `false` layer runs on the inner backend, and a
    /// `true` layer builds the graph. `None` ⇒ no cut anywhere.
    ///
    /// The mask can turn off and on again any number of times: once for a
    /// [`Schedule::Cyclic`] stack, once per real layer for a
    /// [`Schedule::Stretched`] stack, and arbitrarily for a
    /// [`GradHorizon::Mask`].
    ///
    /// Returns `None` off the autodiff backend. `Tensor::inner` and
    /// `Module::valid` are idempotent there, so a cut would buy nothing but its
    /// own round-trip. A horizon left set in a config thus falls through to
    /// the untouched path at inference. The device of the module decides,
    /// because [`Self::prime`] has no input tensor to ask.
    fn grad_tracked(&self, n: usize) -> Option<Vec<bool>> {
        let on_autodiff = self.real_layers[0]
            .norm
            .gamma
            .val()
            .device()
            .is_autodiff();
        if !on_autodiff {
            return None;
        }
        let schedule = self.n_virtual_layers.as_ref().map(|(_, s)| s);
        let tracked = self
            .grad_horizon
            .as_ref()?
            .tracked(schedule, n, self.n_real_layers);
        // Only its own application reads an untied copy. So an untracked copy
        // would never train.
        for (i, _) in tracked.iter().enumerate().filter(|(_, t)| !**t) {
            let real = self.real_idx(i);
            assert!(
                !self.real_layers[real].has_untied(),
                "grad_horizon leaves virtual layer {i} untracked, but its real layer {real} unties \
                 parameters across its applications. The copies of that application would never \
                 train. Track every application of a layer with untied weights.",
            );
        }
        // An all-tracked mask *is* the untouched stack, so take no cut. This
        // keeps a horizon deeper than the application count of every real
        // layer a no-op, down to the graph that it builds.
        tracked.iter().any(|t| !t).then_some(tracked)
    }

    /// Full-sequence pass through every (virtual) layer.
    ///
    /// [`Layer`] returns only its delta: `F_l = Block(RMSNorm(·))`, plus the
    /// contribution of the feed-forward sub-block when the layer has one. This
    /// function adds the outer residual:
    ///
    /// - [`Residuals::Standard`]: each layer adds the input skip (unless it is
    ///   suppressed).
    /// - [`Residuals::MultiGate`]: no skip. Up to `n_stream` parallel streams
    ///   carry the residual, and `x` seeds the first one. Each layer reads
    ///   their attention-pooled aggregate as input. Its output either
    ///   *becomes* a new stream (while fewer than `n_stream` exist) or is
    ///   gated into every stream (see [`MultiGate`]).
    ///
    /// `ignore_first/last_residual` apply to **both** paths:
    ///
    /// - A skipped first residual restarts the residual carry from the output
    ///   of the first layer. The layer reads the input but does not carry it.
    /// - A skipped last residual makes the stack output the transform `F_l` of
    ///   the last layer alone (no input-dependent carry).
    ///
    /// `class` places the stack-level and the per-layer class latents. `None`
    /// takes `x` as the whole sequence, so every latent lands in this call.
    /// Pass the same [`ClassCursors`] to consecutive chunks to split the
    /// sequence without a move of any latent (see [`ClassCursors`]).
    ///
    /// Both residual paths host the latents. A per-layer latent is spliced
    /// into the token sequence. Under MultiGate, it is also spliced into every
    /// carried stream. The aggregator over the resulting identical streams
    /// reproduces the row. So the layer above reads it back exactly as the
    /// additive skip passes it.
    ///
    /// `pad` (`[batch, sequence]`, `true` at padding, `None` ⇒ none) marks a
    /// right-padded batch. Every slot comes out as its own sequence would come
    /// out alone: the outputs of its real rows, and its caches. Every class
    /// latent is placed against the length of that slot (see
    /// [`crate::utils::padding`]). The rows still come back in the batch-wide
    /// order, with the class latents where the plan for the padded length
    /// puts them.
    ///
    /// [`MultiGate`]: crate::modules::MultiGate
    /// [`Layer`]: crate::modules::Layer
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<M::Caches>,
        options: M::Options,
        class: Option<&mut ClassCursors>,
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, M::Caches) {
        self.forward_padded(x, caches, options, class, pad.map(Padding::new))
    }

    /// [`Self::forward`] over packed rows (see [`crate::utils::packing`]). At
    /// each reset of `packed`, every layer restarts from a zero cache. The
    /// rows before the first reset of a slot continue from `caches`.
    ///
    /// The stack puts its class latents into the opening slots of each
    /// sequence, in place of the input rows. So the output has the shape of
    /// `x`. Every class latent of the stack must be a `Start`. A layer cannot
    /// have class latents of its own, because a splice would move the resets.
    /// The returned caches are those of the last sequence of each slot.
    pub fn forward_packed(
        &self,
        x: Tensor<3>,
        caches: Option<M::Caches>,
        options: M::Options,
        packed: &Packed,
    ) -> (Tensor<3>, M::Caches) {
        assert!(
            self.class_latents.iter().all(|m| matches!(m, ClassLatent::Start)),
            "a packed stack takes only `Start` class latents"
        );
        assert!(
            self.real_layers.iter().all(|l| l.class_latents.is_empty()),
            "a packed stack takes no class latents of a layer"
        );
        let packed = packed.inner();
        let x = match &self.class_latents_emb {
            Some(emb) if !self.class_latents.is_empty() => packed.place(x, emb.val()),
            _ => x,
        };
        self.forward_rows(x, caches, options, None, None, Some(&packed))
    }

    /// [`Self::forward`] with the padding already tracked. A container that
    /// splices its own markers below this stack calls this.
    pub(crate) fn forward_padded(
        &self,
        x: Tensor<3>,
        caches: Option<M::Caches>,
        options: M::Options,
        class: Option<&mut ClassCursors>,
        padding: Option<Padding>,
    ) -> (Tensor<3>, M::Caches) {
        self.forward_rows(x, caches, options, class, padding, None)
    }

    /// One layer of the stack, over padded rows or over packed rows.
    fn run_layer(
        layer: &Layer<M>,
        x: Tensor<3>,
        cache: M::Cache,
        options: M::Options,
        padding: Option<&Padding>,
        packed: Option<&Packed>,
    ) -> (Tensor<3>, M::Cache) {
        match packed {
            None => layer.forward(x, Some(cache), options, padding),
            Some(packed) => layer.forward_packed(x, Some(cache), options, packed),
        }
    }

    /// The loop of [`Self::forward_padded`] and [`Self::forward_packed`].
    /// `packed` rows take no padding, and their class latents are already in
    /// their opening slots.
    fn forward_rows(
        &self,
        x: Tensor<3>,
        caches: Option<M::Caches>,
        options: M::Options,
        class: Option<&mut ClassCursors>,
        padding: Option<Padding>,
        packed: Option<&Packed>,
    ) -> (Tensor<3>, M::Caches) {
        let n = self.n_virtual_count();
        // No cursors ⇒ this one call covers the whole sequence.
        let mut whole = ClassCursors::new(x.dims()[1]);
        let class = class.unwrap_or(&mut whole);
        class.fit(n);

        let (mut x, mut padding) = match packed {
            None => self.insert_latents(x, padding, class),
            Some(_) => {
                assert!(padding.is_none(), "packed rows take no padding");
                (x, None)
            }
        };
        // The latents of the stack make the sequence of the layers longer.
        // Each layer then makes it longer for the layers above it.
        let mut full = class
            .full_len
            .map(|l| l + landing_count(&self.class_latents, l));
        // Sized from a view, because the tensors of an untied block hold the
        // copy of every application.
        let caches = caches
            .unwrap_or_else(|| self.real_layers[0].application(0).block.zero_caches_3d(&x, n));
        assert_eq!(caches.slot_count(), n, "one cache per virtual layer");

        // An untracked layer must build no graph. In Burn, that means it runs
        // **off the autodiff backend**. A detach is not sufficient. It cuts the
        // gradient flow, but Burn still registers the untracked op in the
        // graph. Burn keeps an `UntrackedOpsStep` per op, so that a
        // memory-bound op can still retrieve an untracked parent. So the output
        // of the op stays retained. Measured on a 64-virtual-layer stack, a detached
        // prefix saved ~6% of peak memory and still scaled linearly with depth.
        // An inner-backend prefix was flat. A peak-memory probe against a real
        // block reproduces both curves.
        //
        // `Tensor::inner`/`Module::valid` are idempotent off the autodiff
        // backend, so a cut there would cost a round-trip and save nothing. The
        // stack takes a cut only on an autodiff backend. At inference,
        // `grad_horizon` does nothing.
        //
        // The mask is not a single boundary. It can turn off and on again any
        // number of times (once per real layer under `Schedule::Stretched`, see
        // `grad_horizon`). Each transition is a full hop of everything that the
        // loop carries.
        let tracked = self.grad_tracked(n);
        let inner_stack = tracked.is_some().then(|| Module::valid(self));
        let mut slots = caches.into_slots();
        // Straight-through carry (see `grad_horizon`): a value-**zero** tracked
        // tensor that stands in for what entered the current untracked
        // segment. The loop adds it back where the graph resumes, so
        // everything below keeps an identity gradient path across the segment.
        // It is `Some` exactly while inside such a segment. It must be added on
        // the autodiff side of the boundary. An earlier add would make it a
        // tracked input to an untracked layer: a backend mismatch, and the end
        // of the memory saving.
        //
        // It shadows the **shape** of `x`, not its value. The class latents of
        // an untracked layer make the sequence longer, and the carry takes
        // ghost rows at those same positions (see the splice below).
        let mut st: Option<Tensor<3>> = None;

        // MultiGate carries up to `n_stream` parallel streams: the input is the
        // first, and the early layers append the rest. Standard threads the
        // single tensor `x` directly, and `streams` stays `None`.
        let mut streams = self.multi_gate_streams_seed(&x);
        let apps = self.applications();

        // `i` is the virtual-layer index (schedule, cut boundary, residual
        // flags). The slot lookup is incidental. The bound stays `n`, so a
        // mis-sized cache stack still panics.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let track = tracked.as_ref().is_none_or(|t| t[i]);
            match (st.is_some(), track) {
                // Entering an untracked segment: everything that crosses into
                // it goes down to the inner backend. The tokens and their
                // MultiGate streams go here, the cache slot of this layer below.
                (false, false) => {
                    st = Some(x.clone() - x.clone().detach());
                    x = x.inner();
                    streams = streams.map(Tensor::inner);
                }
                // Leaving one: lift what it produced back onto the autodiff
                // backend, as fresh graph roots, and re-attach the carry.
                (true, true) => {
                    x = Tensor::from_inner(x);
                    streams = streams.map(Tensor::from_inner);
                    let st = st.take().expect("inside an untracked segment");
                    // Under MultiGate, the residual lives in the streams, not
                    // in the token. So *every* carrier gets the identity path:
                    // that is what "the segment behaved like identity" means
                    // there. Below the first layer, this is exact. The seed
                    // stream is the input and the pool is convex, so an
                    // identity segment leaves all `k` streams equal to the
                    // input. The pooled `x` is not counted twice: its route
                    // from the input runs through an aggregator whose softmax
                    // weights sum to one. At a boundary further up, the streams
                    // already differ, and the carry is that of the pooled token
                    // alone. Exactness would need one carry per stream, and a
                    // segment that accumulates new streams has nothing to pair
                    // it with. Either way, the gradient reaches every stream
                    // through that same convex aggregation. See `grad_horizon`.
                    streams = streams.map(|s| {
                        let dims = s.dims();
                        s + st.clone().unsqueeze_dim::<4>(2).expand(dims)
                    });
                    x = x + st;
                }
                _ => {}
            }
            let real = self.real_idx(i);
            // `self` on a tracked layer, the inner-backend copy on an untracked
            // one.
            let this = match &inner_stack {
                Some(d) if !track => d,
                _ => self,
            };
            // The layer as its own application sees it, if the layer unties
            // anything.
            let layer = this.real_layers[real].application(apps.index[i]);
            // The slot makes the same hop as its layer. A cache from a tracked
            // segment goes down with an untracked layer, because Burn cannot
            // mix backends within an op. It goes back up below.
            let cache = slots[i].take().unwrap();
            let cache = match track {
                true => cache,
                false => M::Caches::cache_to_inner(cache),
            };
            let first = self.ignore_first_residual && i == 0;
            let last = self.ignore_last_residual && i + 1 == n;

            // Splice the class latents of this layer into the sequence. Under
            // MultiGate, also splice them into every carried stream, because
            // the residual lives there. The (convex, all-scores-equal)
            // aggregator reproduces a row that is in all `k` streams exactly.
            // So the layer above reads the latent back just as the Standard
            // skip passes it.
            let mut cursor = ClassCursor::at(class.per_layer[i], full);
            let whole = cursor.covers_whole(x.dims()[1]);
            let plan = class_chunk_plan(&layer.class_latents, x.dims()[1], &mut cursor, "Layer");
            class.per_layer[i] = cursor.offset;
            full = full.map(|l| l + landing_count(&layer.class_latents, l));
            if !plan.is_empty() {
                padding = padding.map(|p| p.splice(&plan, &layer.class_latents, whole, "Layer"));
                let emb = class_emb_table(
                    &layer.class_latents,
                    layer.class_latents_emb.as_ref(),
                    x.dims()[2],
                );
                x = splice_class_rows(x, &plan, &emb);
                streams = streams.map(|s| splice_class_rows(s, &plan, &emb));
                // Keep the carry aligned with `x`: splice **ghost** rows at the
                // latent positions. A ghost row has value zero like the rest of
                // the carry, but it comes from the *tracked* table. So the class
                // latents of a prefix layer keep an identity gradient path,
                // exactly as the stack input does. They are learnable input
                // rows, not part of the transform of the layer. The transform
                // stays undifferentiated below the cut.
                st = st.map(|st| {
                    let tracked = class_emb_table(
                        &self.real_layers[real].class_latents,
                        self.real_layers[real].class_latents_emb.as_ref(),
                        emb.dims()[1],
                    );
                    let ghost = tracked.clone() - tracked.detach();
                    splice_class_rows(st, &plan, &ghost)
                });
            }

            match &this.residuals {
                Residuals::Standard(_noop) => {
                    // Add the residual (the lengthened input) here. When it is
                    // suppressed, move the input straight in (no clone, no add).
                    let x_l = x;
                    let (out, c_) = if first || last {
                        Self::run_layer(&layer, x_l, cache, options.clone(), padding.as_ref(), packed)
                    } else {
                        let (out, c_) = Self::run_layer(
                            &layer,
                            x_l.clone(),
                            cache,
                            options.clone(),
                            padding.as_ref(),
                            packed,
                        );
                        (out + x_l, c_)
                    };
                    x = out;
                    slots[i] = Some(c_);
                }
                Residuals::MultiGate(mg) => {
                    let (out, c_) =
                        Self::run_layer(&layer, x, cache, options.clone(), padding.as_ref(), packed);
                    slots[i] = Some(c_);
                    let s = streams.take().unwrap();
                    // A skipped residual here drops every carried stream. The
                    // MGR gets the same result when it forces the mixer gate to
                    // β ≡ 1 (`new_streams = out`): the aggregator over the
                    // resulting identical streams collapses to `F_l`. Both
                    // branches take a shortcut to that result.
                    if last {
                        // The output depends only on the transform of the last
                        // layer.
                        x = out;
                        streams = Some(s);
                    } else if first {
                        // Drop the input seed: restart the streams from `F_0`
                        // alone (the accumulation phase refills them).
                        streams = Some(out.clone().unsqueeze_dim::<4>(2));
                        x = out;
                    } else {
                        let mgr = &mg.layers[mg.module_index(i, real)];
                        // Accumulate `F_l` as a new stream while there is room,
                        // then switch to gated mixing (see `MultiGate`).
                        let (new_h, new_streams) = if s.dims()[2] < mg.n_stream {
                            mgr.accumulate(out, s)
                        } else {
                            mgr.forward(out, s)
                        };
                        x = new_h;
                        streams = Some(new_streams);
                    }
                }
            }
            if !track {
                slots[i] = slots[i].take().map(M::Caches::cache_from_inner);
            }
        }
        // The stack ended inside an untracked segment (its top layers were cut):
        // lift the output and re-attach the carry, exactly as the in-loop
        // boundary does. The streams are not returned, so they stay below.
        if let Some(st) = st.take() {
            x = Tensor::from_inner(x) + st;
        }
        (x, M::Caches::from_slots(slots))
    }

    /// Seed the MultiGate streams from a full-sequence input: the **single**
    /// stream `x` as `[batch, sequence, 1, d_model]`. The layers below
    /// `n_stream` widen it (see [`MultiGate`](crate::modules::MultiGate)).
    /// Returns `None` for the Standard path. `x` already carries the
    /// stack-level class latents, so they seed the streams like any other
    /// token.
    fn multi_gate_streams_seed(&self, x: &Tensor<3>) -> Option<Tensor<4>> {
        matches!(&self.residuals, Residuals::MultiGate(_)).then(|| x.clone().unsqueeze_dim::<4>(2))
    }

    /// Single-token step through every (virtual) layer.
    ///
    /// `class` drives two independent class-latent levels:
    ///
    /// - the stack-level [`Self::class_latents`] (`class.stack`), spliced once
    ///   below the first layer, exactly as in `forward`,
    /// - the per-[`Layer`] latents (`class.per_layer[i]`, one cursor per
    ///   virtual layer).
    ///
    /// The class latents of a layer make the sequence of the *next* layer
    /// longer (exactly as in `forward`). So a single user step is a
    /// **cascade**. The bottom input stream (the stack latents that fall on
    /// this step, plus the user token) goes up the stack, and each layer adds
    /// its own class latents to it. So the recurrence of every layer sees the
    /// same token order as in `forward`, and `forward` and `step` agree.
    ///
    /// The step returns the (fully propagated) output of the **last** token of
    /// that stream. This is the user token, unless an `End` latent follows it
    /// (the one kind that closes the sequence, not precedes a token). The
    /// latents emitted *before* the user token are stepped only for their
    /// effect on the state.
    ///
    /// `None` injects nothing at either level. `Middle`/`End` latents then
    /// panic, as they do without a [`ClassCursors::full_len`] hint.
    ///
    /// [`Self::grad_horizon`] applies here exactly as in [`Self::forward`], on
    /// the same virtual layers. So a stack decodes under the truncation that
    /// it trains under. Note: the cut rebuilds an inner-backend view of the
    /// stack once per call, which is once per *token* here, not per sequence.
    /// This is negligible against a training step. But do not leave it set
    /// for plain decoding. (There it does nothing anyway, because the model is
    /// then off the autodiff backend.)
    pub fn step(
        &self,
        x: Tensor<2>,
        caches: Option<M::Caches>,
        mut class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, M::Caches) {
        let [batch, d_model] = x.dims();
        let n = self.n_virtual_count();
        let caches = caches
            .unwrap_or_else(|| self.real_layers[0].application(0).block.zero_caches_2d(&x, n));
        assert_eq!(caches.slot_count(), n, "one cache per virtual layer");
        if let Some(c) = class.as_deref_mut() {
            c.fit(n);
        }
        let mut slots = caches.into_slots();

        // The bottom input stream for this user step: the stack-level class
        // latents that fall on it, around the user token. They go through the
        // whole stack like ordinary inputs. `at == 0` is before the token.
        // `at == 1` is after it (an `End` latent that closes the sequence, and
        // so ends the stream).
        let mut stream: Vec<Tensor<2>> = Vec::with_capacity(1);
        if let Some(class) = class.as_deref_mut() {
            let mut cursor = ClassCursor::at(class.stack, class.full_len);
            let plan = class_chunk_plan(&self.class_latents, 1, &mut cursor, "Layers");
            class.stack = cursor.offset;
            let row = |i: usize| class_row(self.class_latents_emb.as_ref(), i, batch, d_model);
            stream.extend(
                plan.iter()
                    .filter(|&&(at, _)| at == 0)
                    .map(|&(_, i)| row(i)),
            );
            stream.push(x);
            stream.extend(
                plan.iter()
                    .filter(|&&(at, _)| at == 1)
                    .map(|&(_, i)| row(i)),
            );
        } else {
            assert_full_len_known(&self.class_latents, None, "Layers");
            stream.push(x);
        }

        let mut stream = self.cascade(batch, stream, &mut slots, class, false);

        // The stream keeps the token order of `forward`. So its last element is
        // the latest token of the sequence: the user token, or an `End` after
        // it.
        let out = stream.pop().expect("the user token is always emitted");
        (out, M::Caches::from_slots(slots))
    }

    /// Thread `stream` (the tokens that enter the bottom layer) up through
    /// every (virtual) layer. Each layer splices its own class latents into
    /// what it receives, adds its residual, and gives the result to the next
    /// layer. Returns what leaves the top layer. `slots` holds the advanced
    /// caches.
    ///
    /// This is the shared body of [`Self::step`] and [`Self::prime`]. They
    /// differ in two things:
    ///
    /// - The stream that they open with: a user token among the stack
    ///   latents, or stack latents only.
    /// - What a layer emits at the **end** of the stream that it receives. An
    ///   ordinary step leaves those latents for the token that they precede
    ///   (only a closing `End` trails a token). A `prime` (`prime = true`)
    ///   emits the latents that are due before that next token. This is how
    ///   the cascade continues when the stream below it is empty.
    ///
    /// Everything else is common. This is why a `prime` and the `step` after
    /// it run the same sequence as that `step` alone.
    ///
    /// Under [`Residuals::MultiGate`], the residual is not in the token but in
    /// the depth-streams of that token. So each element of `stream` comes with
    /// its `[batch, k, d_model]` stream set. The set is rebuilt per token and
    /// never crosses steps, exactly as the `[batch, sequence, k, d_model]`
    /// streams of [`Self::forward`] are a per-position construct.
    fn cascade(
        &self,
        batch: usize,
        mut stream: Vec<Tensor<2>>,
        slots: &mut [Option<M::Cache>],
        mut class: Option<&mut ClassCursors>,
        prime: bool,
    ) -> Vec<Tensor<2>> {
        let n = slots.len();
        let has_mg = matches!(&self.residuals, Residuals::MultiGate(_));
        let apps = self.applications();

        // The same mask as in `forward`, layer for layer. An untracked virtual
        // layer runs on an inner-backend copy of the stack, so it builds no
        // graph (see `grad_horizon`). Everything that crosses into an untracked
        // segment goes down with it: the token stream, its MultiGate stream
        // sets, and the cache slots of those layers. The loop lifts them back
        // where the graph resumes, as many times as the mask alternates.
        let tracked = self.grad_tracked(n);
        let inner_stack = tracked.is_some().then(|| Module::valid(self));
        // The straight-through carry of `forward`, one entry per token of the
        // stream that enters the current untracked segment (see
        // `grad_horizon`). It is `Some` exactly while inside such a segment.
        // It can open empty. A `prime` whose cut layers receive no token has no
        // input to carry a gradient back to. It has only its own class latents,
        // and the ghost rows below cover them.
        let mut st: Option<Vec<Tensor<2>>> = None;

        // MultiGate: one stream set per token, seeded (like `forward`'s) with
        // the token itself as the single stream. Empty for the Standard path.
        let mut carried: Vec<Tensor<3>> = match has_mg {
            false => Vec::new(),
            true => stream
                .iter()
                .map(|t| t.clone().unsqueeze_dim::<3>(1))
                .collect(),
        };
        // The stream count that enters the current layer. It follows the depth
        // alone (`s.dims()[2]` in `forward`). So the loop tracks it also
        // across layers that no token reaches. A class latent that first
        // appears at layer `pos` must be seeded with exactly the `k` streams of
        // that depth.
        let mut k = 1usize;

        // The full length of the stream that the layers see, with the latents
        // of this stack. Each layer then makes it longer for the layers above
        // it.
        let mut full = class
            .as_deref()
            .and_then(|c| c.full_len)
            .map(|l| l + landing_count(&self.class_latents, l));
        // `pos` is the virtual-layer index (schedule, cut boundary). The slot
        // lookup is incidental.
        #[allow(clippy::needless_range_loop)]
        for pos in 0..n {
            // The boundaries that `forward` crosses, in the shape of a token
            // stream. They are before the empty-layer `continue` below only as
            // a defense: `carried` is empty exactly when `stream` is empty, so
            // a skipped layer has nothing to hop either way. But this keeps the
            // boundary independent of the skip condition.
            let track = tracked.as_ref().is_none_or(|t| t[pos]);
            match (st.is_some(), track) {
                // Entering an untracked segment: take the carry, then send the
                // stream (and its MultiGate stream sets) down.
                (false, false) => {
                    st = Some(
                        stream
                            .iter()
                            .map(|t| t.clone() - t.clone().detach())
                            .collect(),
                    );
                    stream = stream.into_iter().map(Tensor::inner).collect();
                    carried = carried.into_iter().map(Tensor::inner).collect();
                }
                // Leaving one: lift what it produced back onto the autodiff
                // backend, as fresh graph roots, and re-attach the carry.
                (true, true) => {
                    stream = stream.into_iter().map(Tensor::from_inner).collect();
                    carried = carried.into_iter().map(Tensor::from_inner).collect();
                    let st = st.take().expect("inside an untracked segment");
                    debug_assert_eq!(st.len(), stream.len(), "carry tracks the stream");
                    // Every carrier, as in `forward`. Under MultiGate, each
                    // token brings its own `[batch, k, d]` stream set and takes
                    // the carry of the pooled token into all of them.
                    if !carried.is_empty() {
                        debug_assert_eq!(carried.len(), st.len(), "one stream set per token");
                        carried = carried
                            .into_iter()
                            .zip(&st)
                            .map(|(c, s)| {
                                let dims = c.dims();
                                c + s.clone().unsqueeze_dim::<3>(1).expand(dims)
                            })
                            .collect();
                    }
                    stream = stream.into_iter().zip(st).map(|(t, s)| t + s).collect();
                }
                _ => {}
            }
            let real = self.real_idx(pos);
            // `self` on a tracked layer, the inner-backend copy on an untracked
            // one. This holds for layer weights, class-latent embeddings and
            // MultiGate gates.
            let this = match &inner_stack {
                Some(d) if !track => d,
                _ => self,
            };
            // The layer as its own application sees it, if the layer unties
            // anything.
            let layer = this.real_layers[real].application(apps.index[pos]);
            let mg = match &this.residuals {
                Residuals::Standard(_noop) => None,
                Residuals::MultiGate(mg) => Some(mg),
            };
            let first = self.ignore_first_residual && pos == 0;
            let last = self.ignore_last_residual && pos + 1 == n;
            let plan = if let Some(class) = class.as_deref_mut() {
                let mut cursor = ClassCursor::at(class.per_layer[pos], full);
                let markers = &layer.class_latents;
                let plan = if prime {
                    class_prime_plan(markers, stream.len(), &mut cursor, "Layer")
                } else {
                    class_chunk_plan(markers, stream.len(), &mut cursor, "Layer")
                };
                class.per_layer[pos] = cursor.offset;
                plan
            } else {
                // No cursors ⇒ nothing is injected. So `Middle`/`End` cannot be
                // placed at all, because their positions exist only against the
                // whole sequence. (With cursors, the plan above places every
                // kind. This is why the tokens below go through
                // `Layer::step_one`, not the cursorless `Layer::step`.)
                assert_full_len_known(&layer.class_latents, None, "Layer");
                Vec::new()
            };
            full = full.map(|l| l + landing_count(&layer.class_latents, l));
            // The stream count that this layer leaves, as in `forward`. A
            // suppressed last residual leaves the streams untouched. A
            // suppressed first residual restarts them from `F_0`. Otherwise the
            // accumulation phase appends one until `n_stream` is reached.
            let k_next = match mg {
                None => 1,
                Some(_) if last => k,
                Some(_) if first => 1,
                Some(mg) => (k + 1).min(mg.n_stream),
            };
            if plan.is_empty() && stream.is_empty() {
                k = k_next;
                continue; // nothing reaches this layer, and it adds nothing
            }

            // The slot makes the same hop as its layer (see `forward`).
            // `Layer::step_one` fills an empty slot on the backend of the
            // tokens.
            let mut cache = slots[pos].take();
            if !track {
                cache = cache.map(M::Caches::cache_to_inner);
            }
            let emitted = stream.len() + plan.len();
            let mut next: Vec<Tensor<2>> = Vec::with_capacity(emitted);
            let mut next_carried: Vec<Tensor<3>> = Vec::with_capacity(mg.map_or(0, |_| emitted));
            // One token through the layer, then its residual: the plain additive
            // skip, or the Multi-Gate mix into the streams of that token. A
            // suppressed skip moves the token straight in (no clone/add).
            let advance = |token: Tensor<2>,
                           tok_streams: Option<Tensor<3>>,
                           cache: Option<M::Cache>|
             -> (Tensor<2>, Option<Tensor<3>>, M::Cache) {
                let Some(mg) = mg else {
                    return if first || last {
                        let (out, c) = layer.step_one(token, cache);
                        (out, None, c)
                    } else {
                        let (out, c) = layer.step_one(token.clone(), cache);
                        (out + token, None, c)
                    };
                };
                let s = tok_streams.expect("MultiGate carries one stream set per token");
                let (out, c) = layer.step_one(token, cache);
                // As in `forward`, a skipped residual is β ≡ 1 in the mixer
                // (`new_streams = F_l`). The aggregator then collapses to `F_l`.
                if last {
                    (out, Some(s), c) // the output depends only on `F_l`
                } else if first {
                    // Drop the input seed: restart the streams from `F_0` alone.
                    (out.clone(), Some(out.unsqueeze_dim::<3>(1)), c)
                } else {
                    let mgr = &mg.layers[mg.module_index(pos, real)];
                    let (h, s) = if s.dims()[1] < mg.n_stream {
                        mgr.accumulate_step(out, s)
                    } else {
                        mgr.step(out, s)
                    };
                    (h, Some(s), c)
                }
            };
            // A class latent enters the token sequence *and* every stream (see
            // `forward`). Identical streams score alike. So the aggregator
            // reproduces the row, and the layer above reads it back unchanged.
            let row = |i: usize| {
                let emb = layer.class_latents_emb.as_ref();
                let width = class_emb_width(emb);
                let r = class_row(emb, i, batch, width);
                let s = mg.map(|_| r.clone().unsqueeze_dim::<3>(1).expand([batch, k, width]));
                (r, s)
            };
            // The carry comes along. It takes a **ghost** row wherever a class
            // latent is emitted. The row has value zero, so the carry stays
            // index-aligned with the output of this layer. But the row is
            // tracked, so the latent trains (see `grad_horizon`). The row is
            // built on demand, not cloned from the carry. So a layer whose
            // stream is empty still ghosts its own latents.
            let carry_active = st.is_some();
            let mut st_next: Vec<Tensor<2>> =
                Vec::with_capacity(if carry_active { emitted } else { 0 });
            let ghost = |i: usize| {
                let emb = self.real_layers[real].class_latents_emb.as_ref();
                let r = class_row(emb, i, batch, class_emb_width(emb));
                r.clone() - r.detach()
            };
            let mut push = |(out, s): (Tensor<2>, Option<Tensor<3>>), carry: Option<Tensor<2>>| {
                next.push(out);
                next_carried.extend(s);
                st_next.extend(carry);
            };
            let mut tokens_streams = carried.into_iter();
            let mut st_tokens = st.take().map(Vec::into_iter);
            let mut plan = plan.into_iter().peekable();
            for (t, token) in stream.into_iter().enumerate() {
                // This layer's class latents that fall before this token.
                while let Some((_, i)) = plan.next_if(|&(at, _)| at == t) {
                    let (r, rs) = row(i);
                    let (out, s, c) = advance(r, rs, cache);
                    push((out, s), carry_active.then(|| ghost(i)));
                    cache = Some(c);
                }
                let (out, s, c) = advance(token, tokens_streams.next(), cache);
                push((out, s), st_tokens.as_mut().and_then(Iterator::next));
                cache = Some(c);
            }
            // …and the latents after the last token of the stream: an `End`
            // that closes the sequence, or (on a prime) the latents due before
            // the next token.
            for (_at, i) in plan {
                let (r, rs) = row(i);
                let (out, s, c) = advance(r, rs, cache);
                push((out, s), carry_active.then(|| ghost(i)));
                cache = Some(c);
            }
            if carry_active {
                st = Some(st_next);
            }
            slots[pos] = match track {
                true => cache,
                false => cache.map(M::Caches::cache_from_inner),
            };
            stream = next;
            carried = next_carried;
            k = k_next;
        }
        // The stack ended inside an untracked segment: lift what it hands back
        // and re-attach the carry, exactly as the in-loop boundary does.
        if let Some(st) = st.take() {
            debug_assert_eq!(st.len(), stream.len(), "carry tracks the stream");
            stream = stream
                .into_iter()
                .map(Tensor::from_inner)
                .zip(st)
                .map(|(t, s)| t + s)
                .collect();
        }
        stream
    }

    /// Step the class latents that the stack has waiting for its next user
    /// token, with **no** user token. So the call consumes only class data.
    ///
    /// This is the opening half of [`Self::step`] on its own. The stack-level
    /// latents due now open the bottom stream (empty when there are none). The
    /// stream then goes up the stack through the same cascade that `step`
    /// runs. In addition, every layer flushes the latents that are due before
    /// *its* next token. So a `prime` followed by a `step` runs exactly the
    /// sequence that the `step` alone would run.
    ///
    /// `End` latents are never primed. They close the sequence, so they belong
    /// to the step that carries its last user token (this is why that step
    /// returns them). A cursor already at the announced end thus primes
    /// nothing.
    ///
    /// Returns the fully propagated output of the **last** latent emitted, or
    /// `None` when no latent was waiting. This is the entry point of seedless
    /// generation: `prime` → sample → `step` → sample → … `batch` sizes the
    /// latent rows, which are the only inputs.
    ///
    /// When nothing ran, the caches come back as they went in (`None`
    /// included). A partly primed stack gets zero caches for the layers that
    /// stepped nothing, which is exactly the state that they hold. `None`
    /// cursors inject nothing (`Middle`/`End` latents then panic, as in
    /// `step`).
    pub fn prime(
        &self,
        batch: usize,
        caches: Option<M::Caches>,
        class: Option<&mut ClassCursors>,
    ) -> (Option<Tensor<2>>, Option<M::Caches>) {
        let n = self.n_virtual_count();
        let Some(class) = class else {
            assert_full_len_known(&self.class_latents, None, "Layers");
            for layer in &self.real_layers {
                assert_full_len_known(&layer.class_latents, None, "Layer");
            }
            return (None, caches);
        };
        class.fit(n);
        let mut slots: Vec<Option<M::Cache>> = match caches {
            Some(caches) => {
                assert_eq!(caches.slot_count(), n, "one cache per virtual layer");
                caches.into_slots()
            }
            // Possibly nothing runs. Until something runs, no token exists to
            // size the zero caches. So start with empty slots.
            None => (0..n).map(|_| None).collect(),
        };

        // Bottom input stream for this prime: the stack-level class latents due
        // now, with no user token to accompany them (so possibly none at all).
        let mut cursor = ClassCursor::at(class.stack, class.full_len);
        let plan = class_prime_plan(&self.class_latents, 0, &mut cursor, "Layers");
        class.stack = cursor.offset;
        let stream: Vec<Tensor<2>> = if plan.is_empty() {
            Vec::new()
        } else {
            let emb = self.class_latents_emb.as_ref();
            let width = class_emb_width(emb);
            plan.into_iter()
                .map(|(_at, i)| class_row(emb, i, batch, width))
                .collect()
        };
        let mut stream = self.cascade(batch, stream, &mut slots, Some(class), true);

        // A layer always hands up at least what it received. So the cascade
        // comes back empty exactly when nothing ran anywhere. Otherwise its
        // last token is what this prime emitted, and also a `[batch, d_model]`
        // sample that sizes the zero caches below.
        let out = stream.pop();
        let Some(sample) = out.as_ref() else {
            // Not a single latent was due anywhere: no state moved, so the
            // caches go back exactly as they came (`None` included).
            let caches = slots
                .iter()
                .all(Option::is_some)
                .then(|| M::Caches::from_slots(slots));
            return (out, caches);
        };
        if slots.iter().any(Option::is_none) {
            // The call started with no caches, and only some layers ran. The
            // other layers hold the zero state that they started from.
            let zeros = self.real_layers[0]
                .application(0)
                .block
                .zero_caches_2d(sample, n)
                .into_slots();
            for (slot, zero) in slots.iter_mut().zip(zeros) {
                if slot.is_none() {
                    *slot = zero;
                }
            }
        }
        (out, Some(M::Caches::from_slots(slots)))
    }
}

impl<M: Block> Layers<M> {
    /// Reset the untied parameters of every real layer to copies of their
    /// first application ([`Layer::retie`]). A post-build [`InitPolicy`]
    /// redraws a 2-D `weight` element by element, and this call undoes that
    /// for the untied copies.
    pub fn retie(mut self) -> Self {
        self.real_layers = self.real_layers.into_iter().map(Layer::retie).collect();
        self
    }
}

/// The [`Applications`] of a stack of `n_real_layers` under this optional
/// virtual scheduling (none ⇒ each real layer applied once).
fn stack_applications(
    n_virtual_layers: &Option<(usize, Schedule)>,
    n_real_layers: usize,
) -> Applications {
    match n_virtual_layers {
        Some((n, schedule)) => schedule.applications(*n, n_real_layers),
        None => Applications::new(0..n_real_layers, n_real_layers),
    }
}

/// Plain (non-serde) factory for [`Layers`]. The serializable surface of a
/// family is its own `Config` enum, which delegates to this generic builder.
pub struct LayersBuilder<C> {
    /// Number of real (weight-bearing) layers.
    pub n_real_layers: usize,
    /// Optional virtual-layer scheduling.
    pub n_virtual_layers: Option<(usize, Schedule)>,
    /// Shared block config.
    pub block: C,
    /// Zero the first virtual layer's residual.
    pub ignore_first_residual: bool,
    /// Zero the last virtual layer's residual.
    pub ignore_last_residual: bool,
    /// Stack-level class latents (spliced once before the first virtual layer).
    pub class_latents: Vec<ClassLatent>,
    /// Inter-layer residual scheme (defaults to plain additive).
    pub residuals: ResidualsConfig,
    /// Optional SwiGLU feed-forward sub-block per layer, with its own pre-norm
    /// and residual (`d_intermediate > 0` in the reference configs). `None` ⇒
    /// mixer-only layers.
    pub mlp: Option<GatedMlpConfig>,
    /// Back-propagate only some of the (virtual) layers (see
    /// [`Layers::grad_horizon`]). `None` ⇒ track the whole stack.
    pub grad_horizon: Option<GradHorizon>,
    /// The parameters of the layer itself that are held once per application,
    /// not tied (see [`Layer::application`]). The block config names those of
    /// the block.
    pub untied: Vec<LayerUntied>,
}

impl<C: BlockConfig> LayersBuilder<C> {
    /// Builder with no virtual scheduling, no class latents, residuals enabled.
    pub fn new(n_real_layers: usize, block: C) -> Self {
        Self {
            n_real_layers,
            n_virtual_layers: None,
            block,
            ignore_first_residual: false,
            ignore_last_residual: false,
            class_latents: Vec::new(),
            residuals: ResidualsConfig::Standard,
            mlp: None,
            grad_horizon: None,
            untied: Vec::new(),
        }
    }

    /// Hold these layer parameters once per application (see
    /// [`Layer::application`]). Empty ties them all.
    pub fn with_untied(mut self, untied: Vec<LayerUntied>) -> Self {
        self.untied = untied;
        self
    }

    /// Back-propagate only some of the (virtual) layers (see
    /// [`Layers::grad_horizon`]). `None` tracks the whole stack.
    pub fn with_grad_horizon(mut self, grad_horizon: Option<GradHorizon>) -> Self {
        self.grad_horizon = grad_horizon;
        self
    }

    /// Interleave a SwiGLU feed-forward sub-block after each layer's mixer
    /// (see [`Layer`]). `None` keeps layers mixer-only.
    pub fn with_mlp(mut self, mlp: Option<GatedMlpConfig>) -> Self {
        self.mlp = mlp;
        self
    }

    /// Set the optional virtual-layer scheduling.
    pub fn with_n_virtual_layers(mut self, n: Option<(usize, Schedule)>) -> Self {
        self.n_virtual_layers = n;
        self
    }

    /// Set the inter-layer residual scheme (plain additive vs Multi-Gate).
    pub fn with_residuals(mut self, residuals: ResidualsConfig) -> Self {
        self.residuals = residuals;
        self
    }

    /// Suppress the first virtual layer's residual (see [`Layers`]).
    pub fn with_ignore_first_residual(mut self, ignore: bool) -> Self {
        self.ignore_first_residual = ignore;
        self
    }

    /// Suppress the last virtual layer's residual (see [`Layers`]).
    pub fn with_ignore_last_residual(mut self, ignore: bool) -> Self {
        self.ignore_last_residual = ignore;
        self
    }

    /// Set the stack-level class latents.
    pub fn with_class_latents(mut self, class_latents: Vec<ClassLatent>) -> Self {
        self.class_latents = class_latents;
        self
    }

    /// Allocate and initialise the stack on `device`.
    pub fn init(&self, device: &Device) -> Layers<C::Block> {
        let d_model = self.block.d_model();
        let n_virtual = self
            .n_virtual_layers
            .as_ref()
            .map(|(l, _)| *l)
            .unwrap_or(self.n_real_layers);
        // Each real layer holds its untied copies for as many applications as
        // the schedule gives it.
        let real_layers = stack_applications(&self.n_virtual_layers, self.n_real_layers)
            .count
            .into_iter()
            .map(|n_applications| {
                Layer::init(
                    self.block.init_block(n_applications, device),
                    d_model,
                    self.mlp.as_ref(),
                    &self.untied,
                    n_applications,
                    device,
                )
            })
            .collect();
        Layers {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            real_layers,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
            residuals: self
                .residuals
                .init(d_model, self.n_real_layers, n_virtual, device),
            class_latents_emb: init_class_emb(self.class_latents.len(), d_model, device),
            class_latents: self.class_latents.clone(),
            grad_horizon: self.grad_horizon.clone(),
        }
    }
}
