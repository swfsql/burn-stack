use crate::modules::{GatedMlpConfig, Residuals, ResidualsConfig, RmsNormConfig};
use crate::prelude::*;
use crate::utils::{GradHorizon, Schedule};
use crate::utils::class::{
    assert_full_len_known, class_chunk_plan, class_emb_table, class_emb_width,
    class_marker_output_indices, class_prime_plan, class_row, init_class_emb,
    insert_class_markers, splice_class_rows,
};
use crate::utils::{ClassCursor, ClassCursors, ClassLatent};
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
    /// Back-propagate only **some** of the (virtual) layers; the rest run
    /// without building an autodiff graph. `None` (the default) tracks the
    /// whole stack.
    ///
    /// This is the truncated-BPTT knob of TRM/HRM-style deep recursion: with
    /// `n_virtual_layers` far above `n_real_layers`, tracking every pass is what
    /// runs out of memory, and both papers back-propagate only a suffix (TRM one
    /// full recursion, HRM-Text a horizon `K` warmed from 2 to 5). It is counted
    /// **from the top** so it stays meaningful when the stack depth changes, and
    /// so a training loop can move it per step.
    ///
    /// [`GradHorizon::Depth`]`(K)` counts `K` from the top of every **real**
    /// layer — its last `K` applications — rather than of the stack, so which
    /// virtual layers that comes to is the [`Schedule`]'s business:
    /// [`Schedule::Cyclic`] spreads a real layer's applications evenly, so they
    /// are the top `K · n_real_layers` virtual layers and the stack cuts
    /// **once**; [`Schedule::Stretched`] gives each real layer one contiguous
    /// run, so they are the tail of each run and the stack cuts and lifts back
    /// once **per real layer**. A plain suffix — which is all this knob used to
    /// be — leaves a stretched stack's lower real layers with no tracked
    /// application at all, i.e. silently untrained.
    /// [`GradHorizon::Mask`] states the tracked layers outright, one flag per
    /// virtual layer; it is what [`Schedule::Custom`] takes (having no canonical
    /// run of its own), and [`GradHorizon::last`] builds the plain suffix.
    ///
    /// A stack **without** weight sharing applies each real layer exactly once,
    /// so any `Depth(K >= 1)` tracks all of it — use `GradHorizon::last(K, n)`
    /// to cut one of those.
    ///
    /// Under weight sharing the same real layer serves both sides of a cut; the
    /// untracked segments run an inner-backend copy, so each weight still
    /// receives gradient — from its tracked applications only.
    ///
    /// The stack **input** is the exception, and deliberately so. It enters at the
    /// bottom and rides the residual stream upward, so a cut would sever its only
    /// path and a network's `in_proj` (or a vocab net's embedding) would never
    /// train at all — silently. TRM and HRM never meet this because they re-inject
    /// the input at every recursion; this stack reads it once. The boundary
    /// therefore re-attaches it *straight-through* at every boundary: a
    /// value-zero term restores an identity gradient path, which under
    /// [`Residuals::Standard`] is not a guess but the exact leading term of
    /// `∂(x + Σ F_l)/∂x`, the rest being precisely the segment one chose not to
    /// differentiate. What enters an untracked segment is carried this way, not
    /// just the stack input, so the tracked layers *below* a cut keep their
    /// gradient path to the ones above it. Under
    /// [`MultiGate`](crate::modules::MultiGate) the residual lives in the
    /// depth-streams rather than the token, so **every** carrier gets the
    /// identity path. At the bottom of the stack that is exact — the seed stream
    /// is the input and the pool is convex, so an identity segment leaves all
    /// `k` streams equal to it — and correcting only the pooled token would
    /// leave the streams' contribution out of the input's gradient, which under
    /// the carry-biased gate init MGR is built for is most of it. A cut opening
    /// *inside* the stack (a stretched schedule takes one per real layer) has
    /// streams that already differ, and carries the pooled token into each of
    /// them: a carry per stream would be the exact identity, but a segment also
    /// *widens* the streams as it accumulates, so there is no stream to pair a
    /// carry with. Every stream still receives gradient, routed by the convex
    /// aggregation that produced the token rather than one-to-one. Values are
    /// untouched in every case.
    ///
    /// **Every class embedding trains**, at all three levels and on both sides of
    /// a cut: a network's [`ClassToken`]s and this stack's own
    /// [`ClassLatent`]s ride the carry because it is taken *after* they are
    /// spliced, and a per-[`Layer`] latent inside an untracked segment gets a
    /// **ghost** row in the carry (value zero, taken from the tracked table). They are learnable
    /// *input rows*, not part of a layer's transform — which is what stays
    /// undifferentiated below the cut. Anything else would leave a silently dead
    /// parameter.
    ///
    /// [`ClassToken`]: crate::utils::ClassToken
    /// [`ClassLatent`]: crate::utils::ClassLatent
    ///
    /// A horizon that tracks every layer behaves exactly like `None`, and so
    /// does any horizon at all off the autodiff backend. Honoured by
    /// [`Self::forward`], [`Self::step`] and [`Self::prime`] alike, so a cut
    /// stack decodes under the same truncation it trains under.
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
    /// position past the emitted sequence — compare against its length.
    pub fn class_latent_output_indices(&self, orig_len: usize) -> Vec<usize> {
        class_marker_output_indices(&self.class_latents, orig_len)
    }

    /// Splice this stack's own class latents into the chunk `x` (no-op when
    /// there are none), advancing the stack-level cursor.
    fn insert_latents(&self, x: Tensor<3>, class: &mut ClassCursors) -> Tensor<3> {
        let mut cursor = ClassCursor::at(class.stack, class.full_len);
        let x = insert_class_markers(
            x,
            &self.class_latents,
            self.class_latents_emb.as_ref(),
            &mut cursor,
            "Layers",
        );
        class.stack = cursor.offset;
        x
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

    /// Which of the `n` virtual layers back-propagate, per
    /// [`Self::grad_horizon`]: a `false` layer runs on the inner backend, a
    /// `true` one builds the graph. `None` ⇒ no cut anywhere.
    ///
    /// The mask may turn off and on again any number of times — once for a
    /// [`Schedule::Cyclic`] stack, once per real layer for a
    /// [`Schedule::Stretched`] one, arbitrarily for a [`GradHorizon::Mask`].
    ///
    /// Returns `None` off the autodiff backend: a cut is taken with
    /// `Tensor::inner`/`AutodiffModule::valid`, which **panic** there (unlike
    /// `detach`, a documented no-op), so the guard is load-bearing rather than an
    /// optimisation — a horizon left set in a config has to fall through to the
    /// untouched path at inference. The module's own device is what decides,
    /// since [`Self::prime`] has no input tensor to ask.
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
        // An all-tracked mask *is* the untouched stack, so take no cut at all —
        // which is what keeps a horizon deeper than every real layer's
        // application count a no-op down to the graph it builds.
        tracked.iter().any(|t| !t).then_some(tracked)
    }

    /// Full-sequence pass through every (virtual) layer.
    ///
    /// [`Layer`] returns only its delta — `F_l = Block(RMSNorm(·))`, plus the
    /// feed-forward sub-block's contribution when the layer has one; the outer
    /// residual is added here. With [`Residuals::Standard`] each layer adds the input skip (unless
    /// suppressed). With [`Residuals::MultiGate`] the skip is dropped and up to
    /// `n_stream` parallel streams — seeded with `x` as the first one — carry the
    /// residual: each layer reads their attention-pooled aggregate as input, and
    /// its output either *becomes* a new stream (while fewer than `n_stream`
    /// exist) or is gated into every stream (see [`MultiGate`]).
    ///
    /// `ignore_first/last_residual` apply to **both** paths: skipping the first
    /// restarts the residual carry from the first layer's output (the input is
    /// read but not carried); skipping the last makes the stack output the last
    /// layer's transform `F_l` alone (no input-dependent carry).
    ///
    /// `class` places the stack-level and the per-layer class latents; `None`
    /// takes `x` for the whole sequence (so every latent lands in this call).
    /// Passing the same [`ClassCursors`] to consecutive chunks splits the
    /// sequence without moving a single latent — see [`ClassCursors`].
    /// Both residual paths host them: a per-layer latent is spliced into the
    /// token sequence and, under MultiGate, into every carried stream too (the
    /// aggregator over the resulting identical streams reproduces the row, so
    /// the layer above reads it back exactly as the additive skip hands it on).
    ///
    /// [`MultiGate`]: crate::modules::MultiGate
    /// [`Layer`]: crate::modules::Layer
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<M::Caches>,
        options: M::Options,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<3>, M::Caches) {
        let n = self.n_virtual_count();
        // No cursors ⇒ this one call covers the whole sequence.
        let mut whole = ClassCursors::new(x.dims()[1]);
        let class = class.unwrap_or(&mut whole);
        class.fit(n);

        let mut x = self.insert_latents(x, class);
        // The sequence the layers see is longer by the stack's own latents; each
        // layer then lengthens it further for the ones above it.
        let mut full = class.full_len.map(|l| l + self.class_latents.len());
        let caches =
            caches.unwrap_or_else(|| self.real_layers[0].block.zero_caches_3d(&x, n));
        assert_eq!(caches.slot_count(), n, "one cache per virtual layer");

        // An untracked layer must build no graph, and in Burn that means running
        // it **off the autodiff backend** — merely detaching is not enough.
        // Detaching does cut gradient flow, but an untracked op is still
        // registered in the graph (Burn keeps an `UntrackedOpsStep` per op so a
        // memory-bound op can still retrieve an untracked parent), so its output
        // stays retained: measured on a 64-virtual-layer stack, a detached
        // prefix saved ~6% of peak memory and still scaled linearly with depth,
        // while an inner-backend prefix was flat. The memory probe in this
        // module's tests reproduces both curves.
        //
        // `Tensor::inner`/`AutodiffModule::valid` **panic** off the autodiff
        // backend (unlike `detach`, which is a documented no-op there), so a cut
        // is taken only when the input really is on one — at inference
        // `grad_horizon` is simply inert.
        //
        // The mask is not a single boundary: it may turn off and on again any
        // number of times (once per real layer under `Schedule::Stretched`, see
        // `grad_horizon`), and each transition is a full hop of everything the
        // loop carries.
        let tracked = self.grad_tracked(n);
        let inner_stack = tracked
            .is_some()
            .then(|| burn::module::AutodiffModule::valid(self));
        let mut slots = caches.into_slots();
        // Straight-through carry (see `grad_horizon`): a value-**zero** tracked
        // tensor standing in for what entered the untracked segment currently
        // being run, added back where the graph resumes so everything below
        // keeps an identity gradient path across it. `Some` exactly while inside
        // such a segment; it must be added on the autodiff side of the boundary
        // — earlier and it would be a tracked input to an untracked layer, which
        // is both a backend mismatch and the end of the memory saving.
        //
        // It shadows `x`'s **shape**, not its value: an untracked layer's class
        // latents lengthen the sequence, and the carry takes zero rows at those
        // same positions (those latents are that layer's parameters,
        // deliberately not differentiated).
        let mut st: Option<Tensor<3>> = None;

        // MultiGate carries up to `n_stream` parallel streams (the input is the
        // first, the early layers append the rest); Standard threads the single
        // tensor `x` directly (streams stays `None`).
        let mut streams = self.multi_gate_streams_seed(&x);

        // `i` is the virtual-layer index (schedule, cut boundary, residual
        // flags); the slot lookup is incidental, and the bound stays `n` so a
        // mis-sized cache stack still panics.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let track = tracked.as_ref().is_none_or(|t| t[i]);
            match (st.is_some(), track) {
                // Entering an untracked segment: everything crossing into it
                // goes down to the inner backend with it — the tokens and their
                // MultiGate streams here, this layer's cache slot below.
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
                    // Under MultiGate the residual lives in the streams, not
                    // the token, so *every* carrier gets the identity path —
                    // which is what "the segment behaved like identity" means
                    // there. Below the first layer that is exact: the seed
                    // stream is the input and the pool is convex, so an identity
                    // segment leaves all `k` streams equal to it, and the pooled
                    // `x` is not double-counted (its route from the input runs
                    // through an aggregator whose softmax weights sum to one).
                    // At a boundary further up the streams have already
                    // diverged and the carry is the pooled token's alone —
                    // exactness would need one carry per stream, which a segment
                    // that accumulates new streams has nothing to pair with. The
                    // gradient reaches every stream either way, through that
                    // same convex aggregation. See `grad_horizon`.
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
            let layer = &this.real_layers[real];
            // The slot rides the same hop as the layer it belongs to: a cache
            // handed in from a tracked segment comes down with an untracked one
            // (Burn cannot mix backends within an op) and goes back up below.
            let cache = slots[i].take().unwrap();
            let cache = match track {
                true => cache,
                false => M::Caches::cache_to_inner(cache),
            };
            let first = self.ignore_first_residual && i == 0;
            let last = self.ignore_last_residual && i + 1 == n;

            // Splice this layer's class latents into the sequence — and, under
            // MultiGate, into every carried stream, that being where the
            // residual lives. A row present in all `k` streams is reproduced
            // exactly by the (convex, all-scores-equal) aggregator, so the layer
            // above reads the latent back just as the Standard skip hands it on.
            let mut cursor = ClassCursor::at(class.per_layer[i], full);
            let plan = class_chunk_plan(&layer.class_latents, x.dims()[1], &mut cursor, "Layer");
            class.per_layer[i] = cursor.offset;
            full = full.map(|l| l + layer.class_latents.len());
            if !plan.is_empty() {
                let emb = class_emb_table(
                    &layer.class_latents,
                    layer.class_latents_emb.as_ref(),
                    x.dims()[2],
                );
                x = splice_class_rows(x, &plan, &emb);
                streams = streams.map(|s| splice_class_rows(s, &plan, &emb));
                // Keep the carry aligned with `x`, splicing **ghost** rows at the
                // latent positions: value zero like the rest of the carry, but
                // taken from the *tracked* table, so a prefix layer's own class
                // latents keep an identity gradient path exactly as the stack
                // input does. They are learnable input rows, not part of the
                // layer's transform — which stays undifferentiated below the cut.
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
                    // Add the residual (the lengthened input) here — unless
                    // suppressed, in which case the input is moved straight in
                    // (no clone, no add).
                    let x_l = x;
                    let (out, c_) = if first || last {
                        layer.forward(x_l, Some(cache), options.clone())
                    } else {
                        let (out, c_) = layer.forward(x_l.clone(), Some(cache), options.clone());
                        (out + x_l, c_)
                    };
                    x = out;
                    slots[i] = Some(c_);
                }
                Residuals::MultiGate(mg) => {
                    let (out, c_) = layer.forward(x, Some(cache), options.clone());
                    slots[i] = Some(c_);
                    let s = streams.take().unwrap();
                    // A skipped residual here drops every carried stream, which
                    // the MGR reaches by forcing the mixer gate to β ≡ 1
                    // (`new_streams = out`), the aggregator over the resulting
                    // identical streams collapsing to `F_l`. Both branches
                    // shortcut that.
                    if last {
                        // Output depends purely on the last layer's transform.
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

    /// Seed the MultiGate streams from a full-sequence input — the **single**
    /// stream `x` as `[batch, sequence, 1, d_model]` (the layers below
    /// `n_stream` widen it, see [`MultiGate`](crate::modules::MultiGate)) — or
    /// `None` for the Standard path. `x` already carries the stack-level class
    /// latents, so they seed the streams like any other token.
    fn multi_gate_streams_seed(&self, x: &Tensor<3>) -> Option<Tensor<4>> {
        matches!(&self.residuals, Residuals::MultiGate(_)).then(|| x.clone().unsqueeze_dim::<4>(2))
    }

    /// Single-token step through every (virtual) layer.
    ///
    /// `class` drives two independent class-latent levels — the stack-level
    /// [`Self::class_latents`] (`class.stack`, spliced once below the first
    /// layer, exactly as in `forward`) and the per-[`Layer`] latents
    /// (`class.per_layer[i]`, one cursor per virtual layer).
    ///
    /// Because a layer's class latents grow the sequence the *next* layer sees
    /// (exactly as in `forward`), a single user step is a **cascade**: the bottom
    /// input stream (the stack latents falling on this step, plus the user token)
    /// is threaded up the stack, each layer expanding it with its own class
    /// latents. Every layer's recurrence therefore sees the same token order as
    /// `forward`, so `forward` and `step` agree.
    ///
    /// The step returns the (fully propagated) output of the **last** token of
    /// that stream — the user token, unless an `End` latent (the one kind that
    /// closes the sequence rather than preceding a token) follows it. Latents
    /// emitted *before* the user token are stepped for their effect on the state
    /// alone.
    ///
    /// `None` injects nothing at either level (and `Middle`/`End` latents panic,
    /// as they do without a [`ClassCursors::full_len`] hint).
    ///
    /// [`Self::grad_horizon`] applies here exactly as in [`Self::forward`], on
    /// the same virtual layers, so a stack decodes under the truncation it trains
    /// under. Note the cut rebuilds an inner-backend view of the stack once per
    /// call, which is per *token* here rather than per sequence — negligible
    /// against a training step, but not something to leave set for plain decoding
    /// (where it is inert anyway, the model then being off the autodiff backend).
    pub fn step(
        &self,
        x: Tensor<2>,
        caches: Option<M::Caches>,
        mut class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, M::Caches) {
        let [batch, d_model] = x.dims();
        let n = self.n_virtual_count();
        let caches =
            caches.unwrap_or_else(|| self.real_layers[0].block.zero_caches_2d(&x, n));
        assert_eq!(caches.slot_count(), n, "one cache per virtual layer");
        if let Some(c) = class.as_deref_mut() {
            c.fit(n);
        }
        let mut slots = caches.into_slots();

        // Bottom input stream for this user step: the stack-level class latents
        // falling on it (fed through the whole stack like ordinary inputs) around
        // the user token — `at == 0` before it, `at == 1` after (an `End` latent
        // closing the sequence, which then ends the stream).
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

        // The stream keeps `forward`'s token order, so its last element is the
        // latest token of the sequence — the user token, or an `End` after it.
        let out = stream.pop().expect("the user token is always emitted");
        (out, M::Caches::from_slots(slots))
    }

    /// Thread `stream` — the tokens entering the bottom layer — up through every
    /// (virtual) layer, each layer splicing its own class latents into what it
    /// receives, adding its residual, and handing the result to the next one.
    /// Returns what leaves the top layer, `slots` holding the advanced caches.
    ///
    /// The shared body of [`Self::step`] and [`Self::prime`], which differ in the
    /// stream they open with — a user token amid the stack latents, or nothing
    /// but stack latents — and in what a layer emits at the **end** of the stream
    /// it receives: an ordinary step leaves those latents for the token they
    /// precede (only a closing `End` trails one), while a `prime` (`prime =
    /// true`) emits what that next token was going to be preceded by, which is
    /// how the cascade carries on when the stream below it is empty. Everything
    /// else is common — which is what makes a `prime` and the `step` after it run
    /// the same sequence as that `step` alone.
    ///
    /// Under [`Residuals::MultiGate`] the residual is not in the token but in
    /// that token's own depth-streams, so each element of `stream` is carried
    /// alongside its `[batch, k, d_model]` stream set — rebuilt per token, never
    /// crossing steps, exactly as the `[batch, sequence, k, d_model]` streams of
    /// [`Self::forward`] are a per-position construct.
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

        // The same mask `forward` takes, layer for layer: an untracked virtual
        // layer runs on an inner-backend copy of the stack, so it builds no graph
        // (see `grad_horizon`). Everything crossing into an untracked segment
        // goes down with it — the token stream, its MultiGate stream sets, and
        // those layers' own cache slots — and is lifted back where the graph
        // resumes, as many times as the mask alternates.
        let tracked = self.grad_tracked(n);
        let inner_stack = tracked
            .is_some()
            .then(|| burn::module::AutodiffModule::valid(self));
        // The straight-through carry `forward` builds, one entry per token of
        // the stream entering the current untracked segment (see
        // `grad_horizon`); `Some` exactly while inside one. It may open empty —
        // a `prime` whose cut layers receive no token has no input to carry a
        // gradient back to in the first place, only its own class latents, which
        // the ghost rows below cover.
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
        // Stream count entering the current layer. It follows the depth alone
        // (`forward`'s `s.dims()[2]`), so it is tracked even across layers no
        // token reaches — a class latent first appearing at layer `pos` must be
        // seeded with exactly the `k` streams that depth carries.
        let mut k = 1usize;

        // Full length of the stream the layers see, this stack's latents
        // included; each layer then lengthens it further for the ones above it.
        let mut full = class
            .as_deref()
            .and_then(|c| c.full_len)
            .map(|l| l + self.class_latents.len());
        // `pos` is the virtual-layer index (schedule, cut boundary); the slot
        // lookup is incidental.
        #[allow(clippy::needless_range_loop)]
        for pos in 0..n {
            // The boundaries `forward` crosses, token-stream shaped. Placed
            // before the empty-layer `continue` below only defensively —
            // `carried` is empty exactly when `stream` is, so a skipped layer
            // has nothing to hop either way — but that keeps the boundary
            // independent of what the skip condition happens to be.
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
                    // Every carrier, as in `forward`: under MultiGate each token
                    // brings its own `[batch, k, d]` stream set, and takes the
                    // pooled token's carry into all of them.
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
            // one — layer weights, class-latent embeddings and MultiGate gates
            // alike.
            let this = match &inner_stack {
                Some(d) if !track => d,
                _ => self,
            };
            let layer = &this.real_layers[real];
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
                // No cursors ⇒ nothing is injected, so `Middle`/`End` — whose
                // positions only exist against the whole sequence — cannot be
                // placed at all. (With cursors the plan above places every kind,
                // which is why the tokens below go through `Layer::step_one`
                // rather than the cursorless `Layer::step`.)
                assert_full_len_known(&layer.class_latents, None, "Layer");
                Vec::new()
            };
            full = full.map(|l| l + layer.class_latents.len());
            // The stream count this layer leaves behind — mirroring `forward`:
            // a suppressed last residual leaves the streams untouched, a
            // suppressed first restarts them from `F_0`, and otherwise the
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

            // The slot rides the same hop as its layer (see `forward`); an
            // empty one is filled by `Layer::step_one` on whichever backend the
            // tokens are already on.
            let mut cache = slots[pos].take();
            if !track {
                cache = cache.map(M::Caches::cache_to_inner);
            }
            let emitted = stream.len() + plan.len();
            let mut next: Vec<Tensor<2>> = Vec::with_capacity(emitted);
            let mut next_carried: Vec<Tensor<3>> = Vec::with_capacity(mg.map_or(0, |_| emitted));
            // One token through the layer, then its residual: the plain additive
            // skip (unless suppressed — the token is then moved straight in, no
            // clone/add), or the Multi-Gate mix into that token's own streams.
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
                // (`new_streams = F_l`), the aggregator then collapsing to `F_l`.
                if last {
                    (out, Some(s), c) // output depends purely on `F_l`
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
            // `forward`): identical streams score alike, so the aggregator
            // reproduces the row and the layer above reads it back unchanged.
            let row = |i: usize| {
                let emb = layer.class_latents_emb.as_ref();
                let width = class_emb_width(emb);
                let r = class_row(emb, i, batch, width);
                let s = mg.map(|_| r.clone().unsqueeze_dim::<3>(1).expand([batch, k, width]));
                (r, s)
            };
            // The carry rides along, taking a **ghost** row wherever a class
            // latent is emitted: value zero, so it stays index-aligned with what
            // this layer hands up, but tracked, so the latent trains (see
            // `grad_horizon`). Built on demand rather than cloned from the carry,
            // so a layer whose stream is empty still ghosts its own latents.
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
            // …and the latents that follow the stream's last token: an `End`
            // closing the sequence, or (on a prime) the ones the token after it
            // is due to be preceded by.
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

    /// Step the class latents the stack has waiting for its next user token —
    /// with **no** user token, so nothing but class data is consumed.
    ///
    /// This is [`Self::step`]'s opening half on its own: the stack-level latents
    /// due now open the bottom stream (empty when none are), which then goes up
    /// the stack through the very cascade `step` runs, with every layer
    /// additionally flushing the latents *its* next token was going to be
    /// preceded by. A `prime` followed by a `step` therefore runs exactly the
    /// sequence that `step` alone would have.
    /// `End` latents are never primed: closing the sequence, they belong to the
    /// step carrying its last user token (which is why that step returns them).
    /// A cursor already at the announced end therefore primes nothing.
    ///
    /// Returns the fully propagated output of the **last** latent emitted, or
    /// `None` when none were waiting — the seedless-generation entry point:
    /// `prime` → sample → `step` → sample → … `batch` sizes the latent rows,
    /// which are the only inputs there are.
    ///
    /// The caches come back as they went in when nothing ran (`None` included);
    /// a partly primed stack is completed with zero caches for the layers that
    /// stepped nothing, which is exactly the state they hold. `None` cursors
    /// inject nothing at all (`Middle`/`End` latents then panic, as in `step`).
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
            // Nothing may run at all, and there is no token to size zero caches
            // from until something does — so start the slots empty.
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

        // A layer only ever hands up at least what it received, so the cascade
        // comes back empty exactly when nothing ran anywhere — and otherwise its
        // last token is both what this prime emitted and a `[batch, d_model]`
        // sample to size the zero caches below.
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
            // The call started cacheless and only some layers ran; the others
            // hold the zero state they started from.
            let zeros = self.real_layers[0]
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

/// Plain (non-serde) factory for [`Layers`]. A family's serializable surface is
/// its own `Config` enum; this is the generic builder that one delegates to.
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
        }
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
        let real_layers = (0..self.n_real_layers)
            .map(|_| Layer {
                norm: RmsNormConfig::new(d_model).init(device),
                block: self.block.init_block(device),
                // `norm2` exists exactly when `mlp` does — `Layer` relies on it.
                norm2: self
                    .mlp
                    .as_ref()
                    .map(|_| RmsNormConfig::new(d_model).init(device)),
                mlp: self.mlp.as_ref().map(|mlp| mlp.init(device)),
                class_latents: Vec::new(),
                class_latents_emb: None,
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

