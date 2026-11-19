use crate::modules::residuals::{JoinAt, ResidualStreams, StraightThroughCarry};
use crate::modules::{
    GatedMlpConfig, LayerUntied, Residuals, ResidualsConfig, cache_from_inner, cache_to_inner,
};
use crate::prelude::*;
use crate::utils::{Applications, GradHorizon, Schedule};
use crate::utils::class::{
    assert_full_len_known, class_emb_width, class_marker_output_indices, class_prime_plan,
    class_row, class_window_plan, init_class_emb, insert_class_markers_padded,
};
use crate::utils::{ClassCursor, ClassCursors, ClassLatent, Packed, Padding};
use burn::module::Param;
use burn::prelude::*;

/// The layer stack: [`Layer`]s with an optional layer schedule. One struct for
/// every family. After each layer, the residual join of [`Self::residuals`]
/// combines the output of the layer with the residual streams.
#[derive(Module, Debug)]
pub struct Layers<M: Module> {
    /// Number of real (weight-bearing) layers.
    pub n_real_layers: usize,
    /// Optional layer schedule `(n_virtual_layers, schedule)`, for
    /// weight-sharing.
    #[module(skip)]
    pub n_virtual_layers: Option<(usize, Schedule)>,
    /// The weight-bearing layers, length `n_real_layers`.
    pub real_layers: Vec<Layer<M>>,
    /// Zero the first virtual layer's residual when `true`.
    pub ignore_first_residual: bool,
    /// Zero the last virtual layer's residual when `true`.
    pub ignore_last_residual: bool,
    /// The residual scheme (plain additive or Multi-Gate). Its residual join
    /// runs after each layer.
    pub residuals: Residuals,
    /// Positions of the stack-level class latents, spliced into the sequence
    /// once before the first virtual layer. Empty ⇒ none.
    #[module(skip)]
    pub class_latents: Vec<ClassLatent>,
    /// The stack-level class-latent embeddings, `[num_class_latents, d_model]`.
    pub class_latents_emb: Option<Param<Tensor<2>>>,
    /// The gradient horizon: back-propagate only **some** of the virtual
    /// layers. The other layers run without an autodiff graph. `None` (the
    /// default) tracks the whole stack.
    ///
    /// The horizon follows the deep recursion of TRM and HRM. When
    /// `n_virtual_layers` is far above `n_real_layers`, the tracking of every
    /// pass is what runs out of memory. Both papers back-propagate only a
    /// suffix: TRM one full recursion, HRM-Text a horizon `K` warmed from 2 to
    /// 5. The horizon counts **from the top**, so it keeps its meaning when the
    /// stack depth changes, and a training loop can move it per training step.
    ///
    /// [`GradHorizon::Depth`]`(K)` counts `K` from the top of every **real**
    /// layer (its last `K` applications), not from the top of the stack. The
    /// layer schedule ([`Schedule`]) decides which virtual layers these are:
    ///
    /// - [`Schedule::Cyclic`] spreads the applications of a real layer evenly.
    ///   They are the top `K · n_real_layers` virtual layers, and the stack
    ///   cuts **once**.
    /// - [`Schedule::Stretched`] gives each real layer one contiguous run. They
    ///   are the tail of each run, and the stack cuts and lifts back once **per
    ///   real layer**. A plain suffix would leave the lower real layers of a
    ///   stretched stack with no tracked application, so they would silently
    ///   not train. Under MGR, `Depth(1)` can panic (see below).
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
    /// The untracked spans run an inner-backend copy, so each weight still
    /// receives gradient, but only from its tracked applications.
    ///
    /// The backend hop moves the tensors that cross an untracked span to the
    /// other backend at each end of the span. These are the token, the
    /// residual streams and the cache slots. They go to the inner backend
    /// where the span starts, and back to the autodiff backend at its cut
    /// (ADR-0003).
    ///
    /// The stack **input** is the exception, on purpose. It enters at the
    /// bottom and rides the residual stream upward. A cut would sever its only
    /// path, and the `in_proj` of a network (or the embedding of a vocab net)
    /// would silently never train. TRM and HRM do not have this problem,
    /// because they re-inject the input at every recursion. This stack reads
    /// it once.
    ///
    /// So every cut re-attaches the input *straight-through*: a value-zero
    /// term restores the gradient of the carry path of the residual scheme.
    /// Under [`Residuals::Standard`], the carry path is the skip, so this is
    /// not a guess. It is the exact leading term of `∂(x + Σ F_l)/∂x`, and the
    /// rest is the untracked span that the cut chose not to differentiate. The
    /// straight-through carry holds what enters an untracked span, not only
    /// the stack input. So the tracked layers *below* a cut keep their
    /// gradient path to the layers above it.
    ///
    /// Under [`MultiGate`](crate::modules::MultiGate), the residual lives in
    /// the residual streams, not in the token. So each residual stream that
    /// enters the span takes its own straight-through carry, with the gain of
    /// its carry path across the span. A stream that the span appends gets
    /// none. At a cut at the bottom of the stack, only the seed stream takes a
    /// straight-through carry. A cut inside the stack, a span in which the
    /// accumulation phase ends, and one MGR module for each virtual layer
    /// follow the same rule. The module header of
    /// [`crate::modules::residuals`] gives the rule, and when it is exact.
    ///
    /// In every case, the values do not change.
    ///
    /// **A horizon that leaves a parameter with no gradient panics** on the
    /// autodiff backend, before the stack runs. Each case below would let a
    /// parameter silently not train:
    ///
    /// - An untracked application of a layer that unties parameters. Only
    ///   their own application reads the copies.
    /// - An untracked layer 0 under `ignore_first_residual`, or an untracked
    ///   top layer under `ignore_last_residual`. The carry path across a
    ///   suppressed skip has gain 0. So the input of the stack, or every layer
    ///   of the stack, would get no gradient.
    /// - Under MGR, a module that pools in a tracked layer, but only directly
    ///   below untracked layers. Its pooled token takes no straight-through
    ///   carry, so its aggregator query `w_alpha` would never train. With no
    ///   weight sharing, or with one module for each virtual layer, an MGR
    ///   stack can thus take an untracked span only at the bottom, or directly
    ///   above a suppressed first skip. Its layer 0 does not pool. Under
    ///   [`Schedule::Stretched`] with modules for each real layer, `Depth(1)`
    ///   panics when a run other than the bottom one is longer than one layer.
    ///   The module of the run below pools only at its tail, directly below
    ///   the untracked head of that run.
    ///
    /// **Every class embedding trains**, at both marker levels and on both
    /// sides of a cut. The [`ClassToken`]s of a network and the
    /// [`ClassLatent`]s of this stack ride the straight-through carry, because
    /// the loop takes it *after* they are spliced. They are learnable
    /// *inputs*, not part of the transform of a layer (the transform stays
    /// undifferentiated below the cut). Anything else would leave a silently
    /// dead parameter.
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

    /// Whether every class latent of the stack is a `Start`. Then, after the
    /// opening of a sequence has run, no later [`step`](Self::step) emits one.
    /// So every step runs the same launches (a
    /// [`CapturedStep`](crate::utils::capture::CapturedStep) needs this), and
    /// a step with no cursors equals the step with them.
    pub fn only_start_latents(&self) -> bool {
        self.class_latents.iter().all(|m| matches!(m, ClassLatent::Start))
    }

    /// Splice the class latents of this stack into the window `x` (no-op when
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

    /// Number of virtual layers that this stack runs.
    pub fn n_virtual_count(&self) -> usize {
        self.n_virtual_layers
            .as_ref()
            .map(|(l, _)| *l)
            .unwrap_or(self.n_real_layers)
    }

    /// The zero caches that [`Self::forward`], [`Self::step`] and
    /// [`Self::prime`] start from when they get `None`: one
    /// [`Block::zero_cache`] for each virtual layer, for `batch` rows on
    /// `device`.
    pub fn zero_caches(&self, batch: usize, device: &Device) -> Vec<M::Cache> {
        zero_caches_for(&self.real_layers[0], self.n_virtual_count(), batch, device)
    }

    /// The real layer of the virtual layer `virtual_idx`, under the layer
    /// schedule of the stack.
    pub(crate) fn real_idx(&self, virtual_idx: usize) -> usize {
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

    /// The device of the stack (that of its parameters), with its autodiff
    /// context. A cut lifts what it took down back onto this device
    /// ([`lift`](crate::modules::lift)), so the lifted tensors get its
    /// checkpointing strategy.
    fn stack_device(&self) -> Device {
        self.real_layers[0].norm.gamma.val().device()
    }

    /// Which of the `n` virtual layers back-propagate, per
    /// [`Self::grad_horizon`]. A `false` layer runs on the inner backend, and a
    /// `true` layer builds the graph. `None` ⇒ no cut anywhere.
    ///
    /// The mask can turn off and on again any number of times: once for a
    /// [`Schedule::Cyclic`] stack, once per real layer for a
    /// [`Schedule::Stretched`] stack, and as often as a [`GradHorizon::Mask`]
    /// states. The panics below limit where an untracked span can be.
    ///
    /// Returns `None` off the autodiff backend. A cut there would buy nothing:
    /// `Tensor::inner` and [`lift`](crate::modules::lift) do nothing off it. A
    /// horizon left set in a config thus falls through to the untouched path
    /// at inference. The device of the module decides
    /// ([`Self::stack_device`]), because [`Self::prime`] has no input tensor
    /// to ask.
    ///
    /// Panics on a mask that leaves a parameter with no gradient (see
    /// [`Self::grad_horizon`]). The residual scheme checks its own cases
    /// (`Residuals::assert_horizon_keeps_gradients`).
    fn grad_tracked(&self, n: usize) -> Option<Vec<bool>> {
        if !self.stack_device().is_autodiff() {
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
        let joins: Vec<JoinAt> = (0..n).map(|i| self.join_at(i, n)).collect();
        self.residuals.assert_horizon_keeps_gradients(&tracked, &joins);
        // An all-tracked mask *is* the untouched stack, so take no cut. This
        // keeps a horizon deeper than the application count of every real
        // layer a no-op, down to the graph that it builds.
        tracked.iter().any(|t| !t).then_some(tracked)
    }

    /// The forward of a window through every virtual layer.
    ///
    /// [`Layer`] returns only its delta: `F_l = Block(RMSNorm(·))`, plus the
    /// contribution of the feed-forward sub-block when the layer has one. The
    /// residual join of [`Self::residuals`] adds the outer residual (see
    /// [`crate::modules::residuals`]):
    ///
    /// - [`Residuals::Standard`]: each layer adds the input skip (unless it is
    ///   suppressed).
    /// - [`Residuals::MultiGate`]: no skip. Up to `n_stream` residual streams
    ///   carry the residual, and `x` seeds the first one. Each layer reads
    ///   their attention-pooled aggregate as input. Its output either
    ///   *becomes* a new stream (while fewer than `n_stream` exist) or is
    ///   gated into every stream (see [`MultiGate`]).
    ///
    /// `ignore_first/last_residual` apply to **both** paths:
    ///
    /// - A skipped first residual starts the residual again from the output
    ///   of the first layer. The layer reads the input, but its output does
    ///   not contain the input.
    /// - A skipped last residual makes the stack output the transform `F_l` of
    ///   the last layer alone (no residual from the input).
    ///
    /// `class` places the stack-level class latents. `None` takes `x` as the
    /// whole sequence, so every latent lands in this call. Pass the same
    /// [`ClassCursors`] to consecutive windows to split the sequence without a
    /// move of any latent (see [`ClassCursors`]).
    ///
    /// `pad` (`[batch, sequence]`, `true` at padding, `None` ⇒ no padding)
    /// marks a right-padded batch. Every row comes out as its own sequence
    /// would come out alone: the outputs of its real tokens, and its caches.
    /// Every class latent is placed against the length of that row (see
    /// [`crate::utils::padding`]). The positions still come back in the
    /// batch-wide order, with the class latents where the plan for the padded
    /// length puts them.
    ///
    /// [`MultiGate`]: crate::modules::MultiGate
    /// [`Layer`]: crate::modules::Layer
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<Vec<M::Cache>>,
        options: M::Options,
        class: Option<&mut ClassCursors>,
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, Vec<M::Cache>) {
        self.forward_padded(x, caches, options, class, pad.map(Padding::new))
    }

    /// [`Self::forward`] over packed rows (see [`crate::utils::packing`]). At
    /// each reset of `packed` in row `b`, each virtual layer starts again from
    /// row `b` of its origin. The positions before the first reset of a row
    /// continue from `caches`. The returned caches are those of the last
    /// segment of each row.
    ///
    /// `origins` holds one origin for each virtual layer, in the order of
    /// `caches`. `None` gives each virtual layer its zero cache, as for
    /// `caches`. The call does not return the origins. An origin follows the
    /// rules of a cache:
    ///
    /// - It gets a gradient from each segment that starts from it.
    /// - Under [`Self::grad_horizon`], the origin of an untracked layer moves
    ///   to the inner backend with the cache of that layer. It gets no
    ///   gradient.
    /// - An origin from the caller has the checkpointing strategy of the
    ///   stack, or an op panics.
    /// - An origin that an earlier backward used needs a detach first
    ///   ([`cache_detach`](crate::modules::cache_detach)).
    ///
    /// The stack puts its class latents into the opening slots of each
    /// segment, in place of the input tokens. So the output has the shape of
    /// `x`. Every class latent of the stack must be a `Start`, and `packed`
    /// must give each segment one opening slot per class latent.
    pub fn forward_packed(
        &self,
        x: Tensor<3>,
        caches: Option<Vec<M::Cache>>,
        origins: Option<Vec<M::Cache>>,
        options: M::Options,
        packed: &Packed,
    ) -> (Tensor<3>, Vec<M::Cache>) {
        assert!(
            self.only_start_latents(),
            "a packed stack takes only `Start` class latents"
        );
        let packed = packed.inner();
        let x = match &self.class_latents_emb {
            Some(emb) if !self.class_latents.is_empty() => packed.place(x, emb.val()),
            _ => {
                assert_eq!(packed.lead, 0, "a stack with no class latents takes no opening slots");
                x
            }
        };
        let origins = origins.unwrap_or_else(|| self.zero_caches(x.dims()[0], &x.device()));
        assert_eq!(origins.len(), self.n_virtual_count(), "one origin per virtual layer");
        self.forward_rows(x, caches, options, None, None, Some((&packed, origins)))
    }

    /// [`Self::forward`] with the padding already tracked. A container that
    /// splices its own markers below this stack calls this.
    pub(crate) fn forward_padded(
        &self,
        x: Tensor<3>,
        caches: Option<Vec<M::Cache>>,
        options: M::Options,
        class: Option<&mut ClassCursors>,
        padding: Option<Padding>,
    ) -> (Tensor<3>, Vec<M::Cache>) {
        self.forward_rows(x, caches, options, class, padding, None)
    }

    /// One layer of the stack, over a padded batch, or over packed rows with
    /// the origin of the layer.
    fn run_layer(
        layer: &Layer<M>,
        x: Tensor<3>,
        cache: M::Cache,
        options: M::Options,
        padding: Option<&Padding>,
        packed_origin: Option<(&Packed, M::Cache)>,
    ) -> (Tensor<3>, M::Cache) {
        match packed_origin {
            None => layer.forward(x, cache, options, padding),
            Some((packed, origin)) => layer.forward_packed(x, cache, origin, options, packed),
        }
    }

    /// Where the residual join of virtual layer `i` of `n` runs.
    fn join_at(&self, i: usize, n: usize) -> JoinAt {
        JoinAt {
            virtual_idx: i,
            real_idx: self.real_idx(i),
            first: self.ignore_first_residual && i == 0,
            last: self.ignore_last_residual && i + 1 == n,
        }
    }

    /// The loop of [`Self::forward_padded`] and [`Self::forward_packed`]. The
    /// window is one [`ResidualStreams`]. `packed` rows come with one origin
    /// for each virtual layer. They take no padding, and their class latents
    /// are already in their opening slots.
    fn forward_rows(
        &self,
        x: Tensor<3>,
        caches: Option<Vec<M::Cache>>,
        options: M::Options,
        class: Option<&mut ClassCursors>,
        padding: Option<Padding>,
        packed: Option<(&Packed, Vec<M::Cache>)>,
    ) -> (Tensor<3>, Vec<M::Cache>) {
        let n = self.n_virtual_count();
        // No cursors ⇒ this one call covers the whole sequence.
        let mut whole = ClassCursors::new(x.dims()[1]);
        let class = class.unwrap_or(&mut whole);

        let (x, padding) = match packed {
            None => self.insert_latents(x, padding, class),
            Some(_) => {
                assert!(padding.is_none(), "packed rows take no padding");
                (x, None)
            }
        };
        let caches = caches.unwrap_or_else(|| self.zero_caches(x.dims()[0], &x.device()));
        assert_eq!(caches.len(), n, "one cache per virtual layer");
        let (packed, mut origins) = match packed {
            Some((packed, origins)) => (Some(packed), Some(origins.into_iter())),
            None => (None, None),
        };
        let mut hop = BackendHop::new(self, n);
        let mut values = vec![self.residuals.seed::<3, 4>(x)];
        let apps = self.applications();
        let mut advanced = Vec::with_capacity(n);

        // `i` is the virtual-layer index (layer schedule, cut, residual flags).
        for (i, cache) in caches.into_iter().enumerate() {
            let at = self.join_at(i, n);
            // The origin of this layer, on the backend of the layer.
            let packed_origin = packed.map(|packed| {
                let origin = origins.as_mut().and_then(Iterator::next).expect("one origin per virtual layer");
                (packed, hop.to_layer(i, origin))
            });
            let (next, cache) = hop.layer(i, values, cache, |stack, mut values, cache| {
                // The layer as its own application sees it, if the layer
                // unties anything.
                let layer = stack.real_layers[at.real_idx].application(apps.index[i]);
                let value = values.pop().expect("the window is one value");
                let (value, cache) = stack.residuals.join(value, at, |x| {
                    Self::run_layer(&layer, x, cache, options.clone(), padding.as_ref(), packed_origin)
                });
                (vec![value], cache)
            });
            values = next;
            advanced.push(cache);
        }
        let x = hop.finish(values).pop().expect("the window is one value");
        (x, advanced)
    }

    /// The step of one token through every virtual layer.
    ///
    /// `class` places the stack-level [`Self::class_latents`] (`class.stack`),
    /// exactly as in `forward`. A single user step is a **cascade**. The
    /// bottom input stream (the stack latents that fall on this step, plus the
    /// user token) goes up the stack, one token after the other. So the
    /// recurrence of every layer sees the same token order as in `forward`,
    /// and `forward` and `step` agree.
    ///
    /// The step returns the (fully propagated) output of the **last** token of
    /// that stream. This is the user token, unless an `End` latent follows it
    /// (the one kind that closes the sequence, not precedes a token). The
    /// latents emitted *before* the user token are stepped only for their
    /// effect on the state.
    ///
    /// `None` injects nothing. `Middle`/`End` latents then panic, as they do
    /// without a [`ClassCursors::full_len`] hint.
    ///
    /// [`Self::grad_horizon`] applies here exactly as in [`Self::forward`], on
    /// the same virtual layers. So a stack decodes under the truncation that
    /// it trains under. Note: the backend hop rebuilds the `Module::valid` copy
    /// of the stack once per call, which is once per *token* here, not per
    /// sequence.
    /// This is negligible against a training step. But do not leave it set
    /// for plain decoding. (There it does nothing anyway, because the model is
    /// then off the autodiff backend.)
    pub fn step(
        &self,
        x: Tensor<2>,
        caches: Option<Vec<M::Cache>>,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, Vec<M::Cache>) {
        let [batch, d_model] = x.dims();
        let n = self.n_virtual_count();
        let caches = caches.unwrap_or_else(|| self.zero_caches(batch, &x.device()));
        assert_eq!(caches.len(), n, "one cache per virtual layer");

        // The bottom input stream for this user step: the stack-level class
        // latents that fall on it, around the user token. They go through the
        // whole stack like ordinary inputs. `at == 0` is before the token.
        // `at == 1` is after it (an `End` latent that closes the sequence, and
        // so ends the stream).
        let mut stream: Vec<Tensor<2>> = Vec::with_capacity(1);
        if let Some(class) = class {
            let mut cursor = ClassCursor::at(class.stack, class.full_len);
            let plan = class_window_plan(&self.class_latents, 1, &mut cursor, "Layers");
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

        let (mut stream, caches) = self.cascade(stream, caches);

        // The stream keeps the token order of `forward`. So its last element is
        // the latest token of the sequence: the user token, or an `End` after
        // it.
        let out = stream.pop().expect("the user token is always emitted");
        (out, caches)
    }

    /// Thread `stream` (the tokens that enter the bottom layer) up through
    /// every virtual layer, from `caches`. Each layer adds its residual to
    /// each token and gives the result to the next layer. Returns what leaves
    /// the top layer, and the advanced caches.
    ///
    /// This is the shared body of [`Self::step`] and [`Self::prime`]. They
    /// differ only in the stream that they open with: a user token among the
    /// stack latents, or stack latents only. This is why a `prime` and the
    /// `step` after it run the same sequence as that `step` alone.
    ///
    /// Each token of `stream` is one [`ResidualStreams`], with its own
    /// residual streams under MGR. A token rebuilds its residual streams at
    /// each step, so they never cross steps.
    fn cascade(
        &self,
        stream: Vec<Tensor<2>>,
        caches: Vec<M::Cache>,
    ) -> (Vec<Tensor<2>>, Vec<M::Cache>) {
        let n = caches.len();
        let mut hop = BackendHop::new(self, n);
        let mut values: Vec<ResidualStreams<2, 3>> =
            stream.into_iter().map(|token| self.residuals.seed(token)).collect();
        let apps = self.applications();
        let mut advanced = Vec::with_capacity(n);

        // `i` is the virtual-layer index (layer schedule, cut, residual flags).
        for (i, cache) in caches.into_iter().enumerate() {
            let at = self.join_at(i, n);
            let (next, cache) = hop.layer(i, values, cache, |stack, values, mut cache| {
                // The layer as its own application sees it, if the layer
                // unties anything.
                let layer = stack.real_layers[at.real_idx].application(apps.index[i]);
                // The tokens go through the layer one after the other, and
                // each one advances the cache.
                let mut after = Vec::with_capacity(values.len());
                for value in values {
                    let (value, c) = stack.residuals.join(value, at, |token| layer.step(token, cache));
                    after.push(value);
                    cache = c;
                }
                (after, cache)
            });
            values = next;
            advanced.push(cache);
        }
        (hop.finish(values), advanced)
    }

    /// Step the class latents that the stack has waiting for its next user
    /// token, with **no** user token. So the call consumes only class data.
    ///
    /// This is the opening half of [`Self::step`] on its own. The stack-level
    /// latents due now form the bottom stream. The stream then goes up the
    /// stack through the same cascade that `step` runs. So a `prime` followed
    /// by a `step` runs exactly the sequence that the `step` alone would run.
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
    /// included). When a latent runs from `None`, the cascade starts from
    /// [`Self::zero_caches`]. `None` cursors inject nothing (`Middle`/`End`
    /// latents then panic, as in `step`).
    pub fn prime(
        &self,
        batch: usize,
        caches: Option<Vec<M::Cache>>,
        class: Option<&mut ClassCursors>,
    ) -> (Option<Tensor<2>>, Option<Vec<M::Cache>>) {
        let n = self.n_virtual_count();
        let Some(class) = class else {
            assert_full_len_known(&self.class_latents, None, "Layers");
            return (None, caches);
        };
        if let Some(caches) = &caches {
            assert_eq!(caches.len(), n, "one cache per virtual layer");
        }

        // Bottom input stream for this prime: the stack-level class latents due
        // now, with no user token to accompany them.
        let mut cursor = ClassCursor::at(class.stack, class.full_len);
        let plan = class_prime_plan(&self.class_latents, &mut cursor, "Layers");
        class.stack = cursor.offset;
        if plan.is_empty() {
            // Not a single latent was due: no state moved, so the caches go
            // back exactly as they came (`None` included).
            return (None, caches);
        }
        let emb = self.class_latents_emb.as_ref();
        let width = class_emb_width(emb);
        let stream: Vec<Tensor<2>> = plan
            .into_iter()
            .map(|(_at, i)| class_row(emb, i, batch, width))
            .collect();
        // A stack with no caches starts from its zero caches, on the device of
        // the latents.
        let caches = caches.unwrap_or_else(|| self.zero_caches(batch, &stream[0].device()));
        let (mut out, caches) = self.cascade(stream, caches);
        (out.pop(), Some(caches))
    }
}

/// The backend hop of the gradient horizon (see [`Layers::grad_horizon`]).
/// It owns the hop at either end of an untracked span, and the state of that
/// hop:
///
/// - the tracked mask of [`Layers::grad_tracked`],
/// - the `Module::valid` copy of the stack, and the view of each virtual
///   layer,
/// - the device of the stack,
/// - the straight-through carry of each [`ResidualStreams`],
/// - the hop of the cache of each layer, and of its origin in a packed
///   forward,
/// - the lift at the top of the stack.
///
/// The loop of `forward` gives it one `ResidualStreams`: the window. The
/// cascade of `step` and `prime` gives it one `ResidualStreams` for each token
/// of its stream. It knows no residual scheme. At the start of an untracked
/// span, [`Residuals::open_span`] takes the straight-through carry of each
/// value, and moves the value down. At the cut, [`Residuals::close_span`]
/// lifts each value and adds its straight-through carry back. Both methods
/// run on the residual scheme of the tracked stack, not on the view.
///
/// An untracked layer must build no graph. In Burn, that means it runs **off
/// the autodiff backend**. A detach is not sufficient. It cuts the gradient
/// flow, but Burn still registers the untracked op in the graph. Burn keeps an
/// `UntrackedOpsStep` per op, so that a memory-bound op can still retrieve an
/// untracked parent. So the output of the op stays retained (ADR-0003).
/// `grad_horizon_memory_probe` in `src/reference/tests/grad_horizon.rs`
/// measures the peak memory of a detach and of the inner backend.
///
/// The stack takes a cut only on an autodiff backend
/// ([`Layers::grad_tracked`] returns `None` elsewhere). At inference, the
/// horizon does nothing. Each lift goes through
/// [`lift`](crate::modules::lift) onto the device of the stack, never through
/// a bare `Tensor::from_inner`. `from_inner` would enable autodiff also on a
/// plain tensor, and would reset the checkpointing strategy. `Module::valid`
/// also turns off the training flags (dropout, …) of a block that has them. So
/// an untracked span runs such a block in eval mode.
///
/// The mask can turn off and on again any number of times: once per real
/// layer under `Schedule::Stretched`, and as often as a `GradHorizon::Mask`
/// states. Each end of an untracked span is a full hop of everything that
/// crosses it.
struct BackendHop<'a, M: Block, const R: usize, const S: usize> {
    /// The stack. Its residual scheme takes and adds back the
    /// straight-through carry.
    stack: &'a Layers<M>,
    /// The tracked mask of [`Layers::grad_tracked`]. `None`: no cut.
    tracked: Option<Vec<bool>>,
    /// The `Module::valid` copy of the stack: the view of each untracked
    /// layer. `Some` exactly when `tracked` is.
    inner_stack: Option<Layers<M>>,
    /// The device of the stack. Each lift goes onto it.
    device: Device,
    /// The straight-through carry of each value. `Some` exactly inside an
    /// untracked span.
    carry: Option<Vec<StraightThroughCarry<R, S>>>,
}

impl<'a, M: Block, const R: usize, const S: usize> BackendHop<'a, M, R, S>
where
    M::Options: Clone,
{
    /// The hop of `stack` over its `n` virtual layers.
    fn new(stack: &'a Layers<M>, n: usize) -> Self {
        let tracked = stack.grad_tracked(n);
        let inner_stack = tracked.is_some().then(|| Module::valid(stack));
        Self { stack, tracked, inner_stack, device: stack.stack_device(), carry: None }
    }

    /// Run the virtual layer `i` on `values`, with the hop at either end of an
    /// untracked span.
    ///
    /// `run` gets the view of the stack for this layer, the values and the
    /// cache of the layer, all on the backend of the layer. It returns the
    /// values for the next layer and the advanced cache. This call returns
    /// them too, with the cache back on the device of the stack. A tracked
    /// layer runs on the stack, and an untracked one on its `Module::valid`
    /// copy. This holds for the weights of the layer and for the MGR gates.
    /// The cache makes the same hop as its layer. A tracked cache goes down
    /// with an untracked layer, because Burn cannot mix backends within an
    /// op. It goes back up after the layer.
    fn layer(
        &mut self,
        i: usize,
        values: Vec<ResidualStreams<R, S>>,
        cache: M::Cache,
        run: impl FnOnce(
            &Layers<M>,
            Vec<ResidualStreams<R, S>>,
            M::Cache,
        ) -> (Vec<ResidualStreams<R, S>>, M::Cache),
    ) -> (Vec<ResidualStreams<R, S>>, M::Cache) {
        let stack = self.stack;
        let track = self.tracks(i);
        let values = match (self.carry.take(), track) {
            // The start of an untracked span.
            (None, false) => {
                let (values, carry) = values.into_iter().map(|v| stack.residuals.open_span(v)).unzip();
                self.carry = Some(carry);
                values
            }
            // A cut.
            (Some(carry), true) => self.close(values, carry),
            // Inside an untracked span, or inside the graph.
            (carry, _) => {
                self.carry = carry;
                values
            }
        };
        let view = match &self.inner_stack {
            Some(inner) if !track => inner,
            _ => stack,
        };
        let cache = self.to_layer(i, cache);
        let (values, cache) = run(view, values, cache);
        let cache = match track {
            true => cache,
            false => cache_from_inner(cache, &self.device),
        };
        (values, cache)
    }

    /// Whether the virtual layer `i` is a tracked layer.
    fn tracks(&self, i: usize) -> bool {
        self.tracked.as_ref().is_none_or(|t| t[i])
    }

    /// `cache` on the backend of the virtual layer `i`. For an untracked
    /// layer, this moves the cache to the inner backend, because Burn cannot
    /// mix backends within an op. Off the autodiff backend, the move does
    /// nothing. [`Self::layer`] moves the cache of the layer. The packed
    /// forward also moves the origin of the layer. The origin does not move
    /// back to the autodiff backend.
    fn to_layer(&self, i: usize, cache: M::Cache) -> M::Cache {
        match self.tracks(i) {
            true => cache,
            false => cache_to_inner(cache),
        }
    }

    /// The cut: [`Residuals::close_span`] on each value, with its
    /// straight-through carry.
    fn close(
        &self,
        values: Vec<ResidualStreams<R, S>>,
        carry: Vec<StraightThroughCarry<R, S>>,
    ) -> Vec<ResidualStreams<R, S>> {
        debug_assert_eq!(carry.len(), values.len(), "one straight-through carry for each value");
        values
            .into_iter()
            .zip(carry)
            .map(|(value, carry)| self.stack.residuals.close_span(value, carry, &self.device))
            .collect()
    }

    /// The top of the stack: the token of each value. A stack whose top
    /// layers are untracked lifts each token at the top, and adds its
    /// straight-through carry back. The residual streams end at the top layer,
    /// so they stay on the inner backend.
    fn finish(mut self, values: Vec<ResidualStreams<R, S>>) -> Vec<Tensor<R>> {
        let values = match self.carry.take() {
            Some(carry) => self.close(values.into_iter().map(ResidualStreams::into_top).collect(), carry),
            None => values,
        };
        values.into_iter().map(ResidualStreams::into_token).collect()
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

/// The zero caches of a stack of `n` slots: one [`Block::zero_cache`] call for
/// each slot, so each slot has its own buffers. The block is that of the
/// first application of `layer`. The tensors of an untied block hold the copy
/// of every application, and this view holds one copy.
pub(crate) fn zero_caches_for<M: Block>(
    layer: &Layer<M>,
    n: usize,
    batch: usize,
    device: &Device,
) -> Vec<M::Cache> {
    let layer = layer.application(0);
    (0..n).map(|_| layer.block.zero_cache(batch, device)).collect()
}

/// The [`Applications`] of a stack of `n_real_layers` under this optional
/// layer schedule (none ⇒ each real layer applied once).
fn stack_applications(
    n_virtual_layers: &Option<(usize, Schedule)>,
    n_real_layers: usize,
) -> Applications {
    match n_virtual_layers {
        Some((n, schedule)) => schedule.applications(*n, n_real_layers),
        None => Applications::new(0..n_real_layers, n_real_layers),
    }
}

/// Plain (non-serde) factory for [`Layers`]. The serializable config of a
/// consumer is a shape plus a block (ADR-0007), and
/// [`NetworkShape::layers`](crate::modules::NetworkShape::layers) makes this
/// builder from that pair.
pub struct LayersBuilder<C> {
    /// Number of real (weight-bearing) layers.
    pub n_real_layers: usize,
    /// Optional layer schedule `(n_virtual_layers, schedule)`.
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
    /// each layer has only its block.
    pub mlp: Option<GatedMlpConfig>,
    /// The gradient horizon: back-propagate only some of the virtual layers
    /// (see [`Layers::grad_horizon`]). `None` ⇒ track the whole stack.
    pub grad_horizon: Option<GradHorizon>,
    /// The parameters of the layer itself that are held once per application,
    /// not tied (see [`Layer::application`]). The block config names those of
    /// the block.
    pub untied: Vec<LayerUntied>,
}

impl<C: BlockConfig> LayersBuilder<C> {
    /// Builder with no layer schedule, no class latents, residuals enabled.
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

    /// Set the gradient horizon: back-propagate only some of the virtual
    /// layers (see [`Layers::grad_horizon`]). `None` tracks the whole stack.
    pub fn with_grad_horizon(mut self, grad_horizon: Option<GradHorizon>) -> Self {
        self.grad_horizon = grad_horizon;
        self
    }

    /// Add a SwiGLU feed-forward sub-block after the block of each layer (see
    /// [`Layer`]). `None` keeps only the block in each layer.
    pub fn with_mlp(mut self, mlp: Option<GatedMlpConfig>) -> Self {
        self.mlp = mlp;
        self
    }

    /// Set the optional layer schedule.
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
        // the layer schedule gives it.
        let real_layers = stack_applications(&self.n_virtual_layers, self.n_real_layers)
            .count
            .into_iter()
            .map(|n_applications| {
                Layer::init(
                    self.block.init_block(device),
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
