//! The residual schemes of a layer stack, and the residual join of each.
//!
//! A [`Layer`] returns only its transform `F_l`. The
//! residual scheme of the stack combines `F_l` with the residual streams. Its
//! rule at one layer is the **residual join**:
//!
//! - [`Residuals::Standard`]: one residual stream, the token itself. The join
//!   gives `x + F_l(x)`. The skip is the carry path.
//! - [`Residuals::MultiGate`]: up to `n_stream` residual streams. The layer
//!   reads their pooled token. The join appends `F_l` as a new stream while
//!   fewer than `n_stream` exist. After that, it gates `F_l` into each stream
//!   (see [`MultiGate`]). The `1 − β` share of each stream is the carry path.
//!
//! A suppressed skip (`ignore_first_residual` or `ignore_last_residual` of the
//! stack) changes the join at its layer:
//!
//! - At the first layer, the residual starts again from `F_0`. Under MGR, the
//!   residual streams start again from the single stream `F_0`. The
//!   accumulation phase fills them again.
//! - At the last layer, the stack output is `F_l` alone. Under MGR, the
//!   residual streams do not change.
//!
//! Under MGR, the token of a suppressed skip is that of the β gate at
//! `β ≡ 1`. Then each new stream is `F_l`, and the aggregator over equal
//! streams gives `F_l`. The join takes a short path to that token. At the first layer,
//! the single stream `F_0` stands for the equal streams. At the last layer, no
//! layer reads the streams, so the join keeps them.
//!
//! The join is generic over the rank `R` of the token and the rank `S = R + 1`
//! of the residual streams. The stream axis is `R − 1`. A window has the ranks
//! 3 and 4 (`[batch, sequence, d_model]` and `[batch, sequence, k, d_model]`).
//! One token has the ranks 2 and 3.
//!
//! Only the skip of Standard clones the input of the layer. Under a suppressed
//! skip of Standard, and under MGR, the input moves into the layer. A clone
//! keeps the input alive through the block call, so it raises the peak
//! memory.
//!
//! **Straight-through carry.** The backend hop of the gradient horizon (see
//! [`Layers::grad_horizon`]) moves the token and the residual streams. They
//! go down where an untracked span starts, and up at its cut. The scheme takes the
//! straight-through carry before the move down, and adds it after the move up.
//! The straight-through carry follows the carry path of the scheme across the
//! span. The rule holds the transforms and the scores of the gates constant:
//!
//! - Standard: the carry path has gain 1. The token gets the straight-through
//!   carry of the token that entered the span.
//! - MGR: each residual stream `i` that enters the span takes its own
//!   straight-through carry `carryᵢ = sᵢ − detach(sᵢ)`. The token that enters
//!   the span reaches the cut only through the transforms, so it takes none.
//!   Each residual join in the span updates the gain `gᵢ` of each carry path,
//!   on the inner backend:
//!   - The mixing phase multiplies `gᵢ` by the `1 − βᵢ` of its layer.
//!   - The accumulation phase appends a stream with `g = 0`.
//!
//!   At the cut, stream `i` gets `gᵢ · carryᵢ`, and the token gets
//!   `Σᵢ αᵢ · gᵢ · carryᵢ`. `αᵢ` is the aggregator weight of stream `i` in
//!   the last pool. A stream that the span appended gets nothing.
//! - A suppressed skip in a span keeps the gains. So the input still trains
//!   across it, but the gradient there is not exact.
//!
//! The values do not change. The rule is exact when each transform of the
//! span is zero and the scores do not depend on the streams. The init `w = 0`
//! gives such scores. ADR-0006 gives the reason for the straight-through carry.

use crate::modules::bidi::NoOp;
use crate::modules::{MultiGate, MultiGateResidualConfig, lift};
use burn::config::Config;
use burn::prelude::*;

/// How a [`Layers`](crate::modules::Layers) stack threads residuals between
/// layers: the plain additive skip, or Multi-Gate Residuals. Each variant
/// gives the residual join of its scheme (see the module header).
#[derive(Module, Debug)]
pub enum Residuals {
    /// Plain Pre-LN additive residual: the stack adds the skip connection of
    /// each [`Layer`](crate::modules::Layer).
    Standard(NoOp),
    /// Multi-Gate Residuals: `n_stream` residual streams with per-layer gated
    /// mixing + attention pooling.
    MultiGate(MultiGate),
}

/// Configuration / factory for [`Residuals`].
#[derive(Config, Debug)]
pub enum ResidualsConfig {
    /// Plain additive Pre-LN residual.
    Standard,
    /// Multi-Gate Residuals over `n_stream` streams.
    MultiGate {
        /// Number of parallel residual streams `n`.
        n_stream: usize,
        /// Initial gate bias of the first stream (see
        /// [`MultiGateResidualConfig::init_bias`]).
        init_bias: f64,
        /// Per-stream gate-bias offset (see
        /// [`MultiGateResidualConfig::init_bias_offset`]). `0` starts every
        /// stream equal.
        init_bias_offset: f64,
        /// `true` ⇒ one MGR per *virtual* layer. `false` ⇒ one per *real*
        /// layer (reused across virtual passes). See [`MultiGate::per_virtual`].
        per_virtual_layer: bool,
    },
}

impl ResidualsConfig {
    /// Build the runtime [`Residuals`] for a stack of `n_real_layers` real
    /// layers unrolled over `n_virtual_layers` virtual layers. The MGR module
    /// count follows `per_virtual_layer` (one per virtual layer vs one per real
    /// layer).
    pub fn init(
        &self,
        d_model: usize,
        n_real_layers: usize,
        n_virtual_layers: usize,
        device: &Device,
    ) -> Residuals {
        match self {
            ResidualsConfig::Standard => Residuals::Standard(NoOp),
            ResidualsConfig::MultiGate {
                n_stream,
                init_bias,
                init_bias_offset,
                per_virtual_layer,
            } => {
                let count = if *per_virtual_layer {
                    n_virtual_layers
                } else {
                    n_real_layers
                };
                let layers = (0..count)
                    .map(|_| {
                        MultiGateResidualConfig::new(d_model, *n_stream)
                            .with_init_bias(*init_bias)
                            .with_init_bias_offset(*init_bias_offset)
                            .init(device)
                    })
                    .collect();
                Residuals::MultiGate(MultiGate {
                    layers,
                    n_stream: *n_stream,
                    per_virtual: *per_virtual_layer,
                })
            }
        }
    }
}

/// The residual streams that a layer stack passes from one layer to the next,
/// at the token rank `R` and the stream rank `S = R + 1`.
pub(crate) struct ResidualStreams<const R: usize, const S: usize> {
    /// The token: the input of the next layer. Under Standard, it is also the
    /// one residual stream.
    token: Tensor<R>,
    /// Under MGR, the `k` residual streams, `[…, k, d_model]`. `None` under
    /// Standard.
    streams: Option<Tensor<S>>,
    /// Under MGR, the aggregator weights `αᵢ` of the last pool, `[…, k, 1]`.
    /// `None` under Standard, and under MGR before the first pool of the
    /// streams. Then the token is the one stream.
    ///
    /// Each pool sets them, also outside an untracked span. In a span in which
    /// no join pools, the cut uses the weights of the pool below the span.
    alpha: Option<Tensor<S>>,
    /// Inside an untracked span under MGR: the gain of the carry path of each
    /// stream from the start of the span, `[…, k, 1]`. `None` elsewhere.
    span_gain: Option<Tensor<S>>,
}

impl<const R: usize, const S: usize> ResidualStreams<R, S> {
    /// The value of Standard: the token is the one residual stream.
    fn standard(token: Tensor<R>) -> Self {
        Self { token, streams: None, alpha: None, span_gain: None }
    }

    /// The token, the output of the top layer. The residual streams end here.
    pub(crate) fn into_token(self) -> Tensor<R> {
        self.token
    }

    /// The value that leaves the top layer: the token, with no residual
    /// streams. No layer reads the streams above the top layer. So an
    /// untracked span that ends at the top lifts only the token.
    pub(crate) fn into_top(self) -> Self {
        Self { streams: None, ..self }
    }
}

/// The straight-through carry of one [`ResidualStreams`]: what enters an
/// untracked span, with value zero. [`Residuals::open_span`] takes it, and
/// [`Residuals::close_span`] adds it back at the cut. The ranks are those of
/// the `ResidualStreams`.
pub(crate) enum StraightThroughCarry<const R: usize, const S: usize> {
    /// Standard: `token − detach(token)` of the token that enters the span.
    Token(Tensor<R>),
    /// MGR: `streams − detach(streams)` of the `k₀` residual streams that
    /// enter the span, `[…, k₀, d_model]`.
    Streams(Tensor<S>),
}

/// Where a residual join runs in its stack.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JoinAt {
    /// The virtual index of the layer. In a bidirectional stack, the virtual
    /// index of the pair.
    pub(crate) virtual_idx: usize,
    /// The real index of the layer. In a bidirectional stack, the real index of
    /// the pair. With `virtual_idx`, it selects the MGR module
    /// ([`MultiGate::module_index`]).
    pub(crate) real_idx: usize,
    /// `true`: this is the first layer, and its skip is suppressed.
    pub(crate) first: bool,
    /// `true`: this is the last layer, and its skip is suppressed.
    pub(crate) last: bool,
}

/// The stream axis `R − 1` of the residual streams. Panics unless
/// `S == R + 1`.
fn stream_axis<const R: usize, const S: usize>() -> usize {
    assert_eq!(S, R + 1, "the residual streams have one axis more than the token");
    R - 1
}

/// The shape of `x`, with the size `n` on the axis `axis`.
fn resized<const S: usize>(x: &Tensor<S>, axis: usize, n: usize) -> [usize; S] {
    let mut dims = x.dims();
    dims[axis] = n;
    dims
}

impl Residuals {
    /// The residual streams that enter the bottom layer, from the input
    /// `token` of the stack. Under MGR, `token` is the single seed stream, and
    /// the layers below `n_stream` widen the set. `token` already holds the
    /// stack-level class latents, so they seed the streams like any other
    /// token.
    pub(crate) fn seed<const R: usize, const S: usize>(&self, token: Tensor<R>) -> ResidualStreams<R, S> {
        let axis = stream_axis::<R, S>();
        match self {
            Residuals::Standard(_) => ResidualStreams::standard(token),
            Residuals::MultiGate(_) => {
                let streams = Some(token.clone().unsqueeze_dim::<S>(axis));
                ResidualStreams { token, streams, alpha: None, span_gain: None }
            }
        }
    }

    /// The residual join of this scheme at the layer `at` (see the module
    /// header).
    ///
    /// `layer` runs the layer on its input. It returns the transform `F_l` and
    /// one more value, for example the cache of the layer. The join returns
    /// the residual streams for the next layer, and that value.
    pub(crate) fn join<const R: usize, const S: usize, T>(
        &self,
        value: ResidualStreams<R, S>,
        at: JoinAt,
        layer: impl FnOnce(Tensor<R>) -> (Tensor<R>, T),
    ) -> (ResidualStreams<R, S>, T) {
        let axis = stream_axis::<R, S>();
        let ResidualStreams { token, streams, alpha, span_gain } = value;
        match self {
            Residuals::Standard(_) => {
                // Only the skip clones the input. A suppressed skip moves it
                // into the layer.
                let (token, extra) = if at.first || at.last {
                    layer(token)
                } else {
                    let (out, extra) = layer(token.clone());
                    (out + token, extra)
                };
                (ResidualStreams::standard(token), extra)
            }
            Residuals::MultiGate(mg) => {
                let streams = streams.expect("an MGR value has residual streams");
                // The streams hold the residual, so the token moves into the
                // layer.
                let (out, extra) = layer(token);
                // A suppressed skip keeps the gain of the span.
                let value = if at.last {
                    ResidualStreams { token: out, streams: Some(streams), alpha, span_gain }
                } else if at.first {
                    let streams = Some(out.clone().unsqueeze_dim::<S>(axis));
                    ResidualStreams { token: out, streams, alpha: None, span_gain }
                } else {
                    let mgr = &mg.layers[mg.module_index(at.virtual_idx, at.real_idx)];
                    let out = out.unsqueeze_dim::<S>(axis);
                    let pooled = if streams.dims()[axis] < mg.n_stream {
                        mgr.append_pool(out, streams)
                    } else {
                        mgr.mix_pool(out, streams)
                    };
                    let span_gain = span_gain.map(|gain| match pooled.carry_gain {
                        Some(carry_gain) => gain * carry_gain,
                        // The new stream holds only `F_l`, so its carry path
                        // starts at gain 0.
                        None => {
                            let appended = Tensor::zeros(resized(&gain, axis, 1), &gain.device());
                            Tensor::cat(vec![gain, appended], axis)
                        }
                    });
                    ResidualStreams {
                        token: pooled.token.squeeze_dim::<R>(axis),
                        streams: Some(pooled.streams),
                        alpha: Some(pooled.alpha),
                        span_gain,
                    }
                };
                (value, extra)
            }
        }
    }

    /// The start of an untracked span: take the straight-through carry of
    /// `value` on the autodiff backend, then move `value` to the inner backend.
    /// The straight-through carry is `x − detach(x)`: value zero, and the
    /// gradient of `x`. Under Standard, `x` is the token. Under MGR, `x` is
    /// each residual stream, and the gain of each carry path starts at 1.
    pub(crate) fn open_span<const R: usize, const S: usize>(
        &self,
        value: ResidualStreams<R, S>,
    ) -> (ResidualStreams<R, S>, StraightThroughCarry<R, S>) {
        let axis = stream_axis::<R, S>();
        let ResidualStreams { token, streams, alpha, span_gain } = value;
        debug_assert!(span_gain.is_none(), "an untracked span cannot start inside an untracked span");
        match self {
            Residuals::Standard(_) => {
                let carry = StraightThroughCarry::Token(token.clone() - token.clone().detach());
                (ResidualStreams::standard(token.inner()), carry)
            }
            Residuals::MultiGate(_) => {
                let streams = streams.expect("an MGR value has residual streams");
                let carry = StraightThroughCarry::Streams(streams.clone() - streams.clone().detach());
                let streams = streams.inner();
                // One gain for each stream: size 1 on the feature axis.
                let span_gain = Some(Tensor::ones(resized(&streams, axis + 1, 1), &streams.device()));
                let (token, alpha) = (token.inner(), alpha.map(Tensor::inner));
                (ResidualStreams { token, streams: Some(streams), alpha, span_gain }, carry)
            }
        }
    }

    /// The cut: lift `value` onto the autodiff backend of `device`, as new
    /// graph roots ([`lift`]), then add the straight-through carry back (see
    /// the module header).
    ///
    /// The add runs on the autodiff side of the cut. An add before the lift
    /// would give a tracked input to an untracked layer: a backend mismatch,
    /// and no memory saving.
    ///
    /// Under MGR, stream `i` of the `k₀` streams that entered the span gets
    /// `gᵢ · carryᵢ`, and the token gets `Σᵢ αᵢ · gᵢ · carryᵢ` (see the module
    /// header). The gains `gᵢ` and the weights `αᵢ` are constants. A stream
    /// that the span appended gets nothing.
    pub(crate) fn close_span<const R: usize, const S: usize>(
        &self,
        value: ResidualStreams<R, S>,
        carry: StraightThroughCarry<R, S>,
        device: &Device,
    ) -> ResidualStreams<R, S> {
        let axis = stream_axis::<R, S>();
        let ResidualStreams { token, streams, alpha, span_gain } = value;
        let token = lift(token, device);
        match carry {
            StraightThroughCarry::Token(carry) => ResidualStreams::standard(token + carry),
            StraightThroughCarry::Streams(carry) => {
                let gain = span_gain.expect("an untracked span under MGR has the gain of each carry path");
                // The accumulation phase appends after the streams that it
                // finds. So the first `k₀` streams entered the span.
                let k0 = carry.dims()[axis];
                let carried = carry * lift(gain.narrow(axis, 0, k0), device);
                let alpha = alpha.map(|a| lift(a, device));
                let token_carry = match &alpha {
                    Some(a) => (carried.clone() * a.clone().narrow(axis, 0, k0)).sum_dim(axis),
                    // No pool yet: the token is the one stream.
                    None => carried.clone(),
                };
                let streams = streams.map(|streams| {
                    let streams = lift(streams, device);
                    let k = streams.dims()[axis];
                    let carried = match k - k0 {
                        0 => carried,
                        appended => {
                            let zeros = Tensor::zeros(resized(&streams, axis, appended), device);
                            Tensor::cat(vec![carried, zeros], axis)
                        }
                    };
                    streams + carried
                });
                let token = token + token_carry.squeeze_dim::<R>(axis);
                ResidualStreams { token, streams, alpha, span_gain: None }
            }
        }
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
