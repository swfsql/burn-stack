//! [`Segmented`]: run a different optimizer on each sub-projection of a fused
//! projection.
//!
//! The [`Muon`](burn::optim::Muon) of Burn orthogonalises the *whole* matrix
//! that it gets. A fused `in_proj` would then couple the singular values of
//! unrelated maps (a gate, the values, the keys and queries, the per-head
//! scalars, …). [`Segmented`] slices the weight and its gradient along the
//! fused axis, steps each sub-projection with its own optimizer, and
//! concatenates the results. So each sub-projection is orthogonalised (and
//! shape-LR-adjusted) on its own, exactly as if it were a separate `Linear`.
//!
//! The *model* does not change: the forward pass keeps one fused GEMM.
//!
//! A slice along columns is exact for every optimizer used here: AdamW and SGD
//! are elementwise, and the Newton–Schulz of Muon is per-matrix. So a
//! [`Segmented`] with all its sub-projections on AdamW is bit-comparable to
//! plain AdamW on the whole tensor (the test suite of a block family asserts
//! this).

use burn::optim::{
    AdamW, AdamWState, LearningRate, Muon, MuonState, Optimizer, RecordState, Sgd, StateSink,
    StateSource, join_index,
};
use burn::prelude::*;

use super::Fallback;
use super::spec::ProjSpec;

/// The optimizer that owns one sub-projection.
#[derive(Clone)]
enum SubProjectionOptim {
    /// Muon: orthogonalised momentum-SGD, for genuine feature maps.
    Muon(Muon),
    /// The fallback for scalar-producing (or otherwise unsuitable)
    /// sub-projections: AdamW …
    AdamW(AdamW),
    /// … or plain SGD, which has no state and so no entry in a
    /// [`SegmentedState`].
    Sgd(Sgd),
}

impl SubProjectionOptim {
    fn is_stateful(&self) -> bool {
        !matches!(self, Self::Sgd(_))
    }
}

/// The optimizer state of one sub-projection.
// The AdamW variant is the bigger one. Only a few of these exist for each
// fused projection, so the padding is not worth an extra indirection.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum SubProjectionState<const D: usize> {
    /// State of a sub-projection on [`Muon`].
    Muon(MuonState<D>),
    /// State of a sub-projection on [`AdamW`].
    AdamW(AdamWState<D>),
}

/// State of a [`Segmented`] optimizer: one entry for each stateful
/// sub-projection (each one that is not on SGD), in order.
#[derive(Clone)]
pub struct SegmentedState<const D: usize> {
    /// The states of the sub-projections, in the order of the sub-projections
    /// of the [`ProjSpec`].
    pub states: Vec<SubProjectionState<D>>,
}

/// One optimizer for each sub-projection of a fused projection.
///
/// The [`ProjSpec`]-driven assembly of a [`MuonPlan`](super::MuonPlan) builds
/// it. The widths of the sub-projections must sum to the size of the
/// parameter along [`Self::dim`].
#[derive(Clone)]
pub struct Segmented {
    optims: Vec<SubProjectionOptim>,
    widths: Vec<usize>,
    dim: usize,
    tiled: bool,
}

impl Segmented {
    /// Build the optimizers of the sub-projections of `spec`: `muon` on those
    /// that Muon owns, `fallback` on the others. The split is along `dim`
    /// (`1` for a Burn `Linear` weight, whose layout is `[d_input, d_output]`).
    pub fn new(spec: &ProjSpec, muon: Muon, fallback: impl Into<Fallback>, dim: usize) -> Self {
        let fallback = match fallback.into() {
            Fallback::AdamW(adamw) => SubProjectionOptim::AdamW(adamw),
            Fallback::Sgd(sgd) => SubProjectionOptim::Sgd(sgd),
        };
        let optims = spec
            .sub_projections
            .iter()
            .map(|s| {
                if s.muon {
                    SubProjectionOptim::Muon(muon.clone())
                } else {
                    fallback.clone()
                }
            })
            .collect();
        let widths = spec.sub_projections.iter().map(|s| s.width).collect();
        Self { optims, widths, dim, tiled: spec.tiled }
    }

    /// The fused axis, along which the sub-projections lie.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Optimizer for Segmented {
    type State<const D: usize> = SegmentedState<D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        assert!(
            self.dim < D,
            "Segmented: split dim {} out of range for a {D}D parameter",
            self.dim
        );
        let per_copy: usize = self.widths.iter().sum();
        let width = tensor.shape().dims::<D>()[self.dim];
        // An untied weight holds one copy of the sub-projections per
        // application.
        let copies = if self.tiled { width / per_copy } else { 1 };
        assert_eq!(
            width,
            per_copy * copies,
            "Segmented: the dim-{} width of the parameter does not match the projection spec",
            self.dim
        );
        let n_sub_projections = self.optims.len() * copies;
        let n_states = self.optims.iter().filter(|o| o.is_stateful()).count() * copies;
        let widths: Vec<usize> = self.widths.iter().copied().cycle().take(n_sub_projections).collect();

        let tensors = tensor.split_with_sizes(widths.clone(), self.dim);
        let grads = grad.split_with_sizes(widths, self.dim);

        // A missing state (at the first training step), or one whose length
        // differs from the spec, gives every sub-projection a fresh state.
        // This prevents a wrong pairing.
        let mut prev = match state {
            Some(s) if s.states.len() == n_states => s.states,
            _ => Vec::new(),
        }
        .into_iter();

        let mut out = Vec::with_capacity(n_sub_projections);
        let mut states = Vec::with_capacity(n_states);

        for (i, optim) in self.optims.iter().cycle().take(n_sub_projections).enumerate() {
            let (t, g) = (tensors[i].clone(), grads[i].clone());
            match optim {
                SubProjectionOptim::Muon(muon) => {
                    let prev = match prev.next() {
                        Some(SubProjectionState::Muon(s)) => Some(s),
                        _ => None,
                    };
                    let (t, s) = muon.step(lr, t, g, prev);
                    out.push(t);
                    states.extend(s.map(SubProjectionState::Muon));
                }
                SubProjectionOptim::AdamW(adamw) => {
                    let prev = match prev.next() {
                        Some(SubProjectionState::AdamW(s)) => Some(s),
                        _ => None,
                    };
                    let (t, s) = adamw.step(lr, t, g, prev);
                    out.push(t);
                    states.extend(s.map(SubProjectionState::AdamW));
                }
                SubProjectionOptim::Sgd(sgd) => out.push(sgd.step(lr, t, g, None).0),
            }
        }

        // Muon and AdamW always return a state, so `states` is complete.
        assert_eq!(states.len(), n_states);
        (Tensor::cat(out, self.dim), Some(SegmentedState { states }))
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D> {
        let states = state
            .states
            .into_iter()
            .map(|b| match b {
                SubProjectionState::Muon(s) => SubProjectionState::Muon(Muon::to_device(s, device)),
                SubProjectionState::AdamW(s) => SubProjectionState::AdamW(AdamW::to_device(s, device)),
            })
            .collect();
        SegmentedState { states }
    }
}

/// Hand-written, for two reasons. The `RecordState` derive covers
/// `Vec<Tensor>`, but not a `Vec` of nested states. Also, the reload has no
/// access to the spec. So the leaf names identify the kind of each state
/// (`momentum.velocity` for Muon, `momentum.moment_1`/`moment_2` for AdamW).
/// These names never overlap.
impl<const D: usize> RecordState for SegmentedState<D> {
    fn state_flatten(&self, prefix: &str, out: &mut StateSink) {
        for (i, state) in self.states.iter().enumerate() {
            let prefix = join_index(prefix, i);
            match state {
                SubProjectionState::Muon(s) => s.state_flatten(&prefix, out),
                SubProjectionState::AdamW(s) => s.state_flatten(&prefix, out),
            }
        }
    }

    fn state_unflatten(prefix: &str, src: &mut StateSource, device: &Device) -> Option<Self> {
        let mut states = Vec::new();
        for i in 0.. {
            let prefix = join_index(prefix, i);
            if !src.has_under(&prefix) {
                break;
            }
            // A failed attempt consumes nothing (the leaf that it looks for is
            // absent). So it is safe to try Muon first.
            let state = MuonState::state_unflatten(&prefix, src, device)
                .map(SubProjectionState::Muon)
                .or_else(|| {
                    AdamWState::state_unflatten(&prefix, src, device).map(SubProjectionState::AdamW)
                })?;
            states.push(state);
        }

        (!states.is_empty()).then_some(SegmentedState { states })
    }
}
