//! [`Segmented`]: run a different optimizer on each column block of a fused
//! projection weight.
//!
//! The [`Muon`](burn::optim::Muon) of Burn orthogonalises the *whole* matrix
//! that it gets. A fused `in_proj` would then couple the singular values of
//! unrelated maps (a gate, the values, the keys and queries, the per-head
//! scalars, …). [`Segmented`] slices the weight and its gradient along the
//! fused axis, steps each block with its own optimizer, and concatenates the
//! results. So each sub-matrix is orthogonalised (and shape-LR-adjusted) on
//! its own, exactly as if it were a separate `Linear`.
//!
//! The *model* does not change: the forward pass keeps one fused GEMM.
//!
//! A slice along columns is exact for every optimizer used here: AdamW and SGD
//! are elementwise, and the Newton–Schulz of Muon is per-matrix. So a
//! [`Segmented`] whose blocks are all AdamW is bit-comparable to plain AdamW
//! on the whole tensor (the test suite of a block family asserts this).

use burn::optim::{
    AdamW, AdamWState, LearningRate, Muon, MuonState, Optimizer, RecordState, Sgd, StateSink,
    StateSource, join_index,
};
use burn::prelude::*;

use super::Fallback;
use super::spec::ProjSpec;

/// The optimizer owning one column block.
#[derive(Clone)]
enum BlockOptim {
    /// Muon: orthogonalised momentum-SGD, for genuine feature maps.
    Muon(Muon),
    /// The fallback for scalar-producing (or otherwise unsuitable) blocks: AdamW …
    AdamW(AdamW),
    /// … or plain SGD, which has no state and so no entry in a
    /// [`SegmentedState`].
    Sgd(Sgd),
}

impl BlockOptim {
    fn is_stateful(&self) -> bool {
        !matches!(self, Self::Sgd(_))
    }
}

/// The optimizer state of one column block.
// The AdamW variant is the bigger one. Only a few of these exist per fused
// weight, so the padding is not worth an extra indirection.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum BlockState<const D: usize> {
    /// State of a [`Muon`] block.
    Muon(MuonState<D>),
    /// State of an [`AdamW`] block.
    AdamW(AdamWState<D>),
}

/// State of a [`Segmented`] optimizer: one entry per stateful column block
/// (every block but an SGD one), in order.
#[derive(Clone)]
pub struct SegmentedState<const D: usize> {
    /// Per-block states, in the same order as the [`ProjSpec`] segments.
    pub blocks: Vec<BlockState<D>>,
}

/// Per-column-block optimizer over a fused projection weight.
///
/// The [`ProjSpec`]-driven assembly of a [`MuonPlan`](super::MuonPlan) builds
/// it. The widths of the blocks must sum to the size of the parameter along
/// [`Self::dim`].
#[derive(Clone)]
pub struct Segmented {
    optims: Vec<BlockOptim>,
    widths: Vec<usize>,
    dim: usize,
    tiled: bool,
}

impl Segmented {
    /// Build the per-block optimizers for `spec`: `muon` on its Muon segments,
    /// `fallback` on the others. The split is along `dim` (`1` for a Burn
    /// `Linear` weight, whose layout is `[d_input, d_output]`).
    pub fn new(spec: &ProjSpec, muon: Muon, fallback: impl Into<Fallback>, dim: usize) -> Self {
        let fallback = match fallback.into() {
            Fallback::AdamW(adamw) => BlockOptim::AdamW(adamw),
            Fallback::Sgd(sgd) => BlockOptim::Sgd(sgd),
        };
        let optims = spec
            .segments
            .iter()
            .map(|s| {
                if s.muon {
                    BlockOptim::Muon(muon.clone())
                } else {
                    fallback.clone()
                }
            })
            .collect();
        let widths = spec.segments.iter().map(|s| s.width).collect();
        Self { optims, widths, dim, tiled: spec.tiled }
    }

    /// The axis of the blocks.
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
        // An untied weight holds one copy of the segments per application.
        let copies = if self.tiled { width / per_copy } else { 1 };
        assert_eq!(
            width,
            per_copy * copies,
            "Segmented: the dim-{} width of the parameter does not match the projection spec",
            self.dim
        );
        let n_blocks = self.optims.len() * copies;
        let n_states = self.optims.iter().filter(|o| o.is_stateful()).count() * copies;
        let widths: Vec<usize> = self.widths.iter().copied().cycle().take(n_blocks).collect();

        let tensors = tensor.split_with_sizes(widths.clone(), self.dim);
        let grads = grad.split_with_sizes(widths, self.dim);

        // A missing (first-step) state, or one whose length drifted from the
        // spec, restarts every block from scratch. This prevents a wrong
        // pairing.
        let mut prev = match state {
            Some(s) if s.blocks.len() == n_states => s.blocks,
            _ => Vec::new(),
        }
        .into_iter();

        let mut out = Vec::with_capacity(n_blocks);
        let mut blocks = Vec::with_capacity(n_states);

        for (i, optim) in self.optims.iter().cycle().take(n_blocks).enumerate() {
            let (t, g) = (tensors[i].clone(), grads[i].clone());
            match optim {
                BlockOptim::Muon(muon) => {
                    let prev = match prev.next() {
                        Some(BlockState::Muon(s)) => Some(s),
                        _ => None,
                    };
                    let (t, s) = muon.step(lr, t, g, prev);
                    out.push(t);
                    blocks.extend(s.map(BlockState::Muon));
                }
                BlockOptim::AdamW(adamw) => {
                    let prev = match prev.next() {
                        Some(BlockState::AdamW(s)) => Some(s),
                        _ => None,
                    };
                    let (t, s) = adamw.step(lr, t, g, prev);
                    out.push(t);
                    blocks.extend(s.map(BlockState::AdamW));
                }
                BlockOptim::Sgd(sgd) => out.push(sgd.step(lr, t, g, None).0),
            }
        }

        // Muon and AdamW always return a state, so `blocks` is complete.
        assert_eq!(blocks.len(), n_states);
        (Tensor::cat(out, self.dim), Some(SegmentedState { blocks }))
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D> {
        let blocks = state
            .blocks
            .into_iter()
            .map(|b| match b {
                BlockState::Muon(s) => BlockState::Muon(Muon::to_device(s, device)),
                BlockState::AdamW(s) => BlockState::AdamW(AdamW::to_device(s, device)),
            })
            .collect();
        SegmentedState { blocks }
    }
}

/// Hand-written, for two reasons. The `RecordState` derive covers
/// `Vec<Tensor>`, but not a `Vec` of nested states. Also, the reload has no
/// access to the spec. So the leaf names tell the two block kinds apart
/// (`momentum.velocity` for Muon, `momentum.moment_1`/`moment_2` for AdamW).
/// These names never overlap.
impl<const D: usize> RecordState for SegmentedState<D> {
    fn state_flatten(&self, prefix: &str, out: &mut StateSink) {
        for (i, block) in self.blocks.iter().enumerate() {
            let prefix = join_index(prefix, i);
            match block {
                BlockState::Muon(s) => s.state_flatten(&prefix, out),
                BlockState::AdamW(s) => s.state_flatten(&prefix, out),
            }
        }
    }

    fn state_unflatten(prefix: &str, src: &mut StateSource, device: &Device) -> Option<Self> {
        let mut blocks = Vec::new();
        for i in 0.. {
            let prefix = join_index(prefix, i);
            if !src.has_under(&prefix) {
                break;
            }
            // A failed attempt consumes nothing (the leaf that it looks for is
            // absent). So it is safe to try Muon first.
            let block = MuonState::state_unflatten(&prefix, src, device)
                .map(BlockState::Muon)
                .or_else(|| {
                    AdamWState::state_unflatten(&prefix, src, device).map(BlockState::AdamW)
                })?;
            blocks.push(block);
        }

        (!blocks.is_empty()).then_some(SegmentedState { blocks })
    }
}
