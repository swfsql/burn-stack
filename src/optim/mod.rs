//! Muon support: which weights the optimizer may touch, and how the fused
//! projections are split before it sees them. (Also [`sgd`]: the plain SGD a
//! captured training step can replay.)
//!
//! [Muon](burn::optim::Muon) replaces a 2-D weight's momentum update with the
//! nearest orthogonal matrix (Newton–Schulz). That only makes sense for a
//! parameter that *is* one linear map, which rules out three things in this
//! crate:
//!
//! 1. **Rank ≠ 2.** Burn's `Muon::step` asserts `D == 2`, so biases, norm gains,
//!    a block's per-head scalars, a depthwise conv weight, and every 3-D
//!    tensor must stay on the fallback optimizer. Passing one to Muon
//!    panics — which is why the plan built here is an **allowlist**: only the
//!    weights named by a [`ProjSpec`] are moved off the fallback (AdamW or
//!    plain SGD, [`FallbackConfig`]).
//! 2. **Embedding-like matrices.** The token embedding, the LM head, the
//!    `LatentNetwork` input/output projections and the class-token/latent tables
//!    are rank-2 but are lookup/readout tables at the model boundary; the usual
//!    Muon recipe keeps them on AdamW. They are simply never listed.
//! 3. **Fused projections.** Every family concatenates several independent maps
//!    into one `Linear` (`in_proj` most of all). Orthogonalising that
//!    concatenation ties together maps that share nothing but an allocation, so
//!    the plan carries the column seams and [`Segmented`] applies Muon
//!    **per block**. The model is untouched — the forward pass keeps its single
//!    fused GEMM.
//!
//! Within a fused projection one more distinction applies: blocks that emit
//! *per-head scalars* (a step size, a decay rate) are gains rather than feature
//! maps and stay on AdamW ([`ProjSegment::adamw`]).
//!
//! ## Why 3-D tensors are never "stacked matrices"
//!
//! A `[heads, r, dim]` tensor looks like a stack of matrices a stack-aware Muon
//! could take a slice at a time. In practice it almost never is one: the shapes
//! that show up at rank 3 in a mixer block are per-head *diagonals* (a learnable
//! gain vector broadcast over a projection's output), a bias, an initial
//! condition, or a depthwise filter — orthogonalising any of them would
//! constrain a set of gains rather than a linear map. Whichever part of a block
//! really is an R-fold matrix expansion lives in a fused 2-D projection, and is
//! reachable through a [`ProjSpec`] segment. So the rule stays simple: rank 2,
//! named explicitly, or AdamW.
//!
//! The seams a [`ProjSpec`] carries are therefore **per named sub-projection**,
//! not per head/group/MIMO rank — the same boundaries the forward's
//! `split_into` uses, and the usual Muon convention (a transformer's `W_q` goes
//! to Muon with all heads fused). A caller who wants to test a finer split can
//! just hand [`ProjSpec::block`] more segments.
//!
//! ## Usage
//!
//! ```ignore
//! let plan = model_config.muon_plan();
//! let mut optim = plan.build(&FallbackConfig::AdamW(adamw_config), &muon_config);
//! // ... then the usual `optim.step(lr, model, grads)`.
//! ```
//!
//! ## Learning rate
//!
//! Muon and its fallback share the one learning rate `optim.step` is called with.
//! [`AdjustLrFn::MatchRmsAdamW`] rescales
//! Muon's update so its per-element RMS is `0.2·lr` — AdamW's own ballpark — so
//! an LR schedule tuned for AdamW can be reused as-is. That is what
//! [`muon_config`] returns; the default
//! [`AdjustLrFn::Original`] instead expects a Muon-specific
//! (typically much larger) LR.

/// [`MuonPlan::describe`]: the per-parameter optimizer assignment, for checking
/// a plan against a real model.
pub mod report;
/// [`Segmented`]: a different optimizer per column block of a fused weight.
pub mod segmented;
/// [`SgdConfig`]: plain SGD, the optimizer a captured training step can replay.
pub mod sgd;
/// The column layout of the fused projection weights.
pub mod spec;


pub use segmented::{BlockState, Segmented, SegmentedState};
pub use sgd::SgdConfig;
pub use spec::{BLOCK_CONTAINERS, ProjScope, ProjSegment, ProjSpec};

use burn::grad_clipping::GradientClipping;
use burn::optim::{AdamW, AdamWConfig, AdjustLrFn, ModuleOptimizer, MuonConfig, Sgd};
use burn::prelude::*;

/// The Muon defaults this crate recommends: `MatchRmsAdamW` LR adjustment, so
/// Muon and AdamW can share one learning rate and one weight decay.
///
/// `weight_decay` should mirror the AdamW config's (Muon applies it after
/// orthogonalisation, with the *unadjusted* LR).
pub fn muon_config(weight_decay: f32) -> MuonConfig {
    MuonConfig::new()
        .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(weight_decay)))
}

/// The optimizer of every parameter Muon does not own — or of every parameter,
/// when there is no Muon.
#[derive(Config, Debug)]
pub enum FallbackConfig {
    /// AdamW.
    AdamW(AdamWConfig),
    /// Plain SGD, the one optimizer a captured training step replays.
    Sgd(SgdConfig),
}

impl FallbackConfig {
    /// The whole-module optimizer, gradient clipping included.
    pub fn init(&self) -> ModuleOptimizer {
        match self {
            Self::AdamW(adamw) => adamw.init(),
            Self::Sgd(sgd) => sgd.init(),
        }
    }

    /// The bare optimizer, for a [`Segmented`] block.
    pub fn build(&self) -> Fallback {
        match self {
            Self::AdamW(adamw) => Fallback::AdamW(adamw.build()),
            Self::Sgd(sgd) => Fallback::Sgd(sgd.build()),
        }
    }
}

impl From<AdamWConfig> for FallbackConfig {
    fn from(adamw: AdamWConfig) -> Self {
        Self::AdamW(adamw)
    }
}

impl From<SgdConfig> for FallbackConfig {
    fn from(sgd: SgdConfig) -> Self {
        Self::Sgd(sgd)
    }
}

/// A built [`FallbackConfig`].
#[derive(Clone)]
pub enum Fallback {
    /// AdamW.
    AdamW(AdamW),
    /// Plain SGD: stateless (no momentum).
    Sgd(Sgd),
}

impl From<AdamW> for Fallback {
    fn from(adamw: AdamW) -> Self {
        Self::AdamW(adamw)
    }
}

impl From<Sgd> for Fallback {
    fn from(sgd: Sgd) -> Self {
        Self::Sgd(sgd)
    }
}

/// Which weights Muon owns in a model, and where their fused columns split.
///
/// Built from a *block* config (via [`BlockConfig::muon_projections`]), so
/// it is independent of the network topology: the specs are matched as path
/// substrings, and therefore cover a plain stack, a virtual-layer stack, and a
/// bidirectional stack alike.
///
/// [`BlockConfig::muon_projections`]: crate::modules::BlockConfig::muon_projections
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MuonPlan {
    /// The fused (or plain) weights Muon may touch.
    pub specs: Vec<ProjSpec>,
}

impl MuonPlan {
    /// An empty plan — everything stays on the fallback optimizer.
    pub fn empty() -> Self {
        Self::default()
    }

    /// A plan over the given specs.
    pub fn new(specs: Vec<ProjSpec>) -> Self {
        Self { specs }
    }

    /// Append another plan's specs.
    pub fn extend(mut self, other: Self) -> Self {
        self.specs.extend(other.specs);
        self
    }

    /// Add the [`GatedMlp`](crate::modules::GatedMlp) weights of a layer's
    /// optional feed-forward sub-block.
    pub fn with_mlp(self, mlp: Option<&crate::modules::GatedMlpConfig>) -> Self {
        match mlp {
            None => self,
            Some(mlp) => {
                let hidden = mlp.hidden();
                self.extend(Self::new(vec![
                    // fc1 fuses the SwiGLU value and gate halves.
                    ProjSpec::path(
                        "mlp.fc1.weight",
                        vec![
                            ProjSegment::muon("value", hidden),
                            ProjSegment::muon("gate", hidden),
                        ],
                    ),
                    ProjSpec::path_whole("mlp.fc2.weight", mlp.d_model),
                ]))
            }
        }
    }

    /// Drop every segment named `name` from Muon's ownership (it falls back to
    /// AdamW). Lets a caller opt a sub-projection out without rebuilding the plan.
    pub fn without_segment(mut self, name: &str) -> Self {
        for spec in &mut self.specs {
            for segment in &mut spec.segments {
                if segment.name == name {
                    segment.muon = false;
                }
            }
        }
        self
    }

    /// Assemble the [`ModuleOptimizer`]: the fallback everywhere, Muon on the
    /// planned weights.
    ///
    /// The fallback's group must match everything, so any parameter the plan
    /// does not name — every 1-D and 3-D tensor included — keeps it, as does
    /// every non-Muon segment of a fused weight. The fallback's gradient
    /// clipping is applied to every group.
    pub fn build(&self, fallback: &FallbackConfig, muon: &MuonConfig) -> ModuleOptimizer {
        let mut optim = fallback.init();
        let clipping: Option<GradientClipping> = optim.grad_clipping().cloned();

        for spec in &self.specs {
            if !spec.has_muon() {
                continue;
            }
            let group = spec.param_group();
            optim = if spec.is_whole_muon() {
                // Unfused: stock Muon, stock state.
                optim.with_group(group, muon.build(), clipping.clone())
            } else {
                optim.with_group(
                    group,
                    Segmented::new(spec, muon.build(), fallback.build(), 1),
                    clipping.clone(),
                )
            };
        }

        optim
    }
}
