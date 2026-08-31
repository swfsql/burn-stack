//! Shared training configuration for the examples.
//!
//! [`TrainingConfig`] holds the common hyperparameters (epochs, batch size, LR
//! schedule, seed) plus the [`OptimizerConfig`].  [`optimizer_config`] builds the
//! AdamW defaults shared by the examples (epsilon, grad clipping, cautious
//! weight decay); [`OptimizerConfig::muon`] optionally moves the hidden weight
//! matrices to Muon (see [`crate::optim`]).
//!
//! [`BatchBudget`] is the run-length knob that is *not* part of the config: the
//! `--max-batches` cap, which belongs to the invocation rather than to the
//! persisted hyperparameters.

use burn::{
    optim::{AdamWConfig, ModuleOptimizer, MuonConfig},
    prelude::*,
    train::metric::NumericEntry,
};
use crate::optim::MuonPlan;
pub use crate::utils::scheduler::{ConstantLr, CosineAnnealingLr, Lr};

/// Current value of a metric reading, or `NaN` when the metric has none yet
/// (`Numeric::value` / `running_value` are `None` for metrics that only produce
/// a value at epoch end).
pub fn metric_current(entry: Option<NumericEntry>) -> f64 {
    entry.map_or(f64::NAN, |entry| entry.current())
}

/// How the examples optimize: AdamW everywhere, optionally with Muon on the
/// hidden weight matrices.
///
/// `muon = None` is the plain-AdamW baseline. When set, the model config's
/// [`MuonPlan`] decides which weights move over (and where the fused projections
/// split) — everything else, 1-D and 3-D tensors included, keeps AdamW.
#[derive(Config, Debug)]
pub struct OptimizerConfig {
    /// AdamW: the fallback optimizer, and the one used for every parameter when
    /// `muon` is `None`.
    pub adamw: AdamWConfig,
    /// Muon for the planned hidden matrices. `None` ⇒ AdamW-only.
    pub muon: Option<MuonConfig>,
}

impl OptimizerConfig {
    /// AdamW-only (the baseline).
    pub fn adamw_only(dtype: burn::tensor::DType) -> Self {
        Self::new(optimizer_config(dtype))
    }

    /// AdamW + Muon on `plan`'s weights, sharing AdamW's weight decay and LR
    /// (see [`crate::optim::muon_config`]).
    pub fn with_muon_defaults(self, weight_decay: f32) -> Self {
        self.with_muon(Some(crate::optim::muon_config(weight_decay)))
    }

    /// Build the module optimizer for a model whose Muon plan is `plan`.
    pub fn init(&self, plan: &MuonPlan) -> ModuleOptimizer {
        match &self.muon {
            None => self.adamw.init(),
            Some(muon) => plan.build(&self.adamw, muon),
        }
    }
}

/// A runtime cap on how many training mini-batches a run may take, spanning
/// every epoch (`--max-batches`; `None` ⇒ unlimited).
///
/// It is a *budget*, not a per-epoch limit: the epoch loops take it by `&mut`,
/// bound their `take()` by what is left and spend one per batch, so a cap of
/// 600 stops 600 batches into the run whichever epoch that lands in. Checking
/// [`is_exhausted`](Self::is_exhausted) after each epoch is what breaks the
/// outer loop (the caller still checkpoints and validates first).
///
/// It deliberately lives outside [`TrainingConfig`]: it describes this
/// invocation ("stop early so I can look at it"), not the hyperparameters the
/// artifacts directory persists and a resumed run should inherit.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchBudget {
    /// Batches still allowed, or `None` when uncapped.
    remaining: Option<usize>,
}

impl BatchBudget {
    /// A budget of `max_batches` training mini-batches; `None` ⇒ unlimited.
    pub fn new(max_batches: Option<usize>) -> Self {
        Self {
            remaining: max_batches,
        }
    }

    /// The uncapped budget (what every example ran with before `--max-batches`).
    pub fn unlimited() -> Self {
        Self::new(None)
    }

    /// Whether a cap was given at all.
    pub fn is_capped(&self) -> bool {
        self.remaining.is_some()
    }

    /// Batches still allowed, as an `Iterator::take` count — `usize::MAX` when
    /// uncapped.
    pub fn take_limit(&self) -> usize {
        self.remaining.unwrap_or(usize::MAX)
    }

    /// Batches still allowed, or `None` when uncapped.
    pub fn remaining(&self) -> Option<usize> {
        self.remaining
    }

    /// Whether a cap was given and is now fully spent, i.e. training must stop.
    pub fn is_exhausted(&self) -> bool {
        self.remaining == Some(0)
    }

    /// Charge one mini-batch to the budget (a no-op when uncapped).
    pub fn spend(&mut self) {
        if let Some(remaining) = &mut self.remaining {
            *remaining = remaining.saturating_sub(1);
        }
    }
}

/// Common training hyperparameters shared by the examples.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// The optimizer configuration.
    pub optimizer: OptimizerConfig,
    /// Number of training epochs.
    #[config(default = 1)]
    pub num_epochs: usize,
    /// Mini-batch size.
    #[config(default = 32)]
    pub batch_size: usize,
    /// Number of dataloader worker threads.
    #[config(default = 2)]
    pub num_workers: usize,
    /// Learning-rate schedule.
    #[config(default = "Lr::Constant(ConstantLr::new())")]
    pub lr: Lr,
    /// RNG seed for reproducibility.
    #[config(default = 0)]
    pub seed: u64,
}

/// The AdamW defaults shared by the examples: per-dtype epsilon, gradient
/// clipping at 1.0, and cautious weight decay. `dtype` should be the device's
/// default float dtype (epsilon is sized to it).
pub fn optimizer_config(dtype: burn::tensor::DType) -> AdamWConfig {
    AdamWConfig::new()
        .with_epsilon(crate::utils::div_eps(dtype))
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(
            1.0,
        )))
        .with_cautious_weight_decay(true)
}
