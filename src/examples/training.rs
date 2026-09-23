//! Shared training configuration for the examples.
//!
//! [`TrainingConfig`] holds the common hyperparameters (epochs, batch size, LR
//! schedule, seed), plus the [`OptimizerConfig`]: a fallback optimizer (AdamW
//! or plain SGD, the one that a captured training step replays), optionally
//! with Muon on the hidden weight matrices (see [`crate::optim`]).
//! [`OptimizerKind`] names the four combinations, and [`OptimizerConfig::of`]
//! builds the defaults of each one ([`optimizer_config`] for AdamW,
//! [`sgd_config`] for SGD).
//!
//! [`Budget`] is the run-length knob that is *not* part of the config: the
//! `--max-batches` and `--max-seconds` caps. They belong to the invocation,
//! not to the saved hyperparameters (the loops reach the budget through their
//! [`Session`](crate::examples::session::Session)).

use burn::{
    optim::{AdamWConfig, ModuleOptimizer, MuonConfig},
    prelude::*,
    train::metric::NumericEntry,
};
use crate::optim::{FallbackConfig, MuonPlan, SgdConfig, muon_config};
pub use crate::utils::scheduler::{ConstantLr, CosineAnnealingLr, Lr};
use std::time::{Duration, Instant};

/// Current value of a metric reading, or `NaN` when the metric has none yet
/// (`Numeric::value` / `running_value` are `None` for metrics that only produce
/// a value at epoch end).
pub fn metric_current(entry: Option<NumericEntry>) -> f64 {
    entry.map_or(f64::NAN, |entry| entry.current())
}

/// How the examples optimize: a fallback optimizer, optionally with Muon on
/// the hidden weight matrices.
///
/// `muon = None` puts every parameter on the fallback. When set, the
/// [`MuonPlan`] of the model config decides which weights move to Muon (and
/// where the fused projections split). Everything else, 1-D and 3-D tensors
/// included, keeps the fallback.
#[derive(Config, Debug)]
pub struct OptimizerConfig {
    /// The optimizer of every parameter that Muon does not own.
    pub fallback: FallbackConfig,
    /// Muon for the planned hidden matrices. `None` ⇒ the fallback everywhere.
    pub muon: Option<MuonConfig>,
}

/// The optimizer of a run: the four [`OptimizerConfig`] shapes, and what
/// `--adamw` / `--sgd` / `--muon` select (see
/// [`AppArgs::optimizer`](crate::examples::cli::AppArgs::optimizer)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerKind {
    /// AdamW everywhere.
    AdamW,
    /// Plain SGD everywhere: the one optimizer that a captured training step
    /// replays (see [`crate::optim::sgd`]).
    Sgd,
    /// Muon on the planned hidden matrices, AdamW on the rest.
    MuonAdamW,
    /// Muon on the planned hidden matrices, plain SGD on the rest.
    MuonSgd,
}

impl OptimizerConfig {
    /// The defaults of the examples for `kind`: [`optimizer_config`] or
    /// [`sgd_config`] as the fallback, and a paired Muon ([`muon_config`])
    /// with the weight decay and the LR of the fallback. `dtype` sizes the
    /// epsilon of AdamW.
    pub fn of(kind: OptimizerKind, dtype: burn::tensor::DType) -> Self {
        let adamw = || FallbackConfig::AdamW(optimizer_config(dtype));
        let sgd = sgd_config();
        let sgd_weight_decay = sgd.weight_decay.unwrap_or(0.0);
        match kind {
            OptimizerKind::AdamW => Self::new(adamw()),
            OptimizerKind::Sgd => Self::new(sgd.into()),
            OptimizerKind::MuonAdamW => {
                Self::new(adamw()).with_muon(Some(muon_config(ADAMW_WEIGHT_DECAY)))
            }
            OptimizerKind::MuonSgd => {
                Self::new(sgd.into()).with_muon(Some(muon_config(sgd_weight_decay)))
            }
        }
    }

    /// Which of the four shapes this is.
    pub fn kind(&self) -> OptimizerKind {
        match (&self.fallback, self.muon.is_some()) {
            (FallbackConfig::AdamW(_), false) => OptimizerKind::AdamW,
            (FallbackConfig::Sgd(_), false) => OptimizerKind::Sgd,
            (FallbackConfig::AdamW(_), true) => OptimizerKind::MuonAdamW,
            (FallbackConfig::Sgd(_), true) => OptimizerKind::MuonSgd,
        }
    }

    /// The SGD of an SGD-only config: what a captured training step replays.
    /// `None` under any other shape, Muon + SGD included (Muon has state).
    pub fn plain_sgd(&self) -> Option<&SgdConfig> {
        match (&self.fallback, &self.muon) {
            (FallbackConfig::Sgd(sgd), None) => Some(sgd),
            _ => None,
        }
    }

    /// Build the module optimizer for a model whose Muon plan is `plan`.
    pub fn init(&self, plan: &MuonPlan) -> ModuleOptimizer {
        match &self.muon {
            None => self.fallback.init(),
            Some(muon) => plan.build(&self.fallback, muon),
        }
    }
}

/// How far one training invocation can go, across every epoch: a cap on its
/// mini-batches (`--max-batches`) and one on its wall-clock time
/// (`--max-seconds`). Each `None` ⇒ unlimited.
///
/// It is a *budget*, not a per-epoch limit. The [`Session`] of the epoch loops
/// holds it. The session bounds their `take()` by the remaining batches, and
/// spends one per optimizer step. So a cap of 600 stops 600 steps into the
/// run, in whichever epoch that lands. The clock starts at the first step
/// (data loading and an initial validation are free). A loop sees it run out
/// when it checks [`Session::is_exhausted`] after each step. A check after
/// each epoch breaks the outer loop (the caller still checkpoints and
/// validates first).
///
/// [`Session`]: crate::examples::session::Session
/// [`Session::is_exhausted`]: crate::examples::session::Session::is_exhausted
///
/// It lives outside [`TrainingConfig`] on purpose. It describes this
/// invocation ("stop early so I can look at it"), not the hyperparameters that
/// the artifacts directory saves and that a resumed run should inherit.
#[derive(Debug, Clone, Copy, Default)]
pub struct Budget {
    /// Batches still allowed, or `None` when uncapped.
    remaining: Option<usize>,
    /// The time allowed, or `None` when uncapped.
    max_time: Option<Duration>,
    /// When the first batch was charged.
    started: Option<Instant>,
}

impl Budget {
    /// A budget of `max_batches` training mini-batches and `max_time` of wall
    /// clock. `None` ⇒ unlimited.
    pub fn new(max_batches: Option<usize>, max_time: Option<Duration>) -> Self {
        Self {
            remaining: max_batches,
            max_time,
            started: None,
        }
    }

    /// The uncapped budget.
    pub fn unlimited() -> Self {
        Self::new(None, None)
    }

    /// Whether a cap was given at all.
    pub fn is_capped(&self) -> bool {
        self.remaining.is_some() || self.max_time.is_some()
    }

    /// Batches still allowed, as an `Iterator::take` count: `usize::MAX` when
    /// uncapped.
    pub fn take_limit(&self) -> usize {
        self.remaining.unwrap_or(usize::MAX)
    }

    /// Batches still allowed, or `None` when uncapped.
    pub fn remaining(&self) -> Option<usize> {
        self.remaining
    }

    /// Whether a cap was given and is fully spent, that is, training must stop.
    pub fn is_exhausted(&self) -> bool {
        let out_of_time = match (self.max_time, self.started) {
            (Some(max_time), Some(started)) => started.elapsed() >= max_time,
            _ => false,
        };
        self.remaining == Some(0) || out_of_time
    }

    /// Charge one mini-batch to the budget (the first one starts the clock).
    pub fn spend(&mut self) {
        self.started.get_or_insert_with(Instant::now);
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

/// The weight decay of [`optimizer_config`], which a paired Muon also uses.
pub const ADAMW_WEIGHT_DECAY: f32 = 1e-4;

/// The AdamW defaults shared by the examples: per-dtype epsilon, gradient
/// clipping at 1.0, and cautious weight decay ([`ADAMW_WEIGHT_DECAY`]).
/// `dtype` should be the default float dtype of the device (the epsilon is
/// sized to it).
pub fn optimizer_config(dtype: burn::tensor::DType) -> AdamWConfig {
    AdamWConfig::new()
        .with_epsilon(crate::utils::div_eps(dtype))
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(
            1.0,
        )))
        .with_weight_decay(ADAMW_WEIGHT_DECAY)
        .with_cautious_weight_decay(true)
}

/// The SGD defaults shared by the examples: the gradient clipping of
/// [`optimizer_config`] (at 1.0), and no weight decay.
pub fn sgd_config() -> SgdConfig {
    SgdConfig::new().with_grad_clipping(Some(
        burn::grad_clipping::GradientClippingConfig::Value(1.0),
    ))
}
