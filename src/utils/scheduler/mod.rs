// copied from:
// https://github.com/huy209vn/burn-jepa/blob/588d3654fbcfdcfce2ecdb7bcaf7a2e5bd5a70ea/src/train/scheduler.rs
// slight adaptions: added Config derives and a unified enum.

//! LR schedules: the learning rate of each training step.
//!
//! [`Lr`] is the LR schedule of a run: the rate as a function of the global
//! training step. The primitive kinds are a cosine annealing with a linear
//! warmup, a constant rate and a linear ramp. [`Lr::Chain`] is a chain of
//! LR schedules, one after another. Each schedule of the chain
//! ([`LrLink`]) starts at a global training step, and counts the training
//! steps from there. So one config can hold a whole run: a warmup, constant
//! stages that each go lower, and a cooldown.

use burn::prelude::*;
use std::f64::consts::PI;

/// An LR schedule: one of the primitive LR schedules, or a chain of them.
#[derive(Config, Debug)]
pub enum Lr {
    /// Cosine annealing with linear warmup. See [`CosineAnnealingLr`].
    CosineAnnealing(CosineAnnealingLr),
    /// A fixed learning rate for all training steps. See [`ConstantLr`].
    Constant(ConstantLr),
    /// A linear ramp. See [`LinearLr`].
    Linear(LinearLr),
    /// A chain of LR schedules, one after another. See [`LrLink`].
    Chain(Vec<LrLink>),
}

impl Lr {
    /// Learning rate for the given (0-indexed) `training_step`.
    pub fn get_lr(&self, training_step: usize) -> f64 {
        match self {
            Lr::CosineAnnealing(inner) => inner.get_lr(training_step),
            Lr::Constant(inner) => inner.get_lr(training_step),
            Lr::Linear(inner) => inner.get_lr(training_step),
            Lr::Chain(links) => {
                let link = links
                    .iter()
                    .rev()
                    .find(|link| link.from <= training_step)
                    .expect("the first schedule of an LR chain must start at training step 0");
                link.lr.get_lr(training_step - link.from)
            }
        }
    }

    /// Panic if the LR schedule cannot run. A chain must have at least one
    /// schedule, its first schedule must start at training step 0, and the
    /// starts must increase. The check includes each schedule of the chain.
    pub fn validate(&self) {
        let Lr::Chain(links) = self else {
            return;
        };
        assert!(!links.is_empty(), "an LR chain needs at least one schedule");
        assert_eq!(links[0].from, 0, "the first schedule of an LR chain must start at training step 0");
        for pair in links.windows(2) {
            assert!(
                pair[0].from < pair[1].from,
                "the schedules of an LR chain must start at increasing training steps ({} then {})",
                pair[0].from,
                pair[1].from
            );
        }
        links.iter().for_each(|link| link.lr.validate());
    }

    /// The highest rate that the LR schedule sets.
    pub fn peak(&self) -> f64 {
        match self {
            Lr::CosineAnnealing(inner) => inner.max_lr.max(inner.min_lr),
            Lr::Constant(inner) => inner.lr,
            Lr::Linear(inner) => inner.start.max(inner.end),
            Lr::Chain(links) => links.iter().map(|link| link.lr.peak()).fold(0.0, f64::max),
        }
    }

    /// Replace the peak rate with `peak` (`--max-lr`). A cosine annealing
    /// gets `peak` as its `max_lr` (its `min_lr` does not change). A constant
    /// rate becomes `peak`. A ramp or a chain scales all its rates by one
    /// factor, so its shape does not change.
    pub fn set_peak(&mut self, peak: f64) {
        match self {
            Lr::CosineAnnealing(inner) => inner.max_lr = peak,
            Lr::Constant(inner) => inner.lr = peak,
            Lr::Linear(_) | Lr::Chain(_) => {
                let old = self.peak();
                assert!(old > 0.0, "an LR schedule with a peak of 0 cannot get the peak {peak}");
                self.scale_rates(peak / old);
            }
        }
    }

    /// Multiply every rate of the LR schedule by `factor`.
    pub fn scale_rates(&mut self, factor: f64) {
        match self {
            Lr::CosineAnnealing(inner) => {
                inner.max_lr *= factor;
                inner.min_lr *= factor;
            }
            Lr::Constant(inner) => inner.lr *= factor,
            Lr::Linear(inner) => {
                inner.start *= factor;
                inner.end *= factor;
            }
            Lr::Chain(links) => links.iter_mut().for_each(|link| link.lr.scale_rates(factor)),
        }
    }

    /// Multiply every count of training steps of the LR schedule by
    /// `num / den` (integer division). A cosine annealing outside a chain
    /// keeps its warmup unless `warmup` is set. A chain scales as a whole: the
    /// start of each of its schedules and every length in them, the warmups
    /// too.
    pub fn scale_training_steps(&mut self, num: usize, den: usize, warmup: bool) {
        let scale = |training_steps: &mut usize| *training_steps = *training_steps * num / den;
        match self {
            Lr::CosineAnnealing(inner) => {
                scale(&mut inner.total_training_steps);
                if warmup {
                    scale(&mut inner.warmup_training_steps);
                }
            }
            Lr::Constant(_) => {}
            Lr::Linear(inner) => scale(&mut inner.training_steps),
            Lr::Chain(links) => {
                for link in links {
                    scale(&mut link.from);
                    link.lr.scale_training_steps(num, den, true);
                }
            }
        }
    }
}

/// # Cosine Annealing LR Schedule with Linear Warmup.
///
/// This LR schedule:
/// 1. Linearly increases LR from 0 to `max_lr` during warmup phase
/// 2. Applies cosine annealing from `max_lr` to `min_lr` after warmup
///
/// This is a common pattern in modern deep learning training.
#[derive(Config, Debug)]
pub struct CosineAnnealingLr {
    /// The maximum learning rate (reached after warmup)
    #[config(default = 1e-4)]
    pub max_lr: f64,
    /// The minimum learning rate (reached at end of training)
    #[config(default = 1e-6)]
    pub min_lr: f64,
    /// The total number of training steps
    pub total_training_steps: usize,
    /// The number of training steps in the warmup
    #[config(default = 0)]
    pub warmup_training_steps: usize,
}

impl CosineAnnealingLr {
    /// Get the learning rate for the current training step.
    ///
    /// # Arguments
    /// * `training_step` - Current training step (0-indexed)
    ///
    /// # Returns
    /// * Learning rate for this training step
    pub fn get_lr(&self, training_step: usize) -> f64 {
        // Warmup phase: linear increase from 0 to max_lr
        if training_step < self.warmup_training_steps {
            return self.max_lr * (training_step as f64) / (self.warmup_training_steps as f64);
        }

        // After total_training_steps, return min_lr
        if training_step >= self.total_training_steps {
            return self.min_lr;
        }

        // Cosine annealing phase
        let progress = (training_step - self.warmup_training_steps) as f64
            / (self.total_training_steps - self.warmup_training_steps) as f64;
        self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (PI * progress).cos())
    }
}

/// # Constant LR Schedule.
///
/// Returns a fixed learning rate for all training steps. Use it for simple
/// experiments, or when a run needs no LR schedule.
#[derive(Config, Debug)]
pub struct ConstantLr {
    /// The fixed learning rate returned for every training step.
    #[config(default = 1e-4)]
    pub lr: f64,
}

impl ConstantLr {
    /// Returns the fixed learning rate (independent of `training_step`).
    pub fn get_lr(&self, _training_step: usize) -> f64 {
        self.lr
    }
}

/// # Linear Learning Rate Ramp.
///
/// The rate goes linearly from `start` at training step 0 to `end` at
/// training step `training_steps`, and stays at `end` after it. A warmup is a
/// ramp from 0. A linear cooldown is a ramp down.
#[derive(Config, Debug)]
pub struct LinearLr {
    /// The rate at training step 0.
    pub start: f64,
    /// The rate at training step `training_steps` and after it.
    pub end: f64,
    /// The length of the ramp, in training steps. With 0, every training step
    /// gets `end`.
    pub training_steps: usize,
}

impl LinearLr {
    /// The rate at `training_step`.
    pub fn get_lr(&self, training_step: usize) -> f64 {
        if training_step >= self.training_steps {
            return self.end;
        }
        self.start + (self.end - self.start) * training_step as f64 / self.training_steps as f64
    }
}

/// One schedule of an [`Lr::Chain`]: the LR schedule `lr` from the
/// global training step `from` until the next schedule of the chain starts.
#[derive(Config, Debug)]
pub struct LrLink {
    /// The global training step where this schedule of the chain starts.
    pub from: usize,
    /// The LR schedule. It counts the training steps from `from`: its training
    /// step 0 is the global training step `from`.
    pub lr: Lr,
}

#[cfg(test)]
mod tests;
