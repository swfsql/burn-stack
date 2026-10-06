// copied from:
// https://github.com/huy209vn/burn-jepa/blob/588d3654fbcfdcfce2ecdb7bcaf7a2e5bd5a70ea/src/train/scheduler.rs
// slight adaptions: added Config derives and a unified enum.

//! Learning rate schedulers for controlling the optimization process.
//!
//! [`Lr`] is the schedule of a run: the rate as a function of the global
//! step. The primitive kinds are a cosine annealing with a linear warmup, a
//! constant rate and a linear ramp. [`Lr::Sequence`] puts schedules one after
//! another. Each segment starts at a global step, and its schedule counts the
//! steps from there. So one config can hold a whole run: a warmup, constant
//! stages that step down, and a cooldown.

use burn::prelude::*;
use std::f64::consts::PI;

/// A learning-rate schedule, dispatching to one of the concrete schedulers.
#[derive(Config, Debug)]
pub enum Lr {
    /// Cosine annealing with linear warmup. See [`CosineAnnealingLr`].
    CosineAnnealing(CosineAnnealingLr),
    /// Fixed learning rate for all steps. See [`ConstantLr`].
    Constant(ConstantLr),
    /// A linear ramp. See [`LinearLr`].
    Linear(LinearLr),
    /// Schedules one after another. See [`LrSegment`].
    Sequence(Vec<LrSegment>),
}

impl Lr {
    /// Learning rate for the given (0-indexed) training `step`.
    pub fn get_lr(&self, step: usize) -> f64 {
        match self {
            Lr::CosineAnnealing(inner) => inner.get_lr(step),
            Lr::Constant(inner) => inner.get_lr(step),
            Lr::Linear(inner) => inner.get_lr(step),
            Lr::Sequence(segments) => {
                let segment = segments
                    .iter()
                    .rev()
                    .find(|segment| segment.from <= step)
                    .expect("the first segment of a sequence starts at step 0");
                segment.lr.get_lr(step - segment.from)
            }
        }
    }

    /// Panic if the schedule cannot run. A sequence must have segments, its
    /// first segment must start at step 0, and the starts must increase. The
    /// check includes the schedule of each segment.
    pub fn validate(&self) {
        let Lr::Sequence(segments) = self else {
            return;
        };
        assert!(!segments.is_empty(), "an LR sequence needs at least one segment");
        assert_eq!(segments[0].from, 0, "the first segment of an LR sequence must start at step 0");
        for pair in segments.windows(2) {
            assert!(
                pair[0].from < pair[1].from,
                "the segments of an LR sequence must start at increasing steps ({} then {})",
                pair[0].from,
                pair[1].from
            );
        }
        segments.iter().for_each(|segment| segment.lr.validate());
    }

    /// The highest rate that the schedule sets.
    pub fn peak(&self) -> f64 {
        match self {
            Lr::CosineAnnealing(inner) => inner.max_lr.max(inner.min_lr),
            Lr::Constant(inner) => inner.lr,
            Lr::Linear(inner) => inner.start.max(inner.end),
            Lr::Sequence(segments) => segments.iter().map(|segment| segment.lr.peak()).fold(0.0, f64::max),
        }
    }

    /// Replace the peak rate with `peak` (`--max-lr`). A cosine annealing
    /// gets `peak` as its `max_lr` (its `min_lr` does not change). A constant
    /// rate becomes `peak`. A ramp or a sequence scales all its rates by one
    /// factor, so its shape does not change.
    pub fn set_peak(&mut self, peak: f64) {
        match self {
            Lr::CosineAnnealing(inner) => inner.max_lr = peak,
            Lr::Constant(inner) => inner.lr = peak,
            Lr::Linear(_) | Lr::Sequence(_) => {
                let old = self.peak();
                assert!(old > 0.0, "a schedule with a peak of 0 cannot get the peak {peak}");
                self.scale_rates(peak / old);
            }
        }
    }

    /// Multiply every rate of the schedule by `factor`.
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
            Lr::Sequence(segments) => segments.iter_mut().for_each(|segment| segment.lr.scale_rates(factor)),
        }
    }

    /// Multiply every step count of the schedule by `num / den` (integer
    /// division). A cosine annealing at the top level keeps its warmup unless
    /// `warmup` is set. A sequence scales as a whole: the starts of its
    /// segments and every length in them, the warmups too.
    pub fn scale_steps(&mut self, num: usize, den: usize, warmup: bool) {
        let scale = |steps: &mut usize| *steps = *steps * num / den;
        match self {
            Lr::CosineAnnealing(inner) => {
                scale(&mut inner.total_steps);
                if warmup {
                    scale(&mut inner.warmup_steps);
                }
            }
            Lr::Constant(_) => {}
            Lr::Linear(inner) => scale(&mut inner.steps),
            Lr::Sequence(segments) => {
                for segment in segments {
                    scale(&mut segment.from);
                    segment.lr.scale_steps(num, den, true);
                }
            }
        }
    }
}

/// # Cosine Annealing Learning Rate Scheduler with Linear Warmup.
///
/// This scheduler:
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
    pub total_steps: usize,
    /// The number of warmup steps
    #[config(default = 0)]
    pub warmup_steps: usize,
}

impl CosineAnnealingLr {
    /// Get the learning rate for the current training step.
    ///
    /// # Arguments
    /// * `step` - Current training step (0-indexed)
    ///
    /// # Returns
    /// * Learning rate for this step
    pub fn get_lr(&self, step: usize) -> f64 {
        // Warmup phase: linear increase from 0 to max_lr
        if step < self.warmup_steps {
            return self.max_lr * (step as f64) / (self.warmup_steps as f64);
        }

        // After total_steps, return min_lr
        if step >= self.total_steps {
            return self.min_lr;
        }

        // Cosine annealing phase
        let progress =
            (step - self.warmup_steps) as f64 / (self.total_steps - self.warmup_steps) as f64;
        self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (PI * progress).cos())
    }
}

/// # Constant Learning Rate Scheduler.
///
/// Simply returns a fixed learning rate for all steps.
/// Useful for simple experiments or when learning rate scheduling is not needed.
#[derive(Config, Debug)]
pub struct ConstantLr {
    /// The fixed learning rate returned for every step.
    #[config(default = 1e-4)]
    pub lr: f64,
}

impl ConstantLr {
    /// Returns the fixed learning rate (independent of `step`).
    pub fn get_lr(&self, _step: usize) -> f64 {
        self.lr
    }
}

/// # Linear Learning Rate Ramp.
///
/// The rate goes linearly from `start` at step 0 to `end` at step `steps`,
/// and stays at `end` after it. A warmup is a ramp from 0. A linear cooldown
/// is a ramp down.
#[derive(Config, Debug)]
pub struct LinearLr {
    /// The rate at step 0.
    pub start: f64,
    /// The rate at step `steps` and after it.
    pub end: f64,
    /// The length of the ramp. With 0, every step gets `end`.
    pub steps: usize,
}

impl LinearLr {
    /// The rate at `step`.
    pub fn get_lr(&self, step: usize) -> f64 {
        if step >= self.steps {
            return self.end;
        }
        self.start + (self.end - self.start) * step as f64 / self.steps as f64
    }
}

/// One segment of [`Lr::Sequence`]: the schedule `lr` from the global step
/// `from` until the next segment starts.
#[derive(Config, Debug)]
pub struct LrSegment {
    /// The global step where the segment starts.
    pub from: usize,
    /// The schedule of the segment. It counts the steps from `from`: its step
    /// 0 is the global step `from`.
    pub lr: Lr,
}

#[cfg(test)]
mod tests;
