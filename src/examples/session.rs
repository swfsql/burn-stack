//! The control state of a training invocation, threaded through the epoch
//! loops as one [`Session`]:
//!
//! - **where the run stands**: [`TrainingProgress`] (the count of training
//!   steps, the epoch, and the batch within the epoch). It is saved next to
//!   every optimizer checkpoint and restored by `--resume`.
//! - **how far this invocation can go**: the `--max-batches` /
//!   `--max-seconds` [`Budget`].
//! - **how often it checkpoints and validates**: [`Cadence`], the defaults of
//!   each example under the `--checkpoint-every` / `--valid-every` /
//!   `--valid-batches` overrides.
//! - **what it measured**: the [`MetricsLog`], one JSON object per line in
//!   `metrics.jsonl` in the artifacts directory.
//!
//! # Resuming mid-epoch
//!
//! The dataloaders shuffle inside each of their worker threads, and the
//! batches of the workers interleave in the order that they finish. So no run
//! can replay the exact batches that an earlier run trained. A resumed epoch
//! thus trains only its remaining batches, drawn from a **new** shuffle
//! ([`TrainingProgress::shuffle_seed`]): statistically the rest of the epoch,
//! never a replay of its start.

use crate::examples::training::{Budget, Lr};
use burn::data::dataloader::Progress;
use burn::prelude::*;
use burn::train::metric::MetricMetadata;
use std::io::Write;
use std::ops::RangeInclusive;
use std::path::Path;

/// Where a training run stands. It is saved next to every optimizer checkpoint
/// (a position has a meaning only for the state that it was taken with), and
/// read back only under `--resume`.
#[derive(Config, Debug)]
pub struct TrainingProgress {
    /// The count of training steps: the training steps taken, plus the
    /// training steps that [`Session::skip_training_steps`] skipped. The LR
    /// schedule gives the LR of the last training step at this count.
    #[config(default = 0)]
    pub training_step: usize,
    /// The epoch in progress (1-based).
    #[config(default = 1)]
    pub epoch: usize,
    /// Batches of `epoch` already trained.
    #[config(default = 0)]
    pub batch: usize,
}

impl TrainingProgress {
    /// The shuffle seed of the training dataloader: `seed` itself for a new
    /// run, offset by the count of training steps for a resumed run. So the
    /// rest of a resumed epoch is a new draw, not a replay of the batches that
    /// its start already trained.
    pub fn shuffle_seed(&self, seed: u64) -> u64 {
        seed.wrapping_add(self.training_step as u64)
    }
}

/// How often a training loop checkpoints and validates, in training steps.
///
/// Each example sets its own defaults. `--checkpoint-every`, `--valid-every`
/// and `--valid-batches` override them for one invocation (see
/// [`AppArgs::cadence`](crate::examples::cli::AppArgs::cadence)). The
/// end-of-epoch checkpoint is not part of the cadence: it always happens.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cadence {
    /// Training steps between mid-epoch checkpoints. `None` ⇒ only at epoch
    /// ends (and when the budget stops the run).
    pub checkpoint_every: Option<usize>,
    /// Training steps between periodic validations. `None` ⇒ none.
    pub valid_every: Option<usize>,
    /// The batches that a periodic validation reads. `None` ⇒ the whole split.
    pub valid_batches: Option<usize>,
}

/// The training state of one invocation (see the [module docs](self)).
///
/// Built by [`AppArgs::session`](crate::examples::cli::AppArgs::session). A loop
/// calls [`begin_batch`](Self::begin_batch) per dataloader item and
/// [`begin_training_step`](Self::begin_training_step) per training step (the
/// same thing, except for the character LM, whose item is a run of windows).
pub struct Session {
    progress: TrainingProgress,
    budget: Budget,
    cadence: Cadence,
    lr_schedule: Lr,
    /// The LR of the last training step taken.
    lr: f64,
    meta: MetricMetadata,
    log: MetricsLog,
    /// The training steps of the last mid-epoch checkpoint and of the last
    /// periodic validation.
    last_checkpoint: usize,
    last_valid: usize,
}

impl Session {
    /// A session continuing from `progress`, following `lr_schedule`, over a
    /// training split of `items_total` items.
    pub fn new(
        progress: TrainingProgress,
        budget: Budget,
        cadence: Cadence,
        lr_schedule: Lr,
        items_total: usize,
        mut log: MetricsLog,
    ) -> Self {
        lr_schedule.validate();
        let lr = lr_schedule.get_lr(progress.training_step);
        let meta = MetricMetadata {
            progress: Progress::new(0, items_total, None),
            iteration: Some(progress.training_step),
            lr: Some(lr.into()),
        };
        log.start(&progress);
        Self {
            last_checkpoint: progress.training_step,
            last_valid: progress.training_step,
            progress,
            budget,
            cadence,
            lr_schedule,
            lr,
            meta,
            log,
        }
    }

    /// Where the run stands: what [`AppArgs::save_optim`](crate::examples::cli::AppArgs::save_optim)
    /// persists.
    pub fn progress(&self) -> &TrainingProgress {
        &self.progress
    }

    /// The effective cadence (the defaults of the example under the CLI
    /// overrides).
    pub fn cadence(&self) -> Cadence {
        self.cadence
    }

    /// The metric metadata for the updates of the burn metrics.
    pub fn meta(&self) -> &MetricMetadata {
        &self.meta
    }

    /// The epochs left to run, the one in progress first.
    pub fn epochs(&self, num_epochs: usize) -> RangeInclusive<usize> {
        self.progress.epoch..=num_epochs
    }

    /// The dataloader items that this epoch can still take: its remaining
    /// items, capped by the budget.
    pub fn batch_limit(&self, batches_per_epoch: usize) -> usize {
        let left = batches_per_epoch.saturating_sub(self.progress.batch);
        left.min(self.budget.take_limit())
    }

    /// Start the next dataloader item. Returns its 1-based index in the epoch.
    pub fn begin_batch(&mut self) -> usize {
        self.progress.batch += 1;
        self.progress.batch
    }

    /// Start the next training step over `items` items: charge the budget and
    /// advance the LR schedule. Returns the count of training steps and the LR
    /// of this one.
    pub fn begin_training_step(&mut self, items: usize) -> (usize, f64) {
        self.budget.spend();
        self.progress.training_step += 1;
        self.lr = self.lr_schedule.get_lr(self.progress.training_step);
        self.meta.iteration = Some(self.progress.training_step);
        self.meta.progress.items_processed += items;
        self.meta.lr = Some(self.lr.into());
        (self.progress.training_step, self.lr)
    }

    /// Add `training_steps` to the count of training steps, with no training,
    /// so the LR schedule advances too (the character LM uses this for the
    /// windows that its frontier gate dropped).
    pub fn skip_training_steps(&mut self, training_steps: usize) {
        self.progress.training_step += training_steps;
        self.meta.iteration = Some(self.progress.training_step);
    }

    /// Close the epoch that the loop just left. It is complete unless the
    /// budget cut it short. Then the progress stays inside it.
    pub fn end_epoch(&mut self, batches_per_epoch: usize) {
        if !self.budget.is_exhausted() || self.progress.batch >= batches_per_epoch {
            self.progress.epoch += 1;
            self.progress.batch = 0;
        }
    }

    /// Whether the budget is spent, that is, training must stop.
    pub fn is_exhausted(&self) -> bool {
        self.budget.is_exhausted()
    }

    /// Whether a mid-epoch checkpoint is due: the count of training steps
    /// crossed a multiple of `checkpoint_every` since the last one. `true`
    /// once per crossing.
    pub fn checkpoint_due(&mut self) -> bool {
        Self::due(self.cadence.checkpoint_every, self.progress.training_step, &mut self.last_checkpoint)
    }

    /// Whether a periodic validation is due, as [`checkpoint_due`](Self::checkpoint_due).
    pub fn valid_due(&mut self) -> bool {
        Self::due(self.cadence.valid_every, self.progress.training_step, &mut self.last_valid)
    }

    /// Whether the current training step is already validated: by a periodic
    /// validation, or by the initial one when no training step was taken.
    pub fn validated_now(&self) -> bool {
        self.last_valid == self.progress.training_step
    }

    fn due(every: Option<usize>, training_step: usize, last: &mut usize) -> bool {
        let due = every.is_some_and(|every| training_step / every > *last / every);
        if due {
            *last = training_step;
        }
        due
    }

    /// Append a training line: the LR of the training step and `fields` (e.g.
    /// its loss).
    pub fn log_train(&mut self, fields: &[(&str, f64)]) {
        let lr = [("lr", self.lr)];
        self.log.line("train", None, &self.progress, &[&lr, fields]);
    }

    /// Append a validation line for `split` (a name for the evaluation set).
    pub fn log_valid(&mut self, split: &str, fields: &[(&str, f64)]) {
        self.log.line("valid", Some(split), &self.progress, &[fields]);
    }
}

/// Base filename (without extension) of the metrics log.
pub const METRICS_LOG_NAME: &str = "metrics";

/// The append-only metrics log: one flat JSON object per line (`jsonl`). So a
/// curve is one `jq`/pandas call away, not a parse of stdout.
///
/// Every line carries `event` (`start` | `train` | `valid`), the
/// `training_step`, `epoch` and `batch` of the run, and `elapsed` seconds since
/// the start of the session. A `valid` line adds its `split`. Sessions append
/// to the same file, and each one opens with a `start` line (which also records
/// the wall-clock `unix_time`). So a resumed run continues the curve, and a run
/// that starts again from training step 0 shows as a new curve. Non-finite
/// values are written as `null`.
pub struct MetricsLog {
    file: std::fs::File,
    start: std::time::Instant,
}

impl MetricsLog {
    /// Open the log in `artifact_dir` to append to it (create it if absent).
    pub fn open(artifact_dir: &Path) -> Self {
        let path = artifact_dir
            .join(METRICS_LOG_NAME)
            .with_added_extension("jsonl");
        println!("Appending metrics to {path:?}");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .expect("Failed to open the metrics log");
        Self {
            file,
            start: std::time::Instant::now(),
        }
    }

    fn start(&mut self, progress: &TrainingProgress) {
        let unix_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(f64::NAN, |time| time.as_secs_f64());
        self.line("start", None, progress, &[&[("unix_time", unix_time)]]);
    }

    fn line(
        &mut self,
        event: &str,
        split: Option<&str>,
        progress: &TrainingProgress,
        fields: &[&[(&str, f64)]],
    ) {
        let mut line = format!(r#"{{"event":{}"#, json_string(event));
        if let Some(split) = split {
            line += &format!(r#","split":{}"#, json_string(split));
        }
        line += &format!(
            r#","training_step":{},"epoch":{},"batch":{},"elapsed":{}"#,
            progress.training_step,
            progress.epoch,
            progress.batch,
            json_number(self.start.elapsed().as_secs_f64()),
        );
        for (key, value) in fields.iter().flat_map(|fields| fields.iter()) {
            line += &format!(",{}:{}", json_string(key), json_number(*value));
        }
        line += "}\n";
        // One write per line, so an interrupted run leaves whole lines behind.
        self.file
            .write_all(line.as_bytes())
            .expect("Failed to write the metrics log");
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out += "\\\"",
            '\\' => out += "\\\\",
            c if c.is_control() => out += &format!("\\u{:04x}", c as u32),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_number(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "null".to_owned()
    }
}
