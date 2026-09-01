//! The character-level language-model loop, shared by the consumers'
//! `tiny-stories` examples.
//!
//! Everything model-specific is behind [`LmModel`]: an example supplies a
//! wrapper that knows how to train one window from a carried cache, hop to the
//! inner backend, apply the optimizer, checkpoint itself, and sample text. The
//! epoch loops, the metric printing (in nats *and* bits per character), the
//! checkpoint cadence and the periodic story samples are the same either way,
//! and live here.
//!
//! The outer `train()` — which reads the configs and builds the model — stays in
//! the example, since that is where the model config actually is; the corpus
//! knobs ([`TinyStoriesConfig`], [`Overrides`]) and the dataloaders they decide
//! ([`dataloaders`]) do not depend on the model, so they are here.
//!
//! # Runs, carried state, and the frontier
//!
//! One dataloader item is a **run** of `run_len` consecutive windows (see
//! [`dataset`](super::dataset)), and [`epoch_train`] walks it window by window,
//! taking one optimizer step per window and **carrying the final state** into
//! the next one — so a window is scored from the state its own prefix produced,
//! the way inference always sees it, instead of from a zero state.
//!
//! The carry is *earned*: after each window a [`Frontier`] gate scores it and
//! the run is abandoned when it fails, discarding the rest of the item. Training
//! the tail of a run on a state the model got lost in is worth less than moving
//! on — and continuing from a zero state (what `run_len = 1` does everywhere) is
//! the thing this loop exists to stop doing.
//!
//! The carried cache is cut off the graph ([`LmModel::detach_caches`]), so the
//! peak memory is one window's activations no matter how deep the run goes;
//! gradients never cross a window boundary. Backpropagation within the window is
//! untouched.

use crate::examples::cli::AppArgs;
use crate::examples::tiny_stories::dataset::{
    Split, TinyStoriesBatch, TinyStoriesBatcher, TinyStoriesDataset, VOCAB_SIZE,
};
use crate::examples::training::{BatchBudget, TrainingConfig, metric_current};
use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder, Progress},
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
    train::{ClassificationOutput, TrainOutput},
};
use std::ffi::OsString;

/// A batched next-character-window dataloader.
pub type Dataloader = std::sync::Arc<dyn DataLoader<TinyStoriesBatch> + 'static>;

/// The example's configuration: the shared training hyperparameters plus the
/// corpus knobs (which decide both the dataloaders and what gets downloaded).
#[derive(Config, Debug)]
pub struct TinyStoriesConfig {
    /// Optimizer, epochs, batch size, LR schedule, seed.
    pub training: TrainingConfig,
    /// Characters per training window (the BPTT length).
    #[config(default = 256)]
    pub seq_len: usize,
    /// Windows per training item — how far the carried state can reach before
    /// the stream is cut anyway. `1` is the stateless tiling (every window from
    /// a zero state); the gate can only ever shorten a run, never lengthen it.
    #[config(default = 8)]
    pub run_len: usize,
    /// When the final state of a window is worth carrying into the next one.
    #[config(default = "FrontierGate::default()")]
    pub frontier: FrontierGate,
    /// Stories pulled from the train split (~820 characters each).
    #[config(default = 4096)]
    pub train_stories: usize,
    /// Stories pulled from the validation split.
    #[config(default = 256)]
    pub valid_stories: usize,
    /// Characters generated at each sampling point.
    #[config(default = 400)]
    pub sample_chars: usize,
    /// Softmax temperature for those samples.
    #[config(default = 0.8)]
    pub sample_temperature: f64,
}

// ===========================================================================
// The frontier gate
// ===========================================================================

/// When a window's **final state** is worth carrying into the next window (the
/// *frontier* advances) and when the rest of the run should be discarded.
///
/// It is part of the trainer, never of the model: it reads one scalar per
/// window and decides whether a cache is passed on. No gradient goes near it.
#[derive(Config, Debug)]
pub enum FrontierGate {
    /// Never stall — carry the state through the whole run. The pure
    /// stateful-TBPTT baseline, and what a run of `run_len = 1` degenerates to.
    Always,
    /// Advance while the window scored at most `max_bits` bits per character.
    ///
    /// Absolute, so it is a real curriculum: until the model is that good at
    /// all, every run stops at its first window (i.e. today's stateless
    /// training), and depth grows out of the training curve by itself. The price
    /// is one number that has to be picked per corpus.
    Bits {
        /// The threshold, in bits per character.
        max_bits: f64,
    },
    /// Advance while the window scored at most `1 + tol` times the running
    /// baseline of **opening** windows — what this same model scores from a
    /// zero state, EMA'd with `decay` over runs.
    ///
    /// Self-calibrating: the baseline rides the training curve down, so the
    /// question stays the same one from the first epoch to the last — *did the
    /// state we carried do at least as well as starting fresh would have?* A
    /// window that the carried state actively hurt is exactly the one whose
    /// final state is not worth passing on.
    Opening {
        /// Slack over the baseline, as a fraction (`0.05` ⇒ 5% worse still
        /// passes).
        tol: f64,
        /// EMA decay of the baseline, applied once per run.
        decay: f64,
    },
}

impl Default for FrontierGate {
    fn default() -> Self {
        Self::Opening {
            tol: 0.05,
            decay: 0.99,
        }
    }
}

/// The gate's running state: the opening-window baseline it compares against,
/// plus the run-depth statistics [`epoch_train`] reports (the one number that
/// says whether the curriculum is moving at all).
#[derive(Debug)]
pub struct Frontier {
    /// The configured criterion.
    gate: FrontierGate,
    /// EMA of the opening (zero-state) window loss, in nats. `None` until the
    /// first run opens.
    baseline: Option<f64>,
    /// Runs opened so far.
    runs: usize,
    /// Windows actually trained in them.
    windows: usize,
}

impl Frontier {
    /// A fresh gate. The statistics are cumulative over the whole run of
    /// training, not per epoch.
    pub fn new(gate: FrontierGate) -> Self {
        Self {
            gate,
            baseline: None,
            runs: 0,
            windows: 0,
        }
    }

    /// Score the window just trained — `w` is its index inside the run, `loss`
    /// its mean cross-entropy in nats — and report whether the run may advance
    /// to `w + 1`.
    ///
    /// Window `0` also *sets* the baseline: it is the one window in a run that
    /// ran from a zero state, so it is the only fair reference for what the
    /// carried state has to beat. A non-finite loss fails every comparison,
    /// which closes the gate — the safe direction.
    pub fn admit(&mut self, w: usize, loss: f64) -> bool {
        self.windows += 1;
        if w == 0 {
            self.runs += 1;
            self.baseline = Some(match (self.baseline, &self.gate) {
                (Some(prev), FrontierGate::Opening { decay, .. }) => {
                    prev * decay + loss * (1.0 - decay)
                }
                _ => loss,
            });
        }
        match &self.gate {
            FrontierGate::Always => true,
            FrontierGate::Bits { max_bits } => loss / std::f64::consts::LN_2 <= *max_bits,
            FrontierGate::Opening { tol, .. } => {
                let baseline = self.baseline.expect("a run opens at window 0");
                loss <= baseline * (1.0 + tol)
            }
        }
    }

    /// Mean windows trained per run since the last [`reset_stats`](Self::reset_stats)
    /// — `1.0` is a fully stalled frontier, `run_len` a gate that never fires.
    pub fn mean_depth(&self) -> f64 {
        self.windows as f64 / self.runs.max(1) as f64
    }

    /// Clear the depth statistics (the epoch loop does this per epoch, like the
    /// loss and accuracy metrics), **keeping** the baseline: that one tracks the
    /// model's skill, which does not restart with the epoch.
    pub fn reset_stats(&mut self) {
        self.runs = 0;
        self.windows = 0;
    }

    /// The opening-window baseline in bits per character (`NaN` before the first
    /// window). Only meaningful for [`FrontierGate::Opening`].
    pub fn baseline_bits(&self) -> f64 {
        self.baseline.unwrap_or(f64::NAN) / std::f64::consts::LN_2
    }
}

/// Corpus knobs forwarded after `--`; each applies on top of the loaded/created
/// [`TinyStoriesConfig`] (and is then persisted with it).
pub struct Overrides {
    /// `--seq-len <usize>`: characters per training window.
    pub seq_len: Option<usize>,
    /// `--run-len <usize>`: windows per training item (`1` ⇒ stateless).
    pub run_len: Option<usize>,
    /// `--frontier-tol <f64>`: slack of the default (opening-baseline) gate.
    pub frontier_tol: Option<f64>,
    /// `--no-frontier`: carry the state through the whole run, ungated.
    pub no_frontier: bool,
    /// `--train-stories <usize>`: stories pulled from the train split.
    pub train_stories: Option<usize>,
    /// `--valid-stories <usize>`: stories pulled from the validation split.
    pub valid_stories: Option<usize>,
    /// `--epochs <usize>`: passes over the corpus.
    pub epochs: Option<usize>,
    /// `--batch-size <usize>`: windows per optimizer step.
    pub batch_size: Option<usize>,
    /// `--no-muon`: keep the block's hidden weight matrices on AdamW instead of
    /// moving them to Muon (which is the default of both examples).
    pub no_muon: bool,
}

impl Overrides {
    /// Parse the flags out of the arguments forwarded after `--`; anything left
    /// over is a caller error and panics.
    pub fn parse(extra_args: &[OsString]) -> Self {
        let mut pargs = pico_args::Arguments::from_vec(extra_args.to_vec());
        let overrides = Overrides {
            seq_len: pargs.opt_value_from_str("--seq-len").unwrap(),
            run_len: pargs.opt_value_from_str("--run-len").unwrap(),
            frontier_tol: pargs.opt_value_from_str("--frontier-tol").unwrap(),
            no_frontier: pargs.contains("--no-frontier"),
            train_stories: pargs.opt_value_from_str("--train-stories").unwrap(),
            valid_stories: pargs.opt_value_from_str("--valid-stories").unwrap(),
            epochs: pargs.opt_value_from_str("--epochs").unwrap(),
            batch_size: pargs.opt_value_from_str("--batch-size").unwrap(),
            no_muon: pargs.contains("--no-muon"),
        };
        let remaining = pargs.finish();
        assert!(remaining.is_empty(), "unused extra arguments: {remaining:?}");
        overrides
    }

    /// Apply the parsed flags onto `config`.
    pub fn apply(&self, config: &mut TinyStoriesConfig) {
        if let Some(seq_len) = self.seq_len {
            config.seq_len = seq_len;
        }
        if let Some(run_len) = self.run_len {
            config.run_len = run_len;
        }
        if let Some(tol) = self.frontier_tol {
            let FrontierGate::Opening { decay, .. } = config.frontier else {
                panic!("--frontier-tol applies to the opening-baseline gate");
            };
            config.frontier = FrontierGate::Opening { tol, decay };
        }
        if self.no_frontier {
            config.frontier = FrontierGate::Always;
        }
        if let Some(train_stories) = self.train_stories {
            config.train_stories = train_stories;
        }
        if let Some(valid_stories) = self.valid_stories {
            config.valid_stories = valid_stories;
        }
        if let Some(epochs) = self.epochs {
            config.training.num_epochs = epochs;
        }
        if let Some(batch_size) = self.batch_size {
            config.training.batch_size = batch_size;
        }
        if self.no_muon {
            // Muon reuses AdamW's LR and weight decay (`MatchRmsAdamW` sizes its
            // update to AdamW's RMS), so only the optimizer of the planned
            // matrices changes between the two arms.
            config.training.optimizer = config.training.optimizer.clone().with_muon(None);
        }
    }
}

/// The seam the shared loops need from an example's language model.
///
/// Implemented on the example's own wrapper, which is what holds the network and
/// knows its forward path — and, since the loops carry state across windows, its
/// cache type.
pub trait LmModel: Sized {
    /// The inner-backend counterpart used for validation and sampling.
    type Valid;

    /// The network's cache collection, carried from one window of a run to the
    /// next.
    type Caches;

    /// Move to the inner (non-autodiff) backend.
    fn valid(&self) -> Self::Valid;

    /// Train one window: forward it from `caches` (`None` ⇒ a zero state), score
    /// every position against its next character, and back-propagate — returning
    /// the gradients together with the window's **final** state.
    fn train_window(
        &self,
        batch: TinyStoriesBatch,
        caches: Option<Self::Caches>,
    ) -> (TrainOutput<ClassificationOutput>, Self::Caches);

    /// Cut a carried cache off the autodiff graph, so the next window's backward
    /// stops at its own first token *and* this window's activations are freed.
    ///
    /// Implement it with [`CacheStack::detach`](crate::modules::CacheStack::detach):
    /// a plain `Tensor::detach` cuts the gradients but frees nothing (see
    /// [`detach_params`](crate::utils::detach_params)), which would make the peak
    /// memory grow with the run length — the one thing the window loop must not
    /// do.
    fn detach_caches(caches: Self::Caches) -> Self::Caches;

    /// The [`train_window`](Self::train_window) counterpart on the inner
    /// backend: same outputs, no gradients.
    fn valid_window(
        valid: &Self::Valid,
        batch: TinyStoriesBatch,
        caches: Option<Self::Caches>,
    ) -> (ClassificationOutput, Self::Caches);

    /// Apply one optimizer step, returning the updated model.
    fn optim_step(self, optim: &mut ModuleOptimizer, lr: f64, grads: GradientsParams) -> Self;

    /// Checkpoint the wrapped network into the artifacts directory.
    fn save(&self, app_args: &AppArgs);

    /// Continue `prompt` with `n_chars` sampled characters — see
    /// [`sample::generate`](crate::examples::tiny_stories::sample::generate).
    fn generate(
        valid: &Self::Valid,
        device: &Device,
        prompt: &str,
        n_chars: usize,
        temperature: f64,
        seed: u64,
    ) -> String;
}

/// Load (downloading once) the train and validation splits and window them into
/// dataloaders. Training batches must live on `training_device` (to match the
/// model weights); validation runs on its inner backend.
pub fn dataloaders(
    config: &TinyStoriesConfig,
    training_device: &Device,
) -> (Dataloader, Dataloader) {
    let batcher = TinyStoriesBatcher::default();
    let (seq_len, run_len) = (config.seq_len, config.run_len);
    let train_set = TinyStoriesDataset::new(Split::Train, config.train_stories, seq_len, run_len);
    let valid_set = TinyStoriesDataset::new(Split::Valid, config.valid_stories, seq_len, run_len);
    println!(
        "corpus: {} train / {} valid characters ({} / {} windows of {seq_len}, \
         in runs of {run_len})",
        train_set.num_tokens(),
        valid_set.num_tokens(),
        train_set.num_windows(),
        valid_set.num_windows(),
    );
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.training.batch_size)
        .shuffle(config.training.seed)
        .num_workers(config.training.num_workers)
        .set_device(training_device.clone())
        .build(train_set);
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(config.training.batch_size)
        .shuffle(config.training.seed)
        .num_workers(config.training.num_workers)
        .set_device(training_device.clone().inner())
        .build(valid_set);
    (dataloader_train, dataloader_valid)
}

/// Optimizer steps between two checkpoint/validate/sample points. Held in
/// *steps* rather than dataloader iterations so the cadence does not move when
/// [`TinyStoriesConfig::run_len`] does.
const CHECKPOINT_STEPS: usize = 100;

/// Train for a single epoch: walk each run window by window, taking one
/// optimizer step per window and carrying the state into the next window for as
/// long as `frontier` admits it; periodically validate, sample and checkpoint.
/// Returns the updated model.
///
/// The epoch ends early once `batch_budget` (the `--max-batches` cap) runs out;
/// the caller's epoch loop should then stop, seeing
/// [`BatchBudget::is_exhausted`]. The budget is spent per **window** — i.e. per
/// optimizer step, which is what it meant before runs existed.
#[allow(clippy::too_many_arguments)]
pub fn epoch_train<W: LmModel>(
    dataloader_train: Dataloader,
    dataloader_valid: Dataloader,
    mut training_model: W,
    config: &TinyStoriesConfig,
    optim: &mut ModuleOptimizer,
    metric_meta: &mut MetricMetadata,
    frontier: &mut Frontier,
    epoch: usize,
    batch_budget: &mut BatchBudget,
    valid_loop_limit: Option<usize>,
    app_args: &AppArgs,
    valid_device: Device,
) -> W {
    let (seq_len, run_len) = (config.seq_len, config.run_len);
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();
    let mut iteration_speed_metric = burn::train::metric::IterationSpeedMetric::new();
    let checkpoint_batches = CHECKPOINT_STEPS.div_ceil(run_len);
    let batches = dataloader_train.num_items() / config.training.batch_size + 1;
    frontier.reset_stats();

    // training loop: one batch of runs — i.e. up to `run_len` windows, every
    // slot of the batch advancing through its own run in lockstep — per
    // iteration.
    for (mut b, run) in dataloader_train
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .enumerate()
    {
        b += 1;
        let [batch_size, _run_len_seq_len] = run.inputs.dims();
        let mut caches: Option<W::Caches> = None;
        // Windows trained in this run, and the readouts of the last of them.
        let mut depth = 0;
        let mut loss = f64::NAN;
        let mut lr = f64::NAN;

        for w in 0..run_len {
            batch_budget.spend();
            metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
            metric_meta.progress.items_processed += batch_size;

            let (train_output, final_caches) =
                training_model.train_window(run.window(w, seq_len), caches.take());
            let pre_metrics = &train_output.item;

            loss_metric.update(&pre_metrics.adapt(), metric_meta);
            acc_metric.update(&pre_metrics.adapt(), metric_meta);
            iteration_speed_metric.update(&pre_metrics.adapt(), metric_meta);

            lr = config.training.lr.get_lr(metric_meta.iteration.unwrap());
            training_model = training_model.optim_step(optim, lr, train_output.grads);
            depth = w + 1;

            // The gate scores *this* window (`value()` is the last update,
            // `running_value()` the epoch average) and must be consulted on
            // every one of them: window 0 is what sets its baseline.
            loss = metric_current(loss_metric.value());
            let admitted = frontier.admit(w, loss);
            if !admitted || depth == run_len || batch_budget.is_exhausted() {
                break;
            }
            // Advance the frontier: the state's values are kept, the graph that
            // produced them is dropped.
            caches = Some(W::detach_caches(final_caches));
        }

        // Windows the gate dropped are corpus this epoch will not see, so the LR
        // schedule — sized in windows, not in runs — skips them too; otherwise a
        // stalling frontier would leave the cosine unfinished at the last epoch.
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + (run_len - depth));

        println!(
            "Epoch {}/{}, Batch {b:0>4}/{batches}, Windows {depth}/{run_len} (mean {:.2}), \
             Loss {loss:.4} ({:.3} bits/char), Acc {:0>6.2}, lr {lr:0>6.2e}, it/s {:.2}",
            epoch,
            config.training.num_epochs,
            frontier.mean_depth(),
            loss / std::f64::consts::LN_2,
            metric_current(acc_metric.value()),
            metric_current(iteration_speed_metric.value()),
        );

        if b % checkpoint_batches == 0 {
            // save assets
            training_model.save(app_args);
            app_args.save_optim(optim);

            println!("running validation (batch iteration limit: {valid_loop_limit:?})");
            let valid_model = training_model.valid();
            epoch_valid::<W>(
                std::sync::Arc::clone(&dataloader_valid),
                &valid_model,
                config,
                epoch,
                valid_loop_limit,
            );

            // Sample a story into a fresh per-step file, to watch the text
            // sharpen from noise into words into sentences. The sampler is
            // re-seeded identically every time, so successive samples differ by
            // the model alone.
            let sample_path = app_args
                .artifacts_path
                .join(format!("sample-epoch-{epoch}-batch-{b}.txt"));
            let sample = W::generate(
                &valid_model,
                &valid_device,
                crate::examples::tiny_stories::dataset::STORY_SEPARATOR,
                config.sample_chars,
                config.sample_temperature,
                config.training.seed,
            );
            std::fs::write(&sample_path, &sample).expect("failed to write the sample");
            println!("--- sample ---\n{sample}\n--- saved to {sample_path:?} ---");
        }

        if batch_budget.is_exhausted() {
            break;
        }
    }

    // Display the averaged training metrics
    println!(
        "Epoch {}/{}, Avg Loss {:.4}, Avg Acc: {}",
        epoch,
        config.training.num_epochs,
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    );

    training_model
}

/// Run validation over (up to `valid_loop_limit`) runs and report the average
/// loss (also as bits per character) and next-character accuracy **twice**:
///
/// - `[fresh]` — every window scored from a zero state. Independent of
///   `run_len`, so it is the number that stays comparable across configurations
///   and against the example's README.
/// - `[carried]` — the state threaded through the whole run, *ungated*, which is
///   the regime generation actually runs in (it never restarts mid-story).
///
/// The two coincide at `run_len = 1`, and window `0` of a run is shared between
/// them, so the second reading costs `run_len - 1` extra forwards per run and no
/// backward.
pub fn epoch_valid<W: LmModel>(
    dataloader_valid: Dataloader,
    valid_model: &W::Valid,
    config: &TinyStoriesConfig,
    epoch: usize,
    valid_loop_limit: Option<usize>,
) {
    let valid_loop_limit = valid_loop_limit.unwrap_or(usize::MAX);
    let valid_num_items = dataloader_valid.num_items();
    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, valid_num_items, None),
        iteration: Some(0),
        lr: Some(config.training.lr.get_lr(0).into()),
    };

    let mut fresh_loss_metric = burn::train::metric::LossMetric::new();
    let mut fresh_acc_metric = burn::train::metric::AccuracyMetric::new();
    let mut carried_loss_metric = burn::train::metric::LossMetric::new();
    let mut carried_acc_metric = burn::train::metric::AccuracyMetric::new();

    // validation loop
    for (_r, run) in dataloader_valid
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .enumerate()
        .take(valid_loop_limit)
    {
        let [batch_size, _run_len_seq_len] = run.inputs.dims();
        let mut caches: Option<W::Caches> = None;

        for w in 0..config.run_len {
            metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
            metric_meta.progress.items_processed += batch_size;
            let window = run.window(w, config.seq_len);

            let (carried, final_caches) =
                W::valid_window(valid_model, window.clone(), caches.take());
            carried_loss_metric.update(&carried.adapt(), &metric_meta);
            carried_acc_metric.update(&carried.adapt(), &metric_meta);
            caches = Some(final_caches);

            // Window 0 already ran from a zero state: it *is* the fresh reading.
            let fresh = match w {
                0 => carried,
                _ => W::valid_window(valid_model, window, None).0,
            };
            fresh_loss_metric.update(&fresh.adapt(), &metric_meta);
            fresh_acc_metric.update(&fresh.adapt(), &metric_meta);
        }
    }

    // Display the averaged validation metrics, one line per state regime.
    for (tag, loss_metric, acc_metric) in [
        ("fresh", &fresh_loss_metric, &fresh_acc_metric),
        ("carried", &carried_loss_metric, &carried_acc_metric),
    ] {
        let loss = metric_current(loss_metric.running_value());
        println!(
            "Epoch {}/{}, Avg Valid Loss {loss:.4} ({:.3} bits/char), \
             Avg Valid Acc: {} [{tag} state]",
            epoch,
            config.training.num_epochs,
            loss / std::f64::consts::LN_2,
            metric_current(acc_metric.running_value()),
        );
    }
}

/// The next-character cross-entropy for a whole window: `[batch, seq, vocab]`
/// logits against `[batch, seq]` targets, flattened so one window contributes
/// `seq_len` classification examples (which is also what makes the accuracy
/// metric per-character).
pub fn lm_output(logits: Tensor<3>, targets: Tensor<2, Int>) -> ClassificationOutput {
    let [batch_size, seq_len] = targets.dims();
    assert_eq!([batch_size, seq_len, VOCAB_SIZE], logits.dims());

    let logits = logits.reshape([batch_size * seq_len, VOCAB_SIZE]);
    let targets = targets.reshape([batch_size * seq_len]);

    let loss = burn::nn::loss::CrossEntropyLossConfig::new()
        .init(&logits.device())
        .forward(logits.clone(), targets.clone());

    ClassificationOutput::new(loss, logits, targets)
}
