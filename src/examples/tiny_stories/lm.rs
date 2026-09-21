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
//! One dataloader item is one **story** (see [`dataset`](super::dataset)), and
//! [`epoch_train`] walks it window by window — the *run* — taking one optimizer
//! step per window and **carrying the final state** into the next one, so a
//! window is scored from the state its own prefix produced, the way inference
//! always sees it, instead of from a zero state. The run's length is the story's,
//! capped by `run_len`.
//!
//! The carry is *earned*: after each window a [`Frontier`] gate scores it and
//! the run is abandoned when it fails, discarding the rest of the story. Training
//! the tail of a run on a state the model got lost in is worth less than moving
//! on. A gate that never opens is not a *wrong* regime — window `0` is the
//! story's own beginning, so the zero state it starts from is the right one, and
//! the recurrence still runs the whole window. What a closed gate costs is
//! **reach**: the model is then only ever trained on the first `seq_len`
//! characters of an example, which is what `run_len = 1` pins it to.
//!
//! A story is an example in its own right, so the state never crosses from one
//! into the next: a run begins where the story does and ends with it. The
//! [`ClassCursors`] threaded through a run's windows are the model's opportunity
//! to open that sequence with something of its own — they place whatever markers
//! it carries in window `0` and in no other. A model carrying none is unaffected
//! by all of this, and every loop here runs the same.
//!
//! The carried cache is cut off the graph ([`LmModel::detach_caches`]), so the
//! peak memory is one window's activations no matter how deep the run goes;
//! gradients never cross a window boundary. Backpropagation within the window is
//! untouched.

use crate::examples::cli::AppArgs;
use crate::examples::tiny_stories::dataset::{
    Split, TinyStoriesBatch, TinyStoriesBatcher, TinyStoriesDataset, VOCAB_SIZE,
};
use crate::examples::session::{Cadence, Session, TrainingProgress};
use crate::examples::training::{TrainingConfig, metric_current};
use crate::utils::ClassCursors;
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
    /// Cap on the windows one story may spend; a longer story is truncated to
    /// that many. `usize::MAX` (the default) lets every story run to its end and
    /// leaves the depth entirely to the [`FrontierGate`], which can only ever
    /// shorten a run. `1` trains each story's opening window only — still its
    /// true beginning, just never more of it.
    #[config(default = "usize::MAX")]
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
    /// Never stall — carry the state through the whole story. The pure
    /// stateful-TBPTT baseline; at `run_len = 1` there is nothing to carry into
    /// and it coincides with every other gate.
    Always,
    /// Advance while the window scored at most `max_bits` bits per character.
    ///
    /// Absolute, so it is a real curriculum, and one that acts from the very
    /// first window: until the model is that good at all, a run trains its
    /// story's opening window and no more, and depth grows out of the training
    /// curve by itself. The price is one number that has to be picked per
    /// corpus — and picked *above* where the model settles, or the curriculum
    /// never starts.
    Bits {
        /// The threshold, in bits per character.
        max_bits: f64,
    },
}

impl Default for FrontierGate {
    fn default() -> Self {
        Self::Bits { max_bits: 1.6 }
    }
}

/// The gate's running state: the run-depth statistics [`epoch_train`] reports
/// (the one number that says whether the curriculum is moving at all).
#[derive(Debug)]
pub struct Frontier {
    /// The configured criterion.
    gate: FrontierGate,
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
            runs: 0,
            windows: 0,
        }
    }

    /// Score the window just trained — `w` is its index inside the run, `loss`
    /// its mean cross-entropy in nats — and report whether the run may advance
    /// to `w + 1`. A non-finite loss fails the comparison, which closes the gate
    /// — the safe direction.
    pub fn admit(&mut self, w: usize, loss: f64) -> bool {
        self.windows += 1;
        if w == 0 {
            self.runs += 1;
        }
        match &self.gate {
            FrontierGate::Always => true,
            FrontierGate::Bits { max_bits } => loss / std::f64::consts::LN_2 <= *max_bits,
        }
    }

    /// Mean windows trained per run since the last [`reset_stats`](Self::reset_stats)
    /// — `1.0` is a frontier that never opens (each story trained to its first
    /// window and no further), the mean story length in windows a gate that never
    /// fires.
    pub fn mean_depth(&self) -> f64 {
        self.windows as f64 / self.runs.max(1) as f64
    }

    /// Clear the depth statistics (the epoch loop does this per epoch, like the
    /// loss and accuracy metrics).
    pub fn reset_stats(&mut self) {
        self.runs = 0;
        self.windows = 0;
    }
}

/// Corpus knobs forwarded after `--`; each applies on top of the loaded/created
/// [`TinyStoriesConfig`] (and is then persisted with it).
pub struct Overrides {
    /// `--seq-len <usize>`: characters per training window.
    pub seq_len: Option<usize>,
    /// `--run-len <usize>`: cap on the windows one story may spend
    /// (`1` ⇒ stateless).
    pub run_len: Option<usize>,
    /// `--frontier-bits <f64>`: threshold of the default (absolute) gate.
    pub frontier_bits: Option<f64>,
    /// `--no-frontier`: carry the state through the whole story, ungated.
    pub no_frontier: bool,
    /// `--train-stories <usize>`: stories pulled from the train split.
    pub train_stories: Option<usize>,
    /// `--valid-stories <usize>`: stories pulled from the validation split.
    pub valid_stories: Option<usize>,
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
            frontier_bits: pargs.opt_value_from_str("--frontier-bits").unwrap(),
            no_frontier: pargs.contains("--no-frontier"),
            train_stories: pargs.opt_value_from_str("--train-stories").unwrap(),
            valid_stories: pargs.opt_value_from_str("--valid-stories").unwrap(),
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
        if let Some(max_bits) = self.frontier_bits {
            config.frontier = FrontierGate::Bits { max_bits };
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
    /// every real position against its next character, and back-propagate —
    /// returning the gradients together with the window's **final** state.
    ///
    /// `class` is the run's class-marker cursor, threaded through its windows in
    /// order. It is offered, not imposed: a model that splices markers into the
    /// sequence gets them placed in window `0` and in no other, and one that
    /// splices none can pass it straight through (or ignore it).
    fn train_window(
        &self,
        batch: TinyStoriesBatch,
        caches: Option<Self::Caches>,
        class: &mut ClassCursors,
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
        class: &mut ClassCursors,
    ) -> (ClassificationOutput, Self::Caches);

    /// Apply one optimizer step, returning the updated model.
    fn optim_step(self, optim: &mut ModuleOptimizer, lr: f64, grads: GradientsParams) -> Self;

    /// Checkpoint the wrapped network into the artifacts directory.
    fn save(&self, app_args: &AppArgs);

    /// Sample `n_chars` characters, continuing `prompt` or — with `None` —
    /// opening a story on the model's own terms, whatever those are: this side
    /// only asks for an unprompted sample and never says how one starts.
    /// [`sample::generate`](crate::examples::tiny_stories::sample::generate) is
    /// one answer.
    fn generate(
        valid: &Self::Valid,
        device: &Device,
        prompt: Option<&str>,
        n_chars: usize,
        temperature: f64,
        seed: u64,
    ) -> String;
}

/// Load (downloading once) the train and validation splits and window them into
/// dataloaders. Training batches must live on `training_device` (to match the
/// model weights) and are shuffled from where `progress` resumes (see
/// [`TrainingProgress::shuffle_seed`]); validation runs on its inner backend.
pub fn dataloaders(
    config: &TinyStoriesConfig,
    training_device: &Device,
    progress: &TrainingProgress,
) -> (Dataloader, Dataloader) {
    let (seq_len, run_len) = (config.seq_len, config.run_len);
    let batcher = TinyStoriesBatcher::new(seq_len);
    let train_set = TinyStoriesDataset::new(Split::Train, config.train_stories, seq_len, run_len);
    let valid_set = TinyStoriesDataset::new(Split::Valid, config.valid_stories, seq_len, run_len);
    let cap = match run_len {
        usize::MAX => "uncapped".to_owned(),
        _ => format!("capped at {run_len}"),
    };
    println!(
        "corpus: {} train / {} valid characters ({} / {} windows of {seq_len}, \
         one run per story, {cap})",
        train_set.num_tokens(),
        valid_set.num_tokens(),
        train_set.num_windows(),
        valid_set.num_windows(),
    );
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.training.batch_size)
        .shuffle(progress.shuffle_seed(config.training.seed))
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

/// The cadence the character LMs default to: checkpoint, run a 10-story
/// validation and sample a story every 300 steps. Held in *steps* rather than
/// dataloader iterations, since a run's window count is the story's and so
/// differs from batch to batch.
pub const CADENCE: Cadence = Cadence {
    checkpoint_every: Some(300),
    valid_every: Some(300),
    valid_batches: Some(10),
};

/// Train for (the rest of) one epoch: walk each story window by window, taking
/// one optimizer step per window and carrying the state into the next window for
/// as long as `frontier` admits it; validate, sample and checkpoint at the
/// `session`'s cadence. Returns the updated model.
///
/// The epoch ends early once the session's budget (the `--max-batches` cap) runs
/// out; the caller's epoch loop should then stop, seeing
/// [`Session::is_exhausted`]. The budget is spent per **window** — i.e. per
/// optimizer step, which is what it meant before runs existed — while the
/// session's position within the epoch counts stories.
#[allow(clippy::too_many_arguments)]
pub fn epoch_train<W: LmModel>(
    dataloader_train: Dataloader,
    dataloader_valid: Dataloader,
    mut training_model: W,
    config: &TinyStoriesConfig,
    optim: &mut ModuleOptimizer,
    session: &mut Session,
    frontier: &mut Frontier,
    epoch: usize,
    app_args: &AppArgs,
    valid_device: Device,
) -> W {
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new().with_pad_token(PAD_TARGET);
    let mut iteration_speed_metric = burn::train::metric::IterationSpeedMetric::new();
    let batches = dataloader_train.num_items().div_ceil(config.training.batch_size);
    frontier.reset_stats();

    // training loop: one batch of stories — every slot advancing through its own
    // story in lockstep, for as many windows as the longest of them spans — per
    // iteration.
    for run in dataloader_train
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .take(session.batch_limit(batches))
    {
        let b = session.begin_batch();
        let [batch_size, _windows_seq_len] = run.inputs.dims();
        let windows = run.num_windows();
        let mut caches: Option<W::Caches> = None;
        // The run opens here: window 0 gets whatever the model splices in front
        // of a sequence, and the cursor is what stops every later window from
        // getting it again. Empty for a model that splices nothing.
        let mut class = ClassCursors::stream();
        // Windows trained in this run, and the readouts of the last of them.
        let mut depth = 0;
        let mut loss = f64::NAN;
        let mut lr = f64::NAN;

        for w in 0..windows {
            let (_step, step_lr) = session.begin_step(batch_size);
            lr = step_lr;

            let (train_output, final_caches) =
                training_model.train_window(run.window(w), caches.take(), &mut class);
            let pre_metrics = &train_output.item;

            loss_metric.update(&pre_metrics.adapt(), session.meta());
            acc_metric.update(&pre_metrics.adapt(), session.meta());
            iteration_speed_metric.update(&pre_metrics.adapt(), session.meta());

            training_model = training_model.optim_step(optim, lr, train_output.grads);
            depth = w + 1;

            // The gate scores *this* window (`value()` is the last update,
            // `running_value()` the epoch average) and must be consulted on
            // every one of them: window 0 is what sets its baseline.
            loss = metric_current(loss_metric.value());
            let acc = metric_current(acc_metric.value());
            session.log_train(&[("loss", loss), ("acc", acc), ("window", w as f64)]);
            let admitted = frontier.admit(w, loss);
            if !admitted || depth == windows || session.is_exhausted() {
                break;
            }
            // Advance the frontier: the state's values are kept, the graph that
            // produced them is dropped.
            caches = Some(W::detach_caches(final_caches));
        }

        // Windows the gate dropped are corpus this epoch will not see, so the LR
        // schedule — sized in windows, not in runs — skips them too; otherwise a
        // stalling frontier would leave the cosine unfinished at the last epoch.
        session.skip_steps(windows - depth);

        println!(
            "Epoch {}/{}, Batch {b:0>4}/{batches}, Windows {depth}/{windows} (mean {:.2}), \
             Loss {loss:.4} ({:.3} bits/char), Acc {:0>6.2}, lr {lr:0>6.2e}, it/s {:.2}",
            epoch,
            config.training.num_epochs,
            frontier.mean_depth(),
            loss / std::f64::consts::LN_2,
            metric_current(acc_metric.value()),
            metric_current(iteration_speed_metric.value()),
        );

        if session.checkpoint_due() {
            training_model.save(app_args);
            app_args.save_optim(optim, session.progress());
        }

        if session.valid_due() {
            let valid_batches = session.cadence().valid_batches;
            println!("running validation (batch iteration limit: {valid_batches:?})");
            let valid_model = training_model.valid();
            epoch_valid::<W>(
                std::sync::Arc::clone(&dataloader_valid),
                &valid_model,
                config,
                epoch,
                valid_batches,
                session,
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
                None,
                config.sample_chars,
                config.sample_temperature,
                config.training.seed,
            );
            std::fs::write(&sample_path, &sample).expect("failed to write the sample");
            println!("--- sample ---\n{sample}\n--- saved to {sample_path:?} ---");
        }

        if session.is_exhausted() {
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
    session.end_epoch(batches);

    training_model
}

/// Run validation over (up to `valid_loop_limit`) stories and report the average
/// loss (also as bits per character) and next-character accuracy, with the state
/// threaded through each whole story, *ungated* — the one regime that exists,
/// since a story is scored the way it is generated: opened once and never
/// restarted part-way through. The averages also go to the `session`'s metrics
/// log.
pub fn epoch_valid<W: LmModel>(
    dataloader_valid: Dataloader,
    valid_model: &W::Valid,
    config: &TinyStoriesConfig,
    epoch: usize,
    valid_loop_limit: Option<usize>,
    session: &mut Session,
) {
    let valid_loop_limit = valid_loop_limit.unwrap_or(usize::MAX);
    let valid_num_items = dataloader_valid.num_items();
    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, valid_num_items, None),
        iteration: Some(0),
        lr: Some(config.training.lr.get_lr(0).into()),
    };

    let mut loss_metric = PerCharLoss::default();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new().with_pad_token(PAD_TARGET);

    // validation loop
    let mut batches = 0;
    for run in dataloader_valid
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .take(valid_loop_limit)
    {
        batches += 1;
        let [batch_size, _windows_seq_len] = run.inputs.dims();
        let mut caches: Option<W::Caches> = None;
        let mut class = ClassCursors::stream();

        for w in 0..run.num_windows() {
            metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
            metric_meta.progress.items_processed += batch_size;

            let (output, final_caches) =
                W::valid_window(valid_model, run.window(w), caches.take(), &mut class);
            loss_metric.update(&output);
            acc_metric.update(&output.adapt(), &metric_meta);
            caches = Some(final_caches);
        }
    }

    // Display the averaged validation metrics.
    let loss = loss_metric.value();
    let bits = loss / std::f64::consts::LN_2;
    let acc = metric_current(acc_metric.running_value());
    session.log_valid(
        "valid",
        &[("loss", loss), ("bits", bits), ("acc", acc), ("batches", batches as f64)],
    );
    println!(
        "Epoch {}/{}, Avg Valid Loss {loss:.4} ({bits:.3} bits/char), Avg Valid Acc: {acc}",
        epoch, config.training.num_epochs,
    );
}

/// The next-character cross-entropy for one window of a batch of stories.
///
/// Two things separate it from a plain flatten-and-score:
///
/// - **A longer output is read out, not trimmed.** When a forward returns more
///   positions than it was given (`lead = out_len - seq_len` — the model spliced
///   something in front of the window), the *last* of those extra positions is
///   the one standing immediately before the first user token, so it is scored
///   against **it**: the story's first character. The earlier `lead - 1` are
///   dropped, having a shorter prefix behind them than the sequence ever has
///   again. `lead = 0` — a model that splices nothing — is the plain case, and
///   nothing here fires.
///
///   Whether that position exists, and what the model puts there, is entirely the
///   model's business; this only makes sure that if it *is* the sequence's
///   opening, it is trained as one. Otherwise the corpus's first character would
///   be the one character never scored — and, for a model that opens sequences
///   that way, the exact position an unprompted sample starts from.
/// - **Padding never reaches the loss.** Stories differ in length, so `scored`
///   says how many leading positions of each batch slot are real; the rest are
///   masked out of the cross-entropy, whose mean is normalized by the real count,
///   and carry [`PAD_TARGET`] so the accuracy stays per real character too.
///
/// Every shape here is the window's, whatever its stories' lengths: a gather of
/// the real positions would give each window its own row count, and every
/// distinct shape a launch sees costs a cached metadata buffer on cubecl
/// backends, slowing every later allocation (tracel-ai/burn#5751). For the same
/// reason the real count is summed on the device rather than passed as a host
/// scalar — kernel scalars are part of that cache's key too.
pub fn lm_output(
    logits: Tensor<3>,
    inputs: Tensor<2, Int>,
    targets: Tensor<2, Int>,
    scored: &[usize],
) -> ClassificationOutput {
    let [batch_size, seq_len] = targets.dims();
    let [_, out_len, _] = logits.dims();
    assert_eq!([batch_size, out_len, VOCAB_SIZE], logits.dims());
    assert_eq!(batch_size, scored.len());
    assert!(out_len >= seq_len, "the forward dropped user positions");

    // The opening readout, when this window's forward spliced anything in.
    let lead = out_len - seq_len;
    let (logits, targets, scored) = match lead {
        0 => (logits, targets, scored.to_vec()),
        _ => (
            logits.narrow(1, lead - 1, seq_len + 1),
            Tensor::cat(vec![inputs.narrow(1, 0, 1), targets], 1),
            scored.iter().map(|&n| (n + 1).min(seq_len + 1)).collect(),
        ),
    };
    let positions = logits.dims()[1];
    let rows = batch_size * positions;
    let device = logits.device();
    assert!(
        scored.iter().any(|&n| n > 0),
        "a scored window holds at least one token"
    );

    // Padding, as a flat mask over the `[batch · positions]` axis: a slot's real
    // positions are its first `scored` ones, the padding being appended.
    let scored: Vec<i32> = scored.iter().map(|&n| n as i32).collect();
    let scored_bp = Tensor::<1, Int>::from_ints(scored.as_slice(), &device)
        .reshape([batch_size, 1])
        .expand([batch_size, positions]);
    let pad = Tensor::<1, Int>::arange(0..positions as i64, &device)
        .reshape([1, positions])
        .expand([batch_size, positions])
        .greater_equal(scored_bp)
        .reshape([rows]);

    let logits = logits.reshape([rows, VOCAB_SIZE]);
    let targets = targets.reshape([rows]);
    let nll = burn::tensor::activation::log_softmax(logits.clone(), 1)
        .gather(1, targets.clone().reshape([rows, 1]))
        .reshape([rows])
        .neg()
        .mask_fill(pad.clone(), 0);
    let real = pad.clone().bool_not().float();
    let loss = nll.sum() / real.sum();
    let targets = targets.mask_fill(pad, PAD_TARGET as i64);

    ClassificationOutput::new(loss, logits, targets)
}

/// The target a padded position carries in [`lm_output`]'s
/// [`ClassificationOutput`]: one past the vocabulary, so no prediction matches
/// it and `AccuracyMetric::with_pad_token(PAD_TARGET)` leaves it out.
pub const PAD_TARGET: usize = VOCAB_SIZE;

/// Validation's mean loss per **character**: each window's (already per-real-
/// character) mean weighted by its real positions, so a late window holding a
/// few characters of one long story weighs by those characters rather than as a
/// whole window — Burn's `LossMetric` weights by the loss tensor's length, which
/// [`lm_output`]'s fixed shapes no longer tie to the real count.
#[derive(Default)]
struct PerCharLoss {
    sum: f64,
    count: f64,
}

impl PerCharLoss {
    fn update(&mut self, output: &ClassificationOutput) {
        let count = output
            .targets
            .clone()
            .not_equal_elem(PAD_TARGET as i64)
            .int()
            .sum()
            .into_scalar::<i64>() as f64;
        self.sum += output.loss.clone().into_scalar::<f64>() * count;
        self.count += count;
    }

    fn value(&self) -> f64 {
        self.sum / self.count
    }
}
