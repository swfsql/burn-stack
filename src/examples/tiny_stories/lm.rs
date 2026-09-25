//! The character-level language-model loop, shared by the `tiny-stories`
//! examples of the consumer crates.
//!
//! [`LmModel`] holds everything model-specific. An example supplies a wrapper
//! that can train one window from a carried cache, hop to the inner backend,
//! apply the optimizer, checkpoint itself, and sample text. The epoch loops,
//! the metric printing (in nats *and* bits per character), the checkpoint
//! cadence and the periodic story samples are the same for every model, so
//! they live here.
//!
//! The outer `train()` reads the configs and builds the model. It stays in the
//! example, because the model config is there. The corpus knobs
//! ([`TinyStoriesConfig`], [`Overrides`]) and the dataloaders that they decide
//! ([`dataloaders`]) do not depend on the model, so they are here.
//!
//! # Runs, carried state, and the frontier
//!
//! One dataloader item is one **story** (see [`dataset`](super::dataset)).
//! [`epoch_train`] walks it window by window: this is the *run*. It takes one
//! optimizer step per window and **carries the final state** into the next
//! window. So a window is scored from the state that its own prefix produced,
//! as inference always sees it, not from a zero state. The length of the run
//! is that of the story, capped by `run_len`.
//!
//! The carry is *earned*. After each window, a [`Frontier`] gate scores it.
//! When the window fails, the run stops and the rest of the story is
//! discarded. To train the tail of a run on a state that the model got lost in
//! is worth less than a move to the next story.
//!
//! A gate that never opens is not a *wrong* regime. Window `0` is the
//! beginning of the story, so its zero start state is the correct one, and
//! the recurrence still runs the whole window. A closed gate costs **reach**:
//! the model then trains only on the first `seq_len` characters of an example.
//! `run_len = 1` pins it to exactly that.
//!
//! A story is an example of its own, so the state never crosses from one story
//! into the next: a run begins and ends with its story. The [`ClassCursors`]
//! threaded through the windows of a run let the model open that sequence with
//! something of its own. They place the markers of the model in window `0` and
//! in no other window. A model with no markers is unaffected, and every loop
//! here runs the same for it.
//!
//! The carried cache is cut off the graph ([`LmModel::detach_caches`]). So the
//! peak memory is the activations of one window, however deep the run goes.
//! Gradients never cross a window boundary. Backpropagation within the window
//! does not change.
//!
//! # Packed training
//!
//! With [`TinyStoriesConfig::pack`], one train item is a **packed row** of
//! whole stories instead (see [`dataset`](super::dataset)), and a batch is one
//! window. A story then trains whole, from a fresh state, with no carry and no
//! frontier. The model reads the layout from [`TinyStoriesBatch::packed`], and
//! scores with [`lm_output_packed`]. The validation is never packed, so its
//! bits per character do not change with the packing.

use crate::examples::cli::AppArgs;
use crate::examples::device::loader_device;
use crate::examples::tiny_stories::dataset::{
    PackLayout, PackedStoriesBatcher, PackedStoriesDataset, Split, TinyStoriesBatch,
    TinyStoriesBatcher, TinyStoriesDataset, VOCAB_SIZE,
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
    /// Cap on the windows that one story can use. A longer story is truncated
    /// to that many. `usize::MAX` (the default) lets every story run to its
    /// end, and leaves the depth fully to the [`FrontierGate`], which can only
    /// make a run shorter. `1` trains only the opening window of each story:
    /// still its true beginning, but never more of it.
    #[config(default = "usize::MAX")]
    pub run_len: usize,
    /// When the final state of a window is worth carrying into the next one.
    #[config(default = "FrontierGate::default()")]
    pub frontier: FrontierGate,
    /// Stories taken from the train split (~820 characters each).
    #[config(default = 4096)]
    pub train_stories: usize,
    /// Stories taken from the validation split.
    #[config(default = 256)]
    pub valid_stories: usize,
    /// Characters generated at each sampling point.
    #[config(default = 400)]
    pub sample_chars: usize,
    /// Softmax temperature for those samples.
    #[config(default = 0.8)]
    pub sample_temperature: f64,
    /// Train on packed rows (see [`Pack`]). `None`: one story per batch slot.
    /// The validation always takes one story per slot, with `seq_len`,
    /// `run_len` and the batch size of [`Self::training`].
    #[config(default = "None")]
    pub pack: Option<Pack>,
}

impl TinyStoriesConfig {
    /// Items per train batch: rows when packed, else stories.
    pub fn train_batch_size(&self) -> usize {
        match &self.pack {
            Some(pack) => pack.rows,
            None => self.training.batch_size,
        }
    }
}

/// The geometry of packed training (see [`dataset`](super::dataset)).
#[derive(Config, Debug)]
pub struct Pack {
    /// Positions per row (the window). A story longer than a row is
    /// truncated to fit. So a width of at least the longest story (4,149
    /// characters) plus the opening slots of the model keeps every story
    /// whole.
    pub width: usize,
    /// Rows per batch.
    pub rows: usize,
    /// Open rows of the first-fit packer ([`pack_rows`]). `1` keeps the
    /// shuffled order of the stories exactly. More rows fill the rows better,
    /// and move a story ahead by at most this many rows.
    ///
    /// [`pack_rows`]: super::dataset::pack_rows
    #[config(default = 8)]
    pub open_rows: usize,
}

// ===========================================================================
// The frontier gate
// ===========================================================================

/// When the **final state** of a window is worth a carry into the next window
/// (the *frontier* advances), and when to discard the rest of the run.
///
/// It is part of the trainer, never of the model. It reads one scalar per
/// window and decides whether to pass a cache on. No gradient goes near it.
#[derive(Config, Debug)]
pub enum FrontierGate {
    /// Never stall: carry the state through the whole story. This is the pure
    /// stateful-TBPTT baseline. At `run_len = 1`, no next window exists, so
    /// this gate is the same as every other gate.
    Always,
    /// Advance while the window scored at most `max_bits` bits per character.
    ///
    /// The threshold is absolute, so this is a real curriculum, and it acts
    /// from the first window. Until the model is that good, a run trains only
    /// the opening window of its story. The depth then grows out of the
    /// training curve by itself. The cost is one number to pick per corpus.
    /// Pick it *above* the loss where the model settles, or the curriculum
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

/// The running state of the gate: the run-depth statistics that
/// [`epoch_train`] reports (the one number that shows whether the curriculum
/// moves at all).
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
    /// A new gate. [`epoch_train`] clears its statistics at the start of each
    /// epoch (see [`reset_stats`](Self::reset_stats)).
    pub fn new(gate: FrontierGate) -> Self {
        Self {
            gate,
            runs: 0,
            windows: 0,
        }
    }

    /// Score the window just trained, and report whether the run can advance
    /// to `w + 1`. `w` is the index of the window inside the run, and `loss` is
    /// its mean cross-entropy in nats. Under [`FrontierGate::Bits`], a
    /// non-finite loss fails the comparison, so it closes the gate: the safe
    /// direction.
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

    /// The mean number of windows trained per run since the last
    /// [`reset_stats`](Self::reset_stats). `1.0` is a frontier that never opens
    /// (each story trained to its first window and no further). A gate that
    /// never fires gives the mean story length in windows.
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

/// Corpus knobs forwarded after `--`. Each one applies on top of the loaded or
/// created [`TinyStoriesConfig`], and is then saved with it. The batch size and
/// the optimizer are flags of [`AppArgs`].
pub struct Overrides {
    /// `--seq-len <usize>`: characters per training window.
    pub seq_len: Option<usize>,
    /// `--run-len <usize>`: cap on the windows that one story can use
    /// (`1` ⇒ stateless).
    pub run_len: Option<usize>,
    /// `--frontier-bits <f64>`: threshold of the default (absolute) gate.
    pub frontier_bits: Option<f64>,
    /// `--no-frontier`: carry the state through the whole story, with no gate.
    pub no_frontier: bool,
    /// `--train-stories <usize>`: stories taken from the train split.
    pub train_stories: Option<usize>,
    /// `--valid-stories <usize>`: stories taken from the validation split.
    pub valid_stories: Option<usize>,
    /// `--pack <usize>`: train on packed rows of this width.
    pub pack: Option<usize>,
    /// `--pack-rows <usize>`: rows per packed batch.
    pub pack_rows: Option<usize>,
    /// `--pack-open <usize>`: open rows of the packer.
    pub pack_open: Option<usize>,
    /// `--no-pack`: train on one story per batch slot.
    pub no_pack: bool,
}

impl Overrides {
    /// The `--help` lines of these flags, for the example's own help text.
    pub const HELP: &str = concat!(
        "    --seq-len <N>          Characters per training window\n",
        "    --run-len <N>          Cap on the windows that one story can use (1: stateless)\n",
        "    --frontier-bits <B>    Carry the state of a window into the next while it scores at most B bits/char\n",
        "    --no-frontier          Carry the state through the whole story, with no gate\n",
        "    --train-stories <N>    Stories taken from the train split\n",
        "    --valid-stories <N>    Stories taken from the validation split\n",
        "    --pack <W>             Train on packed rows of W positions, each row whole stories (needs --pack-rows once)\n",
        "    --pack-rows <N>        Rows per packed batch\n",
        "    --pack-open <K>        Open rows of the packer (default 8; 1 keeps the shuffled order)\n",
        "    --no-pack              Train on one story per batch slot",
    );

    /// Take these flags out of the parser of the example, over the arguments
    /// forwarded after `--` (see [`AppArgs::extra`]). The parser keeps the
    /// other flags.
    pub fn parse(pargs: &mut pico_args::Arguments) -> Self {
        Overrides {
            seq_len: pargs.opt_value_from_str("--seq-len").unwrap(),
            run_len: pargs.opt_value_from_str("--run-len").unwrap(),
            frontier_bits: pargs.opt_value_from_str("--frontier-bits").unwrap(),
            no_frontier: pargs.contains("--no-frontier"),
            train_stories: pargs.opt_value_from_str("--train-stories").unwrap(),
            valid_stories: pargs.opt_value_from_str("--valid-stories").unwrap(),
            pack: pargs.opt_value_from_str("--pack").unwrap(),
            pack_rows: pargs.opt_value_from_str("--pack-rows").unwrap(),
            pack_open: pargs.opt_value_from_str("--pack-open").unwrap(),
            no_pack: pargs.contains("--no-pack"),
        }
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
        assert!(
            !(self.no_pack && self.pack.is_some()),
            "--pack and --no-pack contradict each other"
        );
        if self.no_pack {
            config.pack = None;
        }
        if let Some(width) = self.pack {
            let rows = self
                .pack_rows
                .or(config.pack.as_ref().map(|pack| pack.rows))
                .expect("--pack needs --pack-rows (rows per batch)");
            let open_rows = config.pack.as_ref().map_or(8, |pack| pack.open_rows);
            config.pack = Some(Pack::new(width, rows).with_open_rows(open_rows));
        }
        match config.pack.as_mut() {
            Some(pack) => {
                if let Some(rows) = self.pack_rows {
                    pack.rows = rows;
                }
                if let Some(open_rows) = self.pack_open {
                    pack.open_rows = open_rows;
                }
            }
            None => assert!(
                self.pack_rows.is_none() && self.pack_open.is_none(),
                "--pack-rows and --pack-open need packed training (--pack)"
            ),
        }
    }
}

/// The interface that the shared loops need from the language model of an
/// example.
///
/// The wrapper of the example implements it. The wrapper holds the network
/// and knows its forward path. Because the loops carry state across windows,
/// it also knows its cache type.
pub trait LmModel: Sized {
    /// The inner-backend counterpart used for validation and sampling.
    type Valid;

    /// The cache collection of the network, carried from one window of a run
    /// to the next.
    type Caches;

    /// Move to the inner (non-autodiff) backend.
    fn valid(&self) -> Self::Valid;

    /// Train one window: run it forward from `caches` (`None` ⇒ a zero state),
    /// score every real position against its next character, and
    /// back-propagate. Returns the gradients and the **final** state of the
    /// window.
    ///
    /// `class` is the class-marker cursor of the run, threaded through its
    /// windows in order. It is an offer, not a requirement. A model that
    /// splices markers into the sequence gets them in window `0` and in no
    /// other window. A model that splices none can pass it straight through
    /// (or ignore it).
    fn train_window(
        &self,
        batch: TinyStoriesBatch,
        caches: Option<Self::Caches>,
        class: &mut ClassCursors,
    ) -> (TrainOutput<ClassificationOutput>, Self::Caches);

    /// Cut a carried cache off the autodiff graph. Then the backward of the
    /// next window stops at its own first token, *and* the activations of this
    /// window are freed.
    ///
    /// Implement it with
    /// [`CacheStack::detach`](crate::modules::CacheStack::detach). A plain
    /// `Tensor::detach` cuts the gradients but frees nothing (see
    /// [`detach_params`](crate::utils::detach_params)). The peak memory would
    /// then grow with the run length, and the window loop must prevent exactly
    /// that.
    fn detach_caches(caches: Self::Caches) -> Self::Caches;

    /// The [`train_window`](Self::train_window) counterpart on the inner
    /// backend: same outputs, no gradients.
    fn valid_window(
        valid: &Self::Valid,
        batch: TinyStoriesBatch,
        caches: Option<Self::Caches>,
        class: &mut ClassCursors,
    ) -> (ClassificationOutput, Self::Caches);

    /// Apply one optimizer step. Returns the updated model.
    fn optim_step(self, optim: &mut ModuleOptimizer, lr: f64, grads: GradientsParams) -> Self;

    /// Checkpoint the wrapped network into the artifacts directory.
    fn save(&self, app_args: &AppArgs);

    /// Sample `n_chars` characters. With a `prompt`, continue it. With `None`,
    /// open a story on the terms of the model. This side only asks for an
    /// unprompted sample and never says how one starts.
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

/// [`dataloaders_for`] a model that places no constraint on packed rows
/// ([`PackLayout::default`]).
pub fn dataloaders(
    config: &TinyStoriesConfig,
    training_device: &Device,
    progress: &TrainingProgress,
) -> (Dataloader, Dataloader) {
    dataloaders_for(config, training_device, progress, PackLayout::default())
}

/// Load the train and validation splits (download them once), and window them
/// into dataloaders. Both build their batches on
/// [`loader_device`]`(training_device)`, and the loops move the batches to the
/// device of the model. The training dataloader shuffles from where
/// `progress` resumes (see [`TrainingProgress::shuffle_seed`]).
///
/// With [`TinyStoriesConfig::pack`], the train split is packed into rows with
/// the `layout` of the model, in the order of that same shuffle. The
/// validation split is never packed.
pub fn dataloaders_for(
    config: &TinyStoriesConfig,
    training_device: &Device,
    progress: &TrainingProgress,
    layout: PackLayout,
) -> (Dataloader, Dataloader) {
    let (seq_len, run_len) = (config.seq_len, config.run_len);
    let batcher = TinyStoriesBatcher::new(seq_len);
    let valid_set = TinyStoriesDataset::new(Split::Valid, config.valid_stories, seq_len, run_len);
    let cap = match run_len {
        usize::MAX => "uncapped".to_owned(),
        _ => format!("capped at {run_len}"),
    };
    let seed = progress.shuffle_seed(config.training.seed);
    // The workers build batches on the host, and the loops move them to the
    // device. A worker that uploads to the GPU from its own thread can
    // invalidate a graph under capture (see `loader_device`).
    let dataloader_train: Dataloader = match &config.pack {
        None => {
            let train_set =
                TinyStoriesDataset::new(Split::Train, config.train_stories, seq_len, run_len);
            println!(
                "corpus: {} train / {} valid characters ({} / {} windows of {seq_len}, \
                 one run per story, {cap})",
                train_set.num_tokens(),
                valid_set.num_tokens(),
                train_set.num_windows(),
                valid_set.num_windows(),
            );
            DataLoaderBuilder::new(batcher.clone())
                .batch_size(config.training.batch_size)
                .shuffle(seed)
                .num_workers(config.training.num_workers)
                .set_device(loader_device(training_device))
                .build(train_set)
        }
        Some(pack) => {
            let train_set = PackedStoriesDataset::new(
                Split::Train,
                config.train_stories,
                pack.width,
                layout,
                pack.open_rows,
                seed,
            );
            let (chars, rows) = (train_set.num_tokens(), train_set.num_rows());
            println!(
                "corpus: {chars} train characters in {rows} packed rows of {} \
                 ({:.3} of the positions, {} open rows, stories at multiples of {}, \
                 {} opening slots)",
                pack.width,
                chars as f64 / (rows * pack.width) as f64,
                pack.open_rows,
                layout.align,
                layout.lead,
            );
            println!(
                "corpus: {} valid characters ({} windows of {seq_len}, one run per story, {cap})",
                valid_set.num_tokens(),
                valid_set.num_windows(),
            );
            DataLoaderBuilder::new(PackedStoriesBatcher::new(pack.width, layout))
                .batch_size(pack.rows)
                .shuffle(seed)
                .num_workers(config.training.num_workers)
                .set_device(loader_device(training_device))
                .build(train_set)
        }
    };
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(config.training.batch_size)
        .shuffle(config.training.seed)
        .num_workers(config.training.num_workers)
        .set_device(loader_device(training_device))
        .build(valid_set);
    (dataloader_train, dataloader_valid)
}

/// The default cadence of the character LMs: every 300 steps, checkpoint, run
/// a 10-batch validation, and sample a story. It counts *steps*, not
/// dataloader iterations, because the window count of a run is that of its
/// story, so it differs from batch to batch.
pub const CADENCE: Cadence = Cadence {
    checkpoint_every: Some(300),
    valid_every: Some(300),
    valid_batches: Some(10),
};

/// Train for (the rest of) one epoch. Walk each story window by window, with
/// one optimizer step per window. Carry the state into the next window while
/// `frontier` admits it. Validate, sample and checkpoint at the cadence of the
/// `session`. Returns the updated model.
///
/// The epoch ends early when the budget of the session (the `--max-batches` /
/// `--max-seconds` caps) runs out. The epoch loop of the caller should then
/// stop, because [`Session::is_exhausted`] is true. The budget is spent per
/// **window**, that is, per optimizer step. The position of the session within
/// the epoch counts batches of stories.
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
    let batches = dataloader_train.num_items().div_ceil(config.train_batch_size());
    frontier.reset_stats();

    // Training loop: one batch of stories per iteration. Every slot advances
    // through its own story in lockstep, for as many windows as the longest
    // story spans.
    for run in dataloader_train
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .take(session.batch_limit(batches))
    {
        let b = session.begin_batch();
        // Built on the host by a worker, moved here (see `loader_device`).
        let run = run.to_device(&valid_device);
        let [batch_size, _windows_seq_len] = run.inputs.dims();
        let windows = run.num_windows();
        let mut caches: Option<W::Caches> = None;
        // The run opens here. Window 0 gets what the model splices in front of
        // a sequence, and the cursor stops every later window from getting it
        // again. Empty for a model that splices nothing.
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
            // `running_value()` the epoch average). Call it on every window:
            // window 0 opens the run in its statistics.
            loss = metric_current(loss_metric.value());
            let acc = metric_current(acc_metric.value());
            session.log_train(&[("loss", loss), ("acc", acc), ("window", w as f64)]);
            let admitted = frontier.admit(w, loss);
            if !admitted || depth == windows || session.is_exhausted() {
                break;
            }
            // Advance the frontier: keep the values of the state, and drop the
            // graph that produced them.
            caches = Some(W::detach_caches(final_caches));
        }

        // This epoch will not see the windows that the gate dropped. So the LR
        // schedule (sized in windows, not in runs) skips them too. Otherwise a
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
                &valid_device,
                config,
                epoch,
                valid_batches,
                session,
            );

            // Sample a story into a new per-step file, to watch the text
            // sharpen from noise into words into sentences. The sampler gets
            // the same seed every time, so successive samples differ only by
            // the model.
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

/// Run validation over up to `valid_loop_limit` batches of stories. Report the
/// average loss (also as bits per character) and the next-character accuracy.
/// The state is threaded through each whole story, *ungated*. This is the one
/// regime that exists, because a story is scored as it is generated: opened
/// once, and never restarted part-way through. The averages also go to the
/// metrics log of the `session`. This function moves each batch to `device`,
/// the device of the model.
#[allow(clippy::too_many_arguments)]
pub fn epoch_valid<W: LmModel>(
    dataloader_valid: Dataloader,
    valid_model: &W::Valid,
    device: &Device,
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
        let run = run.to_device(device);
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
/// - **A longer output is read out, not trimmed.** A forward can return more
///   positions than it got: `lead = out_len - seq_len`, when the model spliced
///   something in front of the window. The *last* of those extra positions
///   stands immediately before the first user token. So it is scored against
///   **that token**: the first character of the story. The earlier `lead - 1`
///   positions are dropped, because their prefix is shorter than any prefix
///   that the sequence has later. `lead = 0` (a model that splices nothing) is
///   the plain case, and nothing here applies.
///
///   The model alone decides whether that position exists and what it puts
///   there. This function only makes sure that, if it *is* the opening of the
///   sequence, it trains as one. Otherwise the first character of each story
///   would be the one character that is never scored. For a model that opens
///   sequences that way, it is also the exact position where an unprompted
///   sample starts.
/// - **Padding never reaches the loss.** Stories differ in length, so `scored`
///   gives the number of leading real positions in each batch slot. The other
///   positions are masked out of the cross-entropy, whose mean is normalized
///   by the real count. They carry [`PAD_TARGET`], so the accuracy is also per
///   real character.
///
/// Every shape here is that of the window, whatever the lengths of its
/// stories. A gather of the real positions would give each window its own row
/// count. On cubecl backends, every distinct shape that a launch sees costs a
/// cached metadata buffer, which slows every later allocation
/// (tracel-ai/burn#5751). For the same reason, the real count is summed on the
/// device, not passed as a host scalar: kernel scalars are also part of the
/// key of that cache.
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

    // Padding, as a flat mask over the `[batch · positions]` axis. The real
    // positions of a slot are its first `scored` positions, and the padding
    // comes after them.
    let scored: Vec<i32> = scored.iter().map(|&n| n as i32).collect();
    let scored_bp = Tensor::<1, Int>::from_ints(scored.as_slice(), &device)
        .reshape([batch_size, 1])
        .expand([batch_size, positions]);
    let pad = Tensor::<1, Int>::arange(0..positions as i64, &device)
        .reshape([1, positions])
        .expand([batch_size, positions])
        .greater_equal(scored_bp)
        .reshape([rows]);
    masked_output(logits.reshape([rows, VOCAB_SIZE]), targets.reshape([rows]), pad)
}

/// [`lm_output`] for a batch of packed rows ([`TinyStoriesBatch::packed`]).
/// The forward keeps the shape of the rows (the opening slots are in them),
/// and `score_bs` marks the scored positions.
pub fn lm_output_packed(
    logits: Tensor<3>,
    targets: Tensor<2, Int>,
    score_bs: Tensor<2, Bool>,
) -> ClassificationOutput {
    let [batch_size, width] = targets.dims();
    assert_eq!([batch_size, width, VOCAB_SIZE], logits.dims());
    assert_eq!([batch_size, width], score_bs.dims());
    let rows = batch_size * width;
    let pad = score_bs.bool_not().reshape([rows]);
    masked_output(logits.reshape([rows, VOCAB_SIZE]), targets.reshape([rows]), pad)
}

/// The cross-entropy of `logits` (`[rows, VOCAB_SIZE]`) against `targets`
/// (`[rows]`), with the rows of `pad` (`true`) left out: the mean over the
/// other rows. Their targets become [`PAD_TARGET`].
fn masked_output(logits: Tensor<2>, targets: Tensor<1, Int>, pad: Tensor<1, Bool>) -> ClassificationOutput {
    let [rows] = targets.dims();
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

/// The target of a padded position in the [`ClassificationOutput`] of
/// [`lm_output`]. It is one past the vocabulary, so no prediction matches it,
/// and `AccuracyMetric::with_pad_token(PAD_TARGET)` leaves it out.
pub const PAD_TARGET: usize = VOCAB_SIZE;

/// The mean validation loss per **character**. The mean of each window
/// (already per real character) is weighted by its real positions. So a late
/// window with a few characters of one long story weighs as those characters,
/// not as a whole window. Burn's `LossMetric` weights by the length of the
/// loss tensor, and the fixed shapes of [`lm_output`] do not tie that length
/// to the real count.
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
