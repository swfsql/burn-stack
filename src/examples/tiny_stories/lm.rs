//! The character-level language-model loop, shared by the consumers'
//! `tiny-stories` examples.
//!
//! Everything model-specific is behind [`LmModel`]: an example supplies a
//! wrapper that knows how to take a training step, hop to the inner backend,
//! apply the optimizer, checkpoint itself, and sample text. The epoch loops, the
//! metric printing (in nats *and* bits per character), the checkpoint cadence
//! and the periodic story samples are the same either way, and live here.
//!
//! The outer `train()` — which reads the configs and builds the model — stays in
//! the example, since that is where the model config actually is; the corpus
//! knobs ([`TinyStoriesConfig`], [`Overrides`]) and the dataloaders they decide
//! ([`dataloaders`]) do not depend on the model, so they are here.

use crate::examples::cli::AppArgs;
use crate::examples::tiny_stories::dataset::{
    Split, TinyStoriesBatch, TinyStoriesBatcher, TinyStoriesDataset, VOCAB_SIZE,
};
use crate::examples::training::{TrainingConfig, metric_current};
use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder, Progress},
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
    train::{ClassificationOutput, InferenceStep, TrainStep},
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

/// Corpus knobs forwarded after `--`; each applies on top of the loaded/created
/// [`TinyStoriesConfig`] (and is then persisted with it).
pub struct Overrides {
    /// `--seq-len <usize>`: characters per training window.
    pub seq_len: Option<usize>,
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
/// Implemented on the example's own `TrainStep` wrapper, which is what holds the
/// network and knows its forward path.
pub trait LmModel: TrainStep<Input = TinyStoriesBatch, Output = ClassificationOutput> + Sized {
    /// The inner-backend counterpart used for validation and sampling.
    type Valid: InferenceStep<Input = TinyStoriesBatch, Output = ClassificationOutput>;

    /// Move to the inner (non-autodiff) backend.
    fn valid(&self) -> Self::Valid;

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
    let train_set = TinyStoriesDataset::new(Split::Train, config.train_stories, config.seq_len);
    let valid_set = TinyStoriesDataset::new(Split::Valid, config.valid_stories, config.seq_len);
    println!(
        "corpus: {} train / {} valid characters ({} / {} windows of {})",
        train_set.num_tokens(),
        valid_set.num_tokens(),
        burn_dataset::Dataset::len(&train_set),
        burn_dataset::Dataset::len(&valid_set),
        config.seq_len,
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

/// Train for a single epoch, stepping the optimizer per batch and periodically
/// validating, sampling and checkpointing; returns the updated model.
#[allow(clippy::too_many_arguments)]
pub fn epoch_train<W: LmModel>(
    dataloader_train: Dataloader,
    dataloader_valid: Dataloader,
    mut training_model: W,
    config: &TinyStoriesConfig,
    optim: &mut ModuleOptimizer,
    metric_meta: &mut MetricMetadata,
    epoch: usize,
    training_loop_limit: Option<usize>,
    valid_loop_limit: Option<usize>,
    app_args: &AppArgs,
    valid_device: Device,
) -> W {
    let training_loop_limit = training_loop_limit.unwrap_or(usize::MAX);
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();
    let mut iteration_speed_metric = burn::train::metric::IterationSpeedMetric::new();

    // training loop
    for (mut b, batch) in dataloader_train
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .enumerate()
        .take(training_loop_limit)
    {
        b += 1;
        let [batch_size, _seq_len] = batch.inputs.dims();
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;

        let train_output = TrainStep::step(&training_model, batch);
        let pre_metrics = &train_output.item;

        loss_metric.update(&pre_metrics.adapt(), metric_meta);
        acc_metric.update(&pre_metrics.adapt(), metric_meta);
        iteration_speed_metric.update(&pre_metrics.adapt(), metric_meta);

        let lr = config.training.lr.get_lr(metric_meta.iteration.unwrap());
        training_model = training_model.optim_step(optim, lr, train_output.grads);

        let loss = metric_current(loss_metric.value());
        println!(
            "Epoch {}/{}, Batch {b:0>4}/{}, Loss {loss:.4} ({:.3} bits/char), \
             Acc {:0>6.2}, lr {lr:0>6.2e}, it/s {:.2}",
            epoch,
            config.training.num_epochs,
            dataloader_train.num_items() / config.training.batch_size + 1,
            loss / std::f64::consts::LN_2,
            metric_current(acc_metric.value()),
            metric_current(iteration_speed_metric.value()),
        );

        if b % 100 == 0 {
            // save assets
            training_model.save(app_args);
            app_args.save_optim(optim);

            println!("running validation (batch iteration limit: {valid_loop_limit:?})");
            let valid_model = training_model.valid();
            epoch_valid(
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

/// Run validation over (up to `valid_loop_limit`) batches and report the average
/// loss (also as bits per character) and next-character accuracy.
pub fn epoch_valid<V>(
    dataloader_valid: Dataloader,
    valid_model: &V,
    config: &TinyStoriesConfig,
    epoch: usize,
    valid_loop_limit: Option<usize>,
) where
    V: InferenceStep<Input = TinyStoriesBatch, Output = ClassificationOutput>,
{
    let valid_loop_limit = valid_loop_limit.unwrap_or(usize::MAX);
    let valid_num_items = dataloader_valid.num_items();
    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, valid_num_items, None),
        iteration: Some(0),
        lr: Some(config.training.lr.get_lr(0).into()),
    };

    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();

    // validation loop
    for (_b, batch) in dataloader_valid
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .enumerate()
        .take(valid_loop_limit)
    {
        let [batch_size, _seq_len] = batch.inputs.dims();
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;

        let pre_metrics = InferenceStep::step(valid_model, batch);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);
        acc_metric.update(&pre_metrics.adapt(), &metric_meta);
    }

    // Display the averaged validation metrics
    let loss = metric_current(loss_metric.running_value());
    println!(
        "Epoch {}/{}, Avg Valid Loss {loss:.4} ({:.3} bits/char), Avg Valid Acc: {}",
        epoch,
        config.training.num_epochs,
        loss / std::f64::consts::LN_2,
        metric_current(acc_metric.running_value()),
    );
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
