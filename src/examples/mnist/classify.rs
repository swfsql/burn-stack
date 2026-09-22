//! The sequential-MNIST classification loop, shared by the consumers'
//! `mnist-class` examples.
//!
//! Everything model-specific is behind [`MnistModel`]: an example supplies a
//! wrapper that knows how to take a training step, hop to the inner backend,
//! apply the optimizer, checkpoint itself, and turn images into class
//! probabilities. The epoch loops, the metric printing, the checkpoint cadence
//! and the periodic prediction PNGs are the same either way, and live here.
//!
//! The outer `train()` — which reads the configs and builds the dataloaders —
//! stays in the example, since that is where the model config actually is.

use crate::examples::cli::AppArgs;
use crate::examples::mnist::dataset::{MnistBatch, MnistBatcher, MnistDataset};
use crate::examples::mnist::render;
use crate::examples::session::{Cadence, Session};
use crate::examples::training::{TrainingConfig, metric_current};
use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, Progress, batcher::Batcher},
    data::dataset::Dataset,
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
    train::{ClassificationOutput, InferenceStep, TrainStep},
};

/// A batched sequential-MNIST dataloader.
pub type Dataloader = std::sync::Arc<dyn DataLoader<MnistBatch> + 'static>;

/// The seam the shared loops need from an example's classifier.
///
/// Implemented on the example's own `TrainStep` wrapper, which is what holds the
/// network and knows its forward path and readout position.
pub trait MnistModel: TrainStep<Input = MnistBatch, Output = ClassificationOutput> + Sized {
    /// The inner-backend counterpart used for validation and sampling.
    type Valid: InferenceStep<Input = MnistBatch, Output = ClassificationOutput>;

    /// Move to the inner (non-autodiff) backend.
    fn valid(&self) -> Self::Valid;

    /// Apply one optimizer step, returning the updated model.
    fn optim_step(self, optim: &mut ModuleOptimizer, lr: f64, grads: GradientsParams) -> Self;

    /// One whole training step — forward, backward, optimizer — returning the
    /// updated model and the batch's outputs (for the metrics). The default is
    /// [`TrainStep::step`] then [`optim_step`](Self::optim_step); a model
    /// overrides it to fuse the three, e.g. into one captured graph.
    fn train_step(
        self,
        batch: MnistBatch,
        optim: &mut ModuleOptimizer,
        lr: f64,
    ) -> (Self, ClassificationOutput) {
        let output = TrainStep::step(&self, batch);
        (self.optim_step(optim, lr, output.grads), output.item)
    }

    /// Checkpoint the wrapped network into the artifacts directory.
    fn save(&self, app_args: &AppArgs);

    /// Per-class probabilities `[n, 10]` for `[n, H, W, 1]` images in `[0, 1]`.
    fn predict(valid: &Self::Valid, images_norm: Tensor<4>) -> Tensor<2>;
}

/// Number of fixed test digits sampled for the periodic prediction PNGs.
pub const NUM_SAMPLES: usize = 8;

/// Grab the first `n` test digits (normalized to `[0, 1]`) plus their labels on
/// `device` — a fixed set, so the saved predictions are comparable over time.
pub fn sample_images(n: usize, device: &Device) -> (Tensor<4>, Vec<u8>) {
    let dataset = MnistDataset::test();
    let items: Vec<_> = (0..n).filter_map(|i| dataset.get(i).ok()).collect();
    let labels: Vec<u8> = items.iter().map(|it| it.label).collect();
    let images = MnistBatcher::default().batch(items, device).images_norm();
    (images, labels)
}

/// The cadence the MNIST examples default to: checkpoint and run a 10-batch
/// validation (plus the prediction PNGs) every 100 steps.
pub const CADENCE: Cadence = Cadence {
    checkpoint_every: Some(100),
    valid_every: Some(100),
    valid_batches: Some(10),
};

/// Train for (the rest of) one epoch, stepping the optimizer per batch and
/// checkpointing and validating at the `session`'s cadence; returns the updated
/// model.
///
/// The epoch ends early once the session's budget (the `--max-batches` /
/// `--max-seconds` caps) runs out; the caller's epoch loop should then stop, seeing
/// [`Session::is_exhausted`].
#[allow(clippy::too_many_arguments)]
pub fn epoch_train<W: MnistModel>(
    dataloader_train: Dataloader,
    dataloader_valid: Dataloader,
    mut training_model: W,
    training_config: &TrainingConfig,
    optim: &mut ModuleOptimizer,
    session: &mut Session,
    epoch: usize,
    app_args: &AppArgs,
    valid_device: Device,
) -> W {
    let batches = dataloader_train.num_items().div_ceil(training_config.batch_size);
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();
    let mut iteration_speed_metric = burn::train::metric::IterationSpeedMetric::new();

    // A fixed set of test digits (on the validation backend) classified at every
    // small val check, to watch the predictions sharpen.
    let (sample_imgs, sample_labels) = sample_images(NUM_SAMPLES, &valid_device);

    // training loop
    for batch in dataloader_train
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .take(session.batch_limit(batches))
    {
        let b = session.begin_batch();
        let [batch_size, _, _, _] = batch.images.dims();
        let (_step, lr) = session.begin_step(batch_size);

        let (model, pre_metrics) = training_model.train_step(batch, optim, lr);
        training_model = model;

        loss_metric.update(&pre_metrics.adapt(), session.meta());
        acc_metric.update(&pre_metrics.adapt(), session.meta());
        iteration_speed_metric.update(&pre_metrics.adapt(), session.meta());

        let (loss, acc) = (
            metric_current(loss_metric.value()),
            metric_current(acc_metric.value()),
        );
        session.log_train(&[("loss", loss), ("acc", acc)]);
        println!(
            "Epoch {}/{}, Batch {b:0>4}/{batches}, Loss {loss:.4}, Acc {acc:0>6.2}, lr {lr:0>6.2e}, it/s {:.2}",
            epoch,
            training_config.num_epochs,
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
            epoch_valid(
                std::sync::Arc::clone(&dataloader_valid),
                &valid_model,
                training_config,
                epoch,
                valid_batches,
                session,
            );

            // Save digit + class-probability PNGs into a fresh per-step dir.
            let sample_dir = app_args
                .artifacts_path
                .join(format!("epoch-{epoch}-batch-{b}"));
            let probs = W::predict(&valid_model, sample_imgs.clone());
            render::save_predictions(probs, sample_imgs.clone(), &sample_labels, &sample_dir);
            println!("saved prediction samples to {sample_dir:?}");
        }

        if session.is_exhausted() {
            break;
        }
    }

    // Display the averaged training metrics
    println!(
        "Epoch {}/{}, Avg Loss {:.4}, Avg Acc: {}",
        epoch,
        training_config.num_epochs,
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    );
    session.end_epoch(batches);

    training_model
}

/// Run validation over (up to `valid_loop_limit`) batches, report the average
/// loss and accuracy, and log them into the `session`'s metrics log.
pub fn epoch_valid<V>(
    dataloader_valid: Dataloader,
    valid_model: &V,
    training_config: &TrainingConfig,
    epoch: usize,
    valid_loop_limit: Option<usize>,
    session: &mut Session,
) where
    V: InferenceStep<Input = MnistBatch, Output = ClassificationOutput>,
{
    let valid_loop_limit = valid_loop_limit.unwrap_or(usize::MAX);
    let valid_num_items = dataloader_valid.num_items();
    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, valid_num_items, None),
        iteration: Some(0),
        lr: Some(training_config.lr.get_lr(0).into()),
    };

    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();

    // validation loop
    for batch in dataloader_valid
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .take(valid_loop_limit)
    {
        let [batch_size, _, _, _] = batch.images.dims();
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;

        let pre_metrics = InferenceStep::step(valid_model, batch);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);
        acc_metric.update(&pre_metrics.adapt(), &metric_meta);
    }

    // Display the averaged validation metrics
    let (loss, acc) = (
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    );
    let batches = metric_meta.iteration.unwrap() as f64;
    session.log_valid("valid", &[("loss", loss), ("acc", acc), ("batches", batches)]);
    println!(
        "Epoch {}/{}, Avg Valid Loss {loss:.4}, Avg Valid Acc: {acc}",
        epoch, training_config.num_epochs,
    );
}

/// The cross-entropy classification output for a batch, given the last
/// timestep's `[batch, 10]` logits.
pub fn classification_output(logits: Tensor<2>, targets: Tensor<1, Int>) -> ClassificationOutput {
    let loss = burn::nn::loss::CrossEntropyLossConfig::new()
        .init(&logits.device())
        .forward(logits.clone(), targets.clone());
    ClassificationOutput::new(loss, logits, targets)
}
