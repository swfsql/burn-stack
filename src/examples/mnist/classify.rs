//! The sequential-MNIST classification loop, shared by the `mnist-class`
//! examples of the consumer crates.
//!
//! [`MnistModel`] holds everything model-specific. An example supplies a
//! wrapper that can take a whole training step (typically that of a
//! [`Trainer`](crate::examples::trainer::Trainer)), hop to the inner backend,
//! checkpoint itself, and turn images into class probabilities. The epoch
//! loops, the metric printing, the checkpoint cadence and the periodic
//! prediction PNGs are the same for every model, so they live here.
//!
//! The outer `train()` reads the configs and builds the dataloaders. It stays
//! in the example, because the model config is there.

use crate::examples::cli::AppArgs;
use crate::examples::mnist::dataset::{MnistBatch, MnistBatcher, MnistDataset};
use crate::examples::mnist::render;
use crate::examples::session::{Cadence, Session};
use crate::examples::training::{TrainingConfig, metric_current};
use burn::optim::ModuleOptimizer;
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, Progress, batcher::Batcher},
    data::dataset::Dataset,
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
    train::{ClassificationOutput, InferenceStep},
};

/// A batched sequential-MNIST dataloader.
pub type Dataloader = std::sync::Arc<dyn DataLoader<MnistBatch> + 'static>;

/// The interface that the shared loops need from the classifier of an
/// example.
///
/// The wrapper of the example implements it. The wrapper holds the network,
/// and knows its forward path and readout position.
pub trait MnistModel {
    /// The inner-backend counterpart used for validation and sampling.
    type Valid: InferenceStep<Input = MnistBatch, Output = ClassificationOutput>;

    /// Move to the inner (non-autodiff) backend.
    fn valid(&self) -> Self::Valid;

    /// One whole training step (forward, backward, optimizer). Returns the
    /// outputs of the batch (for the metrics). Typically the step of a
    /// [`Trainer`](crate::examples::trainer::Trainer), which replays it from a
    /// captured graph under plain SGD.
    fn train_step(&mut self, batch: MnistBatch, optim: &mut ModuleOptimizer, lr: f64) -> ClassificationOutput;

    /// Checkpoint the wrapped network into the artifacts directory.
    fn save(&self, app_args: &AppArgs);

    /// Per-class probabilities `[n, 10]` for `[n, H, W, 1]` images in `[0, 1]`.
    fn predict(valid: &Self::Valid, images_norm: Tensor<4>) -> Tensor<2>;
}

/// Number of fixed test digits sampled for the periodic prediction PNGs.
pub const NUM_SAMPLES: usize = 8;

/// Take the first `n` test digits (normalized to `[0, 1]`) and their labels on
/// `device`. It is a fixed set, so the saved predictions are comparable over
/// time.
pub fn sample_images(n: usize, device: &Device) -> (Tensor<4>, Vec<u8>) {
    let dataset = MnistDataset::test();
    let items: Vec<_> = (0..n).filter_map(|i| dataset.get(i).ok()).collect();
    let labels: Vec<u8> = items.iter().map(|it| it.label).collect();
    let images = MnistBatcher::default().batch(items, device).images_norm();
    (images, labels)
}

/// The default cadence of the MNIST examples: every 100 steps, checkpoint and
/// run a 10-batch validation (plus the prediction PNGs).
pub const CADENCE: Cadence = Cadence {
    checkpoint_every: Some(100),
    valid_every: Some(100),
    valid_batches: Some(10),
};

/// Train for (the rest of) one epoch. Step the optimizer per batch, and
/// checkpoint and validate at the cadence of the `session`. Returns the
/// updated model.
///
/// The epoch ends early when the budget of the session (the `--max-batches` /
/// `--max-seconds` caps) runs out. The epoch loop of the caller should then
/// stop, because [`Session::is_exhausted`] is true.
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

    // A fixed set of test digits (on the validation backend), classified at
    // every small validation, to watch the predictions sharpen.
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

        // Built on the host by a worker, moved here (see `loader_device`).
        let pre_metrics = training_model.train_step(batch.to_device(&valid_device), optim, lr);

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
                &valid_device,
                training_config,
                epoch,
                valid_batches,
                session,
            );

            // Save digit + class-probability PNGs into a new per-step dir.
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

/// Run validation over up to `valid_loop_limit` batches. Report the average
/// loss and accuracy, and log them into the metrics log of the `session`. This
/// function moves each batch to `device`, the device of the model.
#[allow(clippy::too_many_arguments)]
pub fn epoch_valid<V>(
    dataloader_valid: Dataloader,
    valid_model: &V,
    device: &Device,
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

        let pre_metrics = InferenceStep::step(valid_model, batch.to_device(device));
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

/// The cross-entropy classification output for a batch, from the
/// `[batch, 10]` logits of the last timestep.
pub fn classification_output(logits: Tensor<2>, targets: Tensor<1, Int>) -> ClassificationOutput {
    let loss = burn::nn::loss::CrossEntropyLossConfig::new()
        .init(&logits.device())
        .forward(logits.clone(), targets.clone());
    ClassificationOutput::new(loss, logits, targets)
}
