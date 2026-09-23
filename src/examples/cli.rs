//! CLI plumbing shared by the examples:
//!
//! - argument parsing into [`AppArgs`],
//! - artifact-directory management,
//! - load and save of the training config, the model config, the model weights
//!   and the optimizer state (with the [`TrainingProgress`] of the run),
//! - the [`Session`] that a training run threads through its epoch loops.
//!
//! See [`HELP`] for the full command-line behaviour.
//!
//! [`AppArgs`] carries every flag that an example can share. The arguments
//! after `--` belong to the example. It parses them from [`AppArgs::extra`]
//! and closes them with [`finish_extra`].
//!
//! This module cannot know *whose* example runs. So [`AppArgs::parse`] takes
//! the prefix of the auto-created artifacts directory (by convention
//! `concat!(env!("CARGO_PKG_NAME"), "-", env!("CARGO_BIN_NAME"), "-")`,
//! evaluated in the example crate).

#[cfg(test)]
mod tests;

use crate::examples::session::{Cadence, MetricsLog, Session, TrainingProgress};
use crate::examples::training::{Budget, Lr, OptimizerConfig, OptimizerKind, TrainingConfig};
use crate::modules::ModelConfigExt;
use burn::optim::ModuleOptimizer;
use burn::prelude::*;
use burn::store::ModuleRecord;
use burn::tensor::DType;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The `--help` text describing every flag and the train/infer/config flow.
pub const HELP: &str = "\
Burn Example

A command-line tool to train machine learning models and/or to run inference with them.
An artifacts directory keeps the models, the optimizers and the configurations.

USAGE:
    example-name [OPTIONS] [-- <EXTRA_ARGS>...]

Without --training or --inference, the program handles the configurations and then exits.

BEHAVIOR OVERVIEW
- The program manages two configurations: the training config and the model config.
- With --training-config or --model-config, the program loads that config from the given file and saves it to the artifacts directory (it overwrites the existing file).
- Without an explicit config file, the program loads the config from the artifacts directory. If that file is absent, the program creates a default config and saves it.
- The program reads and writes the model weights, the optimizer state and the configurations in the artifacts directory (--artifacts-path). Without --artifacts-path, the program creates a new temporary directory and prints its path.
- With --remove-artifacts and --training, the program deletes the model and optimizer files (and the saved progress) in the artifacts directory before training.
- The program loads the model and optimizer weights from the artifacts directory if they are present. Otherwise it creates new ones and saves them.
- --seed, --epochs, --batch-size and --max-lr replace the value in the training config (loaded or created) before the program saves the config. So later runs from the same artifacts directory inherit the value. --epochs also rescales the length of a cosine LR schedule by the same factor, so the schedule still spans the run. --batch-size rescales its length and warmup by the inverse factor (an epoch has that many fewer steps).
- --adamw, --sgd and --muon choose the optimizer. --muon puts the hidden weight matrices of the model on Muon, and the other flag (default --adamw) optimizes every other parameter. --adamw and --sgd are exclusive. Without these flags, a new training config gets the default optimizer of the example.
- In a loaded config, these flags replace the optimizer and keep the LR schedule (see --max-lr). If the new optimizer would ignore the optimizer state that the old optimizer saved, the program panics. (--remove-artifacts with --training removes that state.)
- Only plain SGD (--sgd alone) has a training step that replays from a captured graph.
- An example that supports it replays its fixed-shape passes (training steps under plain SGD, validation, decoding) from captured CUDA graphs. --no-graph runs every pass eagerly.
- The program saves the optimizer state together with the progress of the run: the LR-schedule step, the epoch, and the batch within it. A run that loads this state starts again at step 0 of epoch 1. With --resume, the run continues from the saved progress. The interrupted epoch then trains only its remaining batches, drawn from a new shuffle (the batch order of the dataloader workers cannot be replayed).
- Training makes a checkpoint at every epoch end and when it stops. --checkpoint-every adds a checkpoint every N optimizer steps. --valid-every adds a validation every N optimizer steps. --valid-batches caps the batches that a validation reads. Each example has its own defaults for these three flags.
- Every training step and every validation appends one JSON line to metrics.jsonl in the artifacts directory. Each run starts with a \"start\" line.
- With both --training and --inference, training runs first. Inference then uses the trained model.
- With --max-batches, training stops after that many mini-batches in total (counted across epochs), and makes a checkpoint as usual before it returns. One mini-batch is one optimizer step. For the character LM, this is one window of a run, not one dataloader item. --max-seconds stops training in the same way, when that much wall-clock time has passed since its first step.
- The program forwards all the arguments after -- unchanged to the flags of the example (-- --help lists them).

FLAGS:
    -h, --help                  Show this help message and exit

OPTIONS:
    -t, --training              Run training (creates or updates the model and the optimizer)
    -i, --inference             Run inference: after training (with both flags), or immediately (with this flag only)
    -r, --remove-artifacts      Delete the model and optimizer files (and the progress) in the artifacts directory
                                before training (no effect without --training)
    -c, --training-config <PATH>
                                Load the training config from this file (overrides the config in the artifacts directory)
    -m, --model-config <PATH>   Load the model config from this file (overrides the config in the artifacts directory)
    -b, --max-batches <N>       Stop training after N mini-batches in total (across epochs), for any configured
                                number of epochs. Unlimited when absent.
        --max-seconds <S>       Stop training when S seconds have passed since its first step. Unlimited when absent.
    -s, --seed <N>              Replace the RNG seed of the training config (model init, data shuffle, sampling)
        --epochs <N>            Replace the number of epochs of the training config (rescales a cosine LR schedule)
        --batch-size <N>        Replace the mini-batch size of the training config (rescales a cosine LR schedule)
        --max-lr <LR>           Replace the peak rate of the LR schedule (the only rate of a constant schedule)
        --adamw                 Optimize with AdamW (with --muon: every parameter that Muon does not own)
        --sgd                   Optimize with plain SGD (with --muon: every parameter that Muon does not own)
        --muon                  Put the hidden weight matrices on Muon
        --no-graph              Run every pass eagerly, not from captured CUDA graphs
        --resume                Continue from the progress saved with the optimizer state (schedule step, epoch,
                                batch), not from step 0 (no effect on a new optimizer)
        --checkpoint-every <N>  Also make a checkpoint every N optimizer steps (0: only at epoch ends)
        --valid-every <N>       Validate every N optimizer steps (0: no periodic validation)
        --valid-batches <N>     The batches that a periodic validation reads
    -a, --artifacts-path <PATH>
                                Directory to save and load the configs, the model weights and the optimizer state.
                                The program creates it if it does not exist.
                                Default: a new temporary directory (the program prints its path).

ARGS:
    -- <EXTRA_ARGS>             The program forwards all the arguments after -- unchanged to the flags of the example.
                                -h or --help there shows the help of the example.
";

/// Parsed command-line arguments. For field descriptions, see [`HELP`].
#[derive(Debug)]
pub struct AppArgs {
    /// Whether to run training.
    pub training: bool,
    /// Whether to run inference.
    pub inference: bool,
    /// Whether to delete existing model/optim artifacts before training.
    pub remove_artifacts: bool,
    /// Optional path to load the training config from.
    pub training_config: Option<PathBuf>,
    /// Optional path to load the model config from.
    pub model_config: Option<PathBuf>,
    /// Directory for configs, model weights, and optimizer state.
    pub artifacts_path: PathBuf,
    /// Optional cap on the total number of training mini-batches. See
    /// [`Budget`].
    pub max_batches: Option<usize>,
    /// Optional cap on the training wall-clock time, in seconds. See
    /// [`Budget`].
    pub max_seconds: Option<f64>,
    /// Optional replacement for the seed of the training config. See
    /// [`AppArgs::override_training_config`].
    pub seed: Option<u64>,
    /// Optional replacement for the number of epochs of the training config.
    /// See [`AppArgs::override_training_config`].
    pub epochs: Option<usize>,
    /// Optional replacement for the mini-batch size of the training config.
    /// See [`AppArgs::override_training_config`].
    pub batch_size: Option<usize>,
    /// Optional replacement for the peak rate of the LR schedule. See
    /// [`AppArgs::override_training_config`].
    pub max_lr: Option<f64>,
    /// The optimizer that `--adamw` / `--sgd` / `--muon` chose, if any. It is
    /// the optimizer of a new config (see [`AppArgs::optimizer_or`]), and the
    /// replacement in a loaded config (see
    /// [`AppArgs::override_training_config`]).
    pub optimizer: Option<OptimizerKind>,
    /// Whether captured graphs are off (`--no-graph`). See [`AppArgs::graphs`].
    pub no_graph: bool,
    /// Whether a loaded optimizer continues from its saved [`TrainingProgress`]
    /// (see [`AppArgs::load_or_save_optim`]) rather than from step `0`.
    pub resume: bool,
    /// Optional override of [`Cadence::checkpoint_every`] (`0` ⇒ none).
    pub checkpoint_every: Option<usize>,
    /// Optional override of [`Cadence::valid_every`] (`0` ⇒ none).
    pub valid_every: Option<usize>,
    /// Optional override of [`Cadence::valid_batches`].
    pub valid_batches: Option<usize>,
    /// Arguments after `--`, forwarded verbatim to downstream processing.
    pub extra_args: Vec<OsString>,
}

impl AppArgs {
    /// Parse [`AppArgs`] from `std::env::args_os` (handles `--`, `-h/--help`).
    ///
    /// `artifact_prefix` names the temporary directory created when
    /// `--artifacts-path` is absent, e.g. `"my-crate-mnist-class-"`.
    pub fn parse(artifact_prefix: &str) -> Result<Self, pico_args::Error> {
        let mut args: Vec<_> = std::env::args_os().collect();
        args.remove(0); // remove the executable path.
        Self::parse_from(args, artifact_prefix)
    }

    /// [`parse`](Self::parse) over `args`, the executable path excluded.
    pub fn parse_from(
        mut args: Vec<OsString>,
        artifact_prefix: &str,
    ) -> Result<Self, pico_args::Error> {
        // Find and process `--`.
        let extra_args = if let Some(dash_dash) = args.iter().position(|arg| arg == "--") {
            // Store all arguments following ...
            let later_args = args.drain(dash_dash + 1..).collect();
            // .. then remove the `--`
            args.pop();
            later_args
        } else {
            Vec::new()
        };

        let mut pargs = pico_args::Arguments::from_vec(args);

        // Help has a higher priority and should be handled separately.
        if pargs.contains(["-h", "--help"]) {
            println!("{}", HELP);
            std::process::exit(0);
        }

        let args = AppArgs {
            training_config: pargs
                .opt_value_from_os_str(["-c", "--training-config"], parse_path)?,
            model_config: pargs.opt_value_from_os_str(["-m", "--model-config"], parse_path)?,
            max_batches: pargs.opt_value_from_str(["-b", "--max-batches"])?,
            max_seconds: pargs.opt_value_from_str("--max-seconds")?,
            seed: pargs.opt_value_from_str(["-s", "--seed"])?,
            epochs: pargs.opt_value_from_str("--epochs")?,
            batch_size: pargs.opt_value_from_str("--batch-size")?,
            max_lr: pargs.opt_value_from_str("--max-lr")?,
            checkpoint_every: pargs.opt_value_from_str("--checkpoint-every")?,
            valid_every: pargs.opt_value_from_str("--valid-every")?,
            valid_batches: pargs.opt_value_from_str("--valid-batches")?,
            artifacts_path: pargs
                .opt_value_from_os_str(["-a", "--artifacts-path"], parse_path)?
                .unwrap_or_else(|| {
                    // The example's `extra` prints its help and exits: nothing
                    // to create a directory for.
                    if extra_args.iter().any(|arg| arg == "-h" || arg == "--help") {
                        return PathBuf::new();
                    }
                    // e.g. /tmp/my-crate-mnist-class-abcd-0
                    let tmp = temp_dir::TempDir::with_prefix(artifact_prefix)
                        .expect("Failed to create the temporary directory")
                        .dont_delete_on_drop();
                    let path = tmp.path();
                    println!("new artifacts directory: {path:?}");
                    path.into()
                }),
            // must parse flags after values
            training: pargs.contains(["-t", "--training"]),
            inference: pargs.contains(["-i", "--inference"]),
            remove_artifacts: pargs.contains(["-r", "--remove-artifacts"]),
            resume: pargs.contains("--resume"),
            optimizer: parse_optimizer(&mut pargs),
            no_graph: pargs.contains("--no-graph"),
            extra_args,
        };

        let remaining = pargs.finish();
        if !remaining.is_empty() {
            panic!("unused arguments: {remaining:?}");
        }

        Ok(args)
    }

    /// The arguments after `--`, for the parser of the example. With `-h` /
    /// `--help` among them, prints `help` and exits. Close the parser with
    /// [`finish_extra`].
    pub fn extra(&self, help: &str) -> pico_args::Arguments {
        let mut pargs = pico_args::Arguments::from_vec(self.extra_args.clone());
        if pargs.contains(["-h", "--help"]) {
            println!("{help}");
            std::process::exit(0);
        }
        pargs
    }

    /// The optimizer of a new training config: the choice of the flags, else
    /// the `default` of the example.
    pub fn optimizer_or(&self, default: OptimizerKind) -> OptimizerKind {
        self.optimizer.unwrap_or(default)
    }

    /// Whether to replay passes from captured graphs (unless `--no-graph`).
    pub fn graphs(&self) -> bool {
        !self.no_graph
    }

    /// The `defaults` of the example, under the `--checkpoint-every` /
    /// `--valid-every` / `--valid-batches` overrides.
    pub fn cadence(&self, defaults: Cadence) -> Cadence {
        let every = |flag: Option<usize>, default| match flag {
            Some(0) => None,
            Some(every) => Some(every),
            None => default,
        };
        Cadence {
            checkpoint_every: every(self.checkpoint_every, defaults.checkpoint_every),
            valid_every: every(self.valid_every, defaults.valid_every),
            valid_batches: self.valid_batches.or(defaults.valid_batches),
        }
    }

    /// Start a training [`Session`] from `progress` (the return value of
    /// [`load_or_save_optim`](Self::load_or_save_optim)). The session follows
    /// the LR schedule of `training` over a training split of `items_total`
    /// items, at the `cadence` defaults of the example (see
    /// [`cadence`](Self::cadence)). Its budget is `--max-batches` and
    /// `--max-seconds`. Its log is the metrics log of the artifacts directory.
    pub fn session(
        &self,
        progress: TrainingProgress,
        training: &TrainingConfig,
        cadence: Cadence,
        items_total: usize,
    ) -> Session {
        Session::new(
            progress,
            Budget::new(self.max_batches, self.max_seconds.map(Duration::from_secs_f64)),
            self.cadence(cadence),
            training.lr.clone(),
            items_total,
            MetricsLog::open(&self.artifacts_path),
        )
    }

    /// Create the artifacts directory (removing model/optim first if requested).
    pub fn create_artifact_dir(&self) {
        create_artifact_dir(&self.artifacts_path, self.remove_artifacts && self.training)
    }

    /// Save the training config into the artifacts directory.
    pub fn save_training_config(&self, training_config: &impl Config) {
        let path = self
            .artifacts_path
            .join(TRAINING_CONFIG_NAME)
            .with_added_extension("json");
        save_training_config(&path, training_config)
    }

    /// Load the training config (from `--training-config` or the artifacts dir).
    pub fn load_training_config<TrainingConfig: Config>(&self) -> Option<TrainingConfig> {
        self.training_config
            .as_ref()
            .map(|path| {
                load_training_config(path).unwrap_or_else(|| {
                    panic!("Failed to find the training config file {path:?}")
                })
            })
            // Lazy: with `--training-config`, the saved config is not read.
            .or_else(|| {
                let path = self
                    .artifacts_path
                    .join(TRAINING_CONFIG_NAME)
                    .with_added_extension("json");
                load_training_config(&path)
            })
    }

    /// Apply the overrides of the invocation (`--seed`, `--batch-size`,
    /// `--epochs`, `--max-lr`, the optimizer flags) to `training`, the loaded
    /// or newly created config, before it is saved.
    ///
    /// `--epochs` rescales the `total_steps` of a cosine schedule by the same
    /// factor. So a schedule sized to the run still spans it (a resumed run
    /// then lands where the longer or shorter cosine has it). The warmup does
    /// not change. `--batch-size` rescales both by the inverse factor, because
    /// an epoch then takes that many fewer steps.
    ///
    /// An optimizer flag that disagrees with the optimizer of `training`
    /// replaces it with [`OptimizerConfig::of`] (`dtype` sizes the epsilon of
    /// AdamW), and keeps the LR schedule. It panics instead when both of these
    /// are true:
    ///
    /// - Optimizer state is saved in the artifacts directory. That is, this is
    ///   not a new run, and `--remove-artifacts --training` did not remove the
    ///   state.
    /// - The old optimizer has state (plain SGD has none).
    ///
    /// That state would then load into an optimizer that ignores it.
    pub fn override_training_config(&self, training: &mut TrainingConfig, dtype: DType) {
        if let Some(seed) = self.seed {
            training.seed = seed;
        }
        if let Some(batch_size) = self.batch_size {
            if let Lr::CosineAnnealing(cosine) = &mut training.lr {
                cosine.total_steps = cosine.total_steps * training.batch_size / batch_size;
                cosine.warmup_steps = cosine.warmup_steps * training.batch_size / batch_size;
            }
            training.batch_size = batch_size;
        }
        if let Some(epochs) = self.epochs {
            if let Lr::CosineAnnealing(cosine) = &mut training.lr {
                cosine.total_steps = cosine.total_steps * epochs / training.num_epochs.max(1);
            }
            training.num_epochs = epochs;
        }
        if let Some(max_lr) = self.max_lr {
            match &mut training.lr {
                Lr::CosineAnnealing(cosine) => cosine.max_lr = max_lr,
                Lr::Constant(constant) => constant.lr = max_lr,
            }
        }
        let saved = training.optimizer.kind();
        if let Some(kind) = self.optimizer.filter(|&kind| kind != saved) {
            let optim = self.artifacts_path.join(OPTIM_NAME).with_extension(RECORD_EXT);
            let has_state = saved != OptimizerKind::Sgd
                && std::fs::exists(&optim)
                    .unwrap_or_else(|e| panic!("failed to check {optim:?}: {e}"));
            assert!(
                !has_state,
                "the optimizer of the training config is {saved:?}, and its state {optim:?} \
                 would be ignored under {kind:?}. Remove the optimizer flags to keep the state, \
                 or remove the state (--remove-artifacts --training)."
            );
            println!("Replacing the training config's optimizer ({saved:?}) with {kind:?}");
            training.optimizer = OptimizerConfig::of(kind, dtype);
        }
    }

    /// Save the model config into the artifacts directory.
    pub fn save_model_config(&self, model_config: &impl Config) {
        let path = self
            .artifacts_path
            .join(MODEL_CONFIG_NAME)
            .with_added_extension("json");
        save_model_config(&path, model_config)
    }

    /// Load the model config (from `--model-config` or the artifacts dir).
    pub fn load_model_config<ModelConfig: ModelConfigExt>(&self) -> Option<ModelConfig> {
        self.model_config
            .as_ref()
            .map(|path| {
                load_model_config::<ModelConfig>(path)
                    .unwrap_or_else(|| panic!("Failed to find the model config file {path:?}"))
            })
            // Lazy: with `--model-config`, the saved config is not read.
            .or_else(|| {
                let path = self
                    .artifacts_path
                    .join(MODEL_CONFIG_NAME)
                    .with_added_extension("json");
                load_model_config::<ModelConfig>(&path)
            })
    }

    /// Save the model weights into the artifacts directory.
    pub fn save_model(&self, model: &impl Module) {
        save_model(&self.artifacts_path, model)
    }

    /// Load model weights from the artifacts directory, if present.
    pub fn load_model<ModelConfig: ModelConfigExt>(
        &self,
        model_config: &ModelConfig,
        device: &Device,
    ) -> Option<ModelConfig::Model> {
        load_model(&self.artifacts_path, model_config, device)
    }

    /// Load the model if saved, otherwise initialise a new one and save it.
    pub fn load_or_save_model<ModelConfig: ModelConfigExt>(
        &self,
        model_config: &ModelConfig,
        device: &Device,
    ) -> ModelConfig::Model {
        self.load_model(model_config, device).unwrap_or_else(|| {
            println!("Initializing new model");
            let model_init = model_config.init(device);
            self.save_model(&model_init);
            model_init
        })
    }

    /// Save the optimizer state into the artifacts directory, together with the
    /// `progress` it was taken at (what `--resume` continues from).
    pub fn save_optim(&self, optim: &ModuleOptimizer, progress: &TrainingProgress) {
        save_optim(&self.artifacts_path, optim);
        save_progress(&self.artifacts_path, progress);
    }

    /// Load optimizer state from the artifacts directory into `optim`, if
    /// present. `optim` must already carry the parameter groups the state was
    /// saved with (see [`crate::optim::MuonPlan`]).
    pub fn load_optim(&self, optim: ModuleOptimizer) -> Option<ModuleOptimizer> {
        load_optim(&self.artifacts_path, optim)
    }

    /// Load the optimizer state into `optim` if it is saved. Otherwise, save
    /// `optim` as the initial state. Also returns the progress to start from:
    /// under `--resume`, the progress saved with the loaded state. Otherwise, a
    /// new one.
    pub fn load_or_save_optim(&self, optim: ModuleOptimizer) -> (ModuleOptimizer, TrainingProgress) {
        match self.load_optim(optim.clone()) {
            Some(loaded) => (loaded, self.resumed_progress()),
            None => {
                println!("Initializing new optim");
                let progress = TrainingProgress::new();
                self.save_optim(&optim, &progress);
                (optim, progress)
            }
        }
    }

    /// The progress a loaded optimizer continues from.
    fn resumed_progress(&self) -> TrainingProgress {
        let saved = load_progress(&self.artifacts_path);
        match (self.resume, saved) {
            (true, Some(progress)) => {
                println!(
                    "Resuming at step {}, epoch {}, batch {}",
                    progress.step, progress.epoch, progress.batch
                );
                progress
            }
            (true, None) => panic!(
                "--resume: no {PROGRESS_NAME}.json was saved with the optim in {:?}",
                self.artifacts_path
            ),
            (false, Some(progress)) if progress.step > 0 => {
                println!(
                    "Starting over at step 0 (saved at step {}; --resume continues it)",
                    progress.step
                );
                TrainingProgress::new()
            }
            (false, _) => TrainingProgress::new(),
        }
    }
}

/// `pico-args` value parser turning an `OsStr` into a `PathBuf`.
pub fn parse_path(s: &std::ffi::OsStr) -> Result<std::path::PathBuf, &'static str> {
    Ok(s.into())
}

/// Close the parser of an example over [`AppArgs::extra`]. Panics on any
/// argument that it left.
pub fn finish_extra(pargs: pico_args::Arguments) {
    let remaining = pargs.finish();
    assert!(remaining.is_empty(), "unused extra arguments: {remaining:?}");
}

/// `--adamw` / `--sgd` / `--muon` (see [`HELP`]). `None` when none is given.
fn parse_optimizer(pargs: &mut pico_args::Arguments) -> Option<OptimizerKind> {
    let (adamw, sgd, muon) = (
        pargs.contains("--adamw"),
        pargs.contains("--sgd"),
        pargs.contains("--muon"),
    );
    match (adamw, sgd, muon) {
        (true, true, _) => panic!("--adamw and --sgd are exclusive"),
        (false, false, false) => None,
        (_, false, false) => Some(OptimizerKind::AdamW),
        (_, true, false) => Some(OptimizerKind::Sgd),
        (_, false, true) => Some(OptimizerKind::MuonAdamW),
        (_, true, true) => Some(OptimizerKind::MuonSgd),
    }
}

/// Create the artifacts directory. When `delete` is set, first remove the
/// `model`, `optim` and progress files.
pub fn create_artifact_dir(artifact_dir: &Path, delete: bool) {
    if delete {
        // Every removal must succeed: a missing file is also an error.
        println!("removing {artifact_dir:?}/{{model,optim}}.{RECORD_EXT} and {PROGRESS_NAME}.json");
        std::fs::remove_file(artifact_dir.join(MODEL_NAME).with_extension(RECORD_EXT))
            .expect("failed to remove the model");
        std::fs::remove_file(artifact_dir.join(OPTIM_NAME).with_extension(RECORD_EXT))
            .expect("failed to remove the optim");
        std::fs::remove_file(artifact_dir.join(PROGRESS_NAME).with_added_extension("json"))
            .expect("failed to remove the progress");
    }
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Base filename (without extension) for the persisted training config.
pub const TRAINING_CONFIG_NAME: &str = "training_config";
/// Save a training config as JSON to `path`.
pub fn save_training_config(path: &Path, training_config: &impl Config) {
    println!("Saving training config into {path:?}");
    training_config
        .save(path)
        .expect("Failed to save the training config");
}

/// Load a training config from `path`, or `None` if the file is absent.
pub fn load_training_config<TrainingConfig: Config>(path: &Path) -> Option<TrainingConfig> {
    let exists = std::fs::exists(path).unwrap_or_else(|e| panic!("failed to check {path:?}: {e}"));
    if exists {
        println!("Loading training config from {path:?}");
        let training_config =
            TrainingConfig::load(path).expect("Failed to load the training config");
        Some(training_config)
    } else {
        None
    }
}

/// Base filename (without extension) for the persisted model config.
pub const MODEL_CONFIG_NAME: &str = "model_config";
/// Save a model config as JSON to `path`.
pub fn save_model_config(path: &Path, model_config: &impl Config) {
    println!("Saving model config into {path:?}");
    model_config
        .save(path)
        .expect("Failed to save the model config");
}

/// Load a model config from `path`, or `None` if the file is absent.
pub fn load_model_config<ModelConfig: Config>(path: &Path) -> Option<ModelConfig> {
    let exists = std::fs::exists(path).unwrap_or_else(|e| panic!("failed to check {path:?}: {e}"));
    if exists {
        println!("Loading model config from {path:?}");
        let model_config = ModelConfig::load(path).expect("Failed to load the model config");
        Some(model_config)
    } else {
        None
    }
}

/// Canonical burnpack file extension appended to the model/optim records.
///
/// The save/load of `ModuleRecord`/`ModuleOptimizer` appends this
/// automatically when the path has no extension. So it is explicit here, and
/// the existence checks and the `--remove-artifacts` cleanup match the files
/// that are written.
pub const RECORD_EXT: &str = "bpk";

/// Base filename (without extension) for the persisted model weights.
pub const MODEL_NAME: &str = "model";
/// Save model weights into `artifact_dir` as a burnpack record.
pub fn save_model(artifact_dir: &Path, model: &impl Module) {
    let path = artifact_dir.join(MODEL_NAME).with_extension(RECORD_EXT);
    println!("Saving model to {path:?}");
    model
        .clone()
        .into_record()
        .save(path)
        .expect("Failed to save the model");
}

/// Load model weights from `artifact_dir`, or `None` if absent.
///
/// `load_record` restores the saved `ParamId` of each parameter. So the
/// optimizer state, keyed by `ParamId`, stays associated across process
/// restarts.
pub fn load_model<ModelConfig: ModelConfigExt>(
    artifact_dir: &Path,
    model_config: &ModelConfig,
    device: &Device,
) -> Option<ModelConfig::Model> {
    let path = artifact_dir.join(MODEL_NAME).with_extension(RECORD_EXT);
    let exists = std::fs::exists(&path).unwrap_or_else(|e| panic!("failed to check {path:?}: {e}"));
    if exists {
        println!("Loading model from {path:?}");
        let record = ModuleRecord::load(&path).expect("Failed to load the model record");
        let model = model_config.init(device).load_record(record);
        Some(model)
    } else {
        None
    }
}

/// Base filename for the persisted optimizer state.
pub const OPTIM_NAME: &str = "optim";
/// Save optimizer state into `artifact_dir` as a burnpack record.
pub fn save_optim(artifact_dir: &Path, optim: &ModuleOptimizer) {
    let path = artifact_dir.join(OPTIM_NAME).with_extension(RECORD_EXT);
    println!("Saving optim to {path:?}");
    optim.save(path).expect("Failed to save the optim");
}

/// Load optimizer state from `artifact_dir`, or `None` if absent.
///
/// Optimizer state is keyed by `ParamId`. [`load_model`] keeps the saved ids,
/// so the loaded state lands on the parameters of the resumed model.
pub fn load_optim(artifact_dir: &Path, optim: ModuleOptimizer) -> Option<ModuleOptimizer> {
    let path = artifact_dir.join(OPTIM_NAME).with_extension(RECORD_EXT);
    let exists = std::fs::exists(&path).unwrap_or_else(|e| panic!("failed to check {path:?}: {e}"));
    if !exists {
        return None;
    }
    println!("Loading initial optim from {path:?}");
    let optim = optim.load(&path).expect("Failed to load the initial optim");
    Some(optim)
}

/// Base filename (without extension) for the persisted [`TrainingProgress`].
pub const PROGRESS_NAME: &str = "progress";
/// Save the training progress as JSON into `artifact_dir`.
pub fn save_progress(artifact_dir: &Path, progress: &TrainingProgress) {
    let path = artifact_dir.join(PROGRESS_NAME).with_added_extension("json");
    println!(
        "Saving progress (step {}, epoch {}, batch {}) into {path:?}",
        progress.step, progress.epoch, progress.batch
    );
    progress.save(path).expect("Failed to save the progress");
}

/// Load the training progress from `artifact_dir`, or `None` if absent.
pub fn load_progress(artifact_dir: &Path) -> Option<TrainingProgress> {
    let path = artifact_dir.join(PROGRESS_NAME).with_added_extension("json");
    let exists = std::fs::exists(&path).unwrap_or_else(|e| panic!("failed to check {path:?}: {e}"));
    exists.then(|| TrainingProgress::load(&path).expect("Failed to load the progress"))
}
