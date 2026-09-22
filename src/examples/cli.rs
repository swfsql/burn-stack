//! CLI plumbing shared by the examples: argument parsing into [`AppArgs`],
//! artifact-directory management, and load/save of the training config, model
//! config, model weights, and optimizer state (with the run's
//! [`TrainingProgress`]), plus the [`Session`] a training run threads through its
//! epoch loops.  See [`HELP`] for the full command-line behaviour.
//!
//! [`AppArgs`] carries every flag an example may share; the arguments after
//! `--` are the example's own, which it parses from [`AppArgs::extra`] and
//! closes with [`finish_extra`].
//!
//! The one thing this module cannot know is *whose* example is running, so
//! [`AppArgs::parse`] takes the prefix of the auto-created artifacts directory
//! (conventionally `concat!(env!("CARGO_PKG_NAME"), "-", env!("CARGO_BIN_NAME"),
//! "-")`, evaluated in the example crate).

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

A command-line tool for training and/or running inference with machine learning models.
Models, optimizers, and configurations are persisted in an artifacts directory.

USAGE:
    example-name [OPTIONS] [-- <EXTRA_ARGS>...]

When no --training or --inference flag is provided, the program exits after handling configuration logic.

BEHAVIOR OVERVIEW
- The program manages two configurations: training config and model config.
- If --training-config or --model-config is given, the corresponding config is loaded from the specified file and saved to the artifacts directory (overwriting any existing file).
- If no explicit config file is provided for a component, the program attempts to load it from the artifacts directory; if absent, a default configuration is created and saved.
- The artifacts directory (--artifacts-path) is used to read/write model weights, optimizer state, and configurations. If not specified, a new temporary directory is created and its path is printed.
- With --remove-artifacts, any existing model and optimizer files (and the saved progress) in the artifacts directory are deleted before training (if --training is active).
- Model and optimizer weights are loaded from the artifacts directory if present; otherwise new ones are created and saved.
- With --seed, --epochs, --batch-size or --max-lr, the given value replaces the training config's (loaded or created) before the config is saved, so later runs from the same artifacts directory inherit it. --epochs also rescales a cosine LR schedule's length by the same factor, so the schedule still spans the run; --batch-size rescales its length and warmup by the inverse one (an epoch has that many fewer steps).
- --adamw, --sgd and --muon choose the optimizer. --muon puts the model's hidden weight matrices on Muon and the other flag (default --adamw) optimizes every other parameter; --adamw and --sgd are exclusive. Without any of them a new training config gets the example's own default. A loaded config's optimizer is replaced by the flags' choice (its LR schedule is kept: see --max-lr), unless optimizer state saved under the old optimizer would then be ignored, which panics instead (the state is removed by --remove-artifacts with --training). Only plain SGD (--sgd alone) has a training step that replays from a captured graph.
- An example that supports it replays its fixed-shape passes (training steps under plain SGD, validation, decoding) from captured CUDA graphs; --no-graph runs every pass eagerly.
- The optimizer state is saved together with the run's progress: the LR-schedule step, the epoch, and the batch within it. A run that loads it starts over at step 0 of epoch 1, unless --resume is given, which continues from the saved progress. The interrupted epoch then trains only the batches it has left, drawn from a fresh shuffle (the dataloader workers' batch order cannot be replayed).
- Training checkpoints at every epoch end and when it stops. --checkpoint-every adds a checkpoint every that many optimizer steps, --valid-every a periodic validation, and --valid-batches caps the batches that validation reads; each example has its own defaults for these three.
- Every training step and validation is appended as one JSON line to metrics.jsonl in the artifacts directory; each run opens with a \"start\" line.
- If both --training and --inference are specified, training executes first, followed by inference using the trained model.
- With --max-batches, training stops after that many mini-batches in total (counted across epochs), checkpointing as usual before it returns. One mini-batch is one optimizer step, which for the character LM is one window of a run rather than one dataloader item. --max-seconds stops it the same way once that much wall-clock time has passed since its first step.
- Any arguments following -- are captured as-is and forwarded to the example's own flags (-- --help lists them).

FLAGS:
    -h, --help                  Show this help message and exit

OPTIONS:
    -t, --training              Run training (creates or updates model / optimizer)
    -i, --inference             Run inference after training (if both flags are used) or immediately (if only inference is requested)
    -r, --remove-artifacts      Delete existing model and optimizer files from the artifacts directory before training
                                (has no effect if --training is not used)
    -c, --training-config <PATH>
                                Load training configuration from this file (overrides any config in artifacts directory)
    -m, --model-config <PATH>   Load model configuration from this file (overrides any config in artifacts directory)
    -b, --max-batches <N>       Stop training after N mini-batches in total (across epochs), regardless of the
                                configured number of epochs. Unlimited when absent.
        --max-seconds <S>       Stop training once S seconds have passed since its first step. Unlimited when absent.
    -s, --seed <N>              Replace the training config's RNG seed (model init, data shuffling, sampling)
        --epochs <N>            Replace the training config's number of epochs (rescaling a cosine LR schedule)
        --batch-size <N>        Replace the training config's mini-batch size (rescaling a cosine LR schedule)
        --max-lr <LR>           Replace the LR schedule's peak rate (a constant schedule's only one)
        --adamw                 Optimize with AdamW (with --muon: every parameter Muon does not own)
        --sgd                   Optimize with plain SGD (with --muon: every parameter Muon does not own)
        --muon                  Put the hidden weight matrices on Muon
        --no-graph              Run every pass eagerly instead of replaying captured CUDA graphs
        --resume                Continue from the progress saved with the optimizer state (schedule step, epoch,
                                batch) instead of from step 0 (has no effect on a new optimizer)
        --checkpoint-every <N>  Also checkpoint every N optimizer steps (0: only at epoch ends)
        --valid-every <N>       Validate every N optimizer steps (0: no periodic validation)
        --valid-batches <N>     Batches a periodic validation reads
    -a, --artifacts-path <PATH>
                                Directory where configurations, model weights, and optimizer state are saved and loaded.
                                If the directory does not exist, it will be created.
                                Defaults to a newly created temporary directory (path will be printed).

ARGS:
    -- <EXTRA_ARGS>             All arguments after -- are forwarded verbatim to the example's own flags.
                                Passing -h or --help there displays its help information.
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
    /// Optional cap on the total number of training mini-batches; see
    /// [`Budget`].
    pub max_batches: Option<usize>,
    /// Optional cap on the training wall-clock time, in seconds; see
    /// [`Budget`].
    pub max_seconds: Option<f64>,
    /// Optional replacement for the training config's seed; see
    /// [`AppArgs::override_training_config`].
    pub seed: Option<u64>,
    /// Optional replacement for the training config's number of epochs; see
    /// [`AppArgs::override_training_config`].
    pub epochs: Option<usize>,
    /// Optional replacement for the training config's mini-batch size; see
    /// [`AppArgs::override_training_config`].
    pub batch_size: Option<usize>,
    /// Optional replacement for the LR schedule's peak rate; see
    /// [`AppArgs::override_training_config`].
    pub max_lr: Option<f64>,
    /// The optimizer `--adamw` / `--sgd` / `--muon` chose, if any: a fresh
    /// config's (see [`AppArgs::optimizer_or`]), and a loaded one's replacement
    /// (see [`AppArgs::override_training_config`]).
    pub optimizer: Option<OptimizerKind>,
    /// Whether captured graphs are off (`--no-graph`); see [`AppArgs::graphs`].
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
    /// `--artifacts-path` is absent, e.g. `"burn-deltanet-mnist-class-"`.
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
                    // e.g. /tmp/burn-mamba-reset-majority-abcd-0
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

    /// The arguments after `--`, for the example's own parser; with `-h` /
    /// `--help` among them, prints `help` and exits. Close it with
    /// [`finish_extra`].
    pub fn extra(&self, help: &str) -> pico_args::Arguments {
        let mut pargs = pico_args::Arguments::from_vec(self.extra_args.clone());
        if pargs.contains(["-h", "--help"]) {
            println!("{help}");
            std::process::exit(0);
        }
        pargs
    }

    /// The optimizer a fresh training config gets: the flags' choice, else the
    /// example's `default`.
    pub fn optimizer_or(&self, default: OptimizerKind) -> OptimizerKind {
        self.optimizer.unwrap_or(default)
    }

    /// Whether to replay passes from captured graphs (unless `--no-graph`).
    pub fn graphs(&self) -> bool {
        !self.no_graph
    }

    /// The example's `defaults` under the `--checkpoint-every` /
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

    /// Start a training [`Session`] from `progress` (what
    /// [`load_or_save_optim`](Self::load_or_save_optim) returned), following
    /// `training`'s LR schedule over a training split of `items_total` items,
    /// at the example's `cadence` defaults (see [`cadence`](Self::cadence)).
    /// Its budget is `--max-batches` and `--max-seconds`; its log, the
    /// artifacts directory's metrics log.
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
                load_training_config(path)
                    .expect("Failed to find the training config file {path:?}")
            })
            .or({
                let path = self
                    .artifacts_path
                    .join(TRAINING_CONFIG_NAME)
                    .with_added_extension("json");
                load_training_config(&path)
            })
    }

    /// Apply the invocation's overrides (`--seed`, `--batch-size`, `--epochs`,
    /// `--max-lr`, the optimizer flags) onto `training`, the loaded or freshly
    /// created config, before it is saved.
    ///
    /// `--epochs` rescales a cosine schedule's `total_steps` by the same factor,
    /// so a schedule sized to the run still spans it (a resumed run then lands
    /// where the longer or shorter cosine has it); the warmup is left alone.
    /// `--batch-size` rescales both by the inverse factor, since an epoch then
    /// takes that many fewer steps.
    ///
    /// An optimizer flag that disagrees with `training`'s replaces it with
    /// [`OptimizerConfig::of`] (`dtype` sizes AdamW's epsilon), keeping the LR
    /// schedule. It panics instead when optimizer state is saved in the
    /// artifacts directory — i.e. when this is not a fresh run, nor one whose
    /// `--remove-artifacts --training` already removed it — and the old
    /// optimizer has any (plain SGD has none), since that state would then be
    /// loaded into an optimizer that ignores it.
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
                && std::fs::exists(&optim).expect("failed to check {optim:?}");
            assert!(
                !has_state,
                "the training config's optimizer is {saved:?}, and its state {optim:?} would be \
                 ignored under {kind:?}: drop the optimizer flags to keep it, or remove it \
                 (--remove-artifacts --training)"
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
                    .expect("Failed to find the model config file {path:?}")
            })
            .or({
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

    /// Load the optimizer state into `optim` if saved, otherwise save `optim` as
    /// the initial state. Also returns the progress to start from: the one saved
    /// with the loaded state under `--resume`, a fresh one otherwise.
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

/// Close an example's parser over [`AppArgs::extra`]: panics on any argument it
/// left over.
pub fn finish_extra(pargs: pico_args::Arguments) {
    let remaining = pargs.finish();
    assert!(remaining.is_empty(), "unused extra arguments: {remaining:?}");
}

/// `--adamw` / `--sgd` / `--muon` (see [`HELP`]); `None` when none is given.
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

/// Create the artifacts directory; when `delete` is set, remove any existing
/// `model`/`optim` files first.
pub fn create_artifact_dir(artifact_dir: &Path, delete: bool) {
    if delete {
        // enforce that the removal should not have errors,
        // including for when files didn't exist
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
    let exists = std::fs::exists(path).expect("failed to check {path:?}");
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
    let exists = std::fs::exists(path).expect("failed to check {path:?}");
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
/// `ModuleRecord`/`ModuleOptimizer` save/load auto-append this when the path
/// carries no extension, so spell it out here for the existence checks and the
/// `--remove-artifacts` cleanup to match the files actually written.
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
/// `load_record` restores each parameter's persisted `ParamId`, so the
/// ParamId-keyed optimizer state stays associated across process relaunches.
pub fn load_model<ModelConfig: ModelConfigExt>(
    artifact_dir: &Path,
    model_config: &ModelConfig,
    device: &Device,
) -> Option<ModelConfig::Model> {
    let path = artifact_dir.join(MODEL_NAME).with_extension(RECORD_EXT);
    let exists = std::fs::exists(&path).expect("failed to check {path:?}");
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
/// Optimizer state is keyed by `ParamId`; [`load_model`] preserves the
/// persisted ids, so the loaded state lands on the resumed model's parameters.
pub fn load_optim(artifact_dir: &Path, optim: ModuleOptimizer) -> Option<ModuleOptimizer> {
    let path = artifact_dir.join(OPTIM_NAME).with_extension(RECORD_EXT);
    let exists = std::fs::exists(&path).expect("failed to check {path:?}");
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
    let exists = std::fs::exists(&path).expect("failed to check {path:?}");
    exists.then(|| TrainingProgress::load(&path).expect("Failed to load the progress"))
}
