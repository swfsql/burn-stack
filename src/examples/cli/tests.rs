//! These tests assert that:
//!
//! - the optimizer flags pick one of the four shapes,
//! - on a loaded config, the flags replace its optimizer, unless the state
//!   saved under that optimizer would then be ignored,
//! - `--batch-size` rescales a cosine schedule with the step count of the
//!   epoch,
//! - the clock of the time budget starts at the first step,
//! - an explicit config file does not read the saved config.

use super::*;
use crate::examples::training::{CosineAnnealingLr, OptimizerConfig};
use temp_dir::TempDir;

fn scratch() -> TempDir {
    TempDir::with_prefix("burn-stack-cli-test-").expect("a scratch directory")
}

fn parse(dir: &Path, flags: &[&str]) -> AppArgs {
    let mut args: Vec<OsString> = vec!["-a".into(), dir.into()];
    args.extend(flags.iter().map(OsString::from));
    AppArgs::parse_from(args, "unused-").expect("valid flags")
}

fn config(kind: OptimizerKind) -> TrainingConfig {
    TrainingConfig::new(OptimizerConfig::of(kind, DType::F32))
        .with_batch_size(16)
        .with_lr(Lr::CosineAnnealing(
            CosineAnnealingLr::new(1000).with_max_lr(1e-2).with_warmup_steps(50),
        ))
}

/// Pretend optimizer state was saved in `dir`.
fn save_state(dir: &Path) {
    std::fs::write(dir.join(OPTIM_NAME).with_extension(RECORD_EXT), b"").unwrap();
}

#[test]
fn the_optimizer_flags_pick_the_shape() {
    let dir = scratch();
    let kind = |flags: &[&str]| parse(dir.path(), flags).optimizer;
    assert_eq!(kind(&[]), None);
    assert_eq!(kind(&["--adamw"]), Some(OptimizerKind::AdamW));
    assert_eq!(kind(&["--sgd"]), Some(OptimizerKind::Sgd));
    assert_eq!(kind(&["--muon"]), Some(OptimizerKind::MuonAdamW));
    assert_eq!(kind(&["--muon", "--adamw"]), Some(OptimizerKind::MuonAdamW));
    assert_eq!(kind(&["--sgd", "--muon"]), Some(OptimizerKind::MuonSgd));
    for kind in [
        OptimizerKind::AdamW,
        OptimizerKind::Sgd,
        OptimizerKind::MuonAdamW,
        OptimizerKind::MuonSgd,
    ] {
        assert_eq!(OptimizerConfig::of(kind, DType::F32).kind(), kind);
    }
}

#[test]
#[should_panic(expected = "exclusive")]
fn adamw_and_sgd_are_exclusive() {
    let dir = scratch();
    parse(dir.path(), &["--adamw", "--sgd"]);
}

#[test]
fn a_flag_replaces_the_optimizer_of_a_run_without_state() {
    let dir = scratch();
    let mut training = config(OptimizerKind::MuonAdamW);
    parse(dir.path(), &["--sgd"]).override_training_config(&mut training, DType::F32);
    assert_eq!(training.optimizer.kind(), OptimizerKind::Sgd);
    assert!(training.optimizer.plain_sgd().is_some());
    let Lr::CosineAnnealing(cosine) = &training.lr else { panic!() };
    assert_eq!(cosine.max_lr, 1e-2, "the LR schedule is kept");
}

#[test]
#[should_panic(expected = "would be ignored")]
fn a_flag_that_would_ignore_saved_state_panics() {
    let dir = scratch();
    save_state(dir.path());
    let mut training = config(OptimizerKind::MuonAdamW);
    parse(dir.path(), &["--adamw"]).override_training_config(&mut training, DType::F32);
}

#[test]
fn a_saved_state_survives_a_flag_that_agrees_with_it() {
    let dir = scratch();
    save_state(dir.path());
    let mut training = config(OptimizerKind::MuonAdamW);
    parse(dir.path(), &["--muon"]).override_training_config(&mut training, DType::F32);
    assert_eq!(training.optimizer.kind(), OptimizerKind::MuonAdamW);
}

#[test]
fn plain_sgd_saves_nothing_to_lose() {
    let dir = scratch();
    save_state(dir.path());
    let mut training = config(OptimizerKind::Sgd);
    parse(dir.path(), &["--muon"]).override_training_config(&mut training, DType::F32);
    assert_eq!(training.optimizer.kind(), OptimizerKind::MuonAdamW);
}

#[test]
fn the_batch_size_rescales_a_cosine_schedule() {
    let dir = scratch();
    let mut training = config(OptimizerKind::AdamW);
    parse(dir.path(), &["--batch-size", "32"]).override_training_config(&mut training, DType::F32);
    assert_eq!(training.batch_size, 32);
    let Lr::CosineAnnealing(cosine) = &training.lr else { panic!() };
    assert_eq!((cosine.total_steps, cosine.warmup_steps), (500, 25));
}

/// With `--training-config`, the config in the artifacts directory is not
/// read, so a stale one cannot make the load panic.
#[test]
fn an_explicit_config_does_not_read_the_saved_one() {
    let dir = scratch();
    let saved = dir.path().join(TRAINING_CONFIG_NAME).with_added_extension("json");
    std::fs::write(saved, b"not a config").unwrap();
    let explicit = dir.path().join("explicit.json");
    config(OptimizerKind::Sgd).save(&explicit).unwrap();
    let args = parse(dir.path(), &["-c", explicit.to_str().unwrap()]);
    let loaded: TrainingConfig = args.load_training_config().expect("the explicit config");
    assert_eq!(loaded.optimizer.kind(), OptimizerKind::Sgd);
}

#[test]
fn the_clock_starts_at_the_first_step() {
    let mut budget = Budget::new(None, Some(Duration::ZERO));
    assert!(budget.is_capped());
    assert!(!budget.is_exhausted(), "no step taken, no time spent");
    budget.spend();
    assert!(budget.is_exhausted());
}
