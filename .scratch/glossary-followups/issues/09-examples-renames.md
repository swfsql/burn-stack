# 09: Rename the examples identifiers that conflict with the glossary

Status: ready-for-agent

Blocked by: 05

**What to build:** The identifiers and the CLI flags of the example
scaffolding use the terms of `GLOSSARY.md`, as their prose does after ticket
05. Ticket 05 lists the identifiers that conflict with a glossary term. The
user approved the names of the rename map below. Breaking changes are
permitted, also for saved configs, `progress.json` and `metrics.jsonl`.

## Rename map

The user chose these names:

| Old | New | Where |
|---|---|---|
| flag `--no-graph` | `--eager` | `cli.rs` (the parser, `HELP`, the docs) |
| field `AppArgs::no_graph` | `AppArgs::eager` | `cli.rs` |
| `AppArgs::graphs()` | `AppArgs::capture()` | `cli.rs`, the doc links |
| parameter `graphs` of `Trainer::new`, the closure parameter `graphs` of the trainer test | `capture` | `trainer.rs`, `trainer/tests.rs` |
| field `TrainingProgress::step` (the key of `progress.json`) | `training_step` | `session.rs`, `cli.rs` |
| key `"step"` of each line of `metrics.jsonl` | `"training_step"` | `MetricsLog::line` (`session.rs`) and its docs |
| `Bucket` (struct, a config) | `Bucketing` | `tiny_stories/lm.rs` |
| field `TinyStoriesConfig::bucket` (a config key) | `bucketing` | `tiny_stories/lm.rs` |
| field `Bucket::slots` (a config key) | `Bucketing::buckets` | `tiny_stories/lm.rs` |
| field `Bucket::chars` (a config key) | `Bucketing::batch_chars` | `tiny_stories/lm.rs` |
| flag `--bucket-chars`, field `Overrides::bucket_chars` | `--batch-chars`, `Overrides::batch_chars` | `tiny_stories/lm.rs` (also `Overrides::HELP` and the two assert messages that name the flag) |
| field `BucketItem::slot` | `BucketItem::bucket` | `tiny_stories/dataset.rs` |
| `parse_slots` | `parse_buckets` | `tiny_stories/lm.rs` |
| `Trainer::step` | `Trainer::train_step` | `trainer.rs`, `trainer/tests.rs` |
| `sgd_step` (private) | `sgd_train_step` | `trainer.rs` |
| `StepFlags` | `TrainingStepFlags` | `cli.rs` |
| `LmModel::optim_step` (a trait method) | `LmModel::optimizer_update` | `tiny_stories/lm.rs` |
| `Session::begin_step` | `Session::begin_training_step` | `session.rs`, `mnist/classify.rs`, `tiny_stories/lm.rs` |
| `Session::skip_steps`, its parameter `steps` | `Session::skip_training_steps`, `training_steps` | `session.rs`, `tiny_stories/lm.rs` |

`--buckets`, `--no-bucket`, `--bucket-probe`, `Overrides::buckets`,
`Overrides::no_bucket`, `Overrides::bucket_probe` and `Bucketing::probe`
stay. They name the buckets or the probe over the buckets.

These names follow the glossary with no choice to make:

| Old | New | Where |
|---|---|---|
| field `Prefill::chunk`, the parameter `chunk` of `Prefill::new`, the locals `chunk` | `window` | `tiny_stories/sample.rs` (also the code spans `[1, chunk, 2]`) |
| `ChunkFn` (private) | `WindowFn` | `tiny_stories/sample.rs` |
| field `Prefill::eager_chunks` | `eager_windows` | `tiny_stories/sample.rs` |
| test constant `CHUNK` | `WINDOW` | `sample/tests.rs` |
| the parameters `slots` of `bucket_batches`, `probe_order` and `BucketedStoriesDataset::new`, the locals `slot`, `slots` | `buckets`, `bucket` | `tiny_stories/dataset.rs`, `dataset/tests.rs` (also the code span `[stories, slot]`) |
| field `PackedItem::stories` | `PackedItem::segments` | `tiny_stories/dataset.rs` |
| field `Session::schedule`, the parameter `schedule` of `Session::new` | `lr_schedule` | `session.rs` |
| the parameter `step` of `Session::due` | `training_step` | `session.rs` |
| the locals `_step`, `step_lr` of the LM `epoch_train` | `_training_step`, `window_lr` | `tiny_stories/lm.rs` |
| the local `_step` of the MNIST `epoch_train` | `_training_step` | `mnist/classify.rs` |
| the local `rows` of `a_prefill_is_the_same_captured_and_eager_and_is_the_prompt_in_one_pass` | `positions` | `sample/tests.rs` |
| `captured_sgd_steps_are_the_eager_steps`, constant `STEPS` | `captured_sgd_training_steps_are_the_eager_ones`, `TRAINING_STEPS` | `trainer/tests.rs` |
| `the_batch_size_rescales_a_cosine_schedule` | `the_batch_size_rescales_a_cosine_lr_schedule` | `cli/tests.rs` |
| `the_flags_rescale_a_sequence` | `the_flags_rescale_a_chain` | `cli/tests.rs` |
| `the_clock_starts_at_the_first_step` | `the_clock_starts_at_the_first_training_step` | `cli/tests.rs` |
| `probe_order_opens_with_each_slot_and_keeps_every_batch` | `probe_order_opens_with_each_bucket_and_keeps_every_batch` | `dataset/tests.rs` |
| `packed_batcher_lays_out_each_story` | `packed_batcher_lays_out_each_segment` | `dataset/tests.rs` |

## Scope

- The identifiers and flags of the maps, at each definition and each use in
  this crate.
- The prose that names one of them. Write it in the style of the
  `asd-ste100` skill (messages and help: strict). This prose is:
  - the doc links and the code spans,
  - `cli::HELP`, `TrainingStepFlags::HELP` and `Overrides::HELP`,
  - the messages that name a flag,
  - the doc of `MetricsLog` (the keys of a line).
- Under `## Comments` of ticket 05, add one line: ticket 09 renames the
  identifiers of its table.
- Edit only this repo. burn-mamba gets its own ticket for the sites that
  break there. burn-deltanet has no issue tracker, so the Comments below list
  its sites.

## Method

- Use the Edit tool. `replace_all` is correct for an identifier in one file.
  Edit a word that also has other meanings (`step`, `slot`, `chunk`,
  `stories`, `schedule`) site by site. The opening slots of a packed row and
  the `chunks` method of slices stay.
- Before you change a message, find the tests that match its text.
- Review every hit with `rg` afterwards. Search `\bStepFlags` and
  `\bBucket\b` with word boundaries, because the new names
  `TrainingStepFlags` and `Bucketing` hold the old ones.
- Bump the crate version to 0.36.0 (`Cargo.toml`), because the rename breaks
  the API, the CLI and the saved files.

## Acceptance criteria

- [ ] `rg` over `src/`, `CLAUDE.md` and `README.md` finds no old name of the
      maps.
- [ ] `cargo check --all-targets`, `cargo check --features examples-common`,
      `cargo test --lib -- --test-threads=1` and
      `cargo test --lib --features examples-common -- --test-threads=1` pass
      (flex).
- [ ] `cargo doc --no-deps --features examples-common` has no warnings.

## Comments

- These saved files do not load or read the same after the rename. The user
  permitted these breaks:
  - A `progress.json` with the key `"step"`.
  - A tiny-stories training config with the key `"bucket"` (and its keys
    `"slots"` and `"chars"`).
  - A script that reads the key `"step"` of `metrics.jsonl`. About 30
    experiment scripts of burn-mamba (`examples/*/tmp/`) read it. The
    burn-mamba ticket names them.
- burn-mamba gets the ticket
  `../burn-mamba/.scratch/glossary-followups/issues/09-burn-stack-examples-renames.md`.
- burn-deltanet has no issue tracker. These files use a renamed name:
  - `Session::begin_step`: `examples/register/carousel/training.rs`,
    `examples/register/majority/training.rs`.
  - `LmModel::optim_step`: `examples/tiny-stories/training.rs`.

  burn-deltanet builds against
  burn-stack rev `984d42d`, which is older than tickets 02 and 07. So it has
  other breaks to follow too. For example, `examples/mnist-class/training.rs`
  implements an `optim_step` of the older `MnistModel`. The `MnistModel` of
  this crate has `train_step` instead.
