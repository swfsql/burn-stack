# 05: Prose sweep: examples

Status: resolved

Blocked by: 01

**What to build:** The prose of the example scaffolding uses the terms of
`GLOSSARY.md`, and no term that it lists under _Avoid_. Use the term map, the
method and the acceptance criteria of ticket 02. Build and test with the
`examples-common` feature.

## Scope

- Doc comments and comments of the examples modules. These are:
  - the CLI, the profiler, the VRAM cap and the session,
  - the device helpers, the trainer and the training config,
  - MNIST and tiny stories.
- Their panic, assert and error messages, and the CLI help strings (strict
  style), with the test check of ticket 02.
- Identifiers and CLI flag names stay. List each one that conflicts with a
  glossary term under `## Comments` (for example, `--bucket-chars` and a
  per-slot field).

## What to look for

- "Story" stays for the data of the tiny-stories dataset. In a packed row, a
  story is a **segment**.
- "Slot" for a batch length of the length-bucketed loader becomes **bucket**.
  "Slot" for one sequence of a batch becomes **row**.
- "Chunk" for a part of a prompt in the prefill becomes **window**.
- "BPTT" stays for the training window.
- Bare "step" for an optimizer update becomes **training step**.

## Acceptance criteria

- [x] For each _Avoid_ term, `rg` over the scope finds only the uses that the
      tables of ticket 02 permit. Each other hit that you keep is listed with
      its reason under `## Comments`.
- [x] `cargo check --features examples-common` and
      `cargo test --lib --features examples-common -- --test-threads=1` pass
      (flex).
- [x] `cargo doc --no-deps --features examples-common` has no warnings.

## Comments

### Scope

The scope is `src/examples/`: 21 files, with their `tests.rs`. Three files
have no term to change: `mnist/mod.rs`, `tiny_stories/mod.rs` and
`mnist/render.rs`.

### Hits that stay

`rg` over the scope still finds these uses. The tables of ticket 02 permit
each one, or it has a different meaning from the _Avoid_ entry:

- **slot**: the opening slots of a packed row (`PackLayout::lead`,
  `PackedRows`, `PackedStoriesBatcher`, the corpus line "{} opening slots").
  They are the **Slot** of the glossary. The other hits are identifiers (see
  below).
- **step**: the step call, a captured step and a training step. Examples of
  the step call: "the decode steps", "an eager step" and "every later
  character costs a `step`" (`sample.rs`). The sample tests say "one step at
  a time".
  Also the identifiers `progress.step`, the key `"step"` of `metrics.jsonl`,
  `SgdConfig::step` and the `InferenceStep::step` of Burn.
- **graph**: the autodiff graph ("cut off the graph", "drop the graph that
  produced them", `lm.rs`). The `--no-graph` and `graphs` names are
  identifiers (see below).
- **chunk**: a chunk of a chunked scan ("its chunk starts",
  `PackLayout::align`), and the `chunks` method of slices (`mnist/dataset.rs`,
  `bucket_batches`, `decode`). The code spans `[1, chunk, 2]` and "windows of
  `chunk` tokens" (`Prefill`) name the identifier.
- **story**: the data of the tiny-stories dataset, as the ticket permits. In
  a packed row, the prose says **segment**: the layout docs of `PackLayout`,
  `PackedRows`, `PackedStoriesBatcher` and `pack_rows`, and the corpus line
  "segments at multiples of {}". The docs of the data side keep "story": the
  stories of a `PackedItem`, the shuffle order of `pack_rows`, the truncation
  of a long story.
- **row**: a row of a batch. Also a row of a matrix or a table. Examples:
  - the parquet rows and row groups (`dataset.rs`, `Split`),
  - the pixel rows of `mnist/render.rs`,
  - the `[rows, VOCAB_SIZE + 1]` matrix of `masked_output` and `PerCharLoss`
    (one row for each flat position),
  - the local `row` of `lm/tests.rs` (one row of logits).
- **memory**: the memory of the device or of the host ("peak memory", "the
  memory pools", "load it in-memory"). It is not a cache.
- **schedule**: "LR schedule" and "a chain of LR schedules". The `schedule`
  field and parameter of `Session` are identifiers (see below).
- **segment**: a segment of a packed row only.
- **carry**: the plain verb ("carry the state into the next window", the help
  of `--frontier-bits` and `--no-frontier`). The locals `carry` of
  `bucket_batches` hold the stories that move up to the next bucket. They are
  not the gradient horizon.
- **BPTT, TBPTT**: the training window of the examples ("the BPTT length",
  "the BPTT span of one forward", "the pure stateful-TBPTT baseline" of
  `FrontierGate::Always`). It is the BPTT from one window to the next, as in
  `CacheStack::detach` (ticket 02), not the gradient horizon.
- **level**: "character-level" (each token is a character). It is not a
  stage.
- **forward/backward**: the passes of autodiff and the forward call. "The
  extra pass" of the profiler is the pass of an auxiliary loss. No hit names
  a direction of a bidirectional pair.
- **Two long log lines**: the linter finds two sentences above the strict
  cap. They are the corpus line of the packed rows (`dataloaders_for`) and the
  progress line of `epoch_train` (`lm.rs`). They are metric lines with format
  fields, and a split changes the log format. They stay.
- **The code span `[stories, slot]`** (`BucketedStoriesBatcher`) stays. It
  names the field `BucketItem::slot`. The doc says that `slot` is the bucket.

### Identifiers and CLI flags that conflict with a glossary term

| Identifier | Where | Conflict |
|---|---|---|
| `Bucket` (struct), `TinyStoriesConfig::bucket` | `tiny_stories/lm.rs` | A **bucket** is one batch length. `Bucket` holds all the buckets, plus `chars` and `probe`: the geometry of the length-bucketed training. |
| field `Bucket::slots`, `parse_slots`, the parameters `slots` of `bucket_batches`, `probe_order` and `BucketedStoriesDataset::new`, the locals `slot` and `slots` | `tiny_stories/lm.rs`, `tiny_stories/dataset.rs`, `dataset/tests.rs` | "slot" for a bucket |
| field `BucketItem::slot` | `tiny_stories/dataset.rs` | "slot" for a bucket (the per-slot field of the ticket) |
| test `probe_order_opens_with_each_slot_and_keeps_every_batch` | `dataset/tests.rs` | "slot" for a bucket |
| flag `--bucket-chars`, field `Bucket::chars`, `Overrides::bucket_chars` | `tiny_stories/lm.rs` | The scored characters are for one batch, not for one bucket (a batch length). |
| test `packed_batcher_lays_out_each_story`, field `PackedItem::stories` | `dataset/tests.rs`, `tiny_stories/dataset.rs` | "story" for a segment of a packed row |
| field `Prefill::chunk` and the parameter `chunk` of `Prefill::new`, `ChunkFn`, `Prefill::eager_chunks`, the locals `chunk`, the test constant `CHUNK` | `tiny_stories/sample.rs`, `sample/tests.rs` | "chunk" for a window of the prefill |
| flag `--no-graph`, field `AppArgs::no_graph`, `AppArgs::graphs`, the parameter `graphs` of `Trainer::new`, the closure parameter `graphs` of the trainer test | `cli.rs`, `trainer.rs`, `trainer/tests.rs` | bare "graph" for a captured step |
| `Trainer::step`, `sgd_step` | `trainer.rs` | bare "step" for a training step |
| `StepFlags` | `cli.rs` | bare "step" for a training step (the flags of a training step) |
| field `TrainingProgress::step` (a config key), the key `"step"` of `metrics.jsonl` | `session.rs` | bare "step" for a training step |
| `Session::begin_step`, `Session::skip_steps` and its parameter `steps`, the parameter `step` of `Session::due` | `session.rs` | bare "step" for a training step |
| field `Session::schedule`, the parameter `schedule` of `Session::new` | `session.rs` | bare "schedule" for an LR schedule |
| `LmModel::optim_step`, the locals `_step` and `step_lr` of `epoch_train`, the local `_step` of the MNIST `epoch_train` | `tiny_stories/lm.rs`, `mnist/classify.rs` | bare "step" for a training step |
| test `captured_sgd_steps_are_the_eager_steps`, the constant `STEPS` | `trainer/tests.rs` | bare "step" for a training step |
| tests `the_batch_size_rescales_a_cosine_schedule`, `the_flags_rescale_a_sequence`, `the_clock_starts_at_the_first_step` | `cli/tests.rs` | bare "schedule", "sequence" for a chain of LR schedules (ticket 08 renamed the same words in the scheduler tests), bare "step" |
| local `rows` of `a_prefill_is_the_same_captured_and_eager_and_is_the_prompt_in_one_pass` | `sample/tests.rs` | "row" for the length of the time axis |

For a later rename:

- The user permits a break of the saved configs for this work.
  `TrainingProgress::step` is a key of `progress.json`, and `Bucket::slots`
  and `Bucket::chars` are keys of the tiny-stories config.
- `--no-graph`, `--bucket-chars` and `StepFlags` are part of the CLI. They
  are also part of the API that the consumers use. burn-mamba and
  burn-deltanet print `cli::HELP` in every example, and burn-mamba also
  prints `StepFlags::HELP` and `Overrides::HELP`.

### Copies outside this crate

This ticket did not edit the consumers (another session works in
burn-mamba). These copies still use the old words:

- `burn-mamba/examples/README.md` (about lines 188 to 262) has a copy of
  `cli::HELP`: "captured CUDA graphs", "the LR-schedule step", "schedule
  step". The examples of burn-mamba also say "captured graph" in their docs
  (`mnist-class`, `mnist-ae`, `tiny-stories/inference.rs`, `hnet-stories`).
- `burn-deltanet/examples/README.md:70` says "continue the schedule step".
- The scripts of burn-mamba grep `VRAM LIMIT`. The message of `vram::check`
  keeps that prefix.

### Choices that this sweep made

- **Bucket.** The doc of `bucket_batches` said "batches of similar length
  (the *buckets*)". But a **bucket** is a batch length, not a batch. The doc
  now says "one bucket (a batch length) for each batch". The doc of
  `BucketedStoriesDataset::summary` and the legend of the corpus lines now
  say `bucket: batches × stories`. The output of `summary` does not change.
- **Packed rows.** The contents of a packed row are its **segments**: the
  docs of `PackedItem`, `PackedStoriesDataset::rows` and the return value of
  `pack_rows`. "Story" stays for the data that the packer places.
- **Training step.** "Optimizer step" and bare "step" for an optimizer
  update are now **training step**. The `opt` phase of the profiler and
  `LmModel::optim_step` are "the optimizer update": the update part of one
  training step. In the AdamW doc, "the step is `m/ε`" (the size of an
  update) is now "the update is `m/ε`". In `sample_token` and its tests,
  "step" for the spacing of floats is now "spacing" and "the adjacent float",
  as `half_spacing` of ticket 02.
- **`--profile`.** The help and the docs said "once per N windows". The
  profiler counts the calls of `prof::opt_end`, one for each training step.
  So they now say "once per N training steps". In the character LM, a
  training step is one window, so the meaning does not change there.
- **Window.** A part of a prompt in `Prefill` is a **window** (it was a
  "chunk"). "One chunkwise pass over the opening and the whole prompt" is now
  "One forward over the opening and the whole prompt". "Chunkwise" named the
  parallel call, and a chunked scan is the shape of one family. The readback
  of `decode` said "one chunk at a time". It now says "`READBACK` characters
  at a time".
- **Recording.** "Graph" for what a capture records is now "recording" or
  "captured step", as in ticket 04. The CLI help said "replays its
  fixed-shape passes … from captured CUDA graphs". It now says that the
  example captures its fixed-shape calls and replays them. The replay needs a
  backend that can record the calls (for example, CUDA).
- **Three calls.** The header of `sample.rs` said "the three execution modes
  of a library". It now says "the three calls of a network", the words of
  `src/lib.rs`.
- **KV cache.** "With no growing KV cache" is now "with a cache that does not
  grow".
- **Reset and restart.** "A **reset** (zero) cache" (`sample.rs`) is now "a
  **new** (zero) cache": a **reset** is the mark on the first token of a
  segment. "A restarted run is visibly a new segment" (`MetricsLog`) is now
  "a run that starts again from training step 0 shows as a new curve".
  "Never restarted part-way through" (`epoch_valid`) is now "never started
  again". "Across process restarts" (`load_model`) is now "from one process
  to the next".
- **Boundary.** "No separator character stands in for its boundary" is now
  "marks its start or its end". "Gradients never cross a window boundary" is
  now "never pass from one window to the next".
- **Real rows.** `Prefill` said "as its real rows alone would come out" and
  "the last real row". These are now "real tokens" and "the last real token".
- **Carry.** The noun "the carry" of the frontier gate is now "the carried
  state". "With no carry" is now "with no carried state".
- **Layer.** "The composition layer" (`examples/mod.rs`) is now "the
  containers". "Layer" names only a Pre-LN residual around one block.
- **Position.** "The last timestep" (MNIST) is now "the last position". "Two
  in a row" (the doc of `STORY_SEPARATOR`) is now "two of them one after the
  other".
- **Messages.** No test of this crate or of a consumer matches a changed
  message. The two `should_panic` tests of `cli/tests.rs` match "exclusive"
  and "would be ignored", and both stay. These messages changed:
  - `vram::check`: "… MiB. The process stops." (no semicolon).
  - The two `expect`s of `Trainer`: "a training step leaves its state
    behind".
  - `AppArgs::resumed_progress` and `save_progress`: "training step". "Starting
    over at step 0 (saved at step {}; --resume continues it)" is now "Starting
    again at training step 0. The saved progress is at training step {}, and
    --resume continues it."
  - `bucket_batches`: "the buckets must ascend strictly", "a story is longer
    than the longest bucket". `BucketedStoriesDataset::new`: "at least one
    bucket". `parse_slots`: "the bucket {s:?} is not a positive integer".
  - `pack_rows`: "a segment starts at a multiple of at least 1".
  - `Prefill::new`: "a prefill window holds at least one token".
    `Prefill::run`: "a prompt fills at least one window".
  - The corpus lines of `dataloaders_for`, and the assert messages of
    `cli/tests.rs` and `dataset/tests.rs`.
  - The help texts `cli::HELP`, `StepFlags::HELP` and `Overrides::HELP`:
    training step, LR schedule, chain of LR schedules, bucket and row. The
    help of `--pack-open` has no semicolon.

### Docs that did not match the code

The review found two docs that did not match the code. The sweep fixed them
to match the code:

- `TrainingProgress::step` said "Optimizer steps taken". But
  `Session::skip_steps` also adds the windows that the frontier gate of the
  character LM dropped, and nothing trains them. The doc now says that the
  count holds the training steps taken plus the skipped ones. The doc of
  `skip_steps` now says that it adds to the count with no training.
- The sample of the character LM (`epoch_train`) went into "a new per-step
  file". The file name holds the epoch and the batch
  (`sample-epoch-{epoch}-batch-{b}.txt`), and one batch spans many training
  steps. The comment now says "a new file for this batch". The MNIST comment
  says the same. There, one batch is one training step.
