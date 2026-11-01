# 04: Prose sweep: utils and optim

Status: resolved

Blocked by: 01

**What to build:** The prose of the utils and optim modules uses the terms of
`GLOSSARY.md`, and no term that it lists under _Avoid_. Use the term map, the
method and the acceptance criteria of ticket 02.

## Scope

- Doc comments and comments of the utils modules. These are:
  - class markers, padding and packing,
  - layer schedules and LR schedules,
  - untied parameters and the captured step,
  - init, detach, the test helpers, the backend macros and the other helpers.
- Doc comments and comments of the optim modules. These are the Muon plan,
  the projection specs, the segmented optimizer, the SGD and the report.
- Their panic, assert and error messages, with the test check of ticket 02.
- Identifiers stay. List each identifier that conflicts with a glossary term
  under `## Comments` (for example, `ProjSegment`, `LrSegment`,
  `Lr::Sequence`).

## What to look for

- "Segment" for a part of a fused projection becomes **sub-projection**. For
  an LR schedule, write "one schedule of the chain".
- "Graph" for a capture becomes **captured step** or **replay**. Bare "graph"
  stays for the autodiff graph.
- "Level" stays for a **marker level**. The class-marker header uses "input
  boundary": write "at its input".
- "Boundary weights" and "the model boundary" in the Muon docs become **I/O
  weights**.
- Bare "schedule" becomes **layer schedule** or "LR schedule".

## Acceptance criteria

- [x] For each _Avoid_ term, `rg` over the scope finds only the uses that the
      tables of ticket 02 permit. Each other hit that you keep is listed with
      its reason under `## Comments`.
- [x] `cargo check` and `cargo test --lib -- --test-threads=1` pass (flex).
- [x] `cargo doc --no-deps` has no warnings.

## Comments

### Scope

The scope is `src/utils/` and `src/optim/`: 25 files, with their `tests.rs`.
Four files have no term to change: `backend_macros.rs`, `combined_grad.rs`,
`fprim.rs` and `test_helpers.rs`.

### Hits that stay

`rg` over the scope still finds these uses. The tables of ticket 02 permit
each one, or it has a different meaning from the _Avoid_ entry:

- **graph**: the autodiff graph (`detach.rs`, `GradHorizon`, `untied::tile`,
  `Weights`), the module path `utils::graph`, and the `Graph` type of Burn.
- **schedule**: the `Schedule` and `BidiSchedule` types, the `schedule`
  locals, and "each of its schedules" (the schedules of a chain,
  `Lr::scale_steps`). The module `scheduler` and two rustdoc headings of
  `scheduler/mod.rs` say "Scheduler". "Scheduler" is not an _Avoid_ term,
  and a heading is a link anchor.
- **segment**: a segment of a packed row (`packing.rs`). The other hits are
  identifiers (see below).
- **step**: the step call and the captured step. Two other uses stay, as in
  ticket 02:
  - "Muon steps", "the optimizer steps": the optimizer verb.
  - "a step size", "the step-size directions", "the bias of a discretisation
    step size": the step size of a discretised recurrence.
- **level**: a marker level (`class/mod.rs`, `padding/tests.rs`), and
  "high-level", "primitive-level" and "lower-level" (`fprim.rs`,
  `utils/mod.rs`).
- **slot**: the **Slot** of the glossary. Examples: the opening slots of a
  packed row (with the locals `slot_bs`, `is_slot_bs` and `slots` of
  `Packed::place`), a cache slot (the message `"{label}: slot {i} …"` of
  `graph/tests.rs`, the doc of `check_each_slot_alone`), and the `slot`
  locals of `CapturedStep` (the `Option` in a `RefCell`).
- **row**: a row of a batch. Also a row of a matrix or a table: the rows of a
  class marker table and of the table of `Packed::place`, `fprim.rs`, the
  rows of a strided cubecl buffer (`graph.rs`, `graph/tests.rs`), and `Row`
  and `rows` of `report.rs` (one line of the report).
- **chunk**: a chunk of a chunked scan ("only at the start of a chunk",
  `packing.rs`). The `chunk_len` code spans are identifiers (see below).
- **carry**: the plain verb.
- **variant**: the variants of a Rust enum (`schedule/mod.rs`,
  `segmented.rs`, `spec.rs`).
- **memory**: the memory of the device (`detach.rs`, "memory-efficient").
- **forward/backward**: the passes of autodiff, and the `forward` method. No
  hit names a direction of a bidirectional pair.
- **unshared**: buffers (`StepInput::assign`), not parameters.
- **generation**: text generation ("Seedless generation", `class/mod.rs`).
- **per-layer**: the `per_layer` field and locals of the class cursors and
  the class latents. They are not parameters.
- **full-length hint**: the message "Middle/End class markers need a
  full-length hint" stays. burn-mamba matches `need a full-length hint` in
  two `should_panic` tests (`src/unified/tests/class.rs:486` and `:1498`).
  It is not an _Avoid_ term. The docs say **length hint**, the glossary term.

### Identifiers that conflict with a glossary term

| Identifier | Where | Conflict |
|---|---|---|
| `ProjSegment` (also in the prelude of `lib.rs`) | `optim/spec.rs` | "segment" for a sub-projection |
| field `ProjSpec::segments`, locals `segment`, `segments` | `optim/` | "segment" for a sub-projection |
| `MuonPlan::without_segment` | `optim/mod.rs` | "segment" for a sub-projection |
| `Segmented`, `SegmentedState`, module `segmented` | `optim/segmented.rs` | "segment" for a sub-projection. Ticket 02 kept these names. |
| `BlockState`, `BlockOptim`, field `SegmentedState::blocks`, local `n_blocks` | `optim/segmented.rs` | "block" for a sub-projection. A **block** mixes the tokens of a sequence. |
| `LrSegment`, locals `segment`, `segments` | `utils/scheduler/mod.rs` | "segment" for one schedule of the chain |
| `Lr::Sequence` | `utils/scheduler/mod.rs` | a **sequence** is a list of tokens. The prose says "a chain of LR schedules". |
| `SgdConfig::step`, the mapper `Step` | `optim/sgd.rs` | bare "step" for a training step |
| parameter `step` of each `get_lr`, fields `total_steps`, `warmup_steps`, `LinearLr::steps`, `Lr::scale_steps` | `utils/scheduler/mod.rs` | bare "step" for a training step |
| module `utils::graph` (`graph.rs`, `graph/`), field `CapturedStep::graph`, locals `graph` | `utils/graph.rs` | bare "graph" for a captured step. The prose says "the recording". |
| `expects_graph` | `utils/graph/tests.rs` | bare "graph" for a captured step |
| `class_chunk_plan`, parameters and locals `chunk_len` | `utils/class/mod.rs` (`covers_whole`, `class_prime_plan`, `class_plan`, `insert_class_markers_padded`, `splice_class_embeddings`) | "chunk" for a window |
| locals `segments` | `splice_class_embeddings`, `splice_columns` | "segment" for a part of a tensor |
| parameters and locals `rows`, `rows_local_bs` | `Packed::from_starts`, `Packed::place`, `Padding` | "row" for the length of the time axis |
| `Row` (enum), `stack_rows`, `padded_rows`, parameters `rows` | `utils/padding/tests.rs` | "row" for a position |
| `check_each_slot_alone` and the tests `padded_layers_are_each_slot_alone`, `padded_multi_gate_layers_are_each_slot_alone`, `padded_layers_under_a_grad_horizon_are_each_slot_alone`, `padded_latent_network_is_each_slot_alone`, `padded_bidi_layers_are_each_slot_alone`, `chunked_padded_layers_are_each_slot_alone` | `utils/padding/tests.rs` | "slot" for a row. "Chunked" for a split into windows. |
| `Schedule`, the tests `schedule`, `bidi_schedule` and `grad_horizon_depth_rejects_a_custom_schedule` | `utils/schedule/` | bare "schedule" for a layer schedule. Ticket 02 kept the type name. |
| `run_schedule`, `a_sequence_counts_each_segment_from_its_start`, `a_sequence_survives_a_json_round_trip`, `an_empty_sequence_is_invalid`, `a_sequence_starts_at_step_zero`, `the_segments_start_at_increasing_steps`, `validate_checks_the_inner_sequences`, `set_peak_scales_a_sequence`, `scale_steps_moves_every_step_of_a_sequence` | `utils/scheduler/tests.rs` | "sequence" and "segment" for the chain, bare "schedule", bare "step" |

For a later rename:

- The user permits a break of the saved configs for this work. `Lr` and
  `LrSegment` derive `Config`. So the variant name `Sequence` and the fields
  `total_steps`, `warmup_steps` and `steps` are serialized keys
  (`a_sequence_survives_a_json_round_trip` matches `"Sequence"`).
- burn-mamba uses `ProjSegment`, `ProjSpec::segments`, `BlockState`,
  `SegmentedState::blocks`, `without_segment`, `class_chunk_plan`,
  `class_prime_plan`, `utils::graph` and `Schedule`. burn-deltanet uses
  `ProjSegment`, `ProjSpec::segments` and `Schedule`. Neither uses
  `LrSegment` or `Lr::Sequence`.
- A proposal, for the user to approve or change:

  | Old | New |
  |---|---|
  | `ProjSegment` | `SubProjection` |
  | `ProjSpec::segments` | `ProjSpec::sub_projections` |
  | `MuonPlan::without_segment` | `MuonPlan::without_sub_projection` |
  | `BlockState`, `BlockOptim`, `SegmentedState::blocks` | `SubProjectionState`, `SubProjectionOptim`, `SegmentedState::states` |
  | `Lr::Sequence`, `LrSegment` | `Lr::Chain`, `LrLink` |
  | `class_chunk_plan`, `chunk_len` | `class_window_plan`, `window_len` |
  | module `utils::graph`, field `graph`, `expects_graph` | `utils::capture`, `recording`, `expects_recording` |
  | the test names and test locals above | the names of the glossary terms (row, position, window, chain) |

  The proposal keeps `Segmented`, `Schedule`, `SgdConfig::step` and the
  step counts of the LR schedules. These are the usual names of an
  optimizer and of an LR schedule. Ticket 02 also kept `Segmented` and
  `Schedule`.

### Choices that this sweep made

- **The recording.** The glossary has no term for what a capture returns
  (the `Graph` of Burn). The prose calls it "the recording", and "a hardware
  recording" when `Graph::is_hardware` is true. Bare "graph" now names only
  the autodiff graph. Open for the user: does the glossary need an entry for
  **recording**?
- **Captured step.** The glossary entry says "a step that the backend records
  once". A `CapturedStep` also holds a fixed-shape `forward` (caches `()`)
  and a training step (`Weights`). The header of `graph.rs` lists them as
  "other calls of a fixed shape". Open for the user: does the glossary entry
  need to name them?
- **LR schedules.** `Lr::Sequence` is "a chain of LR schedules", and one
  `LrSegment` is "one schedule of the chain" (the term map of ticket 02). A
  step of an LR schedule is a "training step". "A step down" of the rate is
  "a drop". "A top-level cosine" is "a cosine outside a chain", because
  "level" stays for a marker level.
- **Sub-projections.** "Segment", "column block" and "block" for a part of a
  fused projection are now **sub-projection**. "Fused weight" is **fused
  projection**. "Mixer block" is **block**.
- **I/O weights.** "Embedding-like matrices … at the model boundary" (the
  Muon docs) are now **I/O weights**. "The boundary projections of
  `LatentNetwork`" (the init tests) are its input and output projections.
- **Layer schedules.** The docs of the variants of `Schedule` and
  `BidiSchedule` said "fills virtual positions by wrapping around the real
  schedule". A **position** is a place on the time axis. The docs now give
  the map of each variant, from its `real_idx`. For example, `StridedCyclic`
  runs the straight pass on the even real layers. The two symmetric variants
  name the **bidirectional pair** of the virtual layers `2i` and `2i + 1`.
  "Weight set" is now **real layer**, and "weight sharing" is "virtual
  layers" or "tied parameters".
- **The worked examples of `schedule/mod.rs`** keep their text. Their line
  breaks are now a backslash at the end of the line, not two trailing spaces.
  Rustdoc renders both as a hard line break. The Edit tool cannot keep
  trailing spaces, so a two-space break goes away at the next edit of its
  line. This sweep lost four of them before the review found it.
- **The heading of `schedule/mod.rs`** is now "Layer schedules: virtual layer
  → real layer". The old heading said "real-weight scheduling" (a real layer
  and bare "schedule"). No link points to its anchor, in this crate or in a
  consumer.
- **Packing and padding.** "Slot" for one sequence of a batch is now **row**.
  "Row" for a place on the time axis is now **position**, and "real row" is
  **real token**. A sequence of a packed row is a **segment**. The shapes
  `[batch, rows]` are now `[batch, sequence]`, after the suffix `_bs`.
  "Chunk" for the part of a sequence that one `forward` takes is now
  **window**.
- **Class markers.** The header no longer cites `[CLS]` or a "register"
  (_Avoid_ terms). "At its input boundary" is now "at its input".
  "Full-length hint" is now **length hint** in the docs.
- `Padding::reversed` said "Its batch-wide order is the forward one"
  ("forward" for a direction). It now says "Its batch-wide order does not
  change".
- These messages changed. For each one, the list gives the test that matches
  its text:
  - `Lr::validate`: "an LR chain needs at least one schedule"
    (`an_empty_sequence_is_invalid`), "the first schedule of an LR chain must
    start at training step 0" (`a_sequence_starts_at_step_zero`), and "the
    schedules of an LR chain must start at increasing training steps"
    (`the_segments_start_at_increasing_steps`,
    `validate_checks_the_inner_sequences`). The tests now match the new
    text.
  - No test matches these: the `expect` of `Lr::get_lr`, "an LR schedule
    with a peak of 0 …" (`Lr::set_peak`), the three changed asserts of
    `Packed`, the shape assert of `Padding::in_row_order`, the three
    `expect`s of `class/mod.rs` ("the class markers have no embedding
    param"), and the two invariant messages of `CapturedStep` ("replay needs
    a recording", "the capture keeps a recording only where it can read the
    buffer ids").
  - The `Middle` assert of `Padding::splice` now ends with "half of the
    length of each row". `a_middle_marker_needs_the_whole_padded_sequence`
    matches "needs the whole padded sequence", which stays.
  - The last line of `MuonPlan::describe` now ends with "(* = sub-projection
    left on the fallback)". No test of this crate or of a consumer reads that
    text.
  - The assert messages of `graph/tests.rs` ("step {k} after set_caches …"),
    `padding/tests.rs` ("row {b} …", "owned by a row") and
    `scheduler/tests.rs`.

### Docs that did not match the code

The sweep fixed two docs to match the code:

- `init.rs` listed "a class-token table" among "the bespoke parameters of a
  block". The class marker tables (`class_tokens_emb`, `class_latents_emb`)
  belong to the networks and the layer containers, not to a block. The list
  no longer says "of a block", and the comment of `Reinit::map_float` names
  both kinds.
- The header of `class/mod.rs` said that a layer inserts the class latents.
  A layer stack also inserts them (`Layers::class_latents`,
  `BidiLayers::class_latents`). The header and the doc of `ClassLatent` now
  say "a layer stack or a layer".
