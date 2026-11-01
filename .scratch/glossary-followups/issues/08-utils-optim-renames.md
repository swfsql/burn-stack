# 08: Rename the utils and optim identifiers that conflict with the glossary

Status: resolved

Blocked by: 04

**What to build:** The identifiers of the utils and optim modules use the
terms of `GLOSSARY.md`, as their prose does after ticket 04. Ticket 04 lists
the identifiers that conflict with a glossary term, and it proposes the
names. The user approved the proposal. Breaking changes are permitted, also
for saved configs.

## Rename map

The approved proposal of ticket 04:

| Old | New | Where |
|---|---|---|
| `ProjSegment` | `SubProjection` | `optim/spec.rs`, the re-exports of `optim/mod.rs` and of the prelude of `lib.rs` |
| field `ProjSpec::segments` | `sub_projections` | `optim/` |
| `MuonPlan::without_segment` | `without_sub_projection` | `optim/mod.rs` |
| `BlockState`, `BlockOptim` | `SubProjectionState`, `SubProjectionOptim` | `optim/segmented.rs`, the re-export of `optim/mod.rs` |
| field `SegmentedState::blocks` | `states` | `optim/segmented.rs` |
| variant `Lr::Sequence` (a config key) | `Lr::Chain` | `utils/scheduler/` and the tests of `examples/` |
| `LrSegment` | `LrLink` | `utils/scheduler/`, the re-export of `utils/mod.rs`, `examples/` |
| `class_chunk_plan` | `class_window_plan` | `utils/class/mod.rs` and its callers |
| parameters and locals `chunk_len` | `window_len` | `utils/class/mod.rs` only. Elsewhere `chunk_len` is the chunk of a chunked scan, and it stays. |
| module `utils::graph` (`graph.rs`, `graph/`) | `utils::capture` (`capture.rs`, `capture/`) | `utils/`, the doc links that name the path, `examples/tiny_stories/sample.rs` |
| field and locals `graph` | `recording` | `CapturedStep` |
| `expects_graph` | `expects_recording` | the tests of the captured step |

The test names and test locals of ticket 04 get the glossary terms:

| Old | New | Where |
|---|---|---|
| `Row` (enum), `stack_rows`, `padded_rows`, parameters and closures `rows` | `Position`, `stack_positions`, `padded_positions`, `positions` | `utils/padding/tests.rs` |
| `check_each_slot_alone` | `check_each_row_alone` | `utils/padding/tests.rs` |
| `padded_layers_are_each_slot_alone`, `padded_multi_gate_layers_are_each_slot_alone`, `padded_layers_under_a_grad_horizon_are_each_slot_alone`, `padded_latent_network_is_each_slot_alone`, `padded_bidi_layers_are_each_slot_alone` | the same names with `each_row_alone` | `utils/padding/tests.rs` |
| `chunked_padded_layers_are_each_slot_alone` | `padded_layers_split_into_windows_are_each_row_alone` | `utils/padding/tests.rs` |
| `schedule`, `bidi_schedule`, `grad_horizon_depth_rejects_a_custom_schedule` | `layer_schedule`, `bidi_layer_schedule`, `grad_horizon_depth_rejects_a_custom_layer_schedule` | `utils/schedule/tests.rs` |
| `run_schedule` | `run_chain` | `utils/scheduler/tests.rs` |
| `a_sequence_counts_each_segment_from_its_start` | `a_chain_counts_each_schedule_from_its_start` | `utils/scheduler/tests.rs` |
| `a_sequence_survives_a_json_round_trip` | `a_chain_survives_a_json_round_trip` | `utils/scheduler/tests.rs` |
| `an_empty_sequence_is_invalid` | `an_empty_chain_is_invalid` | `utils/scheduler/tests.rs` |
| `a_sequence_starts_at_step_zero` | `a_chain_starts_at_training_step_zero` | `utils/scheduler/tests.rs` |
| `the_segments_start_at_increasing_steps` | `the_schedules_of_a_chain_start_at_increasing_training_steps` | `utils/scheduler/tests.rs` |
| `validate_checks_the_inner_sequences` | `validate_checks_the_inner_chains` | `utils/scheduler/tests.rs` |
| `set_peak_scales_a_sequence` | `set_peak_scales_a_chain` | `utils/scheduler/tests.rs` |
| `scale_steps_moves_every_step_of_a_sequence` | `scale_steps_moves_every_training_step_of_a_chain` | `utils/scheduler/tests.rs` |

Ticket 04 also lists private locals with a conflict. They break nothing
outside their function, so they follow the map:

| Old | New | Where |
|---|---|---|
| locals `segment`, `segments` of a `ProjSpec` | `sub_projection`, `sub_projections` | `optim/` |
| locals `n_blocks`, `blocks` | `n_sub_projections`, `states` | `optim/segmented.rs` |
| locals `segment`, `segments` of a chain | `link`, `links` | `utils/scheduler/mod.rs` |
| locals `segments` (the parts of a tensor) | `parts` | `splice_class_embeddings`, `splice_columns` |
| parameters and locals `rows` (the length of the time axis) | `sequence` | `Packed::from_starts`, `Packed::place`, `Padding` |
| local `rows_local_bs` | `positions_local_bs` | `Padding::splice` |

These names stay: `Segmented`, `SegmentedState`, `Schedule`,
`SgdConfig::step`, and the training-step counts of the LR schedules
(`total_steps`, `warmup_steps`, `LinearLr::steps`, `scale_steps`, the
parameter `step` of `get_lr`). They are the usual names of an optimizer and of
an LR schedule.

## Scope

- The identifiers of the map, at each definition and each use in this crate.
  This includes the callers in `src/modules/` and the tests of
  `src/examples/`.
- The prose that names an identifier of the map: doc links, code spans, the
  File Map of `CLAUDE.md`, and the Comments of ticket 04.
- Edit only this repo. burn-mamba gets its own ticket for the sites that
  break there. burn-deltanet has no issue tracker, so the Comments below list
  its sites.

## Method

- Move the files of the module with `git mv`.
- Use the Edit tool. `replace_all` is correct for an identifier in one file.
  Edit a word that also has other meanings (`segments`, `rows`, `graph`,
  `chunk_len`) site by site.
- Review every hit with `rg` afterwards.
- Bump the crate version to 0.35.0 (`Cargo.toml`), because the rename breaks
  the API and the saved configs.

## Acceptance criteria

- [x] `rg` over `src/`, `CLAUDE.md` and `README.md` finds no old name of the
      map.
- [x] `cargo check --all-targets`, `cargo check --features examples-common`,
      `cargo test --lib -- --test-threads=1` and
      `cargo test --lib --features examples-common -- --test-threads=1` pass
      (flex).
- [x] `cargo doc --no-deps` has no warnings.

## Comments

- The commit also bumps the crate version to 0.35.0 (`Cargo.toml`), because
  the rename breaks the API and the saved configs. `Cargo.lock` is not
  tracked.
- `cargo test --lib -- --test-threads=1`: 87 tests pass. With
  `--features examples-common`: 112 tests pass. `cargo doc --no-deps` has no
  warnings, also with `--features examples-common`.
- The examples tests had two copies of `expects_graph`
  (`examples/trainer/tests.rs`, `examples/tiny_stories/sample/tests.rs`).
  Both are now `expects_recording`. The two doc lines that name them now say
  "hardware recording". The other prose of `examples/` stays for ticket 05.
- These names are not in the map, and they stay:
  - `the_flags_rescale_a_sequence` (`examples/cli/tests.rs`) names an
    `Lr::Chain`. It is an identifier of `examples/`, so ticket 05 lists it.
  - `padded_rows_match_each_row_alone` (`modules/hnet/tests.rs`, ticket 07):
    its "padded rows" are rows of a padded batch.
- A saved config with an `Lr` chain under the old key `"Sequence"` does not
  load. The user permitted this break.
- burn-mamba gets the ticket
  `../burn-mamba/.scratch/glossary-followups/issues/08-burn-stack-utils-optim-renames.md`.
  Its experiment scripts (`examples/hnet-stories/tmp/1m-a/gen.py`,
  `lrprof.py`, `lrseq.py`) write the key `"Sequence"`. The ticket names
  them.
- burn-deltanet has no issue tracker. These files use a renamed name:
  `ProjSegment` in `src/delta_product/delta_product.rs`,
  `src/deltanet/deltanet.rs`, `src/gated_deltanet_1/gated_deltanet_1.rs` and
  `src/gated_deltanet_2/gated_deltanet_2.rs`, and `ProjSegment` and
  `ProjSpec::segments` in `src/unified/tests/optim.rs`. burn-deltanet builds
  against burn-stack rev `984d42d`, which is older than tickets 02 and 07. So
  it has other breaks to follow too (for example `init_bias_offset`).
