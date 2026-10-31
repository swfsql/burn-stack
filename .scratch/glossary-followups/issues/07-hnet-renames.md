# 07: Rename the H-Net identifiers that conflict with the glossary

Status: resolved

Blocked by: 03

**What to build:** The identifiers of the H-Net modules use the terms of
`GLOSSARY.md`, as their prose does after ticket 03. Ticket 03 lists the
identifiers that conflict with a glossary term. The user approved the names
of the rename map below. Breaking changes are permitted, also for saved
configs.

## Rename map

| Old | New | Where |
|---|---|---|
| `boundary_bs` (field) | `chunk_boundary_bs` | `Routing` |
| `boundary_bs` (parameter) | `chunk_boundary_bs` | `chunk::plan` |
| local `boundaries` | `chunk_boundaries` | `step_unrolled`, `forward_equals_step_unrolled` (`tests.rs`) |
| `gather_rows` | `gather_positions` | `chunk.rs` (the `gather_rows` of `cache.rs` stays) |
| `after_rows` | `after_positions` | `RouterCache` |
| local `row_bd` | `x_last_bd` | `RouterCache::after_rows` |
| `smooth_block` (field and config key) | `smooth_scan_block` | `HNet`, `HNetShape` (and so `with_smooth_block`) |
| parameter `block` | `scan_block` | `smooth` |
| `n_start_tokens` (config key) | `n_start_embeddings` | `HNetVocabShape` (and so `with_n_start_tokens`) |
| `chunk_plan_points_at_the_boundary_rows` | `chunk_plan_points_at_the_chunk_boundaries` | `tests.rs` |
| `only_a_new_sequence_forces_its_first_boundary` | `only_a_new_sequence_forces_its_first_chunk_boundary` | `tests.rs` |
| `a_call_without_boundaries_skips_the_inner_network` | `a_call_without_chunk_boundaries_skips_the_inner_hnet` | `tests.rs` |
| `forward_is_chunkable_through_the_cache` | `forward_splits_into_windows_through_the_cache` | `tests.rs` |
| `padded_slots_match_each_slot_alone` | `padded_rows_match_each_row_alone` | `tests.rs` |
| `start_rows_forward_equals_prime_then_steps` | `start_embeddings_forward_equals_prime_then_steps` | `tests.rs` |

The user chose to keep the term "start embedding" (ticket 03).

## Scope

- The identifiers of the map, at each definition and each use in
  `src/modules/hnet/`.
- The prose that names an identifier of the map: doc links, code spans and
  the ticket 03 Comments.
- Edit only this repo. burn-mamba gets its own ticket for the sites that
  break there.

## Method

- Use the Edit tool. `replace_all` is correct for an identifier in one file.
  `gather_rows` is an exception: `mod.rs` and `tests.rs` use both
  `chunk::gather_rows` and `cache::gather_rows`, so edit those sites one by
  one.
- Review every hit with `rg` afterwards.

## Acceptance criteria

- [x] `rg` over `src/` finds no old name of the map. `cache::gather_rows`
      stays.
- [x] `cargo check --all-targets` and `cargo test --lib -- --test-threads=1`
      pass (flex).
- [x] `cargo doc --no-deps` has no warnings.

## Comments

- The commit also bumps the crate version to 0.34.0 (`Cargo.toml`), because
  the rename breaks the API and the saved configs.
- `cargo check --features examples-common` also passes. The example
  scaffolding uses no name of the map.
- A saved `HNetShape` or `HNetVocabShape` config with the old keys
  (`smooth_block`, `n_start_tokens`) is not compatible. The user permitted
  this break, also when such a config fails to load.
- burn-mamba gets the ticket
  `../burn-mamba/.scratch/glossary-followups/issues/06-hnet-renames.md`.
  burn-deltanet uses no name of the map.
