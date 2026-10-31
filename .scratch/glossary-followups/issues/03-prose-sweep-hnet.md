# 03: Prose sweep: H-Net

Status: resolved

Blocked by: 01

**What to build:** The prose of the H-Net modules uses the terms of
`GLOSSARY.md`, and no term that it lists under _Avoid_. Its H-Net group is the
most important. Use the term map, the method and the acceptance criteria of
ticket 02.

## Scope

- Doc comments and comments of the H-Net modules. These are the stage, the
  router, the chunk index math, the smoothing, the caches, the vocab network
  and the shapes.
- Their panic, assert and error messages, with the test check of ticket 02.
- Identifiers stay. List each identifier that conflicts with a glossary term
  under `## Comments`.

## What to look for

- H-Net is the main user of "row" for a position ("the chunk-starting rows",
  "a gradient through every row"). Write **position**.
- "Main network", "inner network", "encoder/decoder network" and "level"
  become **main stack**, **inner H-Net**, **encoder/decoder** and **stage**.
- "Boundary" becomes **chunk boundary**.
- "Slot" for one sequence of a batch ("differs from slot to slot") becomes
  **row**.
- The symbols of the paper stay in the formulas (`bₜ`, `STE(c)`, `W_r`, …).

## Acceptance criteria

- [x] For each _Avoid_ term, `rg` over the scope finds only the uses that the
      tables of ticket 02 permit. Each other hit that you keep is listed with
      its reason under `## Comments`.
- [x] `cargo check` and `cargo test --lib -- --test-threads=1` pass (flex).
- [x] `cargo doc --no-deps` has no warnings.

## Comments

### Scope

The scope is `src/modules/hnet/`: the eight files, with `tests.rs`. The
H-Net lines of `src/modules/mod.rs` and `src/modules/cache.rs` already use the
glossary terms (ticket 02).

### Hits that stay

`rg` over the scope still finds these uses. The tables permit each one, or it
has a different meaning from the _Avoid_ entry:

- **row**: one sequence of a batch. Examples: the per-row cache operations
  (`select_rows`, `gather_rows` and `merge_rows` of `cache.rs`), and the rows
  that `step` gives to the inner H-Net. Also `Routing::merged`,
  `Routing::reached` and the `row` closure of the padding test. Also a row of
  a matrix (the `row` local of `HNetVocabNetwork::prime` is a row of
  `start_emb`).
  "Column" stays for the time axis of a `[batch, sequence]` or `[batch, K]`
  tensor ("the last column of `z̄`", "only column 0").
- **slot**: the chunk slots and the dump slots of the scatter in
  `chunk::plan`, and its local `slot_bs`. These are the **Slot** of the
  glossary: one entry of a buffer that has a fixed size
  (`[batch, K + sequence]`).
- **carry**: the start value `z̄₋₁` of the EMA of the smoothing ("the carry of
  the previous call", `Z = [carry, z̄₀, …]`). Also the `carry_bd` parameters
  and locals (`smooth`, `smooth_step`, `HNet::forward_from`,
  `HNet::step_from`). It is the state that the recurrence carries from one
  call to the next, not the straight-through carry of the gradient horizon.
  Also the plain verb ("carry the state into the next window", `cache.rs`).
- **chunk**: the chunks of H-Net, and the chunked scan of the smoothing
  ("chunked like SSD"). Also the chunks of the SSD scan of a block ("the
  start of a chunk of a chunked SSD").
- **graph**: the autodiff graph (`HNetCaches::detach`).
- **network**: the vocab network (`HNetVocabNetwork`), and "the other
  networks of this crate" (`shape.rs`). Also "*network normalization*": the
  paper gives this name to the final RMSNorm of each stack.
- **step**: each use names the step call or a captured step. The verb ("step
  the inner H-Net") means "run the step call on it".

### Identifiers that conflict with a glossary term

| Identifier | Where | Conflict |
|---|---|---|
| `boundary_bs` | `Routing`, `chunk::plan` | bare "boundary" (**chunk boundary**) |
| local `boundaries` | `step_unrolled`, `forward_equals_step_unrolled` (`tests.rs`) | bare "boundary" |
| `gather_rows` | `chunk.rs` | it gathers positions (**position**). The `gather_rows` of `cache.rs` gathers rows of a batch, and its name is correct. |
| `after_rows`, local `row_bd` | `RouterCache::after_rows` | "row" for a position |
| `smooth_block`, parameter `block` | `HNet`, `HNetShape`, `smooth` | a scan block of the smoothing, not a **block**. The prose says "scan block". |
| `n_start_tokens` | `HNetVocabShape` | not a glossary conflict. The prose says "start embedding", after `start_emb` (see below). |
| `chunk_plan_points_at_the_boundary_rows` | `tests.rs` | bare "boundary", "row" for a position |
| `only_a_new_sequence_forces_its_first_boundary` | `tests.rs` | bare "boundary" |
| `a_call_without_boundaries_skips_the_inner_network` | `tests.rs` | bare "boundary", "inner network" (**inner H-Net**) |
| `forward_is_chunkable_through_the_cache` | `tests.rs` | "chunk" for a window. Ticket 02 renamed its twin in `reference/tests.rs` to `block_forward_splits_into_windows_through_the_cache`. |
| `padded_slots_match_each_slot_alone` | `tests.rs` | "slot" for a row |
| `start_rows_forward_equals_prime_then_steps` | `tests.rs` | "row" for a position |

For a later rename:

- `smooth_block` and `n_start_tokens` are serialized config keys (the
  `Config` derive of `HNetShape` and `HNetVocabShape`). A rename stops the
  load of a saved config, as `init_bias_offset` did (ticket 02).
- burn-mamba uses `boundary_bs` (`examples/hnet-stories/inference.rs:199`)
  and `n_start_tokens` (`examples/hnet-stories/model.rs:124`, and the
  generated configs under `examples/hnet-stories/tmp/`). burn-deltanet uses
  none of these identifiers.

### Choices that this sweep made

- The prose calls the learnable vectors that open a sequence of
  `HNetVocabNetwork` **start embeddings**. The old prose called them "start
  rows". The new name follows the field `start_emb` and the rename
  `splice_class_embeddings` of ticket 02. The glossary has no term for them.
  Open for the user: does the glossary need an entry for **start embedding**
  (and for **scan block**)? A start embedding is close to a **class marker**,
  but `HNet` refuses class markers.
- "The chunking layer" and "the dechunking layer" (the names of the paper)
  are now "the chunking" and "the dechunking". In the glossary, a **layer** is
  a Pre-LN residual around one block.
- The blocks of the smoothing scan are now "scan blocks". Bare "block" names
  only the block of a family (ticket 02).
- "Every network ends in an RMSNorm" now names the stacks: "Each encoder,
  each decoder and the main stack end in an RMSNorm". A glossary **network**
  has input and output maps, and an encoder has none.
- "A `forward` does not replay as a captured graph" is now "a captured step
  cannot contain a `forward`".
- "The token boundary of `VocabNetwork`" is now "the token input and output
  maps", as ticket 02 did for "I/O boundary".
- "The single-row step" is now "the step call, on one token of each row"
  (the glossary **Step**).
- The prose says "the rows whose token starts a chunk" for the rows that
  `step` gives to the inner H-Net. A row is a sequence. Only its token starts
  a chunk.
- "A `forward` split into two calls" (the test of the windows) is now "split
  into two windows" (the glossary **Window**).
- These messages changed. No test and no consumer matches their text (no
  `should_panic` and no `contains` check on them):
  - `chunk::plan`: "an H-Net call needs at least one position".
  - `Routing::ratio_loss`: "the target chunk length must be above 1 position,
    got {target}".
  - `smooth`: "the scan block of the smoothing must hold at least 1 chunk".
  - `init_stack` (`shape.rs`): "… they would move the positions that the
    chunks point at". Its `what` for the main stack is now "main stack".
  - `tests.rs`: "row {b}: …" (three asserts), "the absent chunks hold the
    last real chunk", "the network has start embeddings".

### Docs that did not match the code

The sweep fixed two docs to match the code:

- The header of `shape.rs` said "Each of the three stacks of a stage has its
  own `NetworkShape`". `HNetStageShape` has two stacks: the encoder and the
  decoder. The header now says "Each layer stack (the encoder and the decoder
  of each stage, and the main stack) has its own `NetworkShape`".
- `Routing::absent` said "A stage that no row reached in a `step`".
  `HNet::forward_from` calls it too, when no position of the outer stage
  starts a chunk. The doc now says "in a call". The doc of `Routing::pad_bs`
  had the same gap. It now names both calls.

### Stale lines in the consumers

burn-mamba still says "main network" for the main stack of an H-Net, and
"start rows" for the start embeddings. The sites are in `src/unified/hnet.rs`
(with the panic messages at `345` and `362`), `src/unified/mod.rs`,
`src/unified/tests/hnet.rs`, `CLAUDE.md` and `examples/hnet-stories/`. This
list is not complete. To find all of them, run
`rg -i 'main network|inner network|start rows?'` in burn-mamba. A later
session in burn-mamba fixes them.
