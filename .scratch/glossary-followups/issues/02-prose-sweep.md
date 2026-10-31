# 02: Prose sweep: the core containers, README and CLAUDE.md

Status: resolved

Blocked by: 01

**What to build:** The prose of the core containers uses the terms of
`GLOSSARY.md`, and no term that it lists under _Avoid_. A reader finds one name
for each concept. Tickets 03, 04 and 05 do the same for other parts of the
crate, with the term map and the method of this ticket.

## Scope

- Doc comments and comments in the modules directory, except H-Net (ticket
  03). These are:
  - the layer, the layer stack, the networks and the shapes,
  - the bidirectional stacks, MGR and the MLP,
  - the caches, the model config and the block traits,
  - the norm, activation, loss and misc modules.
- The reference block and the crate root.
- README.md, and the CLAUDE.md that ticket 01 wrote.
- Panic, assert and error messages. Before you change one, find the tests that
  match its text (`should_panic(expected = …)`), and keep the matched part or
  update the test.
- Identifiers stay. List each identifier that conflicts with a glossary term
  under `## Comments` for a later decision.

## Term map

`GLOSSARY.md` is the term map: for each _Avoid_ term, use the term of its
entry. The table lists the cases that need a judgment.

From the burn-mamba grill:

| Avoid | Use |
|---|---|
| "slot" for one sequence of a batch ("every slot of a batch", "per batch slot") | **row** |
| "slot" for a batch length of the length-bucketed loader | **bucket** |
| "padded row", "real row", "pad row" (for a time position) | **padded position**, **real token**, **position** |
| "restart" as a synonym of reset | **reset** |
| "document", "story", "episode" for a sequence in a packed row | **segment** (in the tiny-stories dataset, "story" stays for the data) |
| "special token", "CLS", "register token", and a loose mix of the three class words | **class marker** (the general term), **class token** (a network inserts it, raw input space), **class latent** (a layer container inserts it, `d_model` space) |
| "weight set" | **real layer** |
| "mixer", "mixer core", "SSM core", "core block" | **block**. "A particular mixer" becomes "a particular family". |
| "variant", "version", "generation" for an implementation of a block | **family** |
| bare "step" for one recurrence step or one training step | "step" is only the step call. Write "training step" for the optimizer. |
| "KV cache", "memory" for what a block keeps between calls | **cache** |

From the burn-stack grill:

| Avoid | Use |
|---|---|
| "segment" for a part of a fused projection, or for a run of untracked layers | **sub-projection**, **untracked span** |
| "fused weight", "column block" | **fused projection**, **sub-projection** |
| "chunk" for the part of a sequence that one forward takes ("`forward` chunks", the prefill chunks) | **window**. "Chunk" stays for a chunk of a chunked scan and for an H-Net chunk. |
| bare "carry" for the gradient horizon | **straight-through carry**. "Carry-biased" stays for the MGR gates (the Highway Networks term). "Carry" stays as a plain verb ("carry the cache into the next window"). |
| "row" for a place on the time axis, "ghost row" | **position**, **ghost latent**. "Row" stays for one sequence of a batch, and for a row of a matrix. |
| "main network", "inner network", "encoder network", "decoder network", "level" in H-Net | **main stack**, **inner H-Net**, **encoder**, **decoder**, **stage**. "Level" stays for a marker level. |
| bare "schedule" | **layer schedule**, or "LR schedule". In prose, `Lr::Sequence` is "a chain of LR schedules", and one `LrSegment` is "one schedule of the chain". |
| bare "boundary" | **chunk boundary** (H-Net), **cut** (gradient horizon), **I/O weight** (Muon), "at its input" (class markers), **reset** (packed rows) |
| "truncated BPTT" for the gradient horizon | **gradient horizon**. "BPTT" stays for the training window of the examples. |
| "lane", "depth-stream" | **residual stream** |
| "forward" or "backward" for a direction of a bidirectional pair | **straight pass**, **reversed pass** |
| bare "graph" for a capture | **captured step**, **replay**. Bare "graph" stays for the autodiff graph. |
| "unshared parameter", "per-layer parameter" | **untied parameter** |

## Method

Use the Edit tool, site by site. Each site needs a judgment, so no scripted
replacement. Write in the style of the `asd-ste100` skill.

## Acceptance criteria

- [x] For each _Avoid_ term, `rg` over the scope finds only the uses that the
      tables permit. Each other hit that you keep is listed with its reason
      under `## Comments`.
- [x] `cargo check` and `cargo test --lib -- --test-threads=1` pass (flex).
- [x] `cargo doc --no-deps` has no warnings.

## Comments

### Scope

The scope also has `src/reference/tests.rs`. It is the contract suite of the
reference block, and no other ticket covers it.

### Hits that stay

`rg` over the scope still finds these uses. The tables permit each one, or it
has a different meaning from the _Avoid_ entry:

- **slot**: one entry of a cache stack (`CacheStack::into_slots`,
  `from_slots`, `slot_count`, the `slots` locals, "cache slot"), and the
  opening slots of a packed row. Both are the **Slot** of the glossary: one
  entry of a buffer that has a fixed size.
- **step**: each prose use names the step call, a captured step or a
  training step. Two other uses stay:
  - "Muon must step one copy at a time" (`reference.rs`): the optimizer verb,
    as in the glossary entry of the Muon plan.
  - "a discretisation step size `Δ`" (`activation/softplus.rs`): the
    step size of a discretised recurrence.
- **carry**: a plain verb, and the MGR gates ("carry-biased", "towards
  *carry*", "carry budget": the Highway Networks term). Also the locals
  `carried`, `carry`, `carry_active` (`layers.rs`) and `carry`
  (`misc/prefix_sum.rs`).
- **chunk**: a chunk of a chunked scan ("chunkwise", "the start of a chunk",
  `chunk_len`, the chunk selector of `Block::Options`). Also the chunking of
  H-Net, and the `chunk`/`chunks` methods of tensors and slices.
- **segment**: a segment of a packed row, and the segment sum of `segsum`
  (the usual name of that op). Also the `Segmented`/`segmented.rs`
  identifiers.
- **row**: one sequence of a batch, or a row of a matrix. Examples of the
  second: an embedding row, the rows of the RMSNorm and MSE tests,
  "row-major", and the `[batch, width]` rows that `prime` sizes.
- **level**: a marker level ("network-level", "stack-level", "at either
  level"), and "high-level" in CLAUDE.md.
- **graph**: the autodiff graph, and the module path `utils::graph`.
- **memory**: the memory of the device ("peak memory", "runs out of
  memory"), and the auto-memory in CLAUDE.md.
- **variant**: the variants of a Rust enum (`cache.rs`, the
  `large_enum_variant` lint). "generation" is text generation, and
  "conversion" is a false positive of the pattern.
- **truncated BPTT**: `CacheStack::detach` (`cache.rs`). It is the BPTT from
  one window to the next, not the gradient horizon.
- **forward/backward**: the passes of autodiff, "feed-forward", and the
  `forward` methods. No hit names a direction of a bidirectional pair.
- **unshared**: buffers in `cache.rs`, not parameters.
- **schedule**: the `Schedule` and `BidiSchedule` types, the `schedule`
  locals and code spans, and "LR schedules".
- **document**: a verb, and markdown documents in CLAUDE.md.

### Identifiers that conflict with a glossary term

No later ticket renames identifiers (tickets 03 to 05 only list them). So
this ticket renamed these:

| Old name | New name | Where |
|---|---|---|
| `Padding::in_slot_order` | `in_row_order` | `utils/padding.rs` and its callers |
| `splice_class_rows` | `splice_class_embeddings` | `utils/class/mod.rs` and its callers |
| `carried`, `next_carried` | `stream_sets`, `next_stream_sets` | `Layers::cascade` |
| `block_forward_is_chunkable_through_the_cache` | `block_forward_splits_into_windows_through_the_cache` | `reference/tests.rs` |
| `an_untied_stack_is_the_unshared_stack_of_its_application_views`, local `unshared` | `an_untied_stack_equals_the_stack_of_its_application_views`, `view_stack` | `reference/tests.rs` |
| `step_matches_the_matching_forward_row`, local `row` | `step_matches_the_matching_forward_position`, `x_t` | `mlp/tests.rs` |
| locals `step`, `half_step` | `machine_eps`, `half_spacing` | the tests of the losses and the norms |

The renamed functions have no caller in burn-mamba or burn-deltanet. The
docs of the two renamed functions now use the glossary terms. The rest of the
prose of `utils/` stays for ticket 04.

One identifier stays for a decision of the user:

- `init_bias_step` (`MultiGateResidualConfig`, `ResidualsConfig::MultiGate`):
  "step" for the offset between the gate biases of two streams. The prose
  says "offset". The field is part of the serialized config, burn-mamba and
  burn-deltanet set it, and their saved config generators write it as a key.

burn-mamba has a copy of the untied-stack test with the old names
(`src/unified/tests/untied.rs`: `assert_untied_stack_is_its_unshared_views`,
the local `unshared`). A later session in burn-mamba renames it.

### Choices that this sweep made

- The old docs called the per-stream sigmoid gate of MGR the "mixer". But
  "mixer" is an _Avoid_ term for a block. So the prose now calls it the
  **β gate**, after `w_beta` and `b_beta`. `GLOSSARY.md` has the term under
  Residuals, with "mixer" under _Avoid_.
- The glossary names only the end of an untracked span (**cut**). The prose
  now says "backend hop" for both the entry and the cut, where it counts both
  (`reference/tests.rs`, `Layers::cascade`). The entry has no glossary term.
- The test docs said "within two steps of the dtype". They now say "two
  machine epsilons of the dtype". "ε" stays for the `div_eps` of the norms.
- "I/O boundary" of a network is now its "input and output maps", the words
  of the glossary entry **Network**. "Boundary weights" (Muon) are now
  **I/O weights**.
- The `src/lib.rs` heading "Three execution modes" is now "Three calls", the
  words of CLAUDE.md and of the glossary. No link points to its anchor.
- "Block" alone now names only the block of a family. "Sub-block" names one
  of the two Pre-LN parts of a layer: the block, or the feed-forward. So the
  feed-forward is no longer a "block" (`mlp.rs`, `norm/rms_norm.rs`, the
  `Layer` docs).

### A doc that does not match its test

`class_latents_receive_gradients` (`reference/tests.rs`) had the doc "A
per-layer class latent is a learnable input". The test sets
`LayersBuilder::class_latents`, which are the class latents of the stack, not
of a layer. The user chose the test as the intent. The doc now says "A class
latent of the stack".
