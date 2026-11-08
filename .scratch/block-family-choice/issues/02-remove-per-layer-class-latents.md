# 02: Remove the per-layer class latents

Status: ready-for-agent

Blocked by: `01-remove-remote-backend.md`

Spec: `../burn-mamba/.scratch/block-family-choice/spec.md`. Run only one session
on the two repos.

**What to build:** No config sets a per-layer class latent. `Layer::init` leaves
them empty, and only tests assign the fields. The user confirmed that nothing
uses them. Remove the per-layer class latents.

All code that supports them or uses them must adapt: it loses that support, and
it becomes simpler where the removal allows it. The list below names the code
that likely needs this review, with the decision for each item.

- **Delete**, because each one exists only for the latents: the class latent
  fields of `Layer`, `Layer::insert_latents`, the ghost latents of the gradient
  horizon (`src/modules/layers.rs:116`), the `k` tracking of the loops, and
  `ClassCursors::per_layer`.
- `Layer::prime`: delete it. It only steps the latents of the layer, so it has
  no job left. Its only other caller is a burn-mamba test that
  `../burn-mamba/.scratch/block-family-choice/issues/01-follow-the-removals.md`
  deletes.
- `Layer::step`: after the removal, it is identical to `Layer::step_one`. Keep
  one method, `Layer::step`, with the body of `step_one`.
- `BidiLayerPair`: keep it, and keep its re-export in `src/lib.rs`. It loses its
  latent fields, its `insert_latents` and the `class` parameter of its `forward`.
- `BLOCK_CONTAINERS` (`src/optim/spec.rs`): do not change it.
- `insert_class_markers`: delete it. `Layer::insert_latents` is its only caller.
- `splice_class_embeddings`: make it rank 3 only. Its only rank-4 use goes with
  the per-layer latents.

The class latents of a stack and the class tokens of a network stay. They ride
the straight-through carry, so they need no ghost latent. Read
`src/modules/layers.rs:110-120` before you start.

Remove the tests of the removed behaviour, for example
`grad_horizon_ghosts_an_untracked_layers_class_latent`. Update each doc that names
a removed item: the module headers, `GLOSSARY.md`, and the last consequence of
ADR-0006. The doc at `layers.rs:110` says "all three marker levels", which becomes
two.

The class-latent docs must still say that each class marker trains on both sides
of a cut.

**Reference:** `git show 0f9f5bf` in this repo shows a tested removal that the
user discarded. Only the local repo holds it. Use it for the two loops of
`src/modules/layers.rs` and for `Layers::prime`. It deletes `BidiLayerPair`,
changes `BLOCK_CONTAINERS` and keeps `step_one`: do not copy those parts.

A plain search misses three docs:

- the "per-virtual-layer cursors" wording in the doc of `VocabNetwork::step`,
- the "lengthened input" comment in `src/modules/layers.rs`,
- the "Class markers" paragraph in `src/modules/multi_gate.rs`.

The baseline is 90 lib tests: 87 pass and 3 are ignored.

**Done when:**

- `cargo check`, `cargo check --all-targets` and `cargo doc --no-deps` have no
  warnings.
- `cargo test --lib -- --test-threads=1` passes. The only missing tests are those
  of the removed behaviour.

burn-mamba follows in `../burn-mamba/.scratch/block-family-choice/issues/01-follow-the-removals.md`.
