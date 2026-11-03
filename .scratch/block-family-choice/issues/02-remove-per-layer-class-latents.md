# 02: Remove the per-layer class latents

Status: ready-for-agent

Blocked by: `01-remove-remote-backend.md`

Spec: `../burn-mamba/.scratch/block-family-choice/spec.md`. Run only one session
on the two repos.

**What to build:** No config sets a per-layer class latent. `Layer::init` leaves
them empty, and only tests assign the fields. The user confirmed that nothing
uses them. Remove them and what exists only for them:

- the class latent fields of `Layer`, and `Layer::insert_latents`,
- the ghost latents of the gradient horizon (`src/modules/layers.rs:116`),
- the `k` tracking of the loops, and `ClassCursors::per_layer`,
- `BidiLayerPair` and its re-export in `src/lib.rs`,
- `Layer::step` and `Layer::prime`, if a search of both repos finds no other
  caller.

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

**Done when:**

- `cargo check`, `cargo check --all-targets` and `cargo doc --no-deps` have no
  warnings.
- `cargo test --lib -- --test-threads=1` passes. The only missing tests are those
  of the removed behaviour.

burn-mamba follows in `../burn-mamba/.scratch/block-family-choice/issues/01-follow-the-removals.md`.
