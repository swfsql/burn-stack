# 03: The reset alignment, a checked `Packed`, and packed rows on `RefBlock`

Status: ready-for-agent

Blocked by: `02-remove-per-layer-class-latents.md`

Spec: `../burn-mamba/.scratch/block-family-choice/spec.md`. Run only one session
on the two repos.

**What to build:**

- **`Block::reset_align`**: it gives the reset alignment for given `Options`. Its
  default is `None`, which matches the panicking default of
  `block_forward_packed`. "Reset alignment" is in `GLOSSARY.md`.
- **`Packed::from_starts`**: it takes the alignment and asserts that each start is
  a multiple of it. This is one integer operation per segment, on host integers.
- **`Packed` stores `lead`**: `place` asserts that `lead` is equal to the row count
  of the table. It reads no device.
- **`pub(crate)` fields**: `reset_bs` and `latent_bs` become `pub(crate)`. Each
  reader is in this crate. A test that reads them stays in this crate.
- **`RefBlock`**: it gets a packed forward, and a `reset_align` option in
  `RefBlockConfig` (default 1). Its parity with each segment run alone stays
  exact.
- **`tiny_stories/dataset.rs:839`**: the caller passes the alignment.
- **`src/modules/layers.rs:1166-1167`**: this doc says that the surface of a
  consumer "is its own `Config` enum". Each consumer config now is a shape plus a
  block (ADR-0007), so rewrite it.
- **Version**: go to 0.38.0.

**Tests** in `src/reference/tests.rs`, through the new `RefBlock`:

- `Layers`, `Layer` and `VocabNetwork` `forward_packed` against each segment alone.
- `Packed::place` with class latents.
- A start that is not a multiple of the alignment panics.
- A `lead` that differs from the table panics.

**Done when:**

- `cargo check`, `cargo check --all-targets`, `cargo check --features
  examples-common` and `cargo doc --no-deps` have no warnings.
- `cargo test --lib -- --test-threads=1` passes.

burn-mamba follows in `../burn-mamba/.scratch/block-family-choice/issues/03-family-reset-alignments.md`.
