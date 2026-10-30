# 05: Prose sweep: examples

Status: ready-for-agent

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

- [ ] For each _Avoid_ term, `rg` over the scope finds only the uses that the
      tables of ticket 02 permit. Each other hit that you keep is listed with
      its reason under `## Comments`.
- [ ] `cargo check --features examples-common` and
      `cargo test --lib --features examples-common -- --test-threads=1` pass
      (flex).
- [ ] `cargo doc --no-deps --features examples-common` has no warnings.

## Comments
