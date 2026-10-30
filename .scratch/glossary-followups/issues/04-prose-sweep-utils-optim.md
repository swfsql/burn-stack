# 04: Prose sweep: utils and optim

Status: ready-for-agent

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

- [ ] For each _Avoid_ term, `rg` over the scope finds only the uses that the
      tables of ticket 02 permit. Each other hit that you keep is listed with
      its reason under `## Comments`.
- [ ] `cargo check` and `cargo test --lib -- --test-threads=1` pass (flex).
- [ ] `cargo doc --no-deps` has no warnings.

## Comments
