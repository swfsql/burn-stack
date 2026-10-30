# 03: Prose sweep: H-Net

Status: ready-for-agent

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

- [ ] For each _Avoid_ term, `rg` over the scope finds only the uses that the
      tables of ticket 02 permit. Each other hit that you keep is listed with
      its reason under `## Comments`.
- [ ] `cargo check` and `cargo test --lib -- --test-threads=1` pass (flex).
- [ ] `cargo doc --no-deps` has no warnings.

## Comments
