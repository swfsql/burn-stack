# 05: Container tests move to burn-stack

Status: needs-triage

Blocked by: `../../block-family-choice/issues/03-reset-align-and-checked-packed.md`

From the test overview of 2026-11-03: burn-stack has 90 lib tests and burn-mamba has
360. The user decided that these tests move. Do it before
`02-residuals-transition.md`, so that a change to `Layers` meets its tests in this
repo.

**Problem:** ADR-0005 says that the tests of a container use `RefBlock`. But 110
tests of burn-mamba are in `src/unified/tests/`. Most of them test a container
mechanism with one family as a stand-in. burn-stack has 35 container tests of its own
(`reference::tests`, `modules::hnet::tests`). So a regression in `Layers` can pass
here and show only when burn-mamba moves its pin.

| File (burn-mamba) | Tests | What it tests |
|---|---:|---|
| `layers.rs` | 20 | The gradient horizon, with Mamba-3 as a stand-in |
| `multi_gate.rs` | 19 | `MultiGateResidual` (some with no block), and `Layers` with Mamba-2 |
| `class.rs` | 33 | Class markers, mostly with Mamba-2. A few run per family |
| `layer.rs` | 5 | `Layer` with an MLP, with Mamba-3 |
| `bidi.rs` | 2 | `BidiLayers`, with Mamba-2 |
| `optim.rs` | 14 | `Segmented` and `MuonPlan`, and the plan of each family |
| `untied.rs`, `capture.rs`, `packed.rs`, `hnet.rs` | 17 | What each family owes to a container |

**Direction:**

- A test of a container mechanism moves to `src/reference/tests.rs`, over `RefBlock`.
  Skip a test that repeats an existing one.
- A test that reads block internals needs another way to reach its state. For
  example, `layers.rs` has 28 lines that set fields of the block for an identity
  prefix. `RefBlock` can grow for this (ADR-0005).
- burn-mamba keeps what a family owes. This is one generic contract suite, run once
  for each family, plus `untied`, `capture`, `packed`, `hnet` and the plan of each
  family. A consumer cannot run such a suite yet: `reference.rs` has no public
  contract function, and `reference/tests.rs` is private. Add the suite under the
  `test-helpers` feature.

**Keep the catch:** A real block can break a contract in a way that `RefBlock` cannot.
So the contract suite in burn-mamba stays, and it must run under a gradient horizon,
class markers and multi-gate residuals.

**Docs:** `../burn-mamba/docs/agents/testing.md:10` presents these tests as a seam of
burn-mamba. Update that row in this work, and name the contract suite there.

**Not blocked by this:** The `block-family-choice` tickets do not need it. `layers.rs`
and `multi_gate.rs` use `LayersBuilder` with family configs, and these do not change.

Split this ticket by repo at triage.
