# 04: Captured-step rules inside the capture module

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-03. Line numbers are those of commit
`12429e1`. Speculative until measured.

**Files:** `src/utils/capture.rs:33-44,79`, `src/examples/trainer.rs:117,156-158`,
`src/examples/tiny_stories/sample.rs:232`, `src/examples/device.rs:34-45`.

**Problem:** An example must know facts that `Trainer` appears to hide. `capture.rs`
states `WARMUP_STEPS`, and only `sample.rs` follows it. `Trainer` captures at the
first batch. The effect under `dev-autotune` is unknown, because nobody measured it.
A lost recording "runs eagerly, and no error shows it". `loader_device(` is on 24
lines in 11 files, and each use is a hand-added guard against that loss.

**Direction:** The capture module owns the warm-up decision. A lost recording reports
itself once. One place that builds a training dataloader owns the host-device rule.

**First:** Measure `dev-autotune`. A warm-up effect needs a GPU run by the user. The
report path can run on flex.
