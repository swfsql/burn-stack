# 16: The options of a block are a setting of the block

Status: needs-triage

Priority: low

Blocked by: None

From the architecture review of 2026-11-16. Line numbers are those of burn-stack
commit `b198fa5` and burn-mamba commit `7807d31b`. Speculative. The review compiled
nothing.

**Files:** `src/modules/mod.rs` (`Block::Options`, `reset_align`), the `options`
parameter of each forward in `src/modules/{layer,layers,network,bidi}.rs` and
`src/modules/hnet/`, `../burn-mamba/src/family/mod.rs:82-112` (`MambaSsdPath`),
`../burn-mamba/src/family/block.rs:76-132`.

**Problem:** Each forward of each container passes `options` to the next module, with
the bound `M::Options: Clone`. `step` and `prime` take none. In the examples, the
value is a constant of the run (`Run::ssd_path`). The run-time family choice adds the
enum `MambaSsdPath`, 5 functions that extract the path of one family, and a panic for
a path of the wrong family. One `Options` value also makes a mixed stack panic
(ADR-0009 of burn-mamba).

**Direction:** The block holds its options as a non-parameter field, as `Layers` holds
`grad_horizon`. `Block::Options`, the `options` parameters and `MambaSsdPath` go, and
`reset_align` takes no argument.

**Cost, and why this is speculative:**

- A caller needs a way to set the field on each block of a container. That is a new
  interface in each container.
- The agreement tests compare SSD paths on one set of weights. They must clone the
  block for each path.

**First:** Do `../burn-mamba/.scratch/arch-review-backlog/issues/18-pathway-option.md`,
which moves the pathway into the options. Then look at how callers use the options.

Split this ticket by repo at triage.
