# 02: `Residuals` owns the transition, and the horizon hop has one owner

Status: needs-triage

Blocked by: `../../block-family-choice/issues/02-remove-per-layer-class-latents.md`, `05-container-tests-home.md`

From the architecture review of 2026-11-03. Line numbers are those of commit
`12429e1`. Low priority.

**Why it matters:** The user has a future idea, not yet researched, to add more
residual schemes. This change makes that one module.

**Files:** `src/modules/layers.rs:380-633,754-1043`, `src/modules/bidi.rs:300-302,422-461`,
`src/modules/layer.rs:123-124,192-202,294-370`, `src/modules/multi_gate.rs:121-126`.

**Problem:** From outside, `Layers` is deep. Inside, each mechanism exists twice.
`forward_rows` has 254 lines and `cascade` has 290, with 13 back-references such as
"as in `forward`". The Multi-Gate branch has 3 copies (`layers.rs:597-618`,
`:945-958`, `bidi.rs:440-459`). The stream seed has 2 copies. The horizon hop has 2
copies (`layers.rs:467-505`, `:818-861`). `Residuals` is a shallow enum whose callers
hold the behaviour, so one more scheme edits three loops, the seed and the count.
`6f24f0a` (one rule about `lift`) edited 61 lines.

**Direction:** `Residuals` owns one transition. A small private owner holds the
straight-through carry and makes the hops. The class splice stays in both loops,
because `step` interleaves tokens with cache updates. The interface of `Layers` does
not change.

**Holds:** ADR-0003, ADR-0006.

**Tests:** The transition gets point-wise tests with no block. Parity tests stay.

**Cost:** A suppressed residual moves `x` with no clone (`layers.rs:568-572`). A
clone keeps the input alive through the block call and raises the GPU peak.
