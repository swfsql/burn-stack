# 14: One stack loop for a layer and for a bidirectional pair

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-16. Line numbers are those of commit
`b198fa5`. The review compiled nothing.

**Files:** `src/modules/bidi.rs:150-388,390-469`,
`src/modules/layers.rs:403-451,674-782`, `src/modules/shape.rs:284-351` (`BidiShape`),
`src/utils/schedule/` (`BidiSchedule`).

**Problem:** `BidiLayers` repeats the stack loop of `Layers`: the fields, the splice of
the class latents (`insert_latents`), the lookup of the layer schedule, the
applications and the loop of the residual join. A mechanism that `Layers` gets does
not reach the copy. `BidiLayers` has no gradient horizon, no feed-forward and no init
policy, and `BidiShape` is not a `NetworkShape`. `bidi_pair_forward` takes `norm` and
`block` out of a `Layer`, past `Layer::forward`.

**Direction:** The stack loop runs a unit. A layer is one adapter of that seam, and a
bidirectional pair is the other, so the seam is real. A layer uses one cache slot and
has a forward and a step. A pair uses two cache slots, has its merge, and has only a
forward. The class latents, the layer schedule, the residual join and the backend hop
then serve both.

**Against it:** Only `mnist-ae` of burn-mamba (work in progress) uses the
bidirectional stack. If that stack gets no new mechanism, the copy costs little.

**Not checked:** How `BidiSchedule` differs from `Schedule`, and whether the
straight-through carry is correct across a pair.
