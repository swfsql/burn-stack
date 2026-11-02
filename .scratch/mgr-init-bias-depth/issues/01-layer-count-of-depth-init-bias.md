# 01: The layer count of `depth_init_bias`

Status: needs-triage

Priority: low

Blocked by: None (can start immediately)

**What to build:** A decision on the layer count that
`MultiGateResidualConfig::depth_init_bias` takes. Then the code or its doc follows
that decision. This can be a bug or only a doc gap. Nobody has decided yet.

## The mismatch

The formula is `−ln( √(L/21)·(e³+1) − n )`.

- **The source counts `L` in source layers.** `L=21` is the "actual lerping depth"
  at a block depth of 12 and `n=4` (`MGR:266`, `MGR:268`). Each Transformer block
  has two source layers. The "block" entry of `docs/source-terms.md` gives the
  count.
- **Our `L` counts our layers.** One `MultiGateResidual` runs for each layer of
  ours. `n_mixing_layers` is the stack depth minus `n−1` (the doc of
  `depth_init_bias`).
- **`L_BASE = 21` stays in source layers.** The code compares a count of our layers
  to it.

So a stack of 12 of our layers at `n=4` does not get the bias of a source model
with 12 Transformer blocks:

| `L` | Why this value | Bias |
|---|---|---|
| 9 | our count: 12 layers minus `n−1` | −2.28 |
| 12 | a port reads "12 layers" as `L` | −2.48 |
| 21 | the value of the source model | −2.84 |

## The two readings

1. **The code is correct.** `L` is the count of layers that lerp. Our stack has 9
   lerps, and the source model has 21. The law `σ(b) ∝ 1/√L` then gives our stack
   a larger bias. In this reading, only the doc needs a line. The author of this
   ticket prefers this reading, but no test or training run supports it.
2. **The code needs a change.** The reference value comes from "preliminary
   experiments" on the source model (`MGR:268`), which lerps at both parts of a
   block. That reference does not need to apply to a stack that lerps once for each
   layer. In this reading, the crate needs another reference, or the caller sets
   `init_bias` directly.

## Why the priority is low

- No caller exists in `src/` of this crate, or in `src/`, `examples/` and `tests/`
  of burn-mamba. The check is from the date of this ticket, so run it again first.
  Another session can change burn-mamba.
- `init_bias` is an input. A caller can set any value, and the default is `0`.

## Acceptance criteria

- [ ] `## Comments` gives the decision and the reason for it: reading 1, reading 2
      or another reading.
- [ ] The doc of `depth_init_bias` says which layers `L` counts, and how `L_BASE`
      relates to the source.
- [ ] A test pins the value of the formula at the chosen reference. No test of this
      function exists now.
- [ ] If the code changes, `docs/source-terms.md` still matches it (the "layer"
      entry and the "carry behavior" entry).
