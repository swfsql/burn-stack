# 01: The guard of `rms` clamps before the root

Status: resolved

Blocked by: none

Decision: `docs/adr/0010-the-rms-guard-clamps-before-the-root.md`.

**Context:** ticket 02 of `.scratch/ste-sweep/` found docs of the norms that
disagree with the code. The user asked for tests of the guard of `rms` at its
breakage points. A probe found no NaN at zero, but an additive `ε` that
composes into every row. The user then selected a clamp before the root.

**What to build:** `rms` (`src/modules/norm/mod.rs`) computes
`√(max(mean(x²), ε))`, not `√(mean(x²) + ε)`. `ε` stays the `div_eps` of the
dtype that `rms` computes in.

- The tests check `RmsNorm`, `RmsNormGated` (both orders of the gate),
  `normed_score` and `rms_denom` at each scale of a finite input. The scales go
  from zero and the subnormals to the overflow of `x²`, on both sides of `√ε`.
- One input of each dtype holds rows of all these scales together. So a
  row that reads another row fails.
- The reference is the formula in f64, with the clamp.
- The docs of the norms describe the clamp. These docs include the docs that
  ticket 02 lists.

**Out of scope:** the overflow of `x²` at the top of the range. The user
accepts it.

**Seam:** "One module alone" of `docs/agents/testing.md`:
`src/modules/norm/tests.rs`.

- [x] Each planted break of `rms` fails a test. The **Mutation result** under
      `## Comments` gives each break and the tests that it fails.
- [x] The docs of the norms describe the clamp, and no doc gives the additive
      `ε`.
- [x] `./scripts/check.sh` passes. Run it in the background.
- [x] The norm tests pass under `dev-f16`.

## Comments

### The probe of the additive `ε`

On the additive `ε`, no scale gives a NaN. Zero and the subnormals give a
finite forward and gradient for each dtype. The probe also gives these
results:

- The additive `ε` changes every row. Against `√mean(x²)`, a row with
  `rms ≪ √ε` collapses towards 0. At scale `10⁻¹⁰` in f32, the output is
  `1.8·10⁻⁷`, and `√mean(x²)` gives `0.68`.
- The clamp gives the same `1.8·10⁻⁷`, because it maps such a row to `x/√ε`.
  The two forms differ near `ε`: the additive `ε` makes the output of a row
  at `mean(x²) = ε` 29% too small (ADR-0010). The user selects the clamp,
  because its `ε` does not compose into a row with `mean(x²) ≥ ε`.
- At the top of the range, the sum of `x²` overflows. The norms and the score
  then give 0, and `rms_denom` gives `inf`. This starts near `10²⁰` for f32
  and bf16 (`10¹⁸` at width 1024) and near `10¹⁵⁴` for f64. The user accepts
  it.

### Mutation result

Each break replaces the clamp in `rms`. The table gives the tests that fail on
it, on `flex`.

| Planted break | Tests that fail |
|---|---|
| The additive `ε`: `√(mean(x²) + ε)` | The four tests of `src/modules/norm/tests.rs` (near `√ε`, and in each input of mixed scales), and `an_f64_input_computes_in_f64` |
| No guard: `√mean(x²)` | The four tests of `src/modules/norm/tests.rs` (a NaN at zero and where `x²` underflows), and both `…_gradient_finite_on_collapsed_slice` tests |
| The clamp after the root: `max(√mean(x²), √ε)` | The four tests of `src/modules/norm/tests.rs` (a NaN gradient at zero and where `x²` underflows), and both `…_gradient_finite_on_collapsed_slice` tests |
| The mean over axis 0, not over the last axis | The four tests of `src/modules/norm/tests.rs` (`normed_score` and `rms_denom` on the length of the output, the two norms from `√ε` up and in each input of mixed scales), and `an_f64_input_computes_in_f64` |

The clamp after the root gives the same values as the clamp before it. Only
its gradient breaks, at a row where the root sees 0.

### For the pin move of burn-mamba

Two test docs of burn-mamba give the QK-norm of `state_rank = 1` as
`γ·x/√(x²+ε)`, with `|B| = 1` either way:
`src/mamba3/single_ssd/single_ssd/tests.rs:440` and
`src/mamba3/double_ssd/double_ssd/tests.rs:315`. They describe the `RmsNorm`
of the pinned burn-stack. With the clamp, `|B|` is `|x|/√ε` below `√ε`. The
derivative is `γ/√ε` below `√ε`, and 0 above it. The pin move updates both
docs, and it checks the gradient tolerance of these tests.

### Outside this ticket

These users of `div_eps` add it, so their `ε` composes into each value. The
rule of ADR-0010 covers only `rms`. The user decides whether they get a
ticket.

- `src/modules/loss/bce.rs` and `src/modules/loss/cross_entropy.rs` compute
  `log(p + ε)` on probabilities.
- `src/examples/training.rs:230` gives `div_eps` to Adam as its epsilon.
- The doc of `div_eps` (`src/utils/mod.rs:72`) names the form `x / (y + eps)`.
