# 01: The losses on probabilities floor each probability at `ε`, at the log

Status: needs-info

Blocked by: none

**Context:** ADR-0010 moves the `ε` of `rms` to the point where the root
breaks. The user extends the rule to each `ε` of the code of the crate. An
`ε` acts only at its breakage point, and it never composes into another
value. On probabilities, BCE and cross-entropy add `ε` inside the log
(`log(p + ε)`, and `log(1 − p + ε)` for the second term of BCE). So the `ε`
shifts each probability. An exact match of a hard target gives a loss of
`−1.19·10⁻⁷` in f32, not 0, because `1 + ε` rounds up to `1 + 2⁻²³`.

**What to build:** on probabilities, each log of BCE and cross-entropy is
`log(max(q, ε))`. Here `q` is the probability of the term: `p`, or `1 − p`
for the second term of BCE. `ε` stays the `div_eps` of the dtype that the
loss computes in.

- For each `q ≥ ε`, the term is exactly `−log q`, with no `ε` in it. So an
  exact match of a hard target (`p = 1` on a target 1, `p = 0` on a target 0)
  gives a loss of exactly 0.
- For `q < ε`, the term is `−log ε`. The forward and the backward stay
  finite.
- The gradient of a term with `q < ε` waits for the decision of the user (see
  the open question).
- A new ADR states the rule for the code of the crate: an `ε` acts only at
  its breakage point. ADR-0010 is its first case. The ADR lists the AdamW and
  the Muon of Burn as exceptions. Their `ε` is inside a formula of Burn, and
  in AdamW it also damps the update of a parameter with small gradients.
- The doc of `div_eps` names the floor at a breakage point, not
  `x / (y + eps)`.
- The docs of the two losses describe the floor. The two
  `a_small_probability_keeps_its_loss` tests take the floor in their
  reference.

**Open question:** the gradient of a term with `q < ε`. The user decides it
before the implementation.

- Capped: `−1/ε`, with a straight-through floor. A confidently wrong class
  keeps a learning signal. The additive `ε` gives the same gradient at
  `p = 0`.
- Zero: a plain clamp gives no gradient below `ε`. Then the loss gives no
  signal for a class with `p < ε` (`8.2·10⁻⁸` in f32).

**Out of scope:** a loss that is 0 at an exact match of a soft target. The
user has a different ticket in mind for it.

**Seam:** "One module alone" of `docs/agents/testing.md`: the sibling tests of
the two losses.

**References:**

- `docs/adr/0010-the-rms-guard-clamps-before-the-root.md`, and its ticket
  `.scratch/rms-guard/issues/01-clamp-before-the-root.md`. The probe of the
  norms (`src/modules/norm/tests.rs`) is the model of the tests.
- The comments in the two losses on the floor of the `1/x` backward.

- [ ] The tests check both losses on probabilities at each scale of `q`:
      zero, the subnormals, both sides of `ε`, and up to 1. For BCE, `1 − p`
      also goes to zero. The reference is the formula in f64.
- [ ] An exact match of a hard target gives a loss of exactly 0, in each
      dtype.
- [ ] Each planted break fails a test: the additive `ε`, no guard, and the
      other choice of the gradient below `ε`. The **Mutation result** under
      `## Comments` gives each break and the tests that it fails.
- [ ] The docs of the losses, the doc of `div_eps` and the new ADR describe
      the floor. No doc of the crate gives `log(p + ε)` for these losses.
- [ ] `./scripts/check.sh` passes. Run it in the background.
- [ ] The loss tests pass under `dev-f16`.

## Comments
