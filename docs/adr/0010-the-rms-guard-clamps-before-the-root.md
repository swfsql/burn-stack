# The guard of `rms` clamps `mean(x²)` at `ε`, before the root

`rms` computes `√(max(mean(x²), ε))`, with `ε` the `div_eps` of the dtype that
it computes in. The guard acts only where `mean(x²) < ε`. There, the division
by `rms` and the backward of the root would break or grow without bound. Each
other row gets the formula `√mean(x²)`, with no `ε` in it.

## Considered Options

- **An additive `ε` inside the root, `√(mean(x²) + ε)`.** This is the usual
  form of RMSNorm. It also guards the zero row and caps the gain. But the `ε`
  composes into every row. The output of a row at `mean(x²) = ε` is 29% too
  small, and that of a row at `10ε` about 5% too small.
- **No `ε`, and a rescale of each row by its largest value.** Then `x²`
  cannot overflow or underflow, and each nonzero row gets the exact formula.
  But the gradient is `h/rms`, with no bound. An f16 gradient overflows when
  `rms` is below about `1.5·10⁻⁵` (at `|h| = 1`).

## Consequences

- A row with `mean(x²) < ε` maps to `x/√ε`. A zero row has the output 0 and
  the gradient `h·γ/√ε`. The gain of each row is at most about `1/√ε`
  (`≈ 3500` in f32).
- The clamp is before the root, so the backward of the root never sees 0.
  Below `ε`, the clamp gives no gradient to `mean(x²)`. The routing of H-Net
  (`cos_distance`) clamps the same way.
- The sum of `x²` can overflow at the top of the range of the dtype: in f32,
  near a scale of `1.8·10¹⁹/√width`. The crate accepts this.
- `RmsNorm`, `RmsNormGated`, `rms_denom` and `normed_score` share `rms`.
  `src/modules/norm/tests.rs` checks each one at each scale of a finite
  input below that overflow.
