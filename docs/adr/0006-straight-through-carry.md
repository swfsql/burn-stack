# A gradient horizon re-attaches what enters a cut, straight-through

A stack reads its input once, at the bottom. A gradient horizon that cuts the
stack would also cut the only gradient path of that input. Then the embedding
or the `in_proj` of the network does not train, and no error shows it. So at
each cut, the stack adds back what enters the untracked span as a
straight-through carry. It has value zero and the gradient of the carry path
of the residual scheme, so the values do not change. Under
`Residuals::Standard`, the carry path is the skip, and the straight-through
carry is the exact leading term of the gradient of the span.

## Considered Options

- **Feed the input again at every recursion, as TRM and HRM do.** This fits a
  task whose input has a limited length, such as the puzzles of TRM or MNIST
  digits. This crate also serves sequences with no length limit. A gradient
  horizon must not impose that architecture.
- **Accept an input that does not train.**

## Consequences

- A TRM-style or HRM-style model is still possible by hand. It uses no
  gradient horizon. It adds its own detaches and inner-backend regions
  (`Module::valid`, `modules::lift`), and feeds its input again where it
  needs it.
- The gradient horizon does not copy the input across the depth of a stack.
  That is the job of the residual scheme (Multi-Gate Residuals).
- Each residual scheme states how its residual streams take the
  straight-through carry.
- Under `MultiGate`, each stream that enters the span takes its own
  straight-through carry. It follows the gain `1 − βᵢ` of the carry path of
  its stream. The token gets each straight-through carry times the aggregator
  weight `αᵢ` of its stream. A stream that the span appends gets none. The
  rule holds the scores of the gates and of the aggregator constant across
  the span. So it is exact only when the transforms of the span are zero and
  the scores do not depend on the streams.
- Each class marker trains on both sides of a cut. The class tokens of a
  network and the class latents of a stack enter below the first layer. So
  they ride the straight-through carry.
