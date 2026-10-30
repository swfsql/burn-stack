# A gradient horizon re-attaches what enters a cut, straight-through

A stack reads its input once, at the bottom. A gradient horizon that cuts the
stack would also cut the only gradient path of that input. Then the embedding
or the `in_proj` of the network does not train, and no error shows it. So at
each cut, the stack adds back what enters the untracked span as a
straight-through carry. The carry has value zero and an identity gradient, so
the values do not change. Under `Residuals::Standard`, it is the exact
leading term of the gradient of the span.

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
- Under `MultiGate`, every stream gets the carry. At the bottom of the stack,
  this is exact. A cut inside the stack adds the carry of the pooled token to
  each stream.
- Each class marker trains on both sides of a cut. A class latent inside an
  untracked span gets a ghost latent in the carry: a value-zero copy from the
  tracked table.
