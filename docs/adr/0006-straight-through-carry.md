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
- **Keep the straight-through carry across a suppressed skip.** The input
  trains, but with the gradient of a stack that has the skip.

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
  weight `αᵢ` of its stream in the last pool of the span. A stream that the
  span appends gets none.

  The rule holds the scores of the gates and of the aggregator constant
  across the span. So it is exact only when the transforms of the span are
  zero and the scores do not depend on the streams. The same rule applies at
  a cut inside the stack, in a span in which the accumulation phase ends, and
  in `step` and `prime`. With one module for each virtual layer
  (`per_virtual_layer`), each join of the span reads the gate of its own
  virtual layer.
- A horizon that would leave a parameter with no gradient panics on the
  autodiff backend:
  - An untracked layer 0 under `ignore_first_residual`, or an untracked top
    layer under `ignore_last_residual`. The carry path across a suppressed
    skip has gain 0.
  - Under `MultiGate`, a module that pools in a tracked layer, but only
    directly below untracked layers. The pooled token that enters a span
    takes no straight-through carry, so the aggregator query `w_alpha` of
    the module would not train.

  Take `MultiGate` with no weight sharing, or with `per_virtual_layer`.
  There, an untracked span can start only at the bottom of the stack, or
  directly above a suppressed first skip. Under `Schedule::Stretched`, `Depth(1)`
  panics when a run other than the bottom one is longer than one layer.
- Each class marker trains on both sides of a cut. The class tokens of a
  network and the class latents of a stack enter below the first layer. So
  they ride the straight-through carry.
