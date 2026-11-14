# An untracked region runs on the inner backend, not through `detach`

A gradient horizon runs some virtual layers with no gradient. Burn has no
ambient no-grad mode, and `detach` does not give one. A detached tensor stays
on autodiff, and each op after it still registers a step in the graph. The
graph keeps these steps until a backward reaches them. So burn-stack runs an
untracked region on the inner backend. The module goes through
`Module::valid`, the tensors that enter go through `Tensor::inner`, and the
tensors that leave go through `modules::lift`.

The difference is memory. `grad_horizon_memory_probe`
(`src/reference/tests/grad_horizon.rs`) runs a stack of 64 virtual layers
with a horizon of 2 layers. Its peak RSS is 545 MB with `detach` and 42 MB on
the inner backend. With no horizon, it is 762 MB. With no autodiff at all, it
is 17 MB.

## Considered Options

- **`detach`**: it cuts the gradients, but it keeps almost every activation.
- **A bare `Tensor::from_inner` to come back**: since Burn PR #5557,
  `from_inner` is `Tensor::autodiff`. It enables autodiff also on a plain
  tensor, and it always sets the `Disabled` checkpointing strategy. On a
  device with gradient checkpointing, the first op that mixes the result with
  the other tensors of the pass panics. `lift` sets the strategy of the
  device, and it does nothing off autodiff.

## Consequences

- A cache converts by hand. `Module::map` does not visit a plain `Tensor`
  field, and a cache holds only plain tensors.
- `Module::valid` also disables the training flags (dropout and others) in
  the untracked region.
- `is_require_grad` reports only the `Grad` leaves. So a test asserts that a
  gradient reaches a parameter. It probes whether a tensor is in the graph
  with `is_tracked` (Burn PR #5571).

See also: [Burn: devices, autodiff, gradients and
checkpointing](../burn/autodiff-semantics.md).
