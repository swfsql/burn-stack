# The containers are tested against `RefBlock`, never a real family

`reference.rs` holds `RefBlock`, a minimal block (a gated EMA) with its cache
and its config. The tests of burn-stack compose it to check every container.
Its `block_forward` unrolls the same recurrence that `block_step` applies. So
the forward/step parity is exact by construction, and a failed test points at
the container, not at the block.

## Considered Options

- **Test with a real family (burn-mamba)**: burn-mamba depends on burn-stack,
  so the dependency is circular. Its parity holds only within a tolerance.
  The tests would also depend on the numerics of one family.

## Consequences

- When the interface of the containers grows, `RefBlock` grows with it. For
  example, it unties its decay and its gate map on request.
- The `test-helpers` feature exposes `RefBlock` to the tests of a consumer.
