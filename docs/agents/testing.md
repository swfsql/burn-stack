# Test seams

`/to-spec` and `/tdd` agree the test seams of a change with the user. Propose
seams from this table first. Propose a new seam only when no row fits the
change.

| Seam | Tests | Catches | Misses |
|---|---|---|---|
| Containers with `RefBlock` (ADR-0005) | `src/reference/tests.rs` | A container that breaks a contract: forward/step parity, windows through the cache, virtual layers, gradient horizon | A failure that only a real family shows |
| Containers with real blocks | `../burn-mamba/src/unified/tests/` | A change that breaks a real family, for example a change to the interface of a container | The cause: a failure can come from the block or from the container |
| The Muon plan | `src/reference/tests.rs`, `src/modules/hnet/tests.rs`, and with real blocks `../burn-mamba/src/unified/tests/optim.rs`. `src/optim/` has no tests of its own. | An error in the allowlist or in the update of a sub-projection | Whether the plan trains well |
| One module alone | The sibling `tests.rs` of the module, for example in `src/modules/hnet/`, `src/modules/loss/`, `src/modules/norm/` or `src/utils/scheduler/` | An error inside that module | An error in how the containers use it |

A burn-mamba test builds against the pin of burn-stack. To run it against a
local change, enable the `[patch]` of burn-mamba. The "Changes in both repos"
section of `issue-tracker.md` gives the rules.

## Outside the seams

- A claim about training quality (for example, "this option learns faster") has
  no test seam. A run of an example in burn-mamba answers it.
- The tests run on `flex`. A failure that only another backend shows needs a run
  on that backend.
