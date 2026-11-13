# Test seams

`/to-spec` and `/tdd` agree the test seams of a change with the user. Propose
seams from this table first. Propose a new seam only when no row fits the
change.

| Seam | Tests | Catches | Misses |
|---|---|---|---|
| Containers with `RefBlock` (ADR-0005) | `src/reference/tests/`, one file for each mechanism | A container that breaks a contract: forward/step parity, windows through the cache, virtual layers, gradient horizon, packed rows | A failure that only a real family shows |
| Block contracts | `src/block_contract/tests.rs`: each contract on `RefBlock`, and on a private block that breaks it. Each family crate runs each contract on each block setting that it ships. | A block that breaks a contract. A contract function that cannot fail. | A need of a container that no contract states. A coverage compare can show code that only a deleted family test ran. |
| Containers with real blocks | `../burn-mamba/src/family/tests/` | A change that breaks a real family, for example a change to the interface of a container | The cause: a failure can come from the block or from the container |
| The Muon plan | `src/reference/tests/muon.rs`, the Muon contract in `src/block_contract/tests.rs`, `src/modules/hnet/tests.rs`, and with real blocks `../burn-mamba/src/family/tests/optim.rs`. `src/optim/` has no tests of its own. | An error in the allowlist or in the update of a sub-projection | Whether the plan trains well |
| One module alone | The sibling `tests.rs` of the module, for example in `src/modules/hnet/`, `src/modules/loss/`, `src/modules/norm/` or `src/utils/scheduler/` | An error inside that module | An error in how the containers use it |

A burn-mamba test builds against the pin of burn-stack. To run it against a
local change, enable the `[patch]` of burn-mamba. The "Changes in both repos"
section of `issue-tracker.md` gives the rules.

## Outside the seams

- A claim about training quality (for example, "this option learns faster") has
  no test seam. A run of an example in burn-mamba answers it.
- The tests run on `flex`. A failure that only another backend shows needs a run
  on that backend.

## Coverage gate

`scripts/check.sh` runs `scripts/coverage.py`. The script gives the line,
branch and condition coverage of `src/` by the lib tests. It does not count a
test file. Its header defines each item and the compare. A box without
`cargo-llvm-cov` or the `llvm-tools-preview` of nightly skips it with a note. A
family crate can run the same script on itself.

The coverage is a gate, not a seam. It shows the code that the tests run, not
the behavior that they check.

A ticket with a coverage gate states `BASE`: a commit, or a report that
`--save` wrote. After the change, run `scripts/coverage.py --base BASE`. Copy
its lost items into the **Coverage result** of the ticket. Do not write
another compare script. When a ticket needs a compare that the script does not
give, change the script.

A commit base builds in the git worktree `target/coverage-base/`. Its first
build takes some minutes, so run it in the background.
