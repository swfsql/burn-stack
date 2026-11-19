# Test seams

`/mattpocock-skills:to-spec` and `/mattpocock-skills:tdd` agree the test seams
of a change with the user. Propose seams from this table first. Propose a new
seam only when no row fits the change.

| Seam | Tests | Catches | Misses |
|---|---|---|---|
| Containers with `RefBlock` (ADR-0005) | `src/reference/tests/`, one file for each mechanism | A container that breaks a contract: forward/step parity, windows through the cache, virtual layers, gradient horizon, packed rows | A failure that only a real family shows |
| Block contracts | `src/block_contract/tests.rs`: each contract on `RefBlock`, and on a private block that breaks it. Each family crate runs each contract on each block setting that it ships. | A block that breaks a contract. A contract function that cannot fail. | A need of a container that no contract states. A coverage compare can show code that only a deleted family test ran. |
| The Muon plan | `src/optim/segmented/tests.rs`: the update of each sub-projection. `src/reference/tests/muon.rs`: the plans of the containers, the module optimizer and its report. The two Muon contracts in `src/block_contract/tests.rs`, and `src/modules/hnet/tests.rs`. | An error in the allowlist, in the plan of a container or in the update of a sub-projection. A spec of a block that selects a weight of the network. | Whether the plan trains well. Which projections a family gives to Muon: each family crate checks its own list. |
| One module alone | The sibling `tests.rs` of the module, for example in `src/modules/hnet/`, `src/modules/loss/`, `src/modules/norm/` or `src/utils/scheduler/` | An error inside that module | An error in how the containers use it |

A burn-mamba test builds against the pin of burn-stack. To run it against a
local change, enable the `[patch]` of burn-mamba. The "Changes in both repos"
section of `issue-tracker.md` gives the rules.

## A test that cannot fail first

A test fails first, before the change that it checks. A change of a signature
can need the new code before the test compiles, so the test passes at its
first run. Then break the new code once for each such test, and run the test.
It must fail. Undo each break. The **Mutation result** under `## Comments` of
the ticket gives each break and each test that it failed.

## Outside the seams

- A claim about training quality (for example, "this option learns faster") has
  no test seam. A run of an example in burn-mamba answers it.
- The tests run on `flex`. A failure that only another backend shows needs a run
  on that backend.

## Coverage gate

`scripts/check.sh` runs `scripts/coverage.py`. The script gives the line,
branch and condition coverage of `src/` by the lib tests. It does not count a
test file. Its header defines each item and the compare. A box without
`cargo-llvm-cov` or the nightly toolchain skips it with a note. When nightly has
no `llvm-tools-preview`, the run installs it. A family crate can run the same
script on itself.

The coverage is a gate, not a seam. It shows the code that the tests run, not
the behavior that they check.

A ticket with a coverage gate states `BASE`: a commit, or a report that
`--save` wrote. Save the report of a commit at the start of the ticket, in the
background:

```bash
python3 scripts/coverage.py --revision BASE --save tmp/<ticket>/coverage-base.json
```

The run builds `BASE` in the git worktree `target/coverage-base/`, so the edits
of the working tree do not change the report. Its first build takes some
minutes.

After the change, run
`./scripts/check.sh --coverage-base tmp/<ticket>/coverage-base.json`. Its
coverage step compares the working tree with the report. Copy the lost items
into the **Coverage result** of the ticket. Do not write another compare
script. When a ticket needs a compare that the script does not give, change the
script.
