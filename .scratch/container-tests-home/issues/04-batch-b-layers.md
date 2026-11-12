# 04: Batch B, the tests of `Layers`, the gradient horizon, `Layer` and the bidirectional stacks

Status: ready-for-agent

Blocked by: `02-block-contract-suite.md`,
`../burn-mamba/.scratch/container-tests-home/issues/01-coverage-run-and-record.md`

Spec: `../burn-mamba/.scratch/container-tests-home/spec.md`, sections "The sort rule"
and "New homes in burn-stack". Run only one session on the two repos.

**What to build:** each test of this batch gets a home in burn-stack, or a block
contract. Then burn-mamba ticket 04 deletes the moved tests.

**Read:** these files of `src/family/tests/` in burn-mamba: `layers.rs` (18 tests, 1059
lines), `layer.rs` (5 tests, 183 lines) and `bidi.rs` (2 tests, 116 lines). Also read
the matching `RefBlock` tests.

**Steps:**

1. Under `## Comments`, write the sorted list first. It has one line for each test of
   the batch: the rule number, the new home, and the block settings that the test uses.
   When the sort rule does not place a test, stop and ask the user.
2. Rule 1: port each test to `RefBlock`, or to no block, in the `RefBlock` test file of
   its mechanism. Skip a test that repeats a test of burn-stack, and name the repeat in
   the list.
3. Rule 2: add each new contract to `block_contract`, with its run on `RefBlock` and its
   negative test.
4. Delete each `RefBlock` test that repeats a contract.

**Known cases:**

- The two identity-prefix tests zero `out_proj.weight`. `RefBlock` makes that field
  public, so `RefBlock` needs no growth for them.
- `grad_horizon_memory_probe` follows rule 1. It moves over a `RefBlock` that is large
  enough for the activations to set the peak memory. It stays `#[ignore]`. Its doc gives
  the command for each configuration.

- [ ] The sorted list has one line for each of the 25 tests.
- [ ] Each test of rule 1 or 2 has its new home, or the name of the test that it repeats.
- [ ] The memory probe runs over `RefBlock` with the command of its doc.
- [ ] The coverage compare of burn-stack against the base commit of this ticket lists no
      lost item.
- [ ] `cargo check`, `cargo doc --all --no-deps` and `scripts/check.sh` pass with no
      warnings.
