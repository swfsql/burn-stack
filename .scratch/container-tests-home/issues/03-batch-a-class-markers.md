# 03: Batch A, the class-marker tests

Status: ready-for-agent

Blocked by: `02-block-contract-suite.md`,
`../burn-mamba/.scratch/container-tests-home/issues/01-coverage-run-and-record.md`

Spec: `../burn-mamba/.scratch/container-tests-home/spec.md`, sections "The sort rule"
and "New homes in burn-stack". Run only one session on the two repos.

**What to build:** each class-marker test of burn-mamba gets a home in burn-stack, or a
block contract. Then burn-mamba ticket 03 deletes the moved tests.

**Read:** `src/family/tests/class.rs` of burn-mamba (29 tests, 1471 lines), and the
class-marker tests of `RefBlock`.

**Steps:**

1. Under `## Comments`, write the sorted list first. It has one line for each test of
   the batch: the rule number, the new home, and the block settings that the test uses.
   When the sort rule does not place a test, stop and ask the user.
2. Rule 1: port each test to `RefBlock`, or to no block, in the `RefBlock` test file of
   its mechanism. Skip a test that repeats a test of burn-stack, and name the repeat in
   the list.
3. Rule 1, no block: 3 tests use no block (the class plans and the custom index). They
   go to the tests of the class module.
4. Rule 2: add each new contract to `block_contract`, with its run on `RefBlock` and its
   negative test.
5. Delete each `RefBlock` test that repeats a contract.

- [ ] The sorted list has one line for each of the 29 tests.
- [ ] Each test of rule 1 or 2 has its new home, or the name of the test that it repeats.
- [ ] The coverage compare of burn-stack against the base commit of this ticket lists no
      lost item.
- [ ] `cargo check`, `cargo doc --all --no-deps` and `scripts/check.sh` pass with no
      warnings.
