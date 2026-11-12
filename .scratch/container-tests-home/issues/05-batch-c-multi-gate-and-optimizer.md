# 05: Batch C, the tests of Multi-Gate Residuals and the optimizer

Status: ready-for-agent

Blocked by: `02-block-contract-suite.md`,
`../burn-mamba/.scratch/container-tests-home/issues/01-coverage-run-and-record.md`

Spec: `../burn-mamba/.scratch/container-tests-home/spec.md`, sections "The sort rule"
and "New homes in burn-stack". Run only one session on the two repos.

**What to build:** each test of this batch gets a home in burn-stack, or a block
contract. Then burn-mamba ticket 05 deletes the moved tests.

**Read:** these files of `src/family/tests/` in burn-mamba: `multi_gate.rs` (17 tests,
861 lines) and `optim.rs` (16 tests, 597 lines). Also read the matching `RefBlock`
tests.

**Steps:**

1. Under `## Comments`, write the sorted list first. It has one line for each test of
   the batch: the rule number, the new home, and the block settings that the test uses.
   When the sort rule does not place a test, stop and ask the user.
2. Rule 1: port each test to `RefBlock`, or to no block, in the `RefBlock` test file of
   its mechanism. Skip a test that repeats a test of burn-stack, and name the repeat in
   the list.
3. Rule 1, no block:
   - The 6 tests of `MultiGateResidual` alone go to the tests of the Multi-Gate module.
   - The 5 tests of `Segmented` alone go to the tests of the `Segmented` module. This
     gives the optimizer module its first tests. Update the "Muon plan" seam row of
     `docs/agents/testing.md`, because it says that the optimizer module has no tests.
4. Rule 2: add each new contract to `block_contract`, with its run on `RefBlock` and its
   negative test. The Muon plan of each family can match the Muon contract of ticket 02.
   Then mark it "replaced by contract" in the list.
5. Delete each `RefBlock` test that repeats a contract.

- [ ] The sorted list has one line for each of the 33 tests.
- [ ] Each test of rule 1 or 2 has its new home, or the name of the test that it repeats.
- [ ] The optimizer module has its own tests, and the seam row says so.
- [ ] The coverage compare of burn-stack against the base commit of this ticket lists no
      lost item.
- [ ] `cargo check`, `cargo doc --all --no-deps` and `scripts/check.sh` pass with no
      warnings.
