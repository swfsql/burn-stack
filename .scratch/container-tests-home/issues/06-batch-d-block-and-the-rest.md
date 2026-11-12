# 06: Batch D, the tests of `MambaBlock`, the untied copies, packed rows, the H-Net, the captured step, the configs and the reset alignment

Status: ready-for-agent

Blocked by: `02-block-contract-suite.md`,
`../burn-mamba/.scratch/container-tests-home/issues/01-coverage-run-and-record.md`

Spec: `../burn-mamba/.scratch/container-tests-home/spec.md`, sections "The sort rule"
and "New homes in burn-stack". Run only one session on the two repos.

**What to build:** each test of this batch gets a home in burn-stack or a block
contract, or it stays in burn-mamba (rule 3). Then burn-mamba ticket 06 deletes the
moved tests.

**Read:** these files of `src/family/tests/` in burn-mamba:

| File | Tests | Lines |
|---|---:|---:|
| `block.rs` | 16 | 617 |
| `untied.rs` | 5 | 218 |
| `packed.rs` | 4 | 236 |
| `hnet.rs` | 4 | 223 |
| `capture.rs` | 4 | 160 |
| `config.rs` | 5 | 167 |
| `reset_align.rs` | 5 | 82 |

Also read the matching `RefBlock` tests and the H-Net tests of burn-stack.

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

**First guesses (the sort decides):**

- `record_round_trip` and the record tests of `MambaBlock`: rule 2, or a repeat of the
  record contract of ticket 02.
- `mamba1_assign_moves_every_field` and `mamba2_assign_moves_every_field`: rule 2, or a
  repeat of the cache-traversal contract of ticket 02.
- The JSON round trips of the configs: rule 2. Each family crate needs its configs to
  survive JSON.
- The reset alignment of each Mamba path: rule 3. The chunk length is Mamba math.
- `pathway_mismatch_panics` and `mamba3_layout_follows_the_untie_list`: rule 3.

- [ ] The sorted list has one line for each of the 43 tests.
- [ ] Each test of rule 1 or 2 has its new home, or the name of the test that it repeats.
- [ ] The coverage compare of burn-stack against the base commit of this ticket lists no
      lost item.
- [ ] `cargo check`, `cargo doc --all --no-deps` and `scripts/check.sh` pass with no
      warnings.
