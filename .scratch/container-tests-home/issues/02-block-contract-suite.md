# 02: The block contract suite

Status: resolved

Blocked by: None (can start immediately).

Spec: `../burn-mamba/.scratch/container-tests-home/spec.md`, section "The block contract
module (burn-stack)". "Block contract" and "Family crate" are in `GLOSSARY.md`.

**What to build:** a family crate can call one public function for each block contract,
on its own block. Each function passes on `RefBlock`, and it fails on a block that
breaks its contract.

**Prefactor first:** split the `RefBlock` tests into one file for each mechanism, with
no change of a test. The batch tickets add their tests to these files. The File Map line
of `reference.rs` names the new directory.

- A new top-level module `block_contract`, beside `reference`, with the same gate: the
  tests of this crate, or the `test-helpers` feature.
- One public function for each contract, generic over the block config. Its inputs are
  the config, the block options and an fp32 tolerance.
- The function passes the tolerance through `dtype_tol`. The check of a value that can
  be large uses `max_rel_diff`. A failure panics with a message that names the contract.
- A contract checks the bare block when it can. It puts the block in a container only
  when the property needs one.
- A contract that does not apply to a block returns early. The block gives the reason
  through the `Block` trait, for example a `reset_align` of `None`.
- The first contracts:
  - the forward equals the step unrolled,
  - a window split through the cache equals one window,
  - padded rows equal each row alone,
  - packed segments equal each segment alone,
  - a record round trip keeps the block,
  - the cache traversal moves every field of the cache,
  - the Muon projections name rank-2 weights that exist (under the `optim` feature).
- The module header says who proves what. burn-stack proves "block contracts ⇒ the
  containers work". Each family crate proves "my blocks ⇒ block contracts".

**Tests:**

- Each contract runs on `RefBlock` with a near-zero tolerance.
- A private test block breaks one contract at a time. For each contract, one test
  expects the panic of that contract on that block. `RefBlock` does not change for this.
- This ticket deletes no test. The batch tickets delete the `RefBlock` tests that repeat
  a contract.

**Docs:**

- ADR-0005 gets one consequence: "A family crate tests its own blocks with the block
  contracts, through `test-helpers`. It does not test the containers again."
- `docs/agents/testing.md` gets a "Block contracts" seam row.
- The File Map of `CLAUDE.md` gets a line for the module. The line of `test-helpers` in
  `CLAUDE.md` names it.

**Version:** the module adds public items only under `test-helpers`, so no version bump.

- [x] The `RefBlock` tests are in one file for each mechanism, and they all pass.
- [x] Each of the seven contracts is a public function with the inputs above.
- [x] Each contract passes on `RefBlock`.
- [x] Each contract has a test that expects its panic on the private test block.
- [x] A contract that does not apply to a block returns early, and a test shows it.
- [x] `cargo check` and `cargo doc --all --no-deps` give no warnings. `scripts/check.sh`
      passes.
- [x] ADR-0005, the seam row and the File Map name the module.

## Comments

### The implementation

- The `RefBlock` tests are in `src/reference/tests/`: `block`, `layers`,
  `grad_horizon`, `cache`, `multi_gate`, `bidi`, `class`, `untied`, `packed` and
  `muon`. The test bodies did not change. A compare of their sorted lines shows
  only the deleted banners, the header list and the `use`/`mod` lines.
- `src/block_contract.rs` holds the seven contracts. Each one runs its checks
  through `run_contract`, which adds ``block contract `<name>`:`` to each panic.
  So a panic inside the block also names the contract.
- The Muon contract builds a `Layers` of 2 virtual layers over 1 real layer,
  because a spec matches only under a block container. It reads the parameters
  through `optim::report::float_params`, the visit of `MuonPlan::describe`.
- `src/block_contract/tests.rs`: each contract on three `RefBlock` settings with
  a tolerance of `1e-6`, and the private `BrokenBlock`. A `Breach` selects the
  contract that it breaks. Each negative test expects the text of the check
  that its breach breaks. A test shows that the broken block with no breach
  keeps every contract.
- `scripts/check.sh`: 122 lib tests pass (103 before). The coverage compare
  with `ba1fa3f` lists no lost item.

### The review (`tmp/review-cth-02-*.md`)

- Fixed: the panics that did not name the contract, the record check of the
  step, the row select in the traversal contract, the advice on chunks in the
  module header, the name `ParamRow` (the glossary **Row** is a sequence), and
  the doc findings.
- Kept: the two tests of `src/reference/tests/block.rs` repeat contracts. The
  batch tickets delete them. The README names the module for a new family
  crate.
- Limits that the contracts keep, noted for burn-mamba ticket 02:
  - A window has 7 tokens. A family gives a chunked block a short chunk.
  - No contract starts from a cache. So the double-SSD pathway of Mamba-3,
    which its cache selects, stays out of reach.
  - The traversal contract reads outputs. A field that one more step does not
    change stays unseen.
  - The record contract adds noise only to float parameters.
