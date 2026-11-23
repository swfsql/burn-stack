# 04: Sweep: utils, optim, the reference block and the block contracts

Status: ready-for-agent

Blocked by: 01, `../burn-mamba/.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../burn-mamba/.scratch/ste-sweep/spec.md`. Follow its method, its text
that does not change, its kept findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–04 are the batches of burn-stack, and they do not block each other. Ticket
05 sweeps `.scratch/`, `docs/` and the root files last.

**What to build:** the prose of the utils, the optim modules, the reference
block, the block contracts and the crate root follows the `asd-ste100` skill.

- Scope: `src/utils/`, `src/optim/`, the reference block and the block
  contracts with their test directories, and `src/lib.rs`, 42 files.
- At the start: 47 hard findings (9 in messages) and 73 advisory findings.
- burn-mamba calls the block contracts on each of its block settings. Search
  the tests of burn-mamba too before a message changes.

**References:**

- Spec: "The method of a batch", "Text that does not change", "Kept
  findings", "Messages and copies" and "A doc that disagrees with the code".
- The terms: `GLOSSARY.md` and `../burn-mamba/GLOSSARY.md`. In burn-stack,
  ADR-0009 permits a quoted source word.
- Prior art: `.scratch/glossary-followups/issues/04-prose-sweep-utils-optim.md`.

- [ ] `## Comments` gives the count of hard findings before and after.
- [ ] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [ ] Each substring of a changed message that a test of either repo matches
      stays.
- [ ] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [ ] `./scripts/check.sh` passes. Run it in the background.

## Comments
