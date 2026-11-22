# 02: Sweep: the modules

Status: ready-for-agent

Blocked by: 01

Spec: `../burn-mamba/.scratch/ste-sweep/spec.md`. Follow its method, its text
that does not change, its kept findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–04 are the batches of burn-stack, and they do not block each other. Ticket
05 sweeps `.scratch/`, `docs/` and the root files last.

**What to build:** the prose of the modules follows the `asd-ste100` skill.

- Scope: `src/modules/`, 48 files.
- At the start: 46 hard findings and 79 advisory findings. No message has a
  finding. `src/modules/layers.rs` alone has 14 hard findings.
- burn-stack code is block-agnostic (ADR-0002). A doc can cite a family, but
  the rewrite adds no Mamba-specific claim.

**References:**

- Spec: "The method of a batch", "Text that does not change", "Kept
  findings", "Messages and copies" and "A doc that disagrees with the code".
- The terms: `GLOSSARY.md` and `../burn-mamba/GLOSSARY.md`. In burn-stack,
  ADR-0009 permits a quoted source word.
- Prior art: `.scratch/glossary-followups/issues/02-prose-sweep.md`.

- [ ] `## Comments` gives the count of hard findings before and after.
- [ ] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [ ] Each substring of a changed message that a test of either repo matches
      stays.
- [ ] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [ ] `./scripts/check.sh` passes. Run it in the background.

## Comments
