# 06: Recount the swept batches on the new linter

Status: ready-for-agent

Blocked by: `../burn-mamba/.scratch/ste-lint-misses/` 01 and 02 (the linter), and the install of the changed skill

Spec: `../burn-mamba/.scratch/ste-sweep/spec.md`. Follow its method, its text
that does not change, its kept findings and its rules for messages and copies.

**Context:** tickets 02 and 03 swept `src/modules/` and `src/examples/` to zero
hard findings, on the linter before the two tickets of
`../burn-mamba/.scratch/ste-lint-misses/`. These tickets add hard findings:
more phrasal verbs, and the history phrases across a line break. Ticket 04
waits for the new linter, so only the scopes of tickets 02 and 03 need a
recount. Ticket 01 has no sweep scope: its new tickets are in the scope of
ticket 05. Ticket 05 goes last, and it waits for this ticket.

**What to build:** the scopes of tickets 02 and 03 have zero hard findings on
the new linter.

- Scope: `src/modules/` and `src/examples/`.
- `scripts/check.sh` runs the installed copy of the skill. Before the first
  count, check that the installed copy holds the change of both linter
  tickets: run the repro of each ticket.
- `## Comments` of each linter ticket lists the new hits of both repos. Start
  from these lists.

**References:**

- Spec: "The method of a batch", "Text that does not change", "Kept
  findings", "Messages and copies" and "A doc that disagrees with the code".
- The terms: `GLOSSARY.md` and `../burn-mamba/GLOSSARY.md`. In burn-stack,
  ADR-0009 permits a quoted source word.
- Prior art: tickets 02 and 03.

- [ ] The repro of each linter ticket gives its new findings on the installed
      copy of the skill.
- [ ] `## Comments` gives the count of hard findings before and after, for each
      scope.
- [ ] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [ ] Each substring of a changed message that a test of either repo matches
      stays.
- [ ] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [ ] `./scripts/check.sh` passes. Run it in the background.

## Comments
