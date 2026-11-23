# 06: Recount the swept batches on the new linter

Status: resolved

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

- [x] The repro of each linter ticket gives its new findings on the installed
      copy of the skill.
- [x] `## Comments` gives the count of hard findings before and after, for each
      scope.
- [x] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [x] Each substring of a changed message that a test of either repo matches
      stays.
- [x] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [x] `./scripts/check.sh` passes. Run it in the background.

## Comments

### The two recounts

The user ran each recount from the ticket of its scope. So the record of each
scope is in that ticket, and this ticket gives the result:

- `src/modules/`: commit `1d096e0`, and
  [its recount](02-sweep-modules.md#the-recount-on-the-new-linter) in
  ticket 02.
- `src/examples/`: commit `94ce28c`, and
  [its recount](03-sweep-examples.md#the-recount-on-the-new-linter) in
  ticket 03.

Each recount ran the repro of both linter tickets on the installed copy of
the skill. Each repro gave its findings, and `--selftest` passed.

### The counts

The linter runs with `--enable history --parts docs,comments,messages`. The
column "Before" gives the count at the start of each recount.

| Scope | Files | Hard before | Hard after | Advisory before | Advisory after |
|---|---|---|---|---|---|
| `src/modules/` | 49 | 2 | 0 | 38 | 35 |
| `src/examples/` | 21 | 3 | 0 | 7 | 3 |

At `94ce28c`, the linter gives the counts of the "after" columns for both
scopes. Neither scope has a kept finding.

In `src/examples/`, the commit that resolves ticket 03 (`2332385`) has 5 hard
findings on the new linter. Commit `c1741a1` removes two of them from
`src/examples/tiny_stories/lm.rs`, both phrasal verbs. The git history of this
ticket gives them. See ticket 02,
[The detach of a cache](02-sweep-modules.md#the-detach-of-a-cache).

### Messages, docs and checks

- One message changes, in `read_parquet` of
  `src/examples/tiny_stories/dataset.rs`. No test or script of either repo
  matches a part of the old string.
- The sweep of the two scopes finds no doc that disagrees with the code. The
  review of ticket 02 finds one: the docs said that the detach of a carried
  cache frees memory. The user decided it, and commit `c1741a1` changes the
  docs. See ticket 02,
  [The detach of a cache](02-sweep-modules.md#the-detach-of-a-cache).
- At `94ce28c`, `./scripts/check.sh` passes, with its prose step. In the
  recount of ticket 02, the prose step first failed on two ticket files (see
  that ticket).
