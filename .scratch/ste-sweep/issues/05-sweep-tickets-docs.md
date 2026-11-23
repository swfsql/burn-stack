# 05: Sweep: the tickets, the docs and the root files

Status: ready-for-agent

Blocked by: 01, 02, 03, 04, 06, `../burn-mamba/.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../burn-mamba/.scratch/ste-sweep/spec.md`. Follow its method, its text
that does not change and its kept findings.

**Context:** ticket 01 extends the prose step of `scripts/check.sh` to
`.scratch/`. Tickets 02–04 sweep `src/` and write their records under
`## Comments` here. This ticket goes last, so it also reads those records.
burn-mamba ticket 08 records each changed copy of a shared `CLAUDE.md` rule
under `## Comments` here.

**What to build:** the prose of the tickets, the docs and the root files
follows the `asd-ste100` skill.

- Scope: all of `.scratch/` (resolved tickets and the tickets of this sweep
  included), `docs/`, `CLAUDE.md`, `README.md` and `GLOSSARY.md`, 64 files
  before this sweep.
- At the start: 66 hard findings and 91 advisory findings. The tickets of
  this sweep add their own findings, if any.
- `## Comments` and `## Answer` are records. A history phrase there stays only
  as a "Ticket record" (spec, "Kept findings").
- Keep the state of each checkbox, each `Status:` line and each `Blocked by:`
  line. The title of a ticket is a heading.
- `docs/burn/autodiff-semantics.md` keeps each rule and each trap.
- An ADR keeps each decision, each option and each reason.
- `GLOSSARY.md` keeps its terms, their meanings and its _Avoid_ entries.
- Each rule of `CLAUDE.md` keeps its exact meaning. Read
  `/mattpocock-skills:writing-for-agents` before you change `CLAUDE.md`.
- Apply each shared `CLAUDE.md` rule that burn-mamba ticket 08 changed, so
  that the two copies stay equal.

**References:**

- The spec sections "The method of a batch", "Text that does not change" and
  "Kept findings".
- The terms: `GLOSSARY.md` and `../burn-mamba/GLOSSARY.md`. In burn-stack,
  ADR-0009 permits a quoted source word.
- `docs/agents/issue-tracker.md`: the shape of a ticket.
- Prior art: `.scratch/glossary-followups/issues/02-prose-sweep.md`.

- [ ] `## Comments` gives the count of hard findings before and after.
- [ ] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [ ] `## Comments` lists the false positives of tickets 02–04 in one place.
- [ ] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [ ] `./scripts/check.sh` passes. Run it in the background.

## Comments

### The push

At the end of this ticket, push `main` to `origin`. The push also takes the
commits of tickets 01–04 and the two commits before them (`e3a6c7b` and
`dcbaf82`). This is the answer of the maintainer on ticket 01.
