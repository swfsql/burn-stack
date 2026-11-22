# 01: The prose ratchet in `scripts/check.sh`

Status: resolved

Blocked by: None (can start immediately)

**What to build:** `scripts/check.sh` fails when a changed `.rs` or `.md` file
has more hard STE violations than it has on `origin/main`. The step also reads
the files of `.scratch/`. It skips when `CI` is set, as the host-path step does,
because the linter is a file of the box.

`../burn-mamba/scripts/check.sh` has this step as `check_prose`, but its copy
skips `.scratch/`. `../burn-mamba-example/scripts/check.sh` has a copy that
reads `.scratch/`. The maintainer made this decision in a burn-mamba-example
session. The burn-mamba tracker has a ticket for its own copy:
`../burn-mamba/.scratch/check-script-followups/issues/01-prose-ratchet-reads-scratch.md`.

## Acceptance criteria

- [x] `scripts/check.sh` has the prose step, and the step reads `.scratch/`.
- [x] The step skips when `CI` is set. On a box without `ste-lint.py`, the step
      gives a note and stops.
- [x] `scripts/check.sh` passes. A file that fails gets a fix of its prose, or
      a comment in its ticket.
- [x] The `CLAUDE.md` line of `./scripts/check.sh` names the prose step.

## Notes

- The linter is part of the `asd-ste100` skill:
  `~/.claude/skills/asd-ste100/scripts/ste-lint.py`.
- A file that is new since `origin/main` starts at zero hard violations.
- A name next to its synonym is a hard violation (`synonym-rotation`). For
  example, the glossary term `Erase` in a text that also says `delete`. A code
  span around the name stops the violation.

## Comments

`scripts/check.sh` has this step as `check_prose` (`e3a6c7b`). The ticket
`.scratch/ste-sweep/issues/01-prose-step-scratch-messages.md` makes the step
read `.scratch/` and the message strings, and it gives the counts. No file
fails the step. `CLAUDE.md:44` names the step ("+ prose").
