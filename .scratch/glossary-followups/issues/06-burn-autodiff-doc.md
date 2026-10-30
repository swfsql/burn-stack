# 06: The Burn autodiff doc in burn-stack

Status: ready-for-agent

Blocked by: 01

**What to build:** An agent in burn-stack or in a consumer finds the semantics
of Burn for devices, autodiff, gradients and checkpointing in this repo. The
doc is correct for the Burn version of this crate. Today the doc is in the
burn-bug workspace (`/shared/claude/burn-bug/info/burn-autodiff-semantics.md`)
by accident, and it describes an older Burn revision.

## Decisions from the burn-stack grill

- The doc goes into burn-stack, under `docs/burn/`. burn-stack is the shared
  base of burn-mamba and burn-deltanet, and they already link into
  `../burn-stack/` for the glossary. A separate repo is for later, when a user
  outside these crates needs the doc.
- Copy the doc with `cp`. Do not edit or delete the burn-bug copy.
- ADR-0003 stays self-contained. It gets one "see also" link to the doc.

## What to change in the copy

- The version line names the Burn of this crate: 0.22.1 from crates.io (see
  `Cargo.lock`), not the `8f08fdad` git revision.
- §14 lists the differences at `8f08fdad`. Check each row of §13 and §14
  against the Burn 0.22.1 source. Rewrite §14 for 0.22.1, or delete it when no
  difference is left. Fix the anchors that point to it.
- Delete each reference that exists only in the burn-bug workspace
  (`projects/autodiff-retention`, `projects/untied-leak`, `issues/issue9.md`,
  "this workspace"). Keep each measured number and its **Measured** mark, and
  give its setup in one line.

## Acceptance criteria

- [ ] The doc is under `docs/burn/`, and its version line names Burn 0.22.1.
- [ ] `rg` finds no path outside this repo in the doc (`projects/`,
      `burn-bug`, `issues/issue`).
- [ ] Each row of §14 is checked against the 0.22.1 source, or §14 is gone
      with its reason under `## Comments`.
- [ ] ADR-0003 and the "where to look" map of CLAUDE.md link to the doc, and
      the links resolve.
- [ ] The asd-ste100 lint (`ste-lint.py`) passes on the doc, or each
      remaining hit is listed under `## Comments`.

## Comments
