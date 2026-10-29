# 01: Docs redo: a short CLAUDE.md with a short File Map

Status: ready-for-agent

Blocked by: the burn-stack grill (an interactive session with the user, not a
ticket). The grill adds the ADRs of burn-stack and the rest of its glossary.

**What to build:** An agent that starts a session reads a short CLAUDE.md that
points to where each kind of knowledge lives. Today, CLAUDE.md (546 lines, with
a File Map of about 200 lines) repeats much of the glossary, the ADRs and the
module headers.

## Decisions from the burn-mamba grill

- Load the `mattpocock-skills:writing-for-agents` and `asd-ste100` skills
  before you write. Read `GLOSSARY.md` and `docs/adr/` first, and use their
  terms in all new prose.
- CLAUDE.md keeps only what every session needs:
  - a summary of the project in about 3 lines,
  - the build and test commands, the feature flags and the test-device rules,
  - the rules for edits, waiting, memory and commit messages,
  - a "where to look" map: `GLOSSARY.md`, `docs/adr/`, the module headers,
    `docs/agents/`, and the consumers (burn-mamba, burn-deltanet),
  - the notation,
  - a short File Map.
- The File Map has 40 lines or fewer. It has one line for each directory, and
  one line for each file whose name does not tell what it holds. It has no
  `tests.rs` lines.
- The Architecture and Key Design Decisions sections go to their sources.
  Terms go to the glossary, decisions with a trade-off to an ADR, and
  mechanisms to the module header of their file. Delete a paragraph only after you find its
  content in one of these places. If you find it in none, move it to the
  module header. If it holds a new term or a new decision, also add a note
  under `## Comments` for the user. Do not edit the glossary or the ADRs.
- The block-agnostic rule follows the burn-stack ADR from the grill (the copy
  of burn-mamba's ADR-0002). Code is block-agnostic by default, and docs can
  cite a family. A rare exception is named after the mechanism. burn-mamba and
  burn-deltanet are consumers.
- The doc-maintenance rules:
  - Keep the docs small, and never use them as a changelog (keep this rule).
  - A change that adds, deletes or renames a file updates its File Map line
    in the same change.
  - A new term goes to `GLOSSARY.md` (through `/domain-modeling`). A new
    decision with a trade-off goes to an ADR.
  - Delete the `tmp.md` rule and the "do not read big files at the end" rule.
- Custom Commands: `mv` and `cp` are allowed to move and copy files. Content
  edits still use the Edit/Write tools. Keep the other rules.
- Do not edit README.md. Ticket 02 sweeps its prose.

## Acceptance criteria

- [ ] CLAUDE.md has about 200 lines or fewer, and its File Map has 40 lines or
      fewer.
- [ ] Each deleted CLAUDE.md paragraph has its content in the glossary, an ADR
      or a module header, or is listed under `## Comments`.
- [ ] No doc links to a CLAUDE.md anchor that is gone (check this repo and
      `../burn-mamba`).
- [ ] `cargo check` passes, and `cargo doc --no-deps` has no warnings.
- [ ] The asd-ste100 lint (`ste-lint.py`) passes on CLAUDE.md.

## Comments
