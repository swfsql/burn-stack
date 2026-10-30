# 01: Docs redo: a short CLAUDE.md with a short File Map

Status: ready-for-agent

Blocked by: None (can start immediately)

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
  mechanisms to the module header of their file. Delete a paragraph only after
  you find its content in one of these places. If you find it in none, move it
  to the module header. If it holds a new term or a new decision, also add a
  note under `## Comments` for the user. Do not edit the glossary or the ADRs.
- The block-agnostic rule is ADR-0002. Code is block-agnostic by default, and
  docs can cite a family. A rare exception is named after the mechanism.
  burn-mamba and burn-deltanet are consumers.
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

## Decisions from the burn-stack grill

- The grill wrote ADR-0001 to ADR-0008 in `docs/adr/`, and added the open terms
  to `GLOSSARY.md`. Read both before you start.
- The user agreed again to three decisions above. These are the File Map of
  40 lines or fewer, the deletion of the `tmp.md` rule, and `mv` and `cp`.
- ADR-0002 replaces the old rule that a doc reference to one family goes to
  the crate of that family.
- The table gives the home of each part of the current CLAUDE.md. It is a
  start, not a proof. Check each paragraph against its home before you delete
  it.

| CLAUDE.md content | Home |
|---|---|
| What the project is, the block-agnostic rule | the 3-line summary, ADR-0002 |
| No optimized kernels | ADR-0001 |
| A no-grad region means the inner backend | ADR-0003 |
| Muon sees split projections | ADR-0004, the header of the optim module |
| `reference.rs` is load-bearing | ADR-0005 |
| The gradient horizon and its straight-through carry | the glossary (Gradient horizon), ADR-0006, the doc of the layer stack |
| A model config is the pair `{ shape, block }` | ADR-0007 |
| Untied parameters | the glossary, ADR-0008, the header of the untied module |
| The plug-in surface, the three calls, the forward/step contract | the glossary (Calls), the header of the block traits |
| Captured steps, the captured training step, no device work during a capture | the glossary (Calls), the headers of the captured step, the SGD and the trainer |
| Padded batches, packed rows | the glossary (Batches, Packing), the headers of padding and packing |
| Virtual layers, bidirectional stacks, class markers, MGR, H-Net | the glossary, the header of each module |
| The Dispatch backend | one line in CLAUDE.md (a session that writes code needs it), or the crate root header |
| `#![warn(missing_docs)]` | a rule in CLAUDE.md |
| The notation | CLAUDE.md |

- Ticket 06 brings the Burn autodiff doc into this repo and adds its line to
  the "where to look" map. Do not add that line here.
- Edit only this repo. burn-mamba and burn-deltanet name
  `../burn-stack/CLAUDE.md`, and some of their lines describe the old File Map
  rule. List those lines under `## Comments` for a later session in each repo.

## Acceptance criteria

- [ ] CLAUDE.md has about 200 lines or fewer, and its File Map has 40 lines or
      fewer.
- [ ] Each deleted CLAUDE.md paragraph has its content in the glossary, an ADR
      or a module header, or is listed under `## Comments`.
- [ ] No doc of this repo links to a CLAUDE.md anchor that is gone. Each line
      of `../burn-mamba` or `../burn-deltanet` that the change makes stale is
      listed under `## Comments`.
- [ ] `cargo check` passes, and `cargo doc --no-deps` has no warnings.
- [ ] The asd-ste100 lint (`ste-lint.py`) passes on CLAUDE.md.

## Comments
