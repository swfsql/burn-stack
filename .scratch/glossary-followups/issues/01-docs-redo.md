# 01: Docs redo: a short CLAUDE.md with a short File Map

Status: resolved

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

- [x] CLAUDE.md has about 200 lines or fewer, and its File Map has 40 lines or
      fewer.
- [x] Each deleted CLAUDE.md paragraph has its content in the glossary, an ADR
      or a module header, or is listed under `## Comments`.
- [x] No doc of this repo links to a CLAUDE.md anchor that is gone. Each line
      of `../burn-mamba` or `../burn-deltanet` that the change makes stale is
      listed under `## Comments`.
- [x] `cargo check` passes, and `cargo doc --no-deps` has no warnings.
- [x] The asd-ste100 lint (`ste-lint.py`) passes on CLAUDE.md.

## Comments

### What moved where

I checked each paragraph of the old Architecture, Key Design Decisions and
File Map against the glossary, the ADRs and the docs of the code. Three facts
had no home. They moved into a module header:

- `src/lib.rs`: `forward` equals `step` unrolled on the outputs, on the final
  cache and on the gradients. For `RefBlock`, `reference/tests.rs` compares
  the outputs and the final cache. (The old CLAUDE.md said that it asserts
  the gradients too. No test there compares forward and step gradients. The
  H-Net tests do, in `hnet/tests.rs`.)
- `src/utils/graph.rs`: no other thread may do device work while a capture
  records. The header points to `loader_device` for the cause. (The workers
  build on the host only under `backend-flex`.)
- `src/modules/model_config.rs`: `ModelConfigExt` is not behind
  `examples-common`, because a consumer implements it in its library.

None of the three holds a new term or a new decision.

Some facts are in the doc of an item, not in a module header. They stay
there, because the item is where a reader looks for them:

- The gradient horizon and its straight-through carry: the doc of
  `Layers::grad_horizon` (`layers.rs` has no module header). Also the
  glossary and ADR-0006.
- The reversed pass of a padded bidirectional pair reads the real tokens of
  each row backwards: the doc of the pair forward (`bidi.rs:176`).
- The read-back of class markers for a window (minus the cursor before the
  call): `utils/class/mod.rs:224`.
- `loader_device` (dataloader workers build on the host): its item doc in
  `examples/device.rs`. The header of `utils/graph.rs` states the rule and
  points to it.

Two facts of the old Build section are now only in `Cargo.toml`, their source
of truth. These are the features that `optim` implies (`burn/optim`,
`burn/std`), and the crates that `examples-common` pulls in.
- The matching of a `ProjSpec` through `BLOCK_CONTAINERS`: its item doc in
  `optim/spec.rs`. Also ADR-0004.

### New in CLAUDE.md

These lines are not from the old CLAUDE.md:

- The **Waiting** and **Memory** rules, and the `dcg` hook line: copied from
  the Custom Commands of burn-mamba. The `dcg` hook is in the global
  settings, so it applies to this repo too.
- `cargo check --features examples-common` in the commands: the default
  features do not check `examples`.
- `mv` and `cp` in the Bash bullet (a decision of this ticket).
- "A rare exception gets the name of its mechanism" in the block-agnostic
  rule (from ADR-0002).

### Stale lines in the consumers

No doc of this repo links to a CLAUDE.md anchor. These consumer lines are now
stale. A later session in each repo fixes them.

burn-mamba (`CLAUDE.md`):

- 16–22: "no name, no shape assumption, no doc reference" in burn-stack.
  ADR-0002 lets the docs of burn-stack cite a family. The burn-mamba ticket 01
  already plans this change.
- 88–89 and 159: "see its File Map" for the generic containers. The File Map
  of burn-stack now lists only directories and non-obvious files. Point to
  `../burn-stack/GLOSSARY.md` and the module headers.
- 491: "documented in `../burn-stack/CLAUDE.md`". The Architecture section is
  gone. These features are in the glossary, the ADRs and the module headers
  of burn-stack.
- 566–575: "A change to a composition type updates `../burn-stack/CLAUDE.md`",
  and the `tmp.md` rule. Now only a change that adds, deletes or renames a
  directory or a listed file updates the File Map of burn-stack. It does so
  in the same change.

burn-mamba (`files.md`):

- 11: "see its `CLAUDE.md`" for the generic containers. Same as line 491
  above. (The burn-mamba ticket 01 moves `files.md` out of the repo.)

burn-deltanet (`CLAUDE.md`):

- 24–25: "a name, a shape assumption, or a doc reference". Same as ADR-0002
  above.
- 80–81: "has its own File Map". Same as burn-mamba 88–89.
- 348–357: the same File Map and `tmp.md` rule as burn-mamba 566–575.

burn-deltanet (`files.md`):

- 8: "has its own" for the composition layer. Weak: burn-stack still has a
  File Map, but it lists the containers only by directory.
