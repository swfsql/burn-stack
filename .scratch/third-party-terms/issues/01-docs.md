# 01: The map from H-Net and MGR source terms to our terms

Status: ready-for-agent

Blocked by: None (can start immediately)

**What to build:** `docs/source-terms.md`. An agent that reads the H-Net paper or
the Multi-Gate Residuals paper looks up a source word there. It finds our term,
the difference, and the wrong port that the word can cause. The docs of this
repo also follow the new definition of `Layer` and the `_Avoid_` rule.

The input is `../burn-mamba/.scratch/third-party-terms/research.md`. Load the
`asd-ste100` and `mattpocock-skills:writing-for-agents` skills before you write.
The user approved reads of `../papers/` for this ticket.

Line numbers are those of commit `4609abe`.

## Decisions from the grill

**Our terms stay.** This doc renames nothing (ADR-0009).

**Placement test.** A word goes in `GLOSSARY.md` if an agent that reads only our
code and docs needs it. A word goes in this doc if only an agent that reads the
sources needs it.

**Sources of this doc.** The H-Net paper and the Multi-Gate Residuals paper:
research parts 2.4 and 2.5. The Mamba rows go to the doc of burn-mamba, whichever
crate owns our term.

**Sections.** The doc of burn-mamba has a section for disagreements between
sources. This doc has none, because the research covers one source for each
mechanism.

1. **False friends and splits.** This section is the focus. Write one row for
   each source word, in the order of risk of research part 3. Each row gives the
   source word, its meaning, our term and the wrong port. The input is the rows
   that research marks `false friend` or `split`.
2. **Source terms with no term of ours.** The H-Net and Multi-Gate Residuals rows
   of research part 4, for example "downsampler" and "lerp". Keep a `renamed` row
   only if a reader cannot decode its name alone.
3. **Letters.** The H-Net and Multi-Gate Residuals rows of research part 5.3.

**Citations.** A source citation has the form `file:line`, and the doc header
gives the path aliases. Our own code is cited by item name, with no line number.

**The `Layer` wording.** The glossary now says "A Pre-LN norm around one block,
with an optional feed-forward part. The layer stack combines its output with its
input." Reword these lines to match:

- `src/lib.rs:3`: "the Pre-LN residual".
- `src/modules/mod.rs:25`: "A single Pre-LN residual layer around one block".

`src/modules/multi_gate.rs:362` stays, because it names a residual scheme.

**The `_Avoid_` rule.** Add one line to the Docs section of `CLAUDE.md`: `_Avoid_`
applies to our own words. A doc can use a source word in quotes or backticks when
it cites that source.

**Not in this ticket.**

- `src/modules/mod.rs:85` keeps "a selective SSM", because it names a class of
  models.
- burn-mamba does not move its pin, because this ticket changes only docs.
- No verification ticket follows, because this doc cites only papers.

## Links

- `docs/agents/domain.md`: tell an agent to read this doc before it reads or
  ports from a source.
- "Where to look" in `CLAUDE.md`: add the doc.
- The File Map of `CLAUDE.md`: add the file if the rule of that section requires
  it.

## Acceptance criteria

- [ ] `docs/source-terms.md` has the three sections.
- [ ] Each word of research part 3 that H-Net or Multi-Gate Residuals uses has a
      row.
- [ ] Each source citation has the form `file:line`, and no citation of our own
      code has a line number.
- [ ] `rg -i 'pre-ln residual' src/lib.rs src/modules/mod.rs` finds no hit.
- [ ] `CLAUDE.md` has the `_Avoid_` rule.
- [ ] `docs/agents/domain.md` and `CLAUDE.md` have the links.
- [ ] `cargo check` and `cargo doc --all --no-deps` give no warnings.
- [ ] `scripts/ste-lint.py` of the `asd-ste100` skill gives no hard violation on
      the doc.
