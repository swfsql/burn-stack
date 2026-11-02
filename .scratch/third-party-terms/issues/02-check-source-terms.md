# 02: Check the map of H-Net and MGR source terms

Status: resolved

Blocked by: 01

**What to build:** Each row of `docs/source-terms.md` matches its citations. Then
the links tell agents to read the doc.

## Decisions from the grill

**How to run it.** Run this ticket in a new session, with several agents. The user
approved reads of `../papers/` for this ticket.

**Scope.** Each row of each section, against each of its citations. The input
`research.md` is a best-effort draft, so no row counts as checked.

**Out of scope.** The doc of burn-mamba. It has its own check ticket.

## Work pipeline

One agent is reliable up to about 150k tokens. So the lead splits the work before
any agent starts.

1. **Split.** The lead reads only `docs/source-terms.md` and numbers its rows. It
   puts each citation into a batch for its source: the H-Net paper or the
   Multi-Gate Residuals paper.
2. **Agents.** One agent takes one batch of at most 25 rows. A source can need
   more than one batch. Expect 2 to 3 agents.
3. **Reads.** An agent reads the cited lines, with enough context to decide. It
   does not read a whole file unless a row needs that.
4. **Limit.** Each agent stays under 150k tokens. At the limit, it stops and
   reports the rows that it did not finish. The lead gives them to a new agent.
5. **Report.** For each citation, an agent gives `confirmed`, `wrong` or
   `not found`, and the lines that prove it. For a wrong row, it also gives the
   corrected text. A batch agent does not edit the doc.
6. **Merge.** The lead edits the doc from the reports. The lead does not read the
   sources again.

## Links

Add these after the merge:

- `docs/agents/domain.md`: tell an agent to read this doc before it reads or
  ports from a source.
- "Where to look" in `CLAUDE.md`: add the doc.

## Acceptance criteria

- [x] Each citation of the doc has a report from an agent.
- [x] Each corrected row cites the lines that prove it.
- [x] `docs/agents/domain.md` and `CLAUDE.md` have the links.
- [x] `## Comments` at the end of this file gives the number of rows checked and
      corrected, and each open question for the user.
- [x] `git diff` shows no change under `src/`.

## Comments

`R<n>` is the n-th table row with a citation, in the order of the doc. `R1` is the
"block" row, and `R55` is the `β` row of Letters.

### Result

- Checked: 55 rows and 153 citations. The results are 141 `confirmed`, 7 `wrong`
  and 5 `not found`.
- Corrected: 42 rows. 13 rows have no change.
- The doc now has 224 citations. Each added citation comes from a report or the
  review, except `methods_trap.tex:138` in R55. The lead checked that line.
- Agents: 4, one for each batch. The limit of 25 rows gives 2 batches for each
  source (35 H-Net rows, 27 MGR rows). Each agent used 85k to 109k tokens.
- Review: a spec agent compared the 42 corrected rows with their cited lines. It
  found one wrong row (R37) and one wrong code statement (R2). Both are corrected.
- The reports are in the session of this ticket, not in the repo.

The reports mark 29 rows as wrong:

- Text, and usually citations: R1, R2, R3, R4, R5, R6, R7, R10, R11, R16, R19, R20, R21,
  R25, R26, R29, R37, R41, R43, R44, R46, R48, R51, R52.
- Citations only: R31, R35, R47, R49, R53.

The other 13 corrected rows take a proved addition from a note in a report: R8,
R9, R18, R22, R30, R32, R33, R36, R39, R40, R42, R45, R55.

### Lead decisions

The reports asked these questions. Our code and glossaries answer them, so the
lead did not read the sources again.

- R2: burn-mamba lists "SSM state" under `_Avoid_` for its State. So "state" is
  not always another thing, and only the "SSM state" is that State. Our
  `RouterCache` keeps the last `x̂`, not the key of the "DC state".
- R8: `HNM:53` uses "main stage" for a general `s`, so the row names **Inner
  H-Net** for that line.
- R10, R32: `Layers` has no end norm. The norm is in `HNetStage` and `HNetMain`.
- R22: the source does not say where the LM head goes. The row says only what the
  source names, so it stays.
- R25, R26: a row gives the whole module, so the second action of each module
  goes in.
- R29: the row gives the `STE` form, because the header of `hnet` uses that form.
- R47: the row has no warning about `s=S`, because R8 gives it.
- R6: "fixed-frequency" stays in the Wrong port cell. It is outside knowledge,
  because `MGR:695` gives only the name and `θ`.

### Open questions for the user

1. **Do we add rows for three words that no row has?** This ticket checks only
   the existing rows, so the doc does not have them yet. I recommend (a) and (b).
   - (a) "layer" and `L` in MGR. "Layer" counts Transformer blocks at `MGR:281`,
     `MGR:552`, `MGR:632` and `MGR:640`. It counts source layers at `MGR:171`,
     `MGR:433` and `MGR:462`. `L` at `MGR:268` counts source layers. A port of
     `n` or of the init formula can use the wrong count.
   - (b) `g` in Letters. `g_l` is a gate at `MGR:237` and `MGR:240`, and `g` is
     `ngroups` in our Notation table.
   - (c) "residual stream" in H-Net, at `HNM:232`. The word means the flow through
     a layer stack, and it is not our **Residual stream**.
2. **Does the sentence of ADR-0009 about "mixer" stay?** _Contradicts ADR-0009
   (Our terms over the terms of the sources) in one sentence:_ it says that "Mixer"
   is our β gate. R43 now says that a "mixer" holds the β gates of one layer, with
   one `w_l` for the layer (`MGR:149`, `MGR:171`, `MGR:172`, `MGR:189`). I did not
   change the ADR or the glossary. I recommend that you keep the glossary and
   change this sentence of the ADR to the R43 form.
3. **Does R5 need a note on the gate of **Feed-forward**?** `GLOSSARY.md` says "a
   gated MLP". `MGR:620` gives a `ReLU²` FFN activation, and no line calls the
   "feedforward layer" gated. I recommend no change, because the row maps the role
   and not the activation.

### Answers of the user

1. Yes. The doc has the three rows, so it now has 58 rows and 237 citations. The
   `R<n>` numbers in this file apply to the doc before these rows.
   - **Correction to (c).** The text of question (c) is wrong. `HNM:232` uses the
     word in the sense of our **Residual stream**. Only
     `HN/sections/A_background.tex:127` has another sense. The row gives both.
2. Yes. ADR-0009 now says that "Mixer" holds the β gates of one layer. The
   glossary has no change.
3. No change.
