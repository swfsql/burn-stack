# 02: Check the map of H-Net and MGR source terms

Status: ready-for-agent

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

- [ ] Each citation of the doc has a report from an agent.
- [ ] Each corrected row cites the lines that prove it.
- [ ] `docs/agents/domain.md` and `CLAUDE.md` have the links.
- [ ] `## Comments` at the end of this file gives the number of rows checked and
      corrected, and each open question for the user.
- [ ] `git diff` shows no change under `src/`.
