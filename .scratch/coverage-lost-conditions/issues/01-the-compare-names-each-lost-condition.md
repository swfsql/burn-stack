# 01: The compare names each lost condition

Status: ready-for-agent

Blocked by: None (can start immediately)

**What to build:** the compare of `scripts/coverage.py` names each condition that a
change lost, beside the lost lines and branches. A reader sees a lost condition in the
compare, and does not have to find it through the lost branches.

**Why:** the compare catches a lost condition, but it lists it only as a lost branch at
the span of the condition. Its count line names only lines and branches. So in burn-mamba
ticket `../burn-mamba/.scratch/container-tests-home/issues/04-batch-b-delete.md`, a review
and the implementer read the compare as a check of the conditions by totals only. The
section "The review" of that ticket gives the test that shows the catch: one lost outcome
of one covered condition gives one lost branch.

- A lost condition: the base covers its two outcomes, and its span maps to a span of the
  working tree. The working tree does not cover the two outcomes of that span.
- A covered condition on a changed line: the base covers its two outcomes, and the change
  edits or deletes its line.
- The listing of a file gets a `conditions` line, beside the `branches` line. It gives the
  start of the span of each condition, in the form of the `branches` line. For example:
  `conditions  165:16`.
- The count line of each section names the three kinds: "N lines, M branches,
  K conditions".
- The header of the script says this: each lost condition also shows as one or two lost
  branches. A lost branch can come with no lost condition, when the base covers only that
  outcome of its condition.
- `../burn-mamba/tmp/coverage-record/dep_report.py` calls the compare of this script. So
  it gets the change with no edit.

- [ ] A selftest: a covered condition that loses one outcome gives 1 lost branch and 1
      lost condition.
- [ ] A selftest: a condition of the base with only one covered outcome loses that
      outcome. The compare gives 1 lost branch and 0 lost conditions.
- [ ] A selftest: a covered condition on a changed line shows under "Covered items on
      changed lines", with its count.
- [ ] The selftests that exist pass with the new count lines.
- [ ] `scripts/coverage.py --selftest` and `scripts/check.sh` pass.
