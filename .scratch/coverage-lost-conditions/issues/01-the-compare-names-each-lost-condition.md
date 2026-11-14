# 01: The compare names each lost condition

Status: resolved

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

- [x] A selftest: a covered condition that loses one outcome gives 1 lost branch and 1
      lost condition.
- [x] A selftest: a condition of the base with only one covered outcome loses that
      outcome. The compare gives 1 lost branch and 0 lost conditions.
- [x] A selftest: a covered condition on a changed line shows under "Covered items on
      changed lines", with its count.
- [x] The selftests that exist pass with the new count lines.
- [x] `scripts/coverage.py --selftest` and `scripts/check.sh` pass.

## Comments

### The implementation

- `compare_items` adds a `conditions` list to each file of the two sections. A
  condition whose two outcomes the base covers goes to `changed` when the change
  touches its span. It goes to `lost` when the working tree does not cover the two
  outcomes of the mapped span.
- `listing` aligns the three labels: `lines`, `branches`, `conditions`.
- The selftests: `a_covered_condition_that_loses_one_outcome_is_lost`,
  `a_lost_branch_of_a_condition_that_the_base_does_not_cover_loses_no_condition` and
  `a_covered_condition_on_a_changed_line_is_changed`. The second test passed at its
  first run, because the code of the first test has the guard `min(pair) == 0`. With
  that guard deleted in a throwaway copy, the second test fails with
  `1 branches, 1 conditions`.
- From `../burn-mamba`, `tmp/coverage-record/dep_report.py compare` of
  `tmp/coverage-record/after-04/burn-stack-by-both.json` and its copy
  `tmp/ct04/one-lost-condition.json` gives `0 lines, 1 branches, 1 conditions`,
  `branches 165:16 false` and `conditions 165:16`. `dep_report.py` did not change.
- `scripts/check.sh`: 6 selftests ok, 169 lib tests passed. The coverage of `src/`:
  lines 3867 of 4415, branches 303 of 432, conditions 121 of 216.

### The review (`tmp/review-lost-conditions-standards.md`, `tmp/review-lost-conditions-spec.md`)

- Fixed:
  - The comment of `KINDS` named the order of a listing. `listing` sets that order
    itself, so it now unpacks the kinds by name.
  - The variable of the dict comprehension in `add` had the name of the parameter
    `kind`.
  - The status and the checkboxes of this ticket.
- Kept:
  - The three loops of `compare_items` have one shape: skip, changed, lost. Each loop
    has a different test for "covered" and a different item. A shared helper needs
    more arguments than it saves.
  - `totals` and the `--files` table give the order of the kinds by hand. This ticket
    does not change them.
- Not from this change: `mapped_span` gives `None` when the change adds a line inside
  a span of more than one line. Branches have the same rule.
