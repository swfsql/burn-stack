# 01: The coverage script

Status: resolved

Blocked by: the box install of `cargo-llvm-cov` and of the nightly `llvm-tools-preview`
(the user does it).

Spec: `../burn-mamba/.scratch/container-tests-home/spec.md`, section "Coverage".

**What to build:** one coverage script for every family crate. Any crate can run it on
itself, and it can show what coverage a change lost.

- The script takes the path of a crate. It runs the lib tests of that crate under
  `cargo llvm-cov` on nightly, with branch and condition coverage.
- It reports the line, branch and condition coverage of the `src/` of that crate. It
  does not count the test files (`tests.rs` and `tests/`).
- Nightly rustc has no MC/DC. `-Zcoverage-options` accepts only `block`, `branch` and
  `condition`, and condition coverage is the nearest to MC/DC.
- A compare mode takes a base: a commit, or a saved report. It lists each item that the
  base covered and the current tree does not cover.
- The compare matches an item of the base to the current tree through the git diff of
  the two trees. So an edit elsewhere in a file does not show as a loss. An item on a
  changed line shows as changed, not as lost.
- The output has the form of `../burn-mamba/scripts/compare_reports.py`. There is no
  stored baseline.
- `scripts/check.sh` runs the script on burn-stack by default. When the tools are
  missing, it prints a note and skips the coverage, as the prose check of burn-mamba
  does.

**Docs:** add the coverage compare to `docs/agents/testing.md` as a gate, beside the
other gates. Update the `scripts/` line of the File Map of `CLAUDE.md`.

- [x] The script reports the line, branch and condition coverage of the `src/` of
      burn-stack.
- [x] A compare against `HEAD` reports no lost item.
- [x] In a throwaway copy of the tree, delete one test. The compare then lists the items
      that only that test covered.
- [x] An edit above a covered line in the same file does not show as a loss.
- [x] `scripts/check.sh` runs the script by default, and skips it with a note when the
      tools are missing.
- [x] The test-seam doc and the File Map name the script.

## Comments

### The implementation

- The script is `scripts/coverage.py`. Its header defines the items and the compare.
- The user chose one run with two views. The run gives `-Zcoverage-options=condition`.
  A branch is one outcome of one condition. The tests cover a condition when they cover
  its two outcomes. The script does not give `cargo llvm-cov --branch`, because its
  `branch` option replaces `condition`.
- On burn-stack: lines 3376 of 4106, branches 241 of 414, conditions 85 of 207. These
  totals differ from the summary of `llvm-cov`, as the header of the script explains.
- The box did not have the nightly `llvm-tools-preview`. `cargo llvm-cov` installed it
  with no question, because the run had no TTY. `--check-tools` reads the components of
  `rustup`, so a run of the script never installs a tool.
- The acceptance checks:
  - `--base HEAD`: no lost item and no changed item.
  - A throwaway worktree of `HEAD`, with the test
    `scale_training_steps_moves_every_training_step_of_a_chain` deleted, and one new
    line above covered code in `src/utils/scheduler/mod.rs` and `src/utils/padding.rs`.
    The compare with a saved report of `HEAD` lists only the covered items of
    `Lr::scale_training_steps`: 13 lines and 2 branches, at their moved lines.
  - The same deletion in a tree with an older run gives the same items. With
    `--no-report`, `cargo llvm-cov` adds the counts of the older runs, so the script
    deletes them before each run.

### For burn-mamba ticket 01

- The script counts only the files in the `src/` of the crate that it runs. A report of
  the `src/` of burn-stack from a run of the burn-mamba tests needs a change of
  `counted_path`.
- A commit base with the `[patch]` path of burn-mamba does not build in
  `target/coverage-base/`. Use a saved report as the base there.
