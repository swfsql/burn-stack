# 01: The coverage script

Status: ready-for-agent

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

- [ ] The script reports the line, branch and condition coverage of the `src/` of
      burn-stack.
- [ ] A compare against `HEAD` reports no lost item.
- [ ] In a throwaway copy of the tree, delete one test. The compare then lists the items
      that only that test covered.
- [ ] An edit above a covered line in the same file does not show as a loss.
- [ ] `scripts/check.sh` runs the script by default, and skips it with a note when the
      tools are missing.
- [ ] The test-seam doc and the File Map name the script.
