# 01: The prose step reads `.scratch/` and the message strings

Status: resolved

Blocked by: none

Spec: `../burn-mamba/.scratch/ste-sweep/spec.md`, section "The prose step of
`scripts/check.sh`".

**Context:** this ticket goes first in burn-stack. Tickets 02–05 are the
batches of the sweep, and this ticket blocks each one. So the prose step guards
each batch. `../burn-mamba/.scratch/ste-sweep/issues/01-prose-step-scratch-messages.md`
makes the same change in burn-mamba.

**What to build:** the prose step of `scripts/check.sh` also checks the
tickets and the strings of `panic!`, `expect` and `assert!` calls.

- The step also reads the files under `.scratch/`.
- The step runs the linter with `--parts docs,comments,messages`, on both
  sides of the compare.
- The rule stays. A changed file fails when it has more hard findings than at
  the merge base with `origin/main`. A new file starts at zero.
- The comments of the script describe what the step reads. Each doc that
  names the prose step agrees with them.
- The ticket plants no violation to prove the change (spec, "Testing
  Decisions").

**References:**

- The prose step of `../burn-mamba/scripts/check.sh` has the same shape. Keep
  the two steps equal.
- The new tickets of `.scratch/ste-sweep/` are new files, so the extended step
  reads them. They start at zero hard findings.

- [x] The prose step reads `.scratch/` and the message strings.
- [x] The comments of `scripts/check.sh` and each doc that names the step
      describe what it reads.
- [x] `./scripts/check.sh` passes. Run it in the background.

## Comments

### The counts of the prose step

The merge base is `7355f1a`. The step reads these files:

| File | Hard findings before | Hard findings after |
|---|---|---|
| The five tickets of `.scratch/ste-sweep/issues/` | new | 0 |
| `CLAUDE.md` | 2 | 2 |
| `docs/agents/testing.md` | 1 | 1 |

### The message strings

With `--parts docs,comments,messages`, four `.rs` files get 11 more hard
findings. All of them are in the message strings. Both sides of the compare
count them, so an unchanged message does not fail the step.

| File | Without messages | With messages |
|---|---|---|
| `src/examples/tiny_stories/lm.rs` | 7 | 9 |
| `src/optim/report.rs` | 1 | 2 |
| `src/utils/capture.rs` | 6 | 13 |
| `src/utils/untied.rs` | 0 | 1 |

The report of a failure also uses `--parts docs,comments,messages`. So it lists
each finding that the count has.

### The calls that the linter reads

The skill and the spec name `panic!`, `expect` and `assert!`. The linter reads
more calls. In a probe `.rs` file, the string of each call has a semicolon:

- The linter reads the strings of `panic!`, `expect`, `assert!`,
  `assert_eq!`, `debug_assert!`, `unreachable!`, `format!`, `println!`,
  `eprintln!`, `write!`, `bail!`, `anyhow!` and `log::warn!`.
- It does not read the strings of `vec!`, a custom macro, a function call,
  `.context(…)`, `Err(…)` and a `help = "…"` attribute, or a plain string.

So the comment of `check_prose` gives these calls as examples. The maintainer
keeps the strings of all these calls in the sweep and in the step. The section
"Scope" of the spec gives these calls. Its section "Out of Scope" gives the
strings that the linter does not read.

### The docs that name the step

- Outside `.scratch/`, `CLAUDE.md:44` is the only doc that names the step. It
  says "prose" and does not say what the step reads. Every session loads
  `CLAUDE.md`, so the line stays short and does not change.
- Line 10 of tickets 02–05 names the step. It agrees with the step.
- `.scratch/check-script-followups/issues/01-prose-ratchet.md` asks for the
  same step. It says that the step reads `.scratch/`, and it does not name the
  message strings. This change and `e3a6c7b` meet each of its criteria. Its
  `Status:` stays `needs-triage` for the maintainer.

### For the burn-mamba ticket

- The burn-mamba ticket can copy `check_prose` of this change, so the two steps
  stay equal.
- `../burn-mamba/docs/agents/code-review.md:8` names the rules of `ste-lint.py`
  that `scripts/check.sh` covers.
- `../burn-mamba/.scratch/check-script-followups/issues/01-prose-ratchet-reads-scratch.md`
  asks for the same step in burn-mamba.

### The `tmp/` directories

After the review of the burn-mamba ticket, the maintainer selects a skip of
each `tmp/` directory, at any depth. So the step reads what the sweep reads. A
second commit adds `grep -v -E '(^|/)tmp/'` to the list of files, and a
sentence to the comment of `check_prose`. In burn-stack, no tracked `.rs` or
`.md` file is under a `tmp/` directory, and `.gitignore` has the root `tmp/`.
So the step reads the same files as before. The two steps stay equal.
