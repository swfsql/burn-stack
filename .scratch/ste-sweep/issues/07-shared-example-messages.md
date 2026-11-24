# 07: The two shared messages of the examples

Status: resolved

Blocked by: None

Spec: item 4 of "The decisions of the user" in
`../burn-mamba/.scratch/ste-sweep/issues/06-sweep-reset-examples.md`, and a
request of the user: the two functions move to burn-stack.

**Context:** burn-mamba ticket 06 gives one form to two messages. A launcher
prints the first one when its flags select no run. A training loop prints the
second one when the budget of its session is spent. Each example of
burn-mamba prints them.

Burn-mamba commit `f55b277e` holds each message once, in the
`examples/common/mod.rs` of burn-mamba. The messages are not specific to
Mamba. They belong beside `cli::HELP` and `Session::is_exhausted`.

**What to build:** burn-stack holds the two messages.

- `examples::cli::print_nothing_to_run` prints "the flags enable neither
  training nor inference", and then `HELP`.
- `examples::session::print_budget_reached` prints "the training reached its
  budget, so it stops".
- The words of each message are those of burn-mamba. No test of either repo
  matches them.

**Out of scope:**

- The loops of `examples::mnist::classify` and `examples::tiny_stories::lm`
  stop at the budget with no message. They stay so.
- The call sites in burn-mamba. They need the pin of burn-stack 0.46.0.
  Burn-mamba ticket 19 plans that pin move (its step 2).

- [x] Each function has a doc comment.
- [x] `./scripts/check.sh` passes. Run it in the background.
- [x] The commit is on `origin/main`.

## Comments

- `print_nothing_to_run` is in `src/examples/cli.rs`, after `HELP`.
  `print_budget_reached` is in `src/examples/session.rs`, after the
  `impl Session` block.
- The words of both messages are those of burn-mamba commit `f55b277e`. A
  search of both repos finds no test that matches them.
- The STE linter finds 0 hard findings in the two changed files and in this
  ticket, as at the base.
- `./scripts/check.sh` passes (exit 0). The log has no warning and no failed
  test.
- The user tells this session to commit and push this change. The
  burn-mamba call sites wait for the pin move (see Out of scope).