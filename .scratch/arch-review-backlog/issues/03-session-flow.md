# 03: A deeper `Session` that owns the flow after a training step

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-03. Line numbers are those of commit
`12429e1`.

**Files:** `src/examples/session.rs:73-237`, `src/examples/mnist/classify.rs:110-177`,
`src/examples/tiny_stories/lm.rs:731-845`, `../burn-mamba/examples/reset/majority/training.rs:132-145,207-218,228`.

**Problem:** `session.rs` has 18 public functions and no test. Ten epoch loops call
them by hand. The order is a hidden interface fact, and the compiler enforces none
of it:

- `checkpoint_due` and `valid_due` change state.
- `validated_now` is correct only if the loop polls `valid_due` at each training
  step.
- `end_epoch` must run before the epoch-end `save_optim`.
- `batch_limit` must bound the `take`.
- `save_model` and `save_optim` must stay a pair.

**Direction:** `Session` takes the decision after each training step and at each
epoch end. The example gives what to save and how to validate. A reader still sees
the whole `for batch` loop, and `Cadence` stays a visible knob.

**Tests:** Resume, budget and cadence scenarios become tests with no model.

**Against it:** No order bug is on record. The window loop of `lm.rs` must fit, and
the actions borrow the trainer and the optimizer during the call.
