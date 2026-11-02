# 10: Add the training terms to the glossary

Status: ready-for-agent

Blocked by: None (can start immediately)

**What to build:** `GLOSSARY.md` gets three entries: the training step, the
optimizer update and the LR schedule. The identifiers and the prose of this
repo then follow them. Ticket 09 uses these terms in identifiers and in
`cli::HELP`, but the glossary does not define them. The note
`_Avoid_: schedule (alone)` is only under the layer schedule.

The rename breaks the API and the saved configs, and the user permits this. A
training config with a key of the rename map does not load after the rename.
No text of `cli::HELP` changes.

burn-mamba moves its pin to the commit of this ticket. Its ticket is
`../burn-mamba/.scratch/glossary-followups/issues/10-burn-stack-training-terms.md`.

## The entries

The user approved this text. Add the entries through the `/domain-modeling`
skill.

In `### Calls`, before **Captured step**, which names the training step:

```markdown
**Training step**:
The forward, the backward and the optimizer update of one batch (one window,
in an LM run).
_Avoid_: optimizer step
```

In `### Optimizer`, after **Fallback optimizer**:

```markdown
**Optimizer update**:
The part of a training step that applies the gradients to the weights.

**LR schedule**:
The map from the count of training steps to the learning rate.
_Avoid_: schedule (alone)
```

## Rename map

| Old | New | Where |
|---|---|---|
| field `CosineAnnealingLr::total_steps` (a config key) | `total_training_steps` | `src/utils/scheduler/mod.rs`, `scheduler/tests.rs`, `src/examples/cli/tests.rs` |
| field `CosineAnnealingLr::warmup_steps` (a config key), `with_warmup_steps` | `warmup_training_steps`, `with_warmup_training_steps` | the same files, `src/utils/capture/tests.rs` |
| field `LinearLr::steps` (a config key) | `LinearLr::training_steps` | `src/utils/scheduler/mod.rs` (also its docs) |
| the parameter `step` of each `get_lr` (`_step` in `ConstantLr`) | `training_step` (`_training_step`) | `src/utils/scheduler/mod.rs` (also its docs) |
| the closure parameter `steps` of `scale` | `training_steps` | `src/utils/scheduler/mod.rs` |

The **Step** entry has `_Avoid_: step (alone, for one recurrence step or for
one training step)`. So these names change.

## Sites

Line numbers are those of commit `1d9aa5f`.

The verb "step" for an optimizer update. Write "update":

- `GLOSSARY.md:247` (**Muon plan**): "that Muon steps".
- `src/optim/mod.rs:1`: "which weights Muon steps".
- `src/optim/spec.rs:85`: "The optimizer steps the sub-projections".
- `src/optim/segmented.rs:8`: "steps each sub-projection".
- `docs/adr/0004-muon-sees-split-projections.md:8,11`: "`Segmented` then steps
  each", "Muon steps only the weights".

The LR schedule under other names:

- `src/examples/training.rs:205`: "Learning-rate schedule." Write "The LR
  schedule."
- `src/utils/scheduler/mod.rs:137,185`: the doc titles "Cosine Annealing
  Learning Rate Scheduler with Linear Warmup" and "Constant Learning Rate
  Scheduler". Write "LR Schedule".
- `src/utils/capture/tests.rs:235,238`: the local `schedule` is `lr_schedule`.
- `src/utils/scheduler/tests.rs`: the local `scheduler` is `lr_schedule`.

## These stay

- The verb "step" for the recurrent call: `src/modules/`,
  `src/examples/tiny_stories/sample.rs`.
- "Schedule" for the layer schedule in code: the type `Schedule`, the module
  `utils::schedule` and their locals. The type gives the sense.
- The module `utils::scheduler`. It is a public path.
- "Schedule" in `src/utils/scheduler/` where the text names an LR chain. The
  chain gives the sense.
- The constant `STEPS` of `src/utils/capture/tests.rs` (see the Comments of
  ticket 09).

## Method

- Use the Edit tool. `replace_all` is correct for a field name in one file.
  Edit "step" and "schedule" in prose site by site, because these words have
  other senses.
- Bump the crate version to 0.37.0 (`Cargo.toml`), because the rename breaks
  the API and the saved configs.
- burn-deltanet has no issue tracker. Under `## Comments`, list its sites that
  use a renamed name, as ticket 09 does.

## Acceptance criteria

- [ ] `GLOSSARY.md` has the three entries, with the text above.
- [ ] `rg -e total_steps -e warmup_steps -e '\.steps\b' -e 'Muon steps'
      -e 'optimizer steps' -e 'steps each' -e 'Learning Rate Scheduler'
      -e 'Learning-rate schedule'` over `src`, `docs`, `CLAUDE.md` and
      `GLOSSARY.md` finds nothing.
- [ ] `cargo check --all-targets`, `cargo check --features examples-common`,
      `cargo test --lib -- --test-threads=1` and
      `cargo test --lib --features examples-common -- --test-threads=1` pass
      (flex).
- [ ] `cargo doc --no-deps --features examples-common` has no warnings.

## Comments
