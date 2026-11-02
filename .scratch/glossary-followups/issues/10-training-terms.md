# 10: Add the training terms to the glossary

Status: resolved

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

- [x] `GLOSSARY.md` has the three entries, with the text above.
- [x] `rg -e total_steps -e warmup_steps -e '\.steps\b' -e 'Muon steps'
      -e 'optimizer steps' -e 'steps each' -e 'Learning Rate Scheduler'
      -e 'Learning-rate schedule'` over `src`, `docs`, `CLAUDE.md` and
      `GLOSSARY.md` finds nothing.
- [x] `cargo check --all-targets`, `cargo check --features examples-common`,
      `cargo test --lib -- --test-threads=1` and
      `cargo test --lib --features examples-common -- --test-threads=1` pass
      (flex).
- [x] `cargo doc --no-deps --features examples-common` has no warnings.

## Comments

- A config that has an old key in place of its new key does not load. This also
  applies to a config with only the old `warmup_steps`, although
  `warmup_training_steps` has a default. The error names the missing new key,
  for example ``missing field `total_training_steps` ``. A throwaway test showed
  this for each key of the rename map.
- The Sites above do not list these prose sites. They have the same meaning, so
  this ticket also changes them:
  - The verb "step" for an optimizer update:
    `docs/adr/0004-muon-sees-split-projections.md:27`,
    `docs/adr/0008-untied-copies-on-an-existing-axis.md:31`,
    `src/optim/mod.rs:171`, `src/reference.rs:99`.
  - `src/optim/sgd.rs:53`: the doc of `SgdConfig::step` said "One training
    step". This call is only the optimizer update, so the doc now says that.
  - `src/examples/trainer.rs:2`: "the update of the optimizer" is "the optimizer
    update".
- The rename map does not name these private names. `CLAUDE.md` asks for the
  glossary terms in code and test names, and a private name cannot break a
  consumer. So this ticket also changes them:
  - `src/reference/tests.rs`: the test
    `muon_steps_an_untied_projection_one_copy_at_a_time` is
    `muon_updates_an_untied_projection_one_copy_at_a_time`, and its local
    `stepped` is `updated`.
  - `src/utils/scheduler/tests.rs`: the local `step` of
    `a_chain_survives_a_json_round_trip` is `training_step`.
  - `src/utils/capture/tests.rs`: the closure `step` that holds a whole training
    step is `training_step`.
  - `src/optim/sgd.rs`: the private struct `Step` is `OptimizerUpdate`.
- These public names still say "step". The rename map does not name them, so
  they stay until the user decides:
  - `Lr::scale_steps` and its test
    `scale_steps_moves_every_training_step_of_a_chain`. burn-mamba and
    burn-deltanet do not call this method.
  - `SgdConfig::step`. It has the name of `Optimizer::step` of Burn.
- burn-deltanet has no issue tracker. These files use a renamed name:
  - `with_warmup_steps`: `examples/mnist-class/main.rs:81`,
    `examples/tiny-stories/main.rs:115`,
    `examples/register/carousel/main.rs:110`,
    `examples/register/majority/main.rs:89`.
  - The local `total_steps`: `examples/register/carousel/main.rs:96,107`,
    `examples/register/majority/main.rs:78,86`. This name is a name of
    burn-deltanet, so the compiler does not find it. It holds the argument of
    `CosineAnnealingLr::new`.

  burn-deltanet builds against burn-stack rev `984d42d`. So the Comments of
  ticket 09 also apply.
- `cargo test --lib -- --test-threads=1`: 87 tests pass. With
  `--features examples-common`: 112 tests pass.
