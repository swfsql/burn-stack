# 03: Sweep: the examples

Status: resolved

Blocked by: 01

Spec: `../burn-mamba/.scratch/ste-sweep/spec.md`. Follow its method, its text
that does not change, its kept findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–04 are the batches of burn-stack, and they do not block each other. Ticket
05 sweeps `.scratch/`, `docs/` and the root files last.

**What to build:** the prose of the example modules follows the `asd-ste100`
skill.

- Scope: `src/examples/`, 21 files.
- At the start: 41 hard findings (2 in messages) and 86 advisory findings.
- The examples of burn-mamba use these modules. Search the tests of
  burn-mamba too before a message changes.

**References:**

- Spec: "The method of a batch", "Text that does not change", "Kept
  findings", "Messages and copies" and "A doc that disagrees with the code".
- The terms: `GLOSSARY.md` and `../burn-mamba/GLOSSARY.md`. In burn-stack,
  ADR-0009 permits a quoted source word.
- Prior art: `.scratch/glossary-followups/issues/05-prose-sweep-examples.md`.

- [x] `## Comments` gives the count of hard findings before and after.
- [x] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [x] Each substring of a changed message that a test of either repo matches
      stays.
- [x] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [x] `./scripts/check.sh` passes. Run it in the background.

## Comments

### The counts

The linter runs with `--enable history --parts docs,comments,messages` on the
21 tracked `.rs` files of `src/examples/`. The scope has no `.md` file.

| Findings | Before | After |
|---|---|---|
| Hard | 41 | 0 |
| Hard, in messages | 2 | 0 |
| Advisory | 86 | 2 |

The scope has no kept finding. Two advisory findings stay:
`src/examples/tiny_stories/dataset.rs:507` and `:1006` say that positions
`are scored`. The scored positions mask the loss and the accuracy, so "the
loss" is not the actor. An active verb needs a new actor, and the passive
keeps the scope.

### The two log lines

The linter read two metric lines of `println!` in
`src/examples/tiny_stories/lm.rs` as one sentence each, above the strict cap:

| Location | Line | Words |
|---|---|---|
| `:631` | the corpus line of the packed rows (`dataloaders_for`) | 25 |
| `:790` | the progress line of `epoch_train` | 21 |

The user selected a full stop in each line (see "The decisions of the user"):

| Location | Old part | New part |
|---|---|---|
| `:632` | `({:.3} of the positions, {} open rows, …, {} opening slots)` | `({:.3} of the positions). {} open rows, …, {} opening slots` |
| `:790` | `Windows {depth}/{windows} (mean {:.2}), Loss …` | `Windows {depth}/{windows} (mean {:.2}). Loss …` |

Each line stays one output line, with the same fields in the same order:

- `../burn-mamba/examples/tiny-stories/tmp/1m-cased/speed.sh:30` reads the
  corpus line with a regex. On a sample of the new line, the regex finds its
  5 fields.
- No script of either repo reads the progress line. The run scripts read
  `metrics.jsonl`.
- `../burn-mamba/examples/tiny-stories/README.md:141` quotes
  `Windows k/n (mean m)`. The new line keeps that text.

### The messages

Four other message strings change. Each one had an advisory finding (a
passive). A search of the tests and the scripts of both repos finds no match
for a part of these strings.

| Location | Old string | New string |
|---|---|---|
| `src/examples/trainer.rs:138` | `only plain SGD is captured` | `the trainer captures only plain SGD` |
| `src/examples/cli.rs:488` | `--resume: no {PROGRESS_NAME}.json was saved with the optim in {:?}` | `--resume: the optim in {:?} has no {PROGRESS_NAME}.json saved with it` |
| `src/examples/cli/tests.rs:74` | `the LR schedule is kept` | `the override keeps the LR schedule` |
| `src/examples/tiny_stories/sample/tests.rs:174` | `the flag input is read: …` | `the model reads the flag input: …` |

The two `should_panic` tests of `src/examples/cli/tests.rs` match `exclusive`
and `would be ignored`. These two messages do not change.

The strings that the linter does not read stay, for example `cli::HELP`. So
the copy of `cli::HELP` in `../burn-mamba/examples/README.md` stays equal.

### The copies

- The headers of `src/examples/mnist/classify.rs` and
  `src/examples/tiny_stories/lm.rs` describe the wrapper that an example
  supplies. Both headers change to the same form: a list of what the wrapper
  can do, then the parts that the module holds.
- The comments before the `println!` of the averaged metrics in
  `src/examples/tiny_stories/lm.rs` and `src/examples/mnist/classify.rs`
  change together (see "Print" below).
- The rest of burn-stack and all of burn-mamba have no copy of a changed
  paragraph.

### Docs that disagree with the code

The sweep changes neither side. The user decides each one.

1. `src/examples/tiny_stories/dataset.rs:86` says "Only the train split is
   bucketed". But with `TinyStoriesConfig::bucketing`, `dataloaders_for` also
   builds the validation dataloader on a `BucketedStoriesDataset`
   (`src/examples/tiny_stories/lm.rs:649`: "The buckets of the training: each
   story whole, in one window"). The header of `lm.rs` ("The validation takes
   the same buckets") and the doc of `dataloaders_for` agree with the code.
   The sentence keeps its advisory finding (a passive).

   The doc of `BucketedStoriesDataset` makes the same claim
   (`src/examples/tiny_stories/dataset.rs:950`: "The train split as batches
   of similar length"). But `dataloaders_for` builds it on `Split::Valid` too
   (`src/examples/tiny_stories/lm.rs:652`).
2. `src/examples/tiny_stories/dataset.rs:784` (the doc of
   `PackedStoriesBatcher`) says "The pair (0, 0) fills the opening slots and
   the gaps, which are not scored". The clause "which are not scored" can also
   apply to the opening slots. In that reading, the doc disagrees with the
   sentence before it and with the code. The batcher scores the last opening
   slot of each segment: `score[row + first - 1] = true` when `lead > 0`
   (`src/examples/tiny_stories/dataset.rs:820`). It scores no gap and no other
   opening slot.
3. `src/examples/session.rs:33` (the doc of `TrainingProgress`) says that the
   examples "read it back only under `--resume`". But `resumed_progress`
   reads the saved progress each time that a saved optimizer state loads
   (`src/examples/cli.rs:478`). Without `--resume`, it prints the saved
   training step and returns a new progress (`src/examples/cli.rs:493`). So
   the code also reads the progress without `--resume`, but it restores the
   progress only under `--resume`. The module header says "`--resume`
   restores it". The rewrite of the sentence keeps the words "read it back".

### The decisions of the user

1. The code is correct. So the two docs of item 1 change.
   `src/examples/tiny_stories/dataset.rs:86` says "The dataloaders bucket the
   train split and the validation split". The doc of `BucketedStoriesDataset`
   says "One split as batches of similar length". No other doc of either repo
   makes the claim. `../burn-mamba/examples/common/long_stories.rs:28` agrees
   with the code.
2. The sentence of item 2 splits in two: "The pair (0, 0) fills the opening
   slots and the gaps. The gaps and the other opening slots are not scored."
   Its paragraph then holds seven sentences. So it splits into two paragraphs:
   the layout of a row, then the targets and the scored positions.
3. The code is correct. So the doc of `TrainingProgress` says that the
   examples "restore it only under `--resume`", the verb of the module header.
4. The user prefers zero hard findings. So each of the two log lines gets a
   full stop (see "The two log lines").
5. The two misses of the linter (see "For the linter") go to a new ticket.
   The user writes it later.

### Choices of the sweep

- **Remove.** The doc of `AppArgs::remove_artifacts` says "remove", as the
  flag `--remove-artifacts`, `create_artifact_dir` and `std::fs::remove_file`
  do. A comment of `stories` (`src/examples/tiny_stories/dataset.rs`) says
  "remove the file by hand". The help text `cli::HELP` says `Delete` and
  `deletes` for the same flag. The linter does not read it, so it stays
  (spec, "Out of Scope").
- **Print.** The comments before the `println!` of the averaged metrics in
  the two `epoch_train` and the two `epoch_valid` say "Print". In `lm.rs`, the
  linter took `display` and `show` for one action.
- **Read back.** The passives `are read back` and `is read back` become
  active: "reads … back" (`sample.rs`, `sample/tests.rs`). The header of
  `sample.rs`, `READBACK` and `read_back` use the same words. In `trainer.rs`,
  the loss "must not read a value back to the host". In `lm_output`,
  `A longer output is read out, not trimmed` becomes "It reads a longer
  output, not a trimmed one".
- **Start.** `a run begins and ends with its story` (`lm.rs`) becomes "a run
  starts and ends with its story".
- **Device.** The old header of `device.rs` said `with the BURN_DEVICE env
  override and a built-in priority list when several are compiled in`. The
  condition could apply to the override too. Burn reads `BURN_DEVICE` in each
  std build (`../burn/crates/burn-dispatch/src/device.rs:238-240`). So the
  rewrite says that the override "can select one of the enabled backends",
  and that the priority list applies "when the build has several".
- **Lists.** Four long sentences become lists: the wrapper of `classify.rs`
  and `lm.rs`, the flags of `TrainingStepFlags` and the order of
  `probe_order`.
- No term of the two glossaries changes.

### For the linter

- The linter does not flag "run out" in "A loop sees it run out"
  (`src/examples/training.rs`). The sweep changes it to "A loop finds the
  budget spent".
- The linter does not flag a passive whose two words are on two lines of a
  comment. An example is "only the row groups up to the requested rows are /
  decompressed" in the header of `src/examples/tiny_stories/dataset.rs`. After
  a rewrap put both words on one line, the linter flagged it.

### Work files

`tmp/ste-sweep-03/` holds the lint reports before and after, the two review
reports, `build.sh`, `check.sh` and the logs of their build steps.
