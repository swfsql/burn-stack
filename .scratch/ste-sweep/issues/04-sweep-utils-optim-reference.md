# 04: Sweep: utils, optim, the reference block and the block contracts

Status: resolved

Blocked by: 01, `../burn-mamba/.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../burn-mamba/.scratch/ste-sweep/spec.md`. Follow its method, its text
that does not change, its kept findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–04 are the batches of burn-stack, and they do not block each other. Ticket
05 sweeps `.scratch/`, `docs/` and the root files last.

**What to build:** the prose of the utils, the optim modules, the reference
block, the block contracts and the crate root follows the `asd-ste100` skill.

- Scope: `src/utils/`, `src/optim/`, the reference block and the block
  contracts with their test directories, and `src/lib.rs`, 42 files.
- At the start: 47 hard findings (9 in messages) and 73 advisory findings.
- burn-mamba calls the block contracts on each of its block settings. Search
  the tests of burn-mamba too before a message changes.

**References:**

- Spec: "The method of a batch", "Text that does not change", "Kept
  findings", "Messages and copies" and "A doc that disagrees with the code".
- The terms: `GLOSSARY.md` and `../burn-mamba/GLOSSARY.md`. In burn-stack,
  ADR-0009 permits a quoted source word.
- Prior art: `.scratch/glossary-followups/issues/04-prose-sweep-utils-optim.md`.

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
42 tracked `.rs` files of the scope. The scope has no `.md` file. The linter
is the installed copy of the skill, with the change of both tickets of
`../burn-mamba/.scratch/ste-lint-misses/`. So the count at the start is
higher than the estimate of this ticket.

| Findings | Before | After |
|---|---|---|
| Hard | 48 | 0 |
| Hard, in messages | 9 | 0 |
| Advisory | 79 | 15 |

The scope has no kept finding.

Fifteen advisory findings stay. Each one is a passive whose actor is not
relevant, or an adjective after `is`:

- `src/optim/spec.rs:26` and `:78` (`is anchored`): "anchored" is the word of
  `ProjScope` and of the doc of the field `ProjSpec::path`.
- `src/reference/tests/grad_horizon.rs:489` (`are untracked`),
  `src/utils/detach.rs:11` (`is untracked`) and `:13` (`are tracked`
  `Param`s): adjectives, as the kept `are untracked` of ticket 02.
- `src/utils/capture.rs:137` (`are unshared`), `src/utils/padding.rs:9`
  (`is unspecified`), `:216` (`is untouched`) and `src/utils/untied.rs:119`
  (`is untied`): adjectives.
- `src/utils/schedule/mod.rs:177` (`is undefined`): an adjective in a
  message. `GradHorizon::Depth is undefined for Schedule::Custom` is the
  `should_panic` text of a test, so the message stays.
- `src/utils/class/mod.rs:126` (`is defined`) and `:420` (`is spliced`): the
  actor is the placement or the caller, and neither is relevant.
- `src/utils/untied.rs:38` (`is fixed`) and `:39` (`was built`): the bold
  lead of a rule, and a clause of the same rule.
- `src/utils/untied.rs:66` (`is unchanged`): the linter reads `is` and
  `unchanged` across the code span `` `module` ``.

### The messages

The hard findings in messages are seven phrasal verbs in
`src/utils/capture.rs`, a semicolon in `src/optim/report.rs` and a long
sentence in `src/utils/untied.rs`. Five other messages change, each for an
advisory finding. A search of the tests of both repos finds a match for one
part of one changed string: `describe_reports_every_parameter`
(`src/reference/tests/muon.rs`) matches `on muon:`, and that part stays. The
test also counts the lines of the report, and the line count stays.

| Location | Old string | New string |
|---|---|---|
| `src/utils/capture.rs`, 7 calls | In the git history of this ticket | `the caches are always present` |
| `src/optim/report.rs:115` | In the git history of this ticket | `total params: {total}, on muon: …` |
| `src/utils/untied.rs:96` | `… built for {n_applications}: an untied parameter has no copy for an application added after the stack was built` | `… built for {n_applications}. An untied parameter has no copy for an application added after the build of the stack.` |
| `src/utils/untied.rs:71` | `a real layer is built for at least one application` | `a real layer needs at least one application` |
| `src/utils/init.rs:94` | `InitPolicy::residual_paths is set but residual_depth is not` | `InitPolicy has residual_paths but no residual_depth` |
| `src/utils/init/tests.rs:93` | `in_proj was rescaled too: {entry}` | `the policy rescaled in_proj too: {entry}` |
| `src/reference/tests/grad_horizon.rs:944` | `nor the caches it hands back` | `nor the caches that it returns` |
| `src/reference/tests/class.rs:62` | `the Start token is emitted before the first user token` | `the network emits the Start token before the first user token` |

The new form of the capture message is the form of its sibling, `the input
is always present`. The message of `src/utils/untied.rs:96` keeps "build" as
a noun, as in "The count is fixed at the build" (`src/utils/untied.rs:38`)
and "applied to a module after its build" (`src/utils/init.rs:1`).

### The copies

- The first sentence of the doc of `fprim::san` is a copy of the first
  sentence of the doc of `sanity` (`src/modules/misc/sanity.rs`, outside the
  scope). Both change from `is set` to `` is `true` ``, because `DENY_NAN` and
  `DENY_INF` are `bool` constants.
- The seven capture messages change together.
- The rest of burn-stack and all of burn-mamba have no copy of a changed
  paragraph. `../burn-mamba/src/family/mod.rs` has a paragraph near the
  rank-3 paragraph of `src/optim/mod.rs`, but it is not a copy.

### Docs that disagree with the code

The sweep changes neither side, and the words of each disputed sentence
stay. The user decides each one.

1. The doc of `MuonPlan` (`src/optim/mod.rs:165`) says that the plan "is
   built from a *block* config (through `BlockConfig::muon_projections`)". But
   the two `muon_plan` of `src/modules/shape.rs` also add specs that the
   shape holds:
   - `NetworkShape::muon_plan` (`:192`) calls `.with_mlp(self.mlp.as_ref())`,
     which adds the specs of the feed-forward (`mlp.fc1.weight`, …).
   - The bidirectional `muon_plan` (`:343`) adds `CatLinear.weight` when an
     output merge is `OutputMergeConfig::CatLinear`.

   The claim "so it is independent of the network topology" can still hold,
   because the specs match as path substrings.
2. The header of `src/utils/padding.rs` says that a block runs through
   `Padding::in_row_order`, "and its output is scattered back". On the way
   out, the code is a gather by `local`: `y_bsd.gather(1, local_bs…)`
   (`src/utils/padding.rs:249`). The inverse permutation, `src_bs`, serves
   the gather on the way in (`:237` and `:244`). The doc of `in_row_order`
   says "a gather on the way in and one on the way out". A gather by `local`
   gives the same result as a scatter by `src`, so only the word differs.
3. The header of `src/optim/segmented.rs:9` says "So each sub-projection is
   orthogonalised (and shape-LR-adjusted) on its own". But a sub-projection
   can stay on the fallback optimizer: `SubProjection::adamw`
   (`src/optim/spec.rs:67`) gives `muon: false`. The same header says that a
   `Segmented` can have "all its sub-projections on AdamW" (`:16`). So only
   the sub-projections on Muon are orthogonalised.

### The decisions of the user

The code is correct in each of the three cases. So the three docs change:

1. The doc of `MuonPlan` says that the plan holds the specs of a block
   config. It also says that the `muon_plan` of a shape adds the specs of
   the shape: the feed-forward (`NetworkShape::muon_plan`, through
   `MuonPlan::with_mlp`) and a `CatLinear` output merge
   (`BidiShape::muon_plan`). The plan stays independent of the network
   topology, because the specs match as path substrings.
2. The header of `src/utils/padding.rs` says "A second gather returns its
   output to the batch-wide order", the word of the doc of `in_row_order`.
3. The header of `src/optim/segmented.rs` says "So Muon orthogonalises each
   sub-projection that it owns (and adjusts its LR to its shape) on its
   own".

No other doc of either repo makes these claims.

### Choices of the sweep

- **Check.** `src/optim/report.rs` and `src/utils/capture.rs` used two
  verbs for one action. Both use `check`, the verb of the module header of
  `report.rs`. The git history of this ticket gives the other verb.
- **Read.** The linter takes `gets` of `src/utils/detach.rs` and the old verb
  of `:28` for one action. They are two actions: `gets the gradient` (`:20`)
  means "receives". So only the old verb of `:28` changes, to `read`. The
  git history of this ticket gives the old verb and its meaning.
- **Return.** `give back`, `gives back` and `hands back` become `return` or
  `returns`. In `src/utils/capture/weights.rs`, a phrasal verb becomes "each
  parameter has … again", so that the doc names no new actor.
- **Restore.** The comments of `CapturedStep::capture` and of the mapper
  `OptimizerUpdate` (`src/optim/sgd.rs`) use `restore` for a phrasal verb.
  The module header of `capture.rs` uses the same verb. The git history of
  this ticket gives the phrasal verbs of this item and of the item before it.
- **Lift.** The doc of `grad_horizon_keeps_the_checkpointing_strategy` uses
  the words of `src/modules/layers.rs:231`: "a cut lifts the values that its
  untracked span took to the inner backend". Ticket 02 names this phrasal
  verb.
- **Must.** The header of `src/utils/capture.rs` keeps the strength of the
  requirements: the step "must keep one batch size", and its control flow
  "must not change". The next sentence says "also must not".
- **Lists.** Three long sentences become lists: the header of `src/lib.rs`,
  the bespoke parameters of `src/utils/init.rs` and the two magnitudes of
  `div_eps`.
- **Paragraphs.** Three long paragraphs split by topic: the rank-3 paragraph
  of `src/optim/mod.rs`, the `prime` paragraph of `src/utils/class/mod.rs`
  and the class-latent paragraph of `src/utils/packing.rs`.
- No term of the two glossaries changes.

### Work files

`tmp/ste-sweep-04/` holds the file list of the scope, the lint reports before
and after, the log of `./scripts/check.sh`, the two review reports and two
width scripts.

### The review

The Standards review (`tmp/ste-sweep-04/review-standards.md`) finds four
hard violations, and the sweep fixes each one:

- The rewrite of the two disputed sentences changed their words. The sweep
  kept their old words until the decisions of the user.
- Six paragraphs had a line wider than the file, or a short line that the
  rewrite did not rewrap. The sweep rewraps them.
- The header of `capture.rs` stated two requirements as facts. It says
  "must" again.
- In the header of `class/mod.rs`, "Otherwise it opens the next window" read
  as the window. It says "the marker".

Of its seven judgement calls, six change:

- "any of these shapes" (`src/optim/mod.rs`),
- the words of `layers.rs` for the lift (`grad_horizon.rs`),
- "Consider every tensor" under `# Safety` (`capture.rs`),
- "the `std` of the policy" (`init/tests.rs`),
- "a clone with `params` narrowed" (`untied.rs`),
- "must have that dtype again" (`src/utils/mod.rs`).

The noun "build" in the message of `view` stays (see
[The messages](#the-messages)).

The Spec review (`tmp/ste-sweep-04/review-spec.md`) finds no scope creep,
and it accepts the reason for the change of `src/modules/misc/sanity.rs`. The
sweep fixes each of its five findings:

- The advisory count at the start is 79, not 81.
- The evidence of disputed doc 2 named the wrong gather as the inverse
  permutation.
- The reason of the choice **Read** was wrong (see
  [Choices of the sweep](#choices-of-the-sweep)).
- In the header of `capture.rs`, "the caches of the capture" could mean the
  caches after the capture. It says "the `caches` argument of the capture".
- The active voice in `src/optim/segmented.rs:9` named Muon as the actor for
  each sub-projection. The sentence disagrees with the code, so the sweep
  kept its old words until the decision of the user (disputed doc 3).
