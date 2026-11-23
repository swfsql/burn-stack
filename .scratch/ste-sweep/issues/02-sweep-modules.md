# 02: Sweep: the modules

Status: resolved

Blocked by: 01

Spec: `../burn-mamba/.scratch/ste-sweep/spec.md`. Follow its method, its text
that does not change, its kept findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–04 are the batches of burn-stack, and they do not block each other. Ticket
05 sweeps `.scratch/`, `docs/` and the root files last.

**What to build:** the prose of the modules follows the `asd-ste100` skill.

- Scope: `src/modules/`, 48 files.
- At the start: 46 hard findings and 79 advisory findings. No message has a
  finding. `src/modules/layers.rs` alone has 14 hard findings.
- burn-stack code is block-agnostic (ADR-0002). A doc can cite a family, but
  the rewrite adds no Mamba-specific claim.

**References:**

- Spec: "The method of a batch", "Text that does not change", "Kept
  findings", "Messages and copies" and "A doc that disagrees with the code".
- The terms: `GLOSSARY.md` and `../burn-mamba/GLOSSARY.md`. In burn-stack,
  ADR-0009 permits a quoted source word.
- Prior art: `.scratch/glossary-followups/issues/02-prose-sweep.md`.

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
48 tracked `.rs` files of `src/modules/`. The scope has no `.md` file.

| Findings | Before | After |
|---|---|---|
| Hard | 46 | 0 |
| Advisory | 79 | 31 |

The scope has no kept finding.

### The messages

Three message strings change. Each one had an advisory finding (a passive or
a technical phrasal verb). A search of the tests of both repos finds no match
for a part of these strings.

| Location | Old string | New string |
|---|---|---|
| `src/modules/bidi.rs:306` | `Bidi virtual layers are used in pairs` | `Bidi uses its virtual layers in pairs` |
| `src/modules/bidi.rs:312` | `Bidi layers are used in pairs` | `Bidi uses its layers in pairs` |
| `src/modules/residuals/tests.rs:93`, `:246` | `the join gives back the extra value of the layer` | `the join returns the extra value of the layer` |

### The copies

These paragraphs have near-copies in the scope. Each set changes together:

- The docs of the untied parameters: `layer.rs`, `layers.rs`, `bidi.rs` and
  two in `shape.rs`. All five use the active voice.
- The note on the dtype of the loss in `bce.rs`, `cross_entropy.rs` and
  `mse.rs`.
- The comment on `eps` inside the log in `bce.rs` and `cross_entropy.rs`.
- The docs of `a_small_probability_keeps_its_loss` in `bce/tests.rs` and
  `cross_entropy/tests.rs`.
- The sentence on the output of each row of a right-padded batch, in the
  `forward` docs of `Layers` and `BidiLayers`.
- The two paragraphs on the untracked spans of the mask in `layers.rs`.

The rest of burn-stack and all of burn-mamba have no copy of a changed
paragraph.

### Docs that disagree with the code

The sweep changes neither side. The user decides each one.

1. `src/modules/norm/rms_norm/tests.rs:106`: the doc of
   `rms_norm_gradient_finite_on_collapsed_slice` says "The forward guards the
   *division* (`rms + div_eps`)". The forward computes `x / rms(x)`
   (`src/modules/norm/rms_norm.rs:59`). `rms` puts `ε` inside the root:
   `((x.clone() * x).mean_dim(D - 1) + eps).sqrt()`
   (`src/modules/norm/mod.rs:30`). So the code has no `rms + div_eps` term.
   The header of `src/modules/norm/rms_norm.rs:3` also gives
   `rms(x) = √(mean(x²))`, with no `ε`.

   The first sentence of the test doc had a hard finding (26 words). The sweep
   splits that sentence and rewraps the paragraph, but the words of the
   disputed sentence stay.
2. `src/modules/mod.rs:8` says "Everything here is parameterised by the block
   `M: `[`Block`]". The shared pieces of the module have no block parameter.
   Examples are `Silu`, `RmsNorm`, `GatedMlp` and `MseLoss`. Only the
   composition types are generic over `M`.

### The decisions of the user

1. The user asks for tests of the guard of `rms` at its breakage points. A
   probe finds no NaN at zero, and the user then selects a clamp before the
   root in place of the additive `ε`.
   `.scratch/rms-guard/issues/01-clamp-before-the-root.md` makes that change.
   It also makes the docs of item 1 describe the clamp.
2. The sentence names only the types that take the block. This ticket
   changes `src/modules/mod.rs:8` to "The layer, the layer stacks and the
   networks take the block `M: `[`Block`] as a type parameter". `Residuals`
   has no block parameter, so the sentence does not name it.

### A term

`src/modules/norm/mod.rs` and `src/modules/norm/rms_norm/tests.rs` use "row"
for one slice along the last axis, for example "a zero row". The glossary
gives **Row** as one sequence of a batch. The term map of
`.scratch/glossary-followups/issues/02-prose-sweep.md` keeps "row" for a row
of a matrix, and these slices are the rows of the matrix view. So the sweep
keeps the word.

### For ticket 04

`src/reference/tests/grad_horizon.rs:1031` uses the same phrasal verb as the
old text of `src/modules/layers.rs:229`. In `layers.rs`, the sweep writes "a
cut lifts the values that its untracked span took to the inner backend". The
two sentences are not copies, so this ticket does not change the test doc.

### For the linter

- The linter does not flag `blow up` in `src/modules/loss/bce.rs`. The sweep
  changes it to "grow without bound".
- In `src/modules/loss/bce.rs`, the linter reads a formula comment and the
  next sentence as one sentence. The split of that sentence is enough, so the
  formula line stays as it is.

### The recount on the new linter

The two tickets of `../burn-mamba/.scratch/ste-lint-misses/` change the linter
after this sweep. So the scope gets a second count and a second rewrite.

The installed copy of the skill holds the change of both tickets. The repro
of each ticket gives its findings, and `--selftest` passes. The repro files
are in `tmp/ste-sweep-02/`.

The scope has 49 tracked `.rs` files. Commit `964345f` (the clamp of
`.scratch/rms-guard/`) adds `src/modules/norm/tests.rs`. The new linter gives
the same counts at `ff44142`, the commit that resolves this ticket.

| Findings | Before | After |
|---|---|---|
| Hard | 2 | 0 |
| Advisory | 38 | 35 |

The two hard findings come from ticket 01 of `ste-lint-misses`:

- `src/modules/cache.rs:44`, the summary of `cache_detach`: `cut off` changes
  to `detached from`, the verb of the name. The paragraph rewraps.
- `src/modules/layers.rs:47`, the doc of `Layers::grad_horizon`:
  `is what runs out of memory` changes to `is what fills the memory`.

Ticket 02 of `ste-lint-misses` gives eight passives in the scope. Three
change to the active voice, because each one has a relevant actor:

- `src/modules/hnet/smooth.rs:49`: `smooth` clamps `p_bk`. Two sentences
  later, `smooth` also replaces `It`.
- `src/modules/layer.rs:35`: every other method acts on `self`.
- `src/modules/loss/l2warp.rs:4`: the penalty pulls the winning logit.

Five stay:

- `src/modules/bidi.rs:8` (`is written`): the actor is the author, and the
  author is not relevant.
- `src/modules/layers.rs:295` (`is suppressed`) and `:300` (`is gated`): these
  are near-copies of `src/modules/bidi.rs:276` and `:281`, and the sweep kept
  those two.
- `src/modules/network.rs:314` (`is rounded`): the sweep kept the three other
  `is rounded` of the scope.
- `src/modules/residuals.rs:309` (`is untracked`): an adjective, as the kept
  `are untracked` of `src/modules/layers.rs:827`.

No message changes. No changed paragraph has a copy in either repo. The
recount finds no doc that disagrees with the code, and it has no kept finding.

The prose step of `scripts/check.sh` fails on two ticket files. Each file is
new at the merge base with `origin/main`, so the step starts it at zero. The
new linter flags the quoted `blow up` of [For the linter](#for-the-linter) and
the two quoted `run out` of ticket 03. A quoted word is a kept finding, but the
step cannot accept one. So the recount puts the three words in code spans, as
ticket 01 of `ste-lint-misses` does.
