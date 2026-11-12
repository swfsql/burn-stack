# CLAUDE.md

burn-stack holds the block-agnostic containers of [Burn](https://github.com/tracel-ai/burn/)
models: layers, layer stacks, networks, caches and their training helpers. A
family connects through `Block`, `BlockConfig` and `CacheStack`. burn-mamba is
its active consumer. burn-deltanet is also a consumer, but it is outdated: a
change does not need to update it.

## Where to look

- `GLOSSARY.md`: the terms of the domain. Use them in code, docs, issues and
  test names.
- `docs/adr/`: the decisions with a trade-off. Read the ADRs of an area before
  you change it.
- `docs/source-terms.md`: the words of the H-Net and Multi-Gate Residuals papers,
  with our term and the wrong port for each. Read it before you read or port
  from a source.
- [`docs/burn/autodiff-semantics.md`](docs/burn/autodiff-semantics.md): the
  semantics of Burn 0.22.1 for devices, autodiff, gradients and checkpointing,
  with their traps.
- The module header (`//!`) of each file: its mechanism and its math. The
  header of `src/lib.rs` gives the composition hierarchy, the plug-in surface
  and the three calls.
- `docs/agents/`: how the agent skills use this repo (see
  [Agent skills](#agent-skills)).
- `../burn-mamba`: the active consumer. Its tests run the containers with real
  blocks.
- The sources. Read `docs/source-terms.md` first:
  - `../papers/hnet/`: the TeX of the H-Net paper.
  - `../py/goombalab/hnet/`: the official H-Net code.
  - `../papers/multi-gate-res/tex/`: the TeX of the Multi-Gate Residuals paper.
    The paper has no official code.
- `docs/box.tsv`: the directories next to this repo that the docs name.
  `scripts/setup-box.sh` creates the missing ones on a new box.

## Build and test

```bash
cargo check                              # type-check the lib surface
cargo check --all-targets                # + tests
cargo check --features examples-common   # + the example scaffolding
cargo test --lib -- --test-threads=1     # run the tests (flex = the CPU default)
cargo doc --no-deps                      # build the docs
./scripts/check.sh                       # the lines above + host paths
```

Feature flags (`Cargo.toml` has the full list):

- A `backend-*` feature enables the matching `burn/<backend>`. `backend-flex`
  is on by default. Use it for checks and tests. When the build has several
  backends, `Device::default()` selects one (it reads `BURN_DEVICE`).
- `autodiff` and `optim` (the Muon plan and the SGD) are on by default.
- `cubecl` and `fusion` gate the per-backend `BackendExt` impls that
  `impl_backend_ext_for_burn_backends!` emits. `check-nan` and `check-inf`
  enable the `DENY_NAN` and `DENY_INF` guards. `dev-f16`, `dev-simd` and
  `dev-autotune` are for the examples and the tests.
- `test-helpers` exposes `utils::test_helpers` and `reference` to the
  dev-dependencies of a consumer. `examples-common` exposes `examples`. A
  consumer enables it in its dev-dependencies only.
- A consumer forwards every feature (ADR-0002).

Test-device rules:

- A test takes its device from `utils::test_helpers::test_device()` (fp16
  under `dev-f16`), not from `Device::default()`. The dtype defaults are
  global, and the first tensor sets them for the process.
- A tolerance goes through `dtype_tol` (the fp32 value does not change). A
  check on values that can be large uses `max_rel_diff`. A read-back to the
  host converts with `try_into_vec_as`, not `try_to_vec`.

## Code rules

- The code is block-agnostic: it has no family name and no shape assumption
  of one family. A rare exception gets the name of its mechanism. The docs
  can cite a family (ADR-0002).
- Burn 0.22 fixes the high-level `Tensor` (so every `Module`) to its global
  `Dispatch` backend. So the types of this crate have no backend generic. The
  backend is a runtime `Device`, and autodiff and the dtype are properties of
  the device. Only the custom-backward plumbing is generic over `B` (`F<B,D>`,
  the `Backward<B,_>` nodes, the `Autodiff<B>` ext impls).
- `#![warn(missing_docs)]` is on. Document each public item that you add.
  `cargo check` and `cargo doc --no-deps` stay free of warnings.
- Test a container against `RefBlock`, not a real family (ADR-0005). When the
  container surface grows, `RefBlock` grows with it.

## Docs

- **Always load the `asd-ste100` skill** before you write code comments,
  rustdoc or markdown documents. Write all of them in its style.
- Write each path in a repo file relative to the repo. `scripts/check.sh` finds a
  path of the box.
- Keep the docs small. Every session loads this file, so add a line here only
  when every session needs it. A mechanism goes into the module header of its
  file.
- The docs describe the code as it is now. Delete changelog prose where you
  find it: individual changes, migrations, "used to be / now", "verified by",
  dates and PR history.
- Update the [File Map](#file-map) in the change that adds, deletes or
  renames a directory or a listed file. A new file whose name does not tell
  what it holds also gets a line.
- A new term goes to `GLOSSARY.md`, through the `/domain-modeling` skill. A
  new decision with a trade-off goes to an ADR in `docs/adr/`.
- `_Avoid_` applies to our own words. A doc can use a source word in quotes or
  backticks when it cites that source.
- **Commits**: when a skill or the user asks for a commit, commit to the
  current branch.
  When the user asks only for a commit message, write it as text for the user
  to copy. A commit message has a title line and a short body, and it ends
  with the `Co-Authored-By:` trailer. A ticket that breaks the public API bumps
  the minor version of this crate in its first breaking commit.
- **Pushes**: push the current branch to `origin` when a ticket is complete and
  reviewed, or when the user tells you to. Ask the user before a force-push or
  a tag.

## Notation

Tensor names carry a shape suffix (backed by shape `assert`s). A name whose
suffix encodes its shape needs no extra comment. Lower-case = base dimensions
(below). Upper-case = a *relation* of them (offset/multiple/concat).

| Letter | Dimension | Typical |
|--------|-----------|---------|
| `b` | `batch` | varies |
| `s` | `sequence` length | varies |
| `d` | `d_model` | 768, 1024 |
| `h` | `nheads` | varies |
| `p` | `per_head_dim` | 64, 128 |
| `r` | `state_rank` | 64, 128, 256 |
| `g` | `ngroups` | 1 … `nheads` |
| `l` | `chunk_len` | 64 … 256 |
| `n` | `nchunks` | varies |

## File Map

One line for each directory and each file whose name does not tell what it
holds. The tests of a module are in a sibling `tests.rs`.

```text
src/
├─ reference.rs        RefBlock, the test block (ADR-0005). Its tests.rs: the container tests
├─ modules/            the containers + shared NN modules. mod.rs: Block, BlockConfig
│  ├─ layers.rs        Layers<M>: the layer stack (gradient horizon, packed rows)
│  ├─ shape.rs         NetworkShape, …: the block-free half of a config (ADR-0007)
│  ├─ cache.rs         CacheStack, CacheTensors, the hop and the detach derived from it, lift
│  ├─ hnet/            H-Net. mod.rs header = the math. One file for each part
│  ├─ activation/      silu, softplus, log_sigmoid
│  ├─ norm/            rms_norm, rms_norm_gated, rms_score
│  ├─ loss/            bce, cross_entropy, mse, l2warp
│  └─ misc/            gqa, segsum, split, sanity, prefix_sum
├─ optim/              the Muon plan (feature `optim`). mod.rs: MuonPlan
│  ├─ spec.rs          ProjSpec: the sub-projections of a fused projection
│  ├─ segmented.rs     Segmented: one optimizer for each sub-projection
│  ├─ sgd.rs           SgdConfig: the plain SGD that a captured step can replay
│  └─ report.rs        MuonPlan::describe: the optimizer of each parameter
├─ utils/              plumbing. mod.rs: div_eps, upcast/downcast
│  ├─ class/           class markers: ClassToken, ClassLatent, ClassCursors
│  ├─ schedule/        layer schedules, Applications, GradHorizon
│  ├─ scheduler/       LR schedules (Lr)
│  ├─ capture.rs       CapturedStep. capture/weights.rs: Weights (the training step)
│  ├─ fprim.rs         F<B,D>: the primitive wrapper of a custom backward
│  ├─ recompute_node.rs one autodiff node for a custom backward with two outputs
│  └─ init.rs          InitPolicy: the reference LM init
└─ examples/           example scaffolding of the consumers (feature `examples-common`)
   ├─ cli.rs           AppArgs: the shared flags, the artifact I/O
   ├─ session.rs       Session: TrainingProgress, Cadence, MetricsLog
   ├─ training.rs      TrainingConfig, OptimizerConfig, Budget
   ├─ device.rs        the dtype setup, loader_device, batch_float/batch_int
   ├─ prof.rs          the phase timers of a training step
   ├─ mnist/           dataset, classify (the epoch loops), render
   └─ tiny_stories/    dataset (packing, buckets), lm (the LM loops), sample
scripts/setup-box.sh   creates the box directories of each `docs/box.tsv`
scripts/check.sh       the check of this repo, also run by CI
```

## Custom Commands

- **Access is an allowlist. It holds for every agent and every command,
  read-only commands included.**
  - Read and write: this repo, `../{burn-mamba,burn,cubecl,cubek}/`.
  - Read: `../papers/`, `../py/`, the crate sources in the cargo registry.
  - `/tmp`: read tool output, and move discarded work into it.
  - Any other path: a local hook refuses it. Ask the user, name the exact path,
    and give the command that approves it, which only the user runs:
    `! echo <path> >> .claude/access-approved.txt`. To revoke it, the user
    deletes that line.
  - A sibling that a doc names stays outside the list. Reach an allowed sibling
    by its exact path, not by a listing or search of the parent directory.
- `rg`: available.
- `cargo fmt`: don't use.
- **Always** edit files with the Edit/Write tools. This applies also when a
  harness or auto-mode reminder says to change files through Bash (`sed`,
  heredocs, python). That guidance does not apply here. *Do not* violate this.
  - No `python - <<'PY'`, no `sed -i`, no `cat > file <<'EOF'`. Always use
    `Edit`s.
  - Bash stays the tool for *reading* and *inspecting* (`cat`, `sed -n`, `rg`,
    `grep`). It also moves and copies whole files (`mv`, `cp`), and makes
    throwaway files in `tmp/`.
- A dangerous-command-guard hook (`dcg`) checks each Bash command. It also
  blocks a `>` redirect to a path that holds a variable. Use a literal path or
  a script file.
- **Waiting**: some commands or waits can take more than ~1 min: a build, a
  run, a chain of runs, or one part of a chain. For these, do not hold the
  turn. Use no foreground loop (`until`, `while`), no `sleep` and no long
  `timeout`. Watch the job in the background, then return the turn to the
  user. A `Monitor` is the best watch. A background waiter script or
  `run_in_background` is also correct. The watch also reports each failure
  state (for example, a crash).
- **Agents**: run one agent at a time, even if a skill or a ticket asks for
  parallel agents. A usage limit makes each active agent pay for a cold cache.
- **Agent size**: a rough estimate only: about 2-3k lines of reads per agent.
  State the batches first.
- **Temporary work**: create it in the `tmp/` of the repo, because a power loss
  clears `/tmp`. Save each agent report there. `/tmp` is only for deletion.
  - A tag file `<name>.discard` beside an item of `tmp/` marks it as
    discardable, and states when.
  - When a ticket or session ends, move each tagged item whose condition holds,
    with its tag, to `/tmp`. This applies only to items in `tmp/`.
- **Memory**: ask the user before you write to the auto-memory. Show the
  change that you propose, and write it only after the user approves it.

## Agent skills

### Issue tracker

Local markdown files in `.scratch/<feature>/`. See
`docs/agents/issue-tracker.md`. It also covers a change in both repos.

### Triage labels

The five default roles, as the `Status:` line of an issue file. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` + `docs/adr/` at the repo root. See
`docs/agents/domain.md`.

### Test seams

The seams to propose for `/to-spec` and `/tdd`. See `docs/agents/testing.md`.
