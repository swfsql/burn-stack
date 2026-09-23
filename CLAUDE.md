# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What This Project Is

`burn-stack` is the block-generic composition layer on top of
[Burn](https://github.com/tracel-ai/burn/). It holds everything *around* a
sequence-mixing block:

- layers, (virtual-)layer stacks, bidirectional pairs,
- latent/vocab networks, multi-gate residuals, class tokens,
- schedules, the serialisable network shapes, the Muon plan.

It holds **nothing** about a particular mixer.

**It must stay family-agnostic.** Put no `mamba`, attention or convolution
specifics here. A block plugs in through `Block` / `BlockConfig` /
`CacheStack`. If a change needs a name, a shape assumption or a doc reference
from one family, put it in the crate of that family.

[`burn-mamba`](../burn-mamba) is its first consumer.

## Build & Test Commands

```bash
cargo check                 # type-check the lib surface
cargo check --all-targets   # + tests
cargo test --lib -- --test-threads=1 # run tests (any backend; flex = CPU default)
cargo doc --no-deps         # build docs
```

- **Feature flags select the backend**:
  `backend-{flex,flex-rayon,cpu,wgpu,webgpu,metal,vulkan,cuda,rocm,tch-cpu,
  tch-gpu,remote,ndarray}` and `backend-ndarray-blas-*`. Use flex for checks
  and tests (it is on by default). Each feature enables the matching
  `burn/<backend>`. Several can be compiled in at once, and
  `Device::default()` resolves which one to use (it reads `BURN_DEVICE`).
- `autodiff`/`optim` are on by default. `optim` (Muon parameter groups)
  implies `burn/optim` + `burn/std`. `cubecl`/`fusion` gate the per-backend
  `BackendExt` impls that the macros emit. `check-nan`/`check-inf` enable the
  `DENY_NAN`/`DENY_INF` guards. `dev-f16`/`dev-simd`/`dev-autotune` are
  example/test conveniences.
- `test-helpers` exposes `utils::test_helpers` and `reference` to the
  dev-dependencies of a downstream crate.
- `examples-common` exposes `examples`. It pulls `burn/train` +
  `burn/dataset` and the download/CLI/image/parquet/rand crates. A consumer
  enables it in its **dev**-dependencies only. `ModelConfigExt` is
  intentionally *not* behind it: consumers implement it in their own libs,
  which cannot depend on a dev-only feature.
- A consumer crate must **forward** every one of these features (see
  `burn-mamba/Cargo.toml`). The `backend-*` cfgs are evaluated where
  `impl_backend_ext_for_burn_backends!` expands, which is in the *calling*
  crate.
- `Cargo.toml` `[patch]`es every burn and cubecl crate to the swfsql forks
  with the tracel-ai/burn#5772 memory fix (a captured graph holds the memory
  of one pass, not ~3). A crate that is missing from the list links a second
  copy.

## Writing Style

- **Always load the `asd-ste100` skill** before you write code comments,
  rustdoc or markdown documents. Write all of them in its style.

## Documentation Maintenance (CLAUDE.md)

- Keep this file **as small as possible**, but still usable. Point to the
  source (each module header carries the details). Do not copy it here. When
  a source file changes, update its one entry. Do not grow this file.
- **Never use this file as a changelog.** It describes the code as it *is
  now*. It must not record individual changes, migrations, "used to be /
  now", "verified by", dates, or PR history. If you find changelog-style
  prose, delete it.
- Always be **extremely succinct** when you add content to this file.
- When a source file is added, removed or changed, prepare an update to its
  entry in the [File Map](#file-map). A change that is specific to one mixer
  family updates the CLAUDE.md of that family's crate instead. Important
  rule: do this at the end of the work. If you have not read this file by
  then, **do not** read it: the context is still big from the work, and
  reading big files then is expensive. Instead, write a `tmp.md` file with the
  new File Map entry and a short overview of the most important aspects of
  the created/removed/updated files. After a full context reset (the user
  triggers it), this file gets its update.
- **Commit messages**: the user can ask for a commit message for the session.
  **Write the message as text only** (a title line + a short body) for the
  user to copy. Do NOT run `git commit` or any git command to create the
  commit. End the message with the `Co-Authored-By:` trailer.

## File Map

Tests are in a sibling `tests.rs` (or `<module>/tests.rs`). This map does not
list them, except the contract suite.

```text
src/
├─ lib.rs            crate root: module decls, prelude, DENY_NAN/DENY_INF guards
├─ reference.rs      RefBlock: a minimal worked `Block` (a gated EMA mixer + its
│                    CacheTensors), the test fixture of the crate.
│                    `reference/tests.rs` is the container contract suite
│                    (feature `test-helpers`)
├─ modules/          composition + shared NN modules
│  ├─ mod.rs         Block / BlockConfig traits (the whole plug-in surface)
│  ├─ layer.rs       Layer<M>: Pre-LN block M(RMSNorm(·)) + optional norm2/mlp.
│  │                 Returns the total delta of the layer (Layers adds the outer
│  │                 residual). application(k): the view that application k runs
│  │                 (LayerUntied)
│  ├─ layers.rs      Layers<M>: virtual-layer stack over real weight sets.
│  │                 grad_horizon truncates BPTT to a tracked-layer mask
│  │                 (forward/step/prime cut alike). only_start_latents (the
│  │                 capture gate)
│  ├─ mlp.rs         GatedMlp: SwiGLU feed-forward interleaved with the mixer.
│  │                 from_hidden_ratio = the Llama ⅔·ratio·d_model sizing rule
│  ├─ model_config.rs ModelConfigExt: config → module + its Muon plan. The seam
│  │                 that a model-agnostic driver builds against (consumers
│  │                 impl it)
│  ├─ multi_gate.rs  Multi-Gate Residuals (Standard | MultiGate): accumulate,
│  │                 then mix
│  ├─ network.rs     LatentNetwork (optional final norm) / VocabNetwork
│  ├─ shape.rs       NetworkShape (+ LatentShape/VocabShape/BidiShape): the
│  │                 serialisable, block-free half of a model config. The
│  │                 builders carry `C` and cannot derive Config, so the config
│  │                 of a consumer is the pair `{ shape, block }`. init/muon_plan
│  │                 over any BlockConfig
│  ├─ bidi.rs        BidiLayers<M> + BidiLayerPair<M> + OutputMerge
│  ├─ cache.rs       CacheStack trait (+ per-slot inner/from_inner, whole-stack
│  │                 detach() to carry state across a gradient boundary).
│  │                 CacheTensors: a pairwise TensorZip traversal per cache type
│  │                 (into_owned_buffers, assign_in_place). Impls: `()` = no
│  │                 cache, `Vec`, `Tensor<D>`, a pair = state beside a cache
│  ├─ activation/    silu, softplus, log_sigmoid (dtype-aware)
│  ├─ norm/          rms_norm (also usable as QK-Norm), rms_norm_gated, rms_score
│  ├─ loss/          bce, cross_entropy, mse, l2warp (max-logit penalty, added
│  │                 to the gradient only)
│  └─ misc/          gqa, segsum, split, sanity
├─ examples/         example scaffolding shared by the consumer crates
│  │                 (feature `examples-common`, off by default, dev-only)
│  ├─ cli.rs         AppArgs: every flag that the examples share (training-config
│  │                 overrides with the optimizer, --no-graph, --resume, budget,
│  │                 cadence), artifact dir, model/optim/config I/O (the optim
│  │                 saved with its TrainingProgress, progress.json). A flag
│  │                 replaces a loaded optimizer, but panics if that ignores a
│  │                 saved optim state. The flags of one example, after `--`:
│  │                 `extra` + `finish_extra`
│  ├─ session.rs     Session, what every epoch loop threads: TrainingProgress
│  │                 (step, epoch, batch. A resumed epoch finishes from a fresh
│  │                 shuffle), Budget, Cadence, MetricsLog (metrics.jsonl)
│  ├─ device.rs      Device dtype configuration (`dev-f16`) + FloatElement.
│  │                 loader_device (dataloader workers build on the host) +
│  │                 batch_float/batch_int (the move of the loop to the device)
│  ├─ trainer.rs     Trainer: one LossFn. It steps eagerly through the module
│  │                 optimizer, or (plain SGD + graphs) is captured whole
│  │                 (Weights + SgdConfig::step, the lr an input). Other shapes
│  │                 step eagerly into the same weights
│  ├─ training.rs    TrainingConfig + OptimizerConfig {fallback: AdamW | SGD,
│  │                 optional Muon}, OptimizerKind (the four, `of` = their
│  │                 defaults). Budget: the --max-batches / --max-seconds run
│  │                 caps (not config state, the clock starts at the first step)
│  ├─ mnist/         dataset.rs (download + batching), classify.rs (the epoch
│  │                 loops + the MnistModel seam, `train_step` = the whole step
│  │                 of a Trainer), render.rs (a digit beside its class
│  │                 distribution, as text or PNG)
│  └─ tiny_stories/  dataset.rs: character corpus (alphabet, whole-parquet
│                    download + text cache, one story per item, batches padded
│                    to whole windows with a per-slot `scored` count).
│                    lm.rs: TinyStoriesConfig + its corpus-flag Overrides, the
│                    FrontierGate, the epoch loops + the cache-carrying LmModel
│                    seam. lm_output scores any extra output positions that the
│                    model spliced in against the first character of the story,
│                    and masks the padding out at the fixed window shape
│                    (PAD_TARGET rows, the mean normalised on device,
│                    PerCharLoss for validation). Class markers are offered,
│                    never assumed.
│                    sample.rs: one prime/prefill/decode sampler over
│                    VocabNetwork<M>. `decode`, the shared loop, draws on the
│                    device (the token is step state) and replays a
│                    CapturedStep. A `Prefill`, held across prompts, feeds
│                    right-padded fixed-shape chunks after a kept opening, with
│                    one captured chunk for all
├─ optim/            Muon parameter groups (feature `optim`): an allowlist, not
│                    a denylist. + the capturable SGD
│  ├─ mod.rs         MuonPlan: specs → ModuleOptimizer (FallbackConfig: AdamW |
│  │                 SGD, + Muon groups)
│  ├─ spec.rs        ProjSpec/ProjSegment: fused-weight column seams → ParamGroup
│  │                 (`tiled`: one copy per application). BLOCK_CONTAINERS = the
│  │                 field names that store a block
│  ├─ segmented.rs   Segmented: one optimizer per column block of a fused weight.
│  │                 An SGD block is stateless and holds no state entry
│  ├─ sgd.rs         SgdConfig: plain SGD (decay, clipping, no momentum). `init`
│  │                 = the Sgd of Burn (`build` = its bare one). `step` = the
│  │                 same op for op, with a `[1]` device LR: the one optimizer
│  │                 that a captured step can replay
│  └─ report.rs      MuonPlan::describe(&module): per-param optimizer assignment
└─ utils/            lower-level plumbing
   ├─ mod.rs         div_eps (per-dtype epsilon)
   ├─ class/         ClassToken / ClassLatent placement (CLS-style registers) +
   │                 ClassCursor(s): offsets + full-length hint, shared by
   │                 forward/step/prime
   ├─ padding.rs     Padding: the mask of a right-padded batch + the place of
   │                 each row in the sequence of its slot. splice (class
   │                 markers), in_slot_order (the rows of a block into slot
   │                 order and back), reversed (bidi). tests.rs: every
   │                 container against each slot run alone
   ├─ schedule/      Schedule + BidiSchedule (virtual → real index mapping) +
   │                 Applications (which application of its real layer each
   │                 virtual layer is) + GradHorizon (which virtual layers
   │                 back-propagate)
   ├─ scheduler/     LR schedulers (cosine + warmup, constant)
   ├─ backend_macros.rs  impl_backend_ext_for_burn_backends! /
   │                     decl_autodiff_backend_ext!: per-backend BackendExt impls
   ├─ combined_grad.rs   flatten/unflatten (y, final_state) for a custom backward
   ├─ detach.rs          detach_params: cuts gradients, does NOT free memory
   ├─ init.rs            InitPolicy: the reference LM init, applied after the
   │                     build. N(0,std) on 2-D weights, zero biases, optional
   │                     residual rescale. The bespoke params of a block do not
   │                     change
   ├─ fprim.rs           F<B,D>: rank-tagged FloatTensor-primitive wrapper
   ├─ graph.rs           CapturedStep: a step captured once (the `capture` of
   │                     burn) and replayed. A stable input (a StepInput: one
   │                     tensor or a tuple) and cache buffers refreshed in place.
   │                     One eager run before `capture` (a cold capture fails
   │                     without it). Caches restored around it. The graph is
   │                     kept only if every buffer id survived (else eager).
   │                     Stateless = caches `()`. WARMUP_STEPS.
   │                     graph/weights.rs: Weights<M>, the params of a module as
   │                     the caches of a training step
   ├─ test_helpers.rs    max_abs_diff + grad-comparison macros
   └─ untied.rs          UntiedParam + tile/view/retie: a parameter held once per
                         application, copies side by side along an existing axis
```

---

## Architecture

### The plug-in surface

A block family supplies three impls and gets every container free:

- **`Block`**: `block_forward` (chunked: training + prefill), `block_step`
  (recurrent: decode), the zero-cache constructors, and `untied_params`
  (empty when nothing is untied). Associated types: `Cache`, `Caches`,
  `Options` (the per-call algorithm/chunk selector, `()` when there is nothing
  to select).
- **`BlockConfig`**: `d_model`, `init_block(n_applications, …)` (tiles what
  it unties), `muon_projections`.
- **`CacheStack`** on its `Caches`: slot count, move-in/move-out, and the
  inner-backend hop that `grad_horizon` does.

`Module + ModuleDisplay` are supertraits of `Block`, so the containers are
themselves `Module`s (the derive of Burn requires `ModuleDisplay` of every
module-typed generic). `Module::valid` lets `grad_horizon` move the stack to
the inner backend.

### Three execution modes

- `forward()`: parallel/chunked.
- `step()`: recurrent, O(state) per token, no growing KV cache.
- `prime()`: `step()` without a user token. It emits the class tokens/latents
  that wait for the next token, and returns the last one (`None` if there
  were none).

`forward()` from any cache equals `step()` unrolled from that same cache, on
the outputs, the final cache **and** the gradients. The containers assume it,
and a family must supply it. `reference/tests.rs` asserts it for `RefBlock`.

A captured graph can replay a `step()` (`utils/graph.rs`, `CapturedStep`):

- A replay reads the buffers that the capture saw. So the closure writes its
  new cache back in place (`CacheTensors`, one per family).
- The graph is kept only where that is verified (cubecl without fusion).
- `capture` is `unsafe`: what the step reads beyond its arguments must stay
  the same buffers.
- A fixed-shape `forward` is the step with caches `()`.
- `capture` first runs the closure once eagerly. The warm-ups of burn never
  run in place (the priming of cubecl holds a second handle on every buffer).
  Without the eager run, the recorded run compiles the in-place kernel
  variants, and loads a module mid-capture, which invalidates it.

A training step is the step whose caches are the weights of the model
(`Weights`): forward, backward and `optim::SgdConfig::step`, with the learning
rate as an input. Only plain SGD replays. The optimizers of Burn bake host
scalars (the rate, the bias correction of Adam) into a graph, and move their
state between buffers (tracel-ai/burn#5779). `examples::trainer::Trainer`
packages it. Its loss must keep the shapes of the batch (a mask, not a
gather).

No other thread may do device work while a capture records. cubecl gives each
thread its own stream over one pool, and an upload from a worker invalidates
the recording. This is silent: the step then runs eagerly. So dataloader
workers build on the host (`examples::device::loader_device`), and the loops
move each batch to the device.

### Padded batches

`forward()` (every container, and `Block::block_forward`) takes
`pad: Option<Tensor<2, Bool>>`: `true` at padding, **right** padding per slot.
A padded row is absent: the real outputs and the final cache of each slot are
those of that slot run alone. A block sees only a right-padded mask, because
the containers keep it so (`utils/padding.rs`):

- `Start`/`Custom` take the padding of the token that they precede.
- `Middle`/`End` land at the own length of each slot. The block then runs on
  its rows gathered into slot order.
- The reversed pass of bidi reads the real rows of each slot backwards.

`Middle` + mask needs the whole sequence in one call. `End` works chunked.
`RefBlock` skips padded rows and asserts the contract.

### Virtual layers, bidirectional, class tokens

- **Virtual layers** (`utils/schedule/`): `Layers<M>` runs `n_virtual_layers`
  logical passes over `n_real_layers` weight sets. Each virtual layer keeps
  its own cache. `Schedule` maps virtual → real (`Cyclic`/`Stretched`/
  `Custom`). `BidiSchedule` pairs forward/backward layers.
- **Gradient horizon** (`grad_horizon: Some(GradHorizon)`): back-propagates
  only *some* virtual layers (truncated BPTT). The others run on the inner
  backend. `forward`/`step`/`prime` cut on the same layers. A shared weight
  collects the gradient of its tracked applications only. It is a **mask**,
  not one boundary:
  - `Depth(K)` keeps the last `K` applications of every **real** layer. Under
    `Cyclic` that is one top suffix. Under `Stretched` it is the tail of each
    run (one cut per real layer). So every weight set trains.
  - `Mask(Vec<bool>)` states the tracked layers directly. `Custom` takes it
    (`Depth` panics on `Custom`).
  - `GradHorizon::last(K, n)` is the plain suffix. It is the only form that
    cuts a stack that shares no weights, where each real layer has a single
    application.

  What enters an untracked segment is re-attached *straight-through* where
  the graph resumes (a value-zero term that restores an identity gradient
  path). The reason is the stack **input**: it enters only at the bottom.
  Without the re-attachment, a cut leaves the `in_proj`/embedding of a network
  untrained. TRM/HRM avoid this: they inject the input again at every
  recursion. This stack does not. Under `MultiGate`, the carry reaches every
  stream. Class embeddings train at every level and on both sides of a cut (a
  per-layer latent inside an untracked segment gets a tracked zero-valued
  *ghost* row in the carry). They are learnable input rows, not part of the
  transform of a layer.
- **Bidirectional** (`modules/bidi.rs`): a pair runs a straight (→) and a
  reversed (← through `flip`) pass. `OutputMerge` (`Mean` | `CatLinear`)
  merges them. `BidiLayers<M>` stacks pairs of its own `Layer`s.
  `BidiLayerPair<M>` is one pair as a standalone module.
- **Class tokens/latents** (`utils/class/`): learnable `[CLS]`-style
  embeddings spliced into the sequence. `ClassToken` is on a *network*,
  `ClassLatent` on a *layer container*.
  - `Start`(0)/`Middle`(`L/2`)/`Custom(k)` are emitted **before** the original
    token at that index, uniformly. So a `Custom(k ≥ L)` never lands (unless
    the caller streams past `L`, and even then it precedes the next token).
    `End` alone **closes** the sequence, after its last token.
  - Placement is streamed and identical for both calls. Each call takes an
    optional `&mut ClassCursors`: one `full_len` hint + one cursor per level
    (`network`/`stack`/`per_layer`). So a sequence can split into any number
    of `forward` chunks and/or `step`s, and no marker moves.
  - A cursor past a marker skips it (`Start` fires once). `Middle`/`End` panic
    without the hint. A level gives the hint to the level above, grown by its
    **landing** markers only (`landing_count`).
  - `step` returns the **last** token that it emitted. `prime` takes no user
    token (`End` is never primed).
  - `None` keeps the defaults: for `forward`, this call is the whole
    sequence. For `step`/`prime`, no injection.
  - Read markers back through `class_*_output_indices` (for a chunk, minus
    the pre-call cursor of the level).
- **Multi-Gate Residuals** (`modules/multi_gate.rs`): `Layers<M>.residuals`
  selects plain additive (`Standard`) or `MultiGate`. `MultiGate` keeps up to
  `n_stream` streams, gated/attention-pooled between layers, in place of one
  additive skip. The input is stream 1, and the first `n_stream−1` layers
  **append** their output as a new stream. Only then does the gated mixing
  start. It is a convex mean-pool, so it gives an `O(1)` output, where the
  additive skip grows with depth. So a latent head wants `final_norm`. Class
  markers ride along at every level. See the module header.
- **Untied parameters** (`utils/untied.rs`): a real layer can hold some
  parameters once per application, not tied: its `LayerUntied` pre-norms,
  and those of its block that the config lists (`Block::untied_params`).
  - The copies tile an existing axis from one draw. So an untied stack starts
    as the tied one (the copy gradients sum to the tied gradient).
  - Containers run `Layer::application(k)` (borrowed when nothing differs),
    and size caches from view 0.
  - A `grad_horizon` cut through an untying layer panics.
  - Shapes `retie` after their `InitPolicy`.
  - `ProjSpec::tiled` steps each copy alone.

---

## Key Design Decisions

- **No optimized kernels**: only the portable tensor ops of Burn, so one code
  path runs on every backend.
- **Dispatch backend (Burn 0.22+)**: the high-level `Tensor` (every `Module`)
  is pinned to the global `Dispatch` backend, so library types are **not
  backend-generic**. The backend is a runtime `Device`. Autodiff and dtype are
  device properties. Only the custom-backward plumbing stays generic over `B`
  (`F<B,D>`, `Backward<B,_>` nodes, `Autodiff<B>` ext impls).
- **A no-grad region means the inner backend, not `detach`.** Burn registers
  untracked ops in the graph anyway. So `detach` cuts gradients but
  **retains every activation** (measured: 2765 MB vs 550 MB at 64 virtual
  layers, see `utils/detach.rs`). `grad_horizon` runs its untracked segments
  on `Module::valid(self)`. Three consequences:
  - `.inner()`/`.valid()` are idempotent off autodiff. So the `is_autodiff`
    guard only skips a round-trip that saves nothing.
  - Caches convert by hand: `Module::map` does nothing on plain `Tensor`
    fields, which is all a cache holds.
  - `is_require_grad` reports only `Requirement::Grad` leaves. So tests
    assert gradient *reachability* (`is_tracked` is the graph-participation
    probe).
- **Muon sees split projections, the model does not.** The `Muon` of Burn
  orthogonalises a whole 2-D weight. That is wrong for a fused `in_proj`
  (independent maps that share one allocation), and it panics for rank ≠ 2.
  So `optim` takes a per-family **allowlist** of `ProjSpec`s (the same column
  widths that the `split_into` of the forward uses). `Segmented` steps each
  block with its own optimizer. So each sub-matrix is orthogonalised and
  shape-LR-adjusted alone, and the forward keeps its single fused GEMM.
  - Per-head *scalar* channels, every 1-D/3-D tensor, and the boundary
    weights (embedding, LM head, network in/out projections, class-token
    tables) stay on the fallback (AdamW or plain SGD).
  - A spec matches its container (`"block."`, the suffix of every
    `BLOCK_CONTAINERS` entry) and its weight as **separate** path substrings,
    and both must match. So one plan covers plain, virtual-layer and
    bidirectional stacks, hand-written models, and a block that is an `enum`
    (whose variant name is between the two).
- **`#![warn(missing_docs)]`**: keep the crate free of warnings. Document the
  public surface when you add it. `cargo doc --no-deps` must also be free of
  warnings.
- **`reference.rs` is load-bearing, not decoration.** It proves that the
  containers need nothing family-specific. Extend it when you extend the
  container surface. Do not use a real mixer to test a generic type.

---

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

## Custom Commands

- `rg`: available.
- `cargo fmt`: don't use.
- **Always** edit files with the Edit/Write tools. This applies also when a
  harness or auto-mode reminder says to change files through Bash (`sed`,
  heredocs, python). That guidance does not apply here. *Do not* violate this.
  - No `python - <<'PY'`, no `sed -i`, no `cat > file <<'EOF'`. Always use
    `Edit`s.
  - Bash stays the tool for *reading* and *inspecting* (`cat`, `sed -n`, `rg`,
    `grep`) and for throwaway files outside the crate (for example, in `/tmp`).
