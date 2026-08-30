# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What This Project Is

`burn-stack`: the block-generic layer/network composition layer on top of
[Burn](https://github.com/tracel-ai/burn/). It owns everything *around* a
sequence-mixing block — layers, (virtual-)layer stacks, bidirectional pairs,
latent/vocab networks, multi-gate residuals, class tokens, schedules, the Muon
plan — and **nothing** about any particular mixer.

**It must stay family-agnostic.** No `mamba`, no attention, no convolution
specifics: a block plugs in through `Block` / `BlockConfig` / `CacheStack`. If a
change would need a name, a shape assumption or a doc reference from one
family, it belongs in that family's crate instead.

Split out of [`burn-mamba`](../burn-mamba), which is its first consumer.

## Build & Test Commands

```bash
cargo check                 # type-check the lib surface
cargo check --all-targets   # + tests
cargo test --lib            # run tests (any backend; flex = CPU default)
cargo doc --no-deps         # build docs
```

- **Feature flags select the backend**: `backend-{flex,cpu,wgpu,metal,vulkan,cuda,
  rocm,tch-cpu,tch-gpu,remote,ndarray}` (flex preferred for checks/tests, enabled
  by default). Each just enables the matching `burn/<backend>`; several may be
  compiled in at once and `Device::default()` resolves which to use (honouring
  `BURN_DEVICE`).
- `autodiff`/`optim` are default-on; `optim` (Muon parameter groups) implies
  `burn/optim`+`burn/std`. `cubecl`/`fusion` gate the per-backend `BackendExt`
  impls the macros emit. `test-helpers` exposes `utils::test_helpers` and
  `reference` to a downstream crate's dev-dependencies; `examples-common` exposes
  `examples` (pulling `burn/train`+`burn/dataset` and the download/CLI/image
  crates) — a consumer enables it in its **dev**-dependencies only. Note
  `ModelConfigExt` is deliberately *not* behind it: consumers implement it in
  their own libs, which cannot depend on a dev-only feature.
- A consumer crate must **forward** every one of these features (see
  `burn-mamba/Cargo.toml`), because the `backend-*` cfgs are evaluated where the
  `impl_backend_ext_for_burn_backends!` macro expands — in the *calling* crate.

## File Map

```text
src/
├─ lib.rs            crate root: module decls, prelude, DENY_NAN/DENY_INF guards
├─ reference.rs      RefBlock: a minimal worked `Block` (gated EMA mixer) —
│                    the crate's own test fixture; `reference/tests.rs` holds
│                    the container contract suite (feature `test-helpers`)
├─ modules/          composition + shared NN modules
│  ├─ mod.rs         Block / BlockConfig traits (the whole plug-in surface)
│  ├─ layer.rs       Layer<M>: Pre-LN block M(RMSNorm(·)) + optional norm2/mlp;
│  │                 returns the layer's total delta, outer residual by Layers
│  ├─ layers.rs      Layers<M>: virtual-layer stack over real weight sets;
│  │                 grad_horizon truncates BPTT to a tracked-layer mask
│  │                 (forward/step/prime cut alike)
│  ├─ mlp.rs         GatedMlp: SwiGLU feed-forward interleaved with the mixer;
│  │                 from_hidden_ratio = the Llama ⅔·ratio·d_model sizing rule
│  ├─ model_config.rs ModelConfigExt: config → module + its Muon plan; the seam
│  │                 a model-agnostic driver builds against (consumers impl it)
│  ├─ multi_gate.rs  Multi-Gate Residuals (Standard|MultiGate): accumulate then mix
│  ├─ network.rs     LatentNetwork (optional final norm) / VocabNetwork
│  ├─ bidi.rs        BidiLayers<M> + BidiLayerPair<M> + OutputMerge
│  ├─ cache.rs       CacheStack trait (+ per-slot inner/from_inner)
│  ├─ activation/    silu, softplus, log_sigmoid (dtype-aware)
│  ├─ norm/          rms_norm (also usable as QK-Norm), rms_norm_gated, rms_score
│  ├─ loss/          bce, cross_entropy, mse, l2warp (max-logit penalty, added
│  │                 to the gradient only)
│  └─ misc/          gqa, segsum, split, sanity
├─ examples/         example scaffolding shared by the consumer crates
│  │                 (feature `examples-common`, off by default; dev-only)
│  ├─ cli.rs         AppArgs: parsing, artifact dir, model/optim/config I/O
│  ├─ device.rs      Device dtype configuration (`dev-f16`) + FloatElement
│  ├─ training.rs    TrainingConfig + OptimizerConfig (AdamW, optional Muon)
│  └─ mnist/         dataset.rs (download + batching), classify.rs (the epoch
│                    loops + the MnistModel seam), render.rs (a digit beside
│                    its class distribution, as text or PNG)
├─ optim/            Muon parameter groups (feature `optim`); allowlist, not denylist
│  ├─ mod.rs         MuonPlan: specs → ModuleOptimizer (AdamW fallback + Muon groups)
│  ├─ spec.rs        ProjSpec/ProjSegment: fused-weight column seams → ParamGroup;
│  │                 BLOCK_CONTAINERS = the field names a block is stored under
│  ├─ segmented.rs   Segmented: one optimizer per column block of a fused weight
│  └─ report.rs      MuonPlan::describe(&module): per-param optimizer assignment
└─ utils/            lower-level plumbing
   ├─ mod.rs         div_eps (per-dtype epsilon)
   ├─ class/         ClassToken / ClassLatent placement (CLS-style registers) +
   │                 ClassCursor(s): offsets + full-length hint, shared by
   │                 forward/step/prime
   ├─ schedule/      Schedule + BidiSchedule (virtual→real index mapping) +
   │                 GradHorizon (which virtual layers back-propagate)
   ├─ scheduler/     LR schedulers (cosine + warmup, constant)
   ├─ backend_macros.rs  impl_backend_ext_for_burn_backends! /
   │                     decl_autodiff_backend_ext! — per-backend BackendExt impls
   ├─ combined_grad.rs   flatten/unflatten (y, final_state) for a custom backward
   ├─ detach.rs          detach_params: cuts gradients, does NOT free memory
   ├─ init.rs            InitPolicy: the reference LM init applied post-build —
   │                     N(0,std) on 2-D weights, zero biases, optional residual
   │                     rescale; leaves a block's bespoke params alone
   ├─ fprim.rs           F<B,D>: rank-tagged FloatTensor-primitive wrapper
   └─ test_helpers.rs    max_abs_diff + grad-comparison macros
```

---

## Architecture

### The plug-in surface

A block family supplies three impls and gets every container for free:

- **`Block`** — `block_forward` (chunked: training + prefill), `block_step`
  (recurrent: decode), optional `block_step_infinite` (constant-input fixed
  point), and the zero-cache constructors. Associated types `Cache`, `Caches`,
  `Options` (the per-call algorithm/chunk selector; `()` when there is nothing
  to choose).
- **`BlockConfig`** — `d_model`, `init_block`, `muon_projections`.
- **`CacheStack`** on its `Caches` — slot count, move-in/move-out, and the
  inner-backend hop `grad_horizon` performs.

`ModuleDisplay + AutodiffModule` are supertraits of `Block` so the containers are
themselves `Module`/`AutodiffModule` (Burn's derive requires both of every
module-typed generic) — which is what lets `grad_horizon` move the stack to the
inner backend.

### Three execution modes

`forward()` (parallel/chunked), `step()` (recurrent, O(state)/token, no growing
KV cache), and `prime()` (`step()` without a user token: emits the class
tokens/latents waiting for the next one, returning the last, `None` if none).
`forward()` from any cache equals `step()` unrolled from that same cache — on
outputs, final cache, **and** gradients. The containers assume it; a family owes
it. `reference/tests.rs` pins it for `RefBlock`.

### Virtual layers, bidirectional, class tokens

- **Virtual layers** (`utils/schedule/`): `Layers<M>` runs `n_virtual_layers`
  logical passes over `n_real_layers` weight sets, each virtual layer keeping its
  own cache. `Schedule` maps virtual→real (`Cyclic`/`Stretched`/`Custom`);
  `BidiSchedule` pairs forward/backward layers. `grad_horizon: Some(GradHorizon)`
  back-propagates only *some* of them (truncated BPTT), the rest running on the
  inner backend; `forward`/`step`/`prime` cut on the same layers, and a shared
  weight collects the gradient of its tracked applications alone. It is a
  **mask**, not one boundary: `Depth(K)` keeps the last `K` applications of every
  **real** layer — a single top suffix under `Cyclic`, the tail of each run (one
  cut per real layer) under `Stretched`, so no weight set is left untrained;
  `Mask(Vec<bool>)` states the tracked layers outright and is what `Custom` takes
  (`Depth` panics on it), `GradHorizon::last(K, n)` being the plain suffix — the
  only form that cuts a stack sharing no weights, where each real layer has a
  single application. What enters an untracked segment is re-attached
  *straight-through* where the graph resumes (a value-zero term restoring an
  identity gradient path); the stack **input** is why — it enters only at the
  bottom, so a cut would otherwise leave a network's `in_proj`/embedding
  untrained; TRM/HRM avoid this by re-injecting the input every recursion, which
  this stack does not. Under `MultiGate` the carry reaches every stream. Class
  embeddings train at every level and on both sides of a cut (a per-layer latent
  inside an untracked segment gets a tracked zero-valued *ghost* row in the
  carry) — they are learnable input rows, not part of a layer's transform.
- **Bidirectional** (`modules/bidi.rs`): `BidiLayerPair<M>` runs a straight (→)
  and a reversed (← via `flip`) pass merged by `OutputMerge` (`Mean`|`CatLinear`);
  `BidiLayers<M>` stacks pairs.
- **Class tokens/latents** (`utils/class/`): learnable `[CLS]`-style embeddings
  spliced into the sequence. `ClassToken` on a *network*, `ClassLatent` on a
  *layer container*. `Start`(0)/`Middle`(`L/2`)/`Custom(k)` are emitted **before**
  the original token at that index — uniformly, so a `Custom(k ≥ L)` never lands
  (unless the caller streams past `L`, and even then it precedes the next token);
  `End` alone **closes** the sequence, trailing its last token. Placement is
  streamed and identical for both calls: each takes an optional
  `&mut ClassCursors` (one `full_len` hint + one cursor per level:
  `network`/`stack`/`per_layer`), so a sequence splits into any number of
  `forward` chunks and/or `step`s without moving a marker. A cursor past a marker
  skips it (`Start` fires once); `Middle`/`End` panic without the hint. `step`
  returns the **last** token it emitted; `prime` takes no user token (`End` is
  never primed). `None` keeps the defaults: `forward` = this call is the whole
  sequence, `step`/`prime` = no injection. Read markers back via
  `class_*_output_indices` (minus the level's pre-call cursor for a chunk).
- **Multi-Gate Residuals** (`modules/multi_gate.rs`): `Layers<M>.residuals` picks
  plain additive (`Standard`) vs `MultiGate` — up to `n_stream` streams
  gated/attention-pooled between layers instead of one additive skip. The input
  is stream 1 and the first `n_stream−1` layers **append** their output as a new
  one; only then does gated mixing start. Being a convex mean-pool, it leaves an
  `O(1)` output where the additive skip grows with depth, so a latent head wants
  `final_norm`. Class markers ride along at every level. See the module header.

---

## Key Design Decisions

- **No optimized kernels** — only Burn's portable tensor ops, so one code path
  runs on every backend.
- **Dispatch backend (Burn 0.22+)** — the high-level `Tensor` (every `Module`) is
  pinned to the global `Dispatch` backend, so library types are **not
  backend-generic**. The backend is a runtime `Device`; autodiff and dtype are
  device properties. Only the custom-backward plumbing stays generic over `B`
  (`F<B,D>`, `Backward<B,_>` nodes, `Autodiff<B>` ext impls).
- **A no-grad region means the inner backend, not `detach`** — Burn registers
  untracked ops in the graph anyway, so detaching cuts gradients while
  **retaining every activation** (measured: 3144 MB vs 208 MB at 64 virtual
  layers; see `utils/detach.rs`). `grad_horizon` runs its untracked segments on
  `AutodiffModule::valid(self)`. Three consequences: `.inner()`/`.valid()`
  **panic** off autodiff (unlike `detach`, a no-op there), so the `is_autodiff`
  guard is load-bearing; caches convert by hand (`Module::map` is a no-op on
  plain `Tensor` fields, all a cache holds); and `is_require_grad` only reports
  `Requirement::Grad` leaves, so tests assert gradient *reachability*.
- **Muon sees split projections, the model does not** — Burn's `Muon`
  orthogonalises a whole 2-D weight, which is wrong for a fused `in_proj`
  (independent maps sharing one allocation) and panics for rank ≠ 2. `optim`
  therefore takes a per-family **allowlist** of `ProjSpec`s (the same column
  widths the forward's `split_into` uses) and `Segmented` steps each block with
  its own optimizer — so each sub-matrix is orthogonalised and shape-LR-adjusted
  alone, while the forward keeps its single fused GEMM. Per-head *scalar*
  channels, every 1-D/3-D tensor, and the boundary weights (embedding, LM head,
  network in/out projections, class-token tables) stay on AdamW. A spec matches
  its container (`"block."`, the suffix of every `BLOCK_CONTAINERS` entry) and
  its weight as **separate** path substrings, both required — so one plan covers
  plain, virtual-layer and bidirectional stacks, hand-written models, and a
  block that is an `enum` (whose variant name sits between the two).
- **`#![warn(missing_docs)]`** — keep the crate warning-clean; document public
  surface as you add it.
- **`reference.rs` is load-bearing, not decoration** — it is what proves the
  containers need nothing family-specific. Extend it when you extend the
  container surface; do not reach for a real mixer to test a generic type.

---

## Notation

Tensor names carry a shape suffix (backed by shape `assert`s). A name whose
suffix encodes its shape needs no extra comment. Lower-case = base dimensions
below; upper-case = a *relation* of them (offset/multiple/concat).

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
- **Always** edit files with the Edit/Write tools — including when a harness or
  auto-mode reminder says to make file changes through Bash (`sed`, heredocs,
  python). That guidance does not apply here. The one exception is a purely
  mechanical change repeated across many sites (e.g. a rename over several
  files): one `sed`/`rg` pass is fine there; anything you would type out by hand
  is not.
