# Burn: devices, autodiff, gradients and checkpointing

A reference for the semantics of the Burn methods on devices, autodiff,
gradients and gradient checkpointing. These methods control where a tensor
lives, whether autodiff records it, and which gradients it keeps. The
document also lists the traps that these semantics set, each with its
symptom and its fix.

- **Version.** Burn 0.22.1 from crates.io (see `Cargo.toml`), with the Dispatch
  API: `Tensor<D, K>` has no backend parameter, and the backend is a runtime
  `Device`.
- **Sources.** Each claim comes from the source or the doc comments of Burn
  0.22.1. A claim marked **Measured** comes from a run on 0.22.1, and its setup
  is next to the number. A measurement on another revision names that revision.
  Paths such as `crates/burn-tensor/...` are relative to the Burn repository at
  tag `v0.22.1`.
- **Older code.** [§13](#13-prs-that-changed-the-meaning-of-these-methods)
  lists PRs that changed what these methods mean, for code from an older Burn.
  The list is not complete. A consumer that pins an older revision must check
  the source of that revision.

## Contents

1. [Rules to remember](#1-rules-to-remember)
2. [The model](#2-the-model)
3. [The three states of a float tensor](#3-the-three-states-of-a-float-tensor)
4. [Device API](#4-device-api)
5. [Tensor API](#5-tensor-api)
6. [One op with mixed inputs](#6-one-op-with-mixed-inputs)
7. [The graph and its memory](#7-the-graph-and-its-memory)
8. [Gradient checkpointing](#8-gradient-checkpointing)
9. [Modules and parameters](#9-modules-and-parameters)
10. [Recipes](#10-recipes)
11. [Traps](#11-traps)
12. [Debugging](#12-debugging)
13. [PRs that changed the meaning of these methods](#13-prs-that-changed-the-meaning-of-these-methods)

## 1. Rules to remember

1. **Autodiff is a property of each tensor, not of a type.** A float tensor
   is plain, or it has autodiff with a checkpointing strategy. A new tensor
   takes the autodiff context of the device that creates it.
2. **One autodiff input makes the output autodiff.** A plain input acts as a
   constant in that op. Two autodiff inputs with different checkpointing
   strategies make the op panic.
3. **`Tensor::from_inner` is `Tensor::autodiff`.** It enables autodiff on
   *any* plain tensor, with the `Disabled` strategy. `Tensor::inner` is
   `Tensor::without_autodiff`, and it does nothing to a plain tensor. So off
   autodiff, `from_inner` is not the inverse of `inner`.
4. **`detach` keeps the tensor on autodiff.** It only starts a new graph
   lineage. To leave autodiff, use `without_autodiff`.
5. **`to_device` never changes the autodiff context.** The tensor keeps its
   own context and strategy, and the context of the target device does not
   apply.
6. **The graph keeps the saved state of each tracked op until a backward
   reaches it.** A forward-only loop on tensors that depend on a trainable
   leaf grows without limit. For inference, use a plain device or
   `Module::valid()`.
7. **One backward for each graph.** Burn has no `retain_graph`.
8. **`Module::valid()` also disables training flags** (dropout, batch norm,
   …) and drops the checkpointing strategies. **`Module::train()` lifts every
   parameter with `from_inner`**, also on a plain device. A plain parameter
   gets the `Disabled` strategy.
9. **A plain `Tensor` field of a module is a constant.** `Module::map` does
   not visit it. `x.no_grad()` on a tensor compiles and does nothing.
10. **`Device` equality ignores autodiff and the strategy.** Ask
    `is_autodiff()` and `gradient_checkpointing_strategy()` when they matter.

## 2. The model

- `Device` wraps a `DispatchDevice`: one backend variant (flex, cuda, …),
  optionally inside an autodiff wrapper that holds a
  `GradientCheckpointingStrategy`.
- A tensor holds its backend primitive and a `DispatchAutodiffContext`:
  `Disabled`, or `Enabled(strategy)`
  (`crates/burn-dispatch/src/tensor.rs`).
- `tensor.device()` reports the context of the *tensor*: an autodiff tensor
  reports an autodiff device with its own strategy. It does not report the
  device that created it.
- A float tensor with autodiff holds an autodiff primitive (a graph node). An
  int or bool tensor can carry an enabled context, but autodiff never tracks
  it.
- `GradientCheckpointingStrategy` has two values: `Disabled` (the default)
  and `Balanced`.

## 3. The three states of a float tensor

| Query | `true` means | Source of truth |
|---|---|---|
| `is_autodiff()` | Ops on it go through the autodiff backend. | The context of the tensor. |
| `is_tracked()` | Its node takes part in a recorded graph. | Node requirement is not `None`. |
| `is_require_grad()` | It is a leaf that keeps its gradient after backward. | Node requirement is `Grad`. |

The node requirement (`crates/burn-autodiff/src/graph/requirement.rs`):

| Requirement | Which tensors | Gradient |
|---|---|---|
| `Grad` | A leaf after `require_grad()` | Computed and kept in `Gradients` |
| `GradInBackward` | The output of an op with at least one tracked input | Computed during backward, not kept |
| `None` | Every input is untracked | None |

So a tensor can have autodiff and still be untracked: for example, a
constant created on an autodiff device. `is_require_grad()` is `false` for
every derived tensor, also when gradients flow through it. A test that wants
to know whether gradients reach a tensor asks `is_tracked()`.

## 4. Device API

`crates/burn-tensor/src/device.rs`

| Method | Semantics |
|---|---|
| `Device::default()` | The first enabled backend, in this order: CUDA, Metal, ROCm, Vulkan, WebGPU, wgpu, CPU, LibTorch, Flex, Remote, NdArray. In a `std` build, the `BURN_DEVICE` env variable (`cuda`, `rocm`, `metal`, `vulkan`, `webgpu`, `wgpu`, `cpu`, `tch`, `remote`, `flex`, `ndarray`) selects one. `metal`, `vulkan` and `webgpu` also pin the graphics API. Panics when no backend feature is enabled, or when `BURN_DEVICE` names an unknown backend or a backend with its feature off. Plain, not autodiff. |
| `Device::flex()`, `Device::cuda(i)`, `Device::cpu()`, … | One backend. Plain. |
| `.autodiff()` | The same device with autodiff. Idempotent: an autodiff device keeps its strategy. First order only. |
| `.gradient_checkpointing()` | Sets the `Balanced` strategy. Panics on a device without autodiff. |
| `.gradient_checkpointing_strategy()` | `Some(strategy)` on an autodiff device, `None` on a plain one. |
| `.without_autodiff()`, `.inner()` | The plain device. Aliases. Idempotent. |
| `.is_autodiff()` | Whether the device has autodiff. |
| `==` | Hardware identity only. Ignores autodiff and the strategy. `Module::devices()` has the same limit. |
| `.configure((FloatDType, IntDType))` | Default dtypes. The first `configure` or the first tensor op on the device locks the settings. A later `configure` returns `DeviceError::AlreadyInitialized`. A query of `settings()` does not lock them. |
| `.supports_dtype(dtype)` | Whether the device can compute in `dtype` (bf16 can be storable but not computable). |
| `.identity()`, `Device::enumerate_physical()` | Hardware queries, each with a feature gate. `identity` reports the part name, the kernel target and the card, or `None` for a backend that does not report them. `enumerate_physical` gives each card once, with the devices that reach it. Both calls open the devices. |
| `.sync()`, `.memory_cleanup()`, `.memory_persistent_allocations(..)` | Device execution and memory. They do not touch autodiff. |

## 5. Tensor API

`crates/burn-tensor/src/tensor/api/{autodiff,float,base}.rs`. The autodiff
methods that are not `#[cfg(feature = "autodiff")]` also compile without
autodiff. With the feature off, no tensor has autodiff.

### 5.1 Context and lineage

| Method | Plain tensor | Autodiff leaf | Autodiff non-leaf |
|---|---|---|---|
| `autodiff()`, `from_inner(t)` | **Enables autodiff**, strategy `Disabled`. Untracked new root. | Unchanged | Unchanged |
| `without_autodiff()`, `inner()` | Unchanged | Plain. Drops its graph reference. | Plain. Drops its graph reference. |
| `detach()` | Unchanged | New node, no parents. **Keeps `require_grad`** and the distributed tag. Stays on autodiff. | New untracked root. Stays on autodiff. |
| `require_grad()`, `set_require_grad(true)` | **Panics** | Becomes a `Grad` leaf | **Panics** (use `detach` first) |
| `set_require_grad(false)` | Unchanged | Stops keeping its gradient | New lineage, as with `detach` |
| `to_device(d)` | Moves. Stays plain. | A tracked leaf moves as a **recorded op**, and the output is a non-leaf. An untracked leaf stays untracked. | Recorded op |
| `gradient_checkpointing_strategy()` | `None` | `Some(strategy)` | `Some(strategy)` |
| `with_gradient_checkpointing_strategy(s)` | **Panics** | Sets `s` | Sets `s` |

Notes:

- In this table, a non-leaf has the requirement `GradInBackward` (§3). The
  output of an op on untracked inputs is untracked, and `require_grad()` does
  not panic on it.
- `from_inner` does not make a tensor require grad. After `from_inner`, call
  `require_grad()` for a trainable leaf.
- `to_device` applies the context of the source tensor to the target device
  (`with_autodiff_context_from`). To make a new trainable leaf on the target,
  use `t.to_device(d).detach().require_grad()`.
- Cross-backend transfers (for example, flex to cuda) go through host memory,
  also for the gradients in the backward.

### 5.2 Backward and gradients

| Method | Semantics |
|---|---|
| `backward()` | Computes the gradients from this tensor and **consumes** the recorded steps that it reaches. Panics when the tensor has no autodiff, is untracked, or depends on a consumed step. A clone does not keep a tape. |
| `grad(&grads)` | The kept gradient of a `Grad` leaf, as a **plain** tensor. `None` when the tensor keeps no gradient. |
| `grad_remove(&mut grads)` | The same, and removes it (lets the optimizer work in place). |
| `grad_replace(&mut grads, g)` | Replaces the entry. `g` must be plain. |

There is no `retain_graph`. A second backward through a consumed graph step
panics with "Cannot run backward: graph tape has already been consumed".
Combine the losses into one before `backward`, or run the forward again. A
later forward can reuse the same leaves.

## 6. One op with mixed inputs

`DispatchAutodiffContext::merge` (`crates/burn-dispatch/src/tensor.rs`)
gives the context of the output:

| Input A | Input B | Output |
|---|---|---|
| plain | plain | plain |
| plain | autodiff (`s`) | autodiff (`s`). The plain input is a constant. |
| autodiff (`s`) | autodiff (`s`) | autodiff (`s`) |
| autodiff (`s1`) | autodiff (`s2`), `s1 ≠ s2` | **Panic**: "Gradient checkpointing strategy mismatch: Disabled vs Balanced. Tensors in the same operation must share a strategy." |

Consequence: a single autodiff tensor in a forward pass (a parameter, a
cache, a constant) moves every op that it reaches onto the autodiff backend.

## 7. The graph and its memory

An op with an autodiff input registers a graph step. It does this
also when the op is untracked (an `UntrackedOpsStep`, in
`crates/burn-autodiff/src/ops/base.rs`). So a memory-bound op can still find
an untracked parent. A tracked step also keeps the state that its backward
needs (for example, `exp` keeps its output).

A dropped output does not free its step. The graph memory manager runs only
inside a backward (`cleanup_orphaned_entries`, in
`crates/burn-autodiff/src/runtime/graph.rs`). A backward frees:

- the steps that it consumes,
- the branches of its own graph that no live tensor needs any more,
- the orphaned entries of other graphs that no longer look useful.
  **Measured** (the setup of the table below): an unrelated backward frees the
  untracked steps of another graph. It does not free the steps that depend on
  a live `Grad` leaf.

**Measured** (one loop on flex over 256 KB tensors, 3000 iterations, with the
output of each iteration dropped):

| Input | Op | Backward | Memory |
|---|---|---|---|
| plain | `x * c` | none | flat (4 MB) |
| autodiff, untracked | `x * c` | none | +1 KB per op (node records only) |
| autodiff, untracked | `x * c` | unrelated, every 100 | flat |
| `Grad` leaf | `(x * c).exp()` | none | **+256 KB per iteration** (760 MB) |
| `Grad` leaf | `(x * c).exp()` | unrelated, every 100 | **+256 KB per iteration** |
| `Grad` leaf | `(x * c).exp()` | through the same leaf, every 100 | bounded (30 MB) |
| `Grad` leaf, `without_autodiff()` first | `x * c` | none | flat |
| recurrent `s = s * x * 0.5 + 1` | | none, or unrelated | **+256 KB per iteration** |

What follows from it:

- A training loop frees what each training step abandoned, because each
  training step runs a backward through the same parameters.
- A forward-only loop over a model that holds `Grad` leaves (inference or
  evaluation on the training model) keeps the saved state of every iteration.
  It never frees it. A state that the loop carries (a decode cache) does not
  change this: it only makes each iteration depend on all the earlier ones.
- `detach` does not help here: the ops after it still run on autodiff and
  register graph steps. `without_autodiff` does help.

## 8. Gradient checkpointing

- **Strategies.** `Disabled` (default): keep the state of every op.
  `Balanced`: during the backward, recompute the output of the ops that Burn
  marks as memory-bound, and keep the output of compute-bound ops. Less peak
  memory, more compute.
- **Where a strategy comes from.** A tensor created on
  `Device::default().autodiff().gradient_checkpointing()` gets `Balanced`.
  `with_gradient_checkpointing_strategy` overrides it for one tensor.
- **The rule.** All autodiff inputs of one op must have the same strategy
  (§6). A tensor with another strategy makes the first op that mixes it
  panic.

What resets the strategy to `Disabled`:

| Operation | Why |
|---|---|
| `Tensor::from_inner` / `autodiff()` on a plain tensor | The new context is always `Enabled(Disabled)`. |
| `Module::valid()` then `Module::train()` | `valid` removes autodiff (and the strategy). `train` lifts with `from_inner`. The docs say: "Keep the original module when continuing training after validation." |
| A `train()` of a `Tensor` field that has no autodiff | `Module for Tensor` implements `train` as `from_inner`. |

What keeps it: `detach`, `to_device`, `from_inner` on an autodiff tensor, the
Burn optimizer (`crates/burn-optim/src/optim/module/parameter_context.rs`),
and an explicit restore:

```rust
let strategy = t.gradient_checkpointing_strategy(); // before going down
let mut t = t.without_autodiff();
// ... work on the inner backend ...
if let Some(strategy) = strategy {
    t = Tensor::from_inner(t).with_gradient_checkpointing_strategy(strategy);
}
```

## 9. Modules and parameters

### 9.1 `Param<T>`

`crates/burn-core/src/module/param/`. A `Param` is an identified,
traversable value that a module owns. `Param<Tensor<D>>` is the trainable
kind. `Param<Flag>` is a training switch (§9.4). A tensor parameter holds:

- `id: ParamId`: the identity that optimizers and records use.
- A lazy state that its clones share. A clone is cheap. The clones share
  only the lazy initialization: after it, each clone changes on its own.
- `is_active`: the **configured** trainability. It is separate from the
  tensor and survives `valid()` → `train()`. On a plain device, a parameter
  can be active while its tensor has no autodiff.
- An optional save/load mapper and an optional reparameterization (LoRA).

| Constructor | Semantics |
|---|---|
| `Param::uninitialized(id, init, device, require_grad, shape)` | Lazy. The initializer gets the device and the *effective* requirement: `false` on a plain device. The requested requirement stays for `train()`. The initializer must create the value on the device that it gets. A move before the first access changes that device, and a record loaded after the move goes to the new device. When a clone shares the lazy state, the parameter initializes on its first device, and the move copies it. |
| `Param::from_tensor(t)` | New id. On autodiff: `require_grad`. On a plain device: no change to the tensor, but `is_active = true`. |
| `Param::from_data(data, device)` | As `from_tensor`, from data. |
| `Param::initialized(id, value)` | The value as it is. `is_active` comes from the value. |
| `Param::from_mapped_value(id, value, mapper)` | As `initialized`, and keeps the mapper and the configured state. Use it in a `ModuleMapper`. |

| Accessor | Semantics |
|---|---|
| `val()` | The effective value (a clone of the handle). Initializes a lazy parameter. Applies the reparameterization. |
| `base()` | The stored value, without the reparameterization. |
| `consume()` | `(id, value, mapper)`. Drops the reparameterization. |
| `map(f)` | Applies `f` to the value. Keeps the configured state. |
| `set_require_grad(b)` | Sets the configured state. On a plain device, the tensor does not change, and `train()` applies the state. |
| `lazy_device()`, `is_initialized()` | Ask without initializing. `Module::devices()` and `Module::num_params()` also do not initialize a parameter. |

### 9.2 Module lifecycle

`crates/burn-core/src/module/base.rs`

| Method | Tensor gradients | Training flags | Autodiff |
|---|---|---|---|
| `valid()` | Off, configured state kept | Off, configured state kept | **Removed** (also the strategies) |
| `train()` | Configured state applied | Configured state applied | **Enabled with `from_inner`**, on every device. A plain parameter gets the strategy `Disabled`. |
| `no_grad()` = `set_require_grad(false)` | Off, **persistent**: `train()` keeps them off | Unchanged | Unchanged |
| `set_require_grad(b)` | All tensor parameters | Unchanged | Unchanged |
| `set_require_grad_group(g, b)` | Matched tensor parameters | Ignored, also when matched | Unchanged |
| `freeze()` / `unfreeze()` | All off / all on, persistent | All off / all on | Unchanged |
| `freeze_group(g)` / `unfreeze_group(g)` | Matched off / on | Matched off / on | Unchanged |
| `to_device(d)` | Kept | Kept | Kept. A recorded op for tracked parameters, so the result cannot train. |
| `fork(d)` | Kept | Kept | Kept. Each parameter is a new leaf on `d`. Use it to train on `d`. |
| `materialize()` | Folds the reparameterizations into the values | Unchanged | Unchanged |

- `valid()` on a module that is already plain still disables its flags. So
  it is idempotent, but it is not always a no-op.
- `valid()` keeps the reparameterizations and their stored bases, and it puts
  their nested modules in validation mode.
- `unfreeze()` sets every flag to *on*. It does not restore earlier values.
- `set_require_grad(true)` and `unfreeze()` override a selective setup, such
  as the frozen base weights of `apply_lora`.
- `LearningResult::model` comes back in validation mode. Call `train()`
  before more training.

### 9.3 Plain `Tensor` fields

`Module for Tensor` (`param/constant.rs`) is a constant:

| Method | On a `Tensor` field |
|---|---|
| `visit`, `map` | Skip it. A `ModuleMapper` never sees it. |
| `valid` | `without_autodiff()` |
| `train` | `Tensor::from_inner(self)`: enables autodiff |
| `to_device`, `fork` | `to_device` |

A `#[derive(Module)]` struct applies this table to each of its plain `Tensor`
fields. So a `ModuleMapper` never reaches such a field, and `valid()` then
`train()` resets its strategy to `Disabled`. A cache or state struct of plain
tensors needs hand-written conversions. A `#[module(skip)]` field takes part
in none of these calls.

### 9.4 Flags

`Param<Flag>` (for example, `Dropout::training`) is a host-side boolean with
an id. Parameter groups can select it by id or path. Records do not save
flags: loading keeps the flag of the destination module. `valid()` disables
the flags of the returned value, `train()` restores them, and `freeze()`
disables them persistently. `no_grad()` and `set_require_grad*` do not
touch them.

## 10. Recipes

**Inference.** Build or load the model on a plain device, or call
`model.valid()`. Check it:

```rust
let y = model.forward(x);
assert!(!y.is_autodiff(), "the forward pass runs on autodiff");
```

**Evaluation during training.** Run it on `model.valid()`, not on the
training model (§7). Burn's `Learner` does this.

**An untracked region inside a tracked pass.** Run the region on the inner
backend. Take its inputs down with `without_autodiff()`, and run it on
`model.valid()`. Lift the outputs back with `from_inner`, and restore the
strategy (§8). `detach` is not sufficient (§7).

**Carry a state to the next window, with no gradient across them.** Round-trip
the state through the inner backend (`without_autodiff`, then `from_inner`
with the strategy). The earlier graph is then unreachable, and the next
backward frees it.

**A new trainable leaf from an existing parameter** (a tiled copy, a
reshaped copy):

```rust
// Works on a plain device and on an autodiff device, and keeps the strategy.
let copy = Param::from_tensor(param.val().repeat_dim(axis, n).detach());
```

Do not write `Tensor::from_inner(param.val().inner().repeat_dim(axis, n))`.
On a plain device it enables autodiff (§11, the first trap), and on autodiff
it resets the strategy.

**A constant inside a tracked op.** Use `x.clone().without_autodiff()`. The
constant then costs no graph step.

**Train on another device.** `model.fork(&device)`. `to_device` gives a
model that cannot train (§9.2).

**Freeze part of a model.** `freeze_group(g)` stops the gradients and the
flags. `set_require_grad_group(g, false)` stops only the gradients.

## 11. Traps

Each trap: the symptom, the cause, and the fix.

1. **Memory grows at each step of a decode, and the decode is slow.** The code
   built a parameter with `Tensor::from_inner(t.inner()…)` on a plain device,
   so the parameter is a `Grad` leaf on autodiff. Every op that reads it runs
   on autodiff (§6) and keeps its state (§7). Fix: `.detach()` after the op,
   not an `inner`/`from_inner` round trip. Check: `param.val().is_autodiff()`
   on a plain device.
   - **Measured** (a recurrent loop on flex that reads a 256 KB slice of such
     a parameter, 1000 iterations): +256 KB per iteration (257 MB). With the
     fix, the memory stays flat (4 MB).
   - **Measured** on Burn `8f08fdad`: a 3.5× slower decode (7.1–8.0 against
     2.0–2.2 ms/char), and +150–170 MB/s in a slower run (12.5 ms/char). The
     setup is a Mamba-3 decode on flex (d164, 3 real layers × 6 applications),
     with untied parameters built this way. An earlier note also gives +200–300
     MB/s for this decode, but no record of that run exists.
2. **"Gradient checkpointing strategy mismatch: Disabled vs Balanced".** A
   tensor went through `from_inner`, or through `valid()` then `train()`,
   and lost the `Balanced` strategy. Fix: restore the strategy (§8).
   **Measured** (on flex, an op between a `Balanced` leaf and
   `Tensor::from_inner(leaf.clone().inner())`): this panic.
3. **`x.no_grad()` has no effect.** `Tensor` has no inherent `no_grad`. The
   call resolves to `Module::no_grad`, which maps nothing on a bare tensor.
   Fix: `x.without_autodiff()` or `x.detach()`.
4. **A `ModuleMapper` skips cache tensors.** `Module::map` does not visit
   plain `Tensor` fields (§9.3). Fix: write the conversion by hand.
5. **`to_device` onto an autodiff device does not enable autodiff**, and
   onto a plain device it does not disable it (§5.1). Fix: `autodiff()` or
   `without_autodiff()` explicitly.
6. **`device_a == device_b` is `true`, but only one has autodiff.** Device
   equality ignores the context (§4).
7. **`detach` does not save memory in an untracked region.** The tensor stays
   on autodiff, and the ops after it register graph steps (§7).
8. **Dropout is off after `valid()`**, also on a plain device. `valid()`
   disables training flags (§9.2). Use `no_grad()` to stop only the
   gradients.
9. **The checkpointing strategy is gone after `valid().train()`.** Keep the
   original training module (§8).
10. **`require_grad()` panics.** The tensor has no autodiff, or it is a
    non-leaf (for example, the output of `to_device` on a tracked tensor).
    Fix: `autodiff()` first, or `detach()` first.
11. **"Cannot run backward: graph tape has already been consumed".** A second
    backward reached a graph step that the first one consumed. For example, a
    cache that the first backward used feeds a second pass. Fix: `detach`
    or round-trip the cache first, or combine the losses.
12. **`configure` returns `AlreadyInitialized`.** A tensor op or an earlier
    `configure` already locked the device settings. Configure the device once,
    before it creates any tensor.
13. **An evaluation during training fills the memory.** It ran on the
    training model: forward-only on `Grad` leaves (§7). Fix: `valid()`.

## 12. Debugging

- **State probes.** `t.is_autodiff()`, `t.is_tracked()`,
  `t.is_require_grad()`, `t.gradient_checkpointing_strategy()`,
  `device.is_autodiff()`. On a model, test a few `param.val()`, and the
  output of a forward pass.
- **Memory that grows with no backward.** Profile the heap with tcmalloc
  (package `google-perftools`) on a binary that keeps its symbols:

  ```sh
  CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build --release ...
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so \
    HEAPPROFILE=/tmp/hp/x HEAP_PROFILE_INUSE_INTERVAL=33554432 <bin> ...
  google-pprof --text --inuse_space --no_strip_temp \
    --base=/tmp/hp/x.0002.heap <bin> /tmp/hp/x.0004.heap
  ```

  The profiler slows the allocations a lot, but the growth keeps its
  shape. An autodiff graph that grows shows as
  `AutodiffTensor::from_parents`, `GraphMemoryManagement::register` and
  `CheckpointerBuilder::checkpoint`, next to the ops whose outputs it keeps
  (for example, `float_mul`).
- **Guard the run.** A leak can fill the RAM in about a minute. Read the
  resident memory (`/proc/self/statm`, or `ps -o rss=`) and stop the process
  above a limit.

## 13. PRs that changed the meaning of these methods

These PRs merged between 2026-08-27 and 2026-09-08. Code for an older Burn
can compile after them and do something different. Before the Dispatch API
(Burn 0.22), `inner` and `from_inner` converted between the types
`Tensor<B>` and `Tensor<B::InnerBackend>`. Code on a plain backend could not
call `from_inner`. Code ported from that API often keeps
`Tensor::from_inner(x.inner())` round trips, which now enable autodiff on
plain tensors.

| PR | Merged | Change | What older code gets wrong |
|---|---|---|---|
| [#5498](https://github.com/tracel-ai/burn/pull/5498) | 2026-08-27 | `Param<Flag>` for Dropout, BatchNorm, GaussianNoise and RRelu training behavior. `ParameterValue` (any value in a `Param`) is separate from `Parameter` (tensor capabilities). The lifecycle state of `Param` is `is_active`, not `require_grad`. Records omit flags. | A training switch is now a parameter that traversal, groups, `valid`, `train` and `freeze` reach. The configured trainability is `is_active`, which can differ from the tensor state. |
| [#5509](https://github.com/tracel-ai/burn/pull/5509) | 2026-08-28 | `Param::valid()` and the way back to training (`from_inner` in that PR, `Param::train` in 0.22.1) keep the save/load mapper (`from_mapped_value`). | Before it, a record saved from a `valid()` model lost the mapper. For example, Burn saved a `LinearLayout::Col` linear transposed, with no error. |
| [#5520](https://github.com/tracel-ai/burn/pull/5520) | 2026-09-01 | One routing for dispatch. An explicit `DispatchAutodiffContext { Disabled, Enabled(strategy) }` replaces the implicit checkpointing field. | Code that read the old field, or relied on implicit context rules. |
| [#5551](https://github.com/tracel-ai/burn/pull/5551) | 2026-09-02 | `merge`: `Disabled` promotes to `Enabled`, and two enabled contexts must share a strategy (§6). Restores the promotion that #5520 lost. Metric targets and predictions go to the inner backend. | Mixed plain/autodiff ops. Mixed strategies now panic. |
| [#5537](https://github.com/tracel-ai/burn/pull/5537) (closes issue [#5524](https://github.com/tracel-ai/burn/issues/5524)) | 2026-09-03 | `Module::no_grad()` controls only gradients. New `freeze()`/`unfreeze()` control gradients and flags. `valid()`/`train()` disable flags and restore them. `set_require_grad_group` ignores flags. LoRA/QLoRA freeze the base tensors and keep the flags. `LearningResult::model` is in validation mode. | Code that used `no_grad()` to also stop dropout. Code that continued to train `LearningResult::model` without `train()`. Code that expected `valid()` to do nothing on a plain device. |
| [#5557](https://github.com/tracel-ai/burn/pull/5557) | 2026-09-03 | `Tensor::no_grad` renamed to `Tensor::without_autodiff`. New `Tensor::autodiff`. `inner`/`from_inner` are aliases of these two. | `x.no_grad()` still compiles, but it now calls `Module::no_grad`, which does nothing to a tensor (§11, trap 3). `from_inner` lifts any plain tensor. |
| [#5571](https://github.com/tracel-ai/burn/pull/5571) | 2026-09-04 | New `Tensor::is_autodiff()` and `is_tracked()`. The docs separate autodiff association, graph participation and gradient retention (§3). | Code that used `is_require_grad()` as the only probe: it is `false` for every derived tensor. |
| [#5618](https://github.com/tracel-ai/burn/pull/5618) | 2026-09-08 | The optimizer (and L-BFGS) lift an updated parameter back with its earlier strategy and `require_grad`. | Before it, training with `Balanced` checkpointing panicked at the second training step. Hand-written update loops that lift with `from_inner` have the same fault (§8). |
