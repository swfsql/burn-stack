# 13: One tiling of the untied parameters, from `untied_params`

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-16. Line numbers are those of burn-stack
commit `b198fa5` and burn-mamba commit `7807d31b`. The review compiled nothing.

**Files:** `src/utils/untied.rs`, `src/modules/mod.rs` (`BlockConfig::init_block`),
`src/modules/layer.rs:76-105`, `src/modules/layers.rs:932-945`,
`src/modules/bidi.rs:417-430`, `src/reference.rs:217-300`,
`../burn-mamba/src/mamba3/mamba3.rs:735-771,1255-1500`,
`../burn-mamba/src/mamba2/mamba2.rs`, `../burn-mamba/src/mamba1/mamba1.rs`,
`../burn-mamba/src/family/adapters.rs`.

**Problem:** Each untied parameter has its axis in two lists. `untied_params` gives it
in `UntiedParam::new(&param, axis)`, and `view` and `retie` read that list. The init
does not read it: `init_applications` calls `untied::tile(param, axis, copies(…))` for
each parameter by hand. The two lists have about 55 call sites: Mamba-3 20, Mamba-2
17, Mamba-1 11, `Layer` 3, `RefBlock` 4. `copy_len` catches a wrong axis only when
the length of that axis is not a multiple of the count. So `init_block` takes
`n_applications`, and each family passes the count to each parameter.

**Direction:** `Layer::init` tiles the layer and its block in one pass over
`untied_params()`. The mapper of `untied::retie` already does this by `ParamId`: it
narrows to the first copy and repeats it along the axis. `init_block` loses
`n_applications`, a family builds the tied block, and the three `init_applications`
go. A new untied parameter then needs one line.

**To check at triage:**

- The order of the random draws stays, so an untied stack still starts equal to the
  tied stack.
- `in_proj_tail` of Mamba-3 and Mamba-2: its weight has the copies along axis 1, and
  its bias along axis 0. Its width at the build is that of one application.
- `require_grad` of the tiled leaf (see the comment in the `Retie` mapper).

**ADR:** ADR-0008 stays. The copies stay side by side on an existing axis.

**Related:** `.scratch/untied-application-axis/issues/01-application-axis.md` changes
the layout of the copies. This ticket makes that change smaller, because the layout
then has one writer.
`../burn-mamba/.scratch/untied-kalman-tropical/issues/01-untied-kalman-and-tropical-tensors.md`
adds untied tensors to the block contracts.

Split this ticket by repo at triage.
