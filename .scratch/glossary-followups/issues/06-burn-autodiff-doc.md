# 06: The Burn autodiff doc in burn-stack

Status: resolved

Blocked by: 01

**What to build:** An agent in burn-stack or in a consumer finds the semantics
of Burn for devices, autodiff, gradients and checkpointing in this repo. The
doc is correct for the Burn version of this crate. Today the doc is in the
burn-bug workspace (`/shared/claude/burn-bug/info/burn-autodiff-semantics.md`)
by accident, and it describes an older Burn revision.

## Decisions from the burn-stack grill

- The doc goes into burn-stack, under `docs/burn/`. burn-stack is the shared
  base of burn-mamba and burn-deltanet, and they already link into
  `../burn-stack/` for the glossary. A separate repo is for later, when a user
  outside these crates needs the doc.
- Copy the doc with `cp`. Do not edit or delete the burn-bug copy.
- ADR-0003 stays self-contained. It gets one "see also" link to the doc.

## What to change in the copy

- The version line names the Burn of this crate: 0.22.1 from crates.io (see
  `Cargo.lock`), not the `8f08fdad` git revision.
- §14 lists the differences at `8f08fdad`. Check each row of §13 and §14
  against the Burn 0.22.1 source. Rewrite §14 for 0.22.1, or delete it when no
  difference is left. Fix the anchors that point to it.
- Delete each reference that exists only in the burn-bug workspace
  (`projects/autodiff-retention`, `projects/untied-leak`, `issues/issue9.md`,
  "this workspace"). Keep each measured number and its **Measured** mark, and
  give its setup in one line.

## Acceptance criteria

- [x] The doc is under `docs/burn/`, and its version line names Burn 0.22.1.
- [x] `rg` finds no path outside this repo in the doc (`projects/`,
      `burn-bug`, `issues/issue`).
- [x] Each row of §14 is checked against the 0.22.1 source, or §14 is gone
      with its reason under `## Comments`.
- [x] ADR-0003 and the "where to look" map of CLAUDE.md link to the doc, and
      the links resolve.
- [x] The asd-ste100 lint (`ste-lint.py`) passes on the doc, or each
      remaining hit is listed under `## Comments`.

## Comments

**§14 is gone.** No row is a difference at 0.22.1. The tag `v0.22.1`
(`907ab111a`) has `8f08fdad` and `f328596` in its history, and each PR of §14.

| PR | Check on 0.22.1 | Before the PR (its parent commit in `../burn`) | Place in the doc |
|---|---|---|---|
| #5739 | Source: `Param::map_to_device` changes the device of an unshared lazy parameter. `burn-store/src/applier.rs` loads to `lazy_device`. | A move initializes the parameter on its first device, then copies it. | §9.1 |
| #5778 | Source: `collect_devices` reads `lazy_device`. Run: `is_initialized` stays `false`. | `collect_devices` reads `base().device()`, which can initialize the parameter. | §9.1 |
| #5746 | Source: the doc of `num_params`. Run: `is_initialized` stays `false`. | `num_params` visits each tensor, which can initialize the parameter. | §9.1 |
| #5821 | Source: `Param::valid` uses `base()` and `valid_dyn`. | `valid()` folds each reparameterization into the value. | §9.2 |
| #5837 | Run: `[1 2 3].repeat_dim(0, 2)` with weights 1 to 6 gives `[5, 7, 9]`. | The backward is interleaved, not tiled. The gradient is wrong when the weights vary along the repeated axis. | None: 0.22.1 has no trap here. |
| #5869 | Source: `get_device_settings`. Run: `configure` after `settings()` returns `Ok`. | `settings()` locks the settings, and a later `configure` fails. | §4 |
| #5722 | Source: `DispatchDevice::default` panics with no backend. | A build with no backend feature gets the Flex backend. | §4 |
| #5688 | Source: `Device::identity`, `Device::enumerate_physical`. | These calls do not exist. | §4 |
| #5838 | Source: the fields of `TensorData` are `pub(in crate::data)`. | The fields are public. | None: `TensorData` is outside the subject of the doc. |

This table replaces the "per the PR title" and "not checked" cells of §14.

**burn-deltanet.** It pins Burn `4a0d726` (2026-09-10). That revision has the
§13 PRs, but none of the PRs of this table. So the "before" column stays true
for burn-deltanet until it moves to 0.22.1. For example, its `repeat_dim`
gradient is wrong. The doc does not list these rows, because the ticket says
to delete §14, and CLAUDE.md says not to keep PR history. The "Older code"
line of the doc says that its list is not complete.

**§13 stays, with no new row.** CLAUDE.md says to delete PR history from docs.
This ticket asks to check each row of §13, and it does not ask to delete §13.
All its PRs and merge dates match `git log v0.22.1`. The maintainer can delete
§13 in a later change.

**File Map.** The map has only `src/`. So the "where to look" line of CLAUDE.md
names `docs/burn/`, as it does for `docs/adr/` and `docs/agents/`.

**Corrections.** The copy had these faults. Each fix has a run on 0.22.1,
unless its item says otherwise:

- §13, #5537: "`list_param_ids` removed" is false. The PR deleted only a test.
  The function is public in 0.22.1.
- §8: `train()` of a `Tensor` field resets the strategy only when the field has
  no autodiff. `from_inner` keeps the strategy of a `Balanced` tensor.
- Trap 12: an earlier `configure` also locks the settings.
- §9.3 and trap 4: "a derived round trip skips every field" is false. Only
  `map` and `visit` skip a plain `Tensor` field. A derived `valid()` removes
  its autodiff, and a derived `train()` enables autodiff with `Disabled`.
- Rule 8, §9.2: `train()` gives `Disabled` only to a plain parameter. A
  parameter on autodiff keeps its strategy (`Balanced` stays `Balanced`).
- §5.1 and trap 10: only a tracked tensor gives a non-leaf through `to_device`.
  `require_grad()` does not panic on an untracked one.
- §4: the `BURN_DEVICE` row follows 0.22.1. `metal`, `vulkan` and `webgpu` pin
  the API, and an unknown name panics. A second `configure` also fails. Only
  the second `configure` has a run. The `BURN_DEVICE` text comes from the source.

The 73 commits from `f328596` to `v0.22.1` change no other claim of the doc.
They do not touch the autodiff graph or the runtime.

**Glossary.** The recipes and trap 7 used three avoided terms. They now use
"untracked region" and "window". ADR-0003 uses "untracked region", but
`GLOSSARY.md` does not have it. "Graph step" names a step of the Burn graph,
and "training step" and "iteration" name the other meanings.

**Measured numbers.** A run on 0.22.1 repeats each number of the §7 table
(flex, `=0.22.1`, the programs of the burn-bug workspace copied to `/tmp`). The
first line of trap 1 and the setup of trap 2 are also runs on 0.22.1.

The Mamba-3 line of trap 1 comes from Burn `8f08fdad`, and the doc names that
revision. A rerun needs a burn-stack with the old `tile`.

- Burn-stack commit `b1ce7a8` records the 3.5× and the ms/char values.
- `handoff/handoff-untied-leak.md` of the burn-bug workspace records
  ~150–170 MB/s at 12.5 ms/char. This number is new in the doc.
- No log records "+200–300 MB/s", and the copy did not mark it as measured. The
  ticket says to keep the number, so the doc keeps it as a note, with no
  **Measured** mark.

Trap 2 of the copy gave no number and no setup. Its new setup is one op on
0.22.1. Burn-stack commit `6f24f0a` records the gradient horizon case, and the
test `grad_horizon_keeps_the_checkpointing_strategy` guards that fix.

**Lint.** `ste-lint.py` has 0 hard hits. These 7 advisory `passive-voice` hits
remain, because each word is a state or a quote:

- Lines 100, 148, 162, 190: "is untracked". This is the state of a tensor.
- Line 113: "is enabled". This is the state of a feature.
- Lines 168, 438: "been consumed". This is the panic message of Burn.
