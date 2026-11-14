# 04: Batch B, the tests of `Layers`, the gradient horizon, `Layer` and the bidirectional stacks

Status: resolved

Blocked by: `02-block-contract-suite.md`,
`../burn-mamba/.scratch/container-tests-home/issues/01-coverage-run-and-record.md`

Spec: `../burn-mamba/.scratch/container-tests-home/spec.md`, sections "The sort rule"
and "New homes in burn-stack". Run only one session on the two repos.

**What to build:** each test of this batch gets a home in burn-stack, or a block
contract. Then burn-mamba ticket 04 deletes the moved tests.

**Read:** these files of `src/family/tests/` in burn-mamba: `layers.rs` (18 tests, 1059
lines), `layer.rs` (5 tests, 183 lines) and `bidi.rs` (2 tests, 116 lines). Also read
the matching `RefBlock` tests.

**Steps:**

1. Under `## Comments`, write the sorted list first. It has one line for each test of
   the batch: the rule number, the new home, and the block settings that the test uses.
   When the sort rule does not place a test, stop and ask the user.
2. Rule 1: port each test to `RefBlock`, or to no block, in the `RefBlock` test file of
   its mechanism. Skip a test that repeats a test of burn-stack, and name the repeat in
   the list.
3. Rule 2: add each new contract to `block_contract`, with its run on `RefBlock` and its
   negative test.
4. Delete each `RefBlock` test that repeats a contract.

**Known cases:**

- The two identity-prefix tests zero `out_proj.weight`. `RefBlock` makes that field
  public, so `RefBlock` needs no growth for them.
- `grad_horizon_memory_probe` follows rule 1. It moves over a `RefBlock` that is large
  enough for the activations to set the peak memory. It stays `#[ignore]`. Its doc gives
  the command for each configuration.

- [x] The sorted list has one line for each of the 25 tests.
- [x] Each test of rule 1 or 2 has its new home, or the name of the test that it repeats.
- [x] The memory probe runs over `RefBlock` with the command of its doc.
- [x] The coverage compare of burn-stack against the base commit of this ticket lists no
      lost item.
- [x] `cargo check`, `cargo doc --all --no-deps` and `scripts/check.sh` pass with no
      warnings.

## Comments

### The sorted list

The base commit of this ticket is `a236fe3`. A home in `src/reference/tests/` is a port
to `RefBlock`. "Mamba-3, `Minimal`" is the setting of `layers.rs` in burn-mamba:
`Mamba3SsdPath::Minimal(Some(4))`, `mimo_rank` 1, `rotation_fraction` 0.5.

`src/family/tests/layers.rs` of burn-mamba (18 tests):

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `full_horizon_matches_no_horizon` | 1 | `src/reference/tests/grad_horizon.rs`: `grad_horizon_cuts_the_prefix_but_not_the_input` gets the check of the gradients. It checked only the values of a full horizon. | Mamba-3, `Minimal` |
| `nothing_tracked_reaches_the_prefix` | 1 | `src/reference/tests/grad_horizon.rs`, as `nothing_tracked_reaches_the_untracked_caches` | Mamba-3, `Minimal` |
| `prefix_parameters_get_no_gradient` | 1 | `src/reference/tests/grad_horizon.rs`, as `untracked_layers_get_no_gradient`. It checks each float parameter of each real layer. | Mamba-3, `Minimal` |
| `shared_weight_grad_counts_tracked_applications_only` | 1 | `src/reference/tests/grad_horizon.rs` | Mamba-3, `Minimal` |
| `horizon_is_inert_without_autodiff` | 1 | `src/reference/tests/grad_horizon.rs` | Mamba-3, `Minimal`, plain device |
| `split_forward_matches_single_forward` | 1 | `src/reference/tests/grad_horizon.rs`: the control of the test below | Mamba-3, `Minimal` |
| `split_forward_matches_single_forward_under_horizon` | 1 | `src/reference/tests/grad_horizon.rs` | Mamba-3, `Minimal` |
| `step_matches_forward` | 1 | `src/reference/tests/grad_horizon.rs`: the control of the test below. `layers_forward_equals_step_unrolled` checks only the outputs. | Mamba-3, `Minimal` |
| `step_matches_forward_under_horizon` | 1 | `src/reference/tests/grad_horizon.rs`. `grad_horizon_mask_alternates_and_keeps_forward_step_parity` checks only the outputs. | Mamba-3, `Minimal` |
| `stepped_prefix_parameters_get_no_gradient` | 1 | `src/reference/tests/grad_horizon.rs`, as `stepped_untracked_layers_get_no_gradient` | Mamba-3, step only |
| `prime_under_a_cut_with_a_latent_below_it` | 1 | `src/reference/tests/grad_horizon.rs`. `grad_horizon_keeps_a_class_latent_of_the_stack_training` runs the forward, not the prime. | Mamba-3, step only |
| `boundary_weights_keep_their_gradient_under_a_cut` | 1 | `src/reference/tests/grad_horizon.rs`, as `the_io_weights_of_a_network_keep_their_gradient_under_a_cut` | Mamba-3, `Minimal`, in a `LatentNetwork` |
| `stepped_input_keeps_its_gradient_under_a_cut` | 1 | `src/reference/tests/grad_horizon.rs`, in `stepped_untracked_layers_get_no_gradient` (the same stack and the same step) | Mamba-3, step only |
| `a_cut_changes_gradients_only` | 1 | `src/reference/tests/grad_horizon.rs`, as `a_cut_keeps_the_outputs_and_the_caches` (the test checks only the values) | Mamba-3, `Minimal` |
| `identity_prefix_is_exact_under_standard_residuals` | 1 | `src/reference/tests/grad_horizon.rs`, as `an_identity_untracked_span_is_exact_under_standard_residuals`. It zeroes `out_proj.weight` of `RefBlock`. | Mamba-3, `Minimal` |
| `identity_prefix_is_exact_under_multi_gate_residuals` | 1 | `src/reference/tests/grad_horizon.rs`, as `an_identity_untracked_span_is_exact_under_multi_gate_residuals`. The source closed every gate (`init_bias` -20), so its tracked layers were near identities too. The port closes only the gates of the untracked layers, and checks that the stack is not an identity. | Mamba-3, `Minimal` |
| `multi_gate_input_keeps_its_gradient_under_a_cut` | 1, repeat | `grad_horizon_mask_alternates_and_keeps_forward_step_parity`: Multi-Gate with 3 streams, and an input that crosses 4 cuts. Its untracked layer 1 is in the accumulation phase. | Mamba-3, `Minimal` |
| `grad_horizon_memory_probe` | 1 | `src/reference/tests/grad_horizon.rs`, over a wide `RefBlock`, `#[ignore]` | Mamba-3, default path |

`src/family/tests/layer.rs` of burn-mamba (5 tests):

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `layer_with_mlp_matches_the_two_residual_reference` | 1, repeat | `layer_mlp_reproduces_two_separate_residuals` | Mamba-3, default path |
| `mlp_layer_forward_step_parity` | 1 | `src/reference/tests/layers.rs`, as `a_layer_with_a_feed_forward_keeps_forward_step_parity` | Mamba-3, default path |
| `block_only_layer_is_unchanged` | 1 | `src/reference/tests/layers.rs`, as `a_layer_without_a_feed_forward_adds_only_its_block` | Mamba-3, default path |
| `mlp_without_norm2_panics` | 1 | `src/reference/tests/layers.rs`, as `a_feed_forward_without_norm2_panics` | Mamba-3, default path |
| `norm_and_norm2_are_independent` | 1 | `src/reference/tests/layers.rs`, as `norm_and_norm2_are_distinct_parameters`. The source writes a new `norm2` and reads the old `norm`, so it cannot fail. The port checks that the two gains have different parameter ids, so a record and an optimizer hold each one. | Mamba-3 |

`src/family/tests/bidi.rs` of burn-mamba (2 tests):

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `virtual_layers_share_per_real_pair_merge` | 1, repeat | `bidi_virtual_pairs_share_the_real_pair_merge`: the same stack, with the default `StridedCyclic`. It gets the check of the merge count. | Mamba-2, default path |
| `virtual_forward_is_deterministic` | 1 | `src/reference/tests/bidi.rs`, as `bidi_virtual_forward_is_deterministic` | Mamba-2, default path |

#### No new contract

Each test of this batch checks a container: `Layers` with a gradient horizon, `Layer` with a
feed-forward, or a bidirectional stack. `RefBlock` takes the place of the Mamba block in
each of them. So no test is rule 2 or rule 3.

The split and step tests also compare gradients across the cache of a block. The block
contracts compare only values. burn-mamba checks the Mamba side of that property in the
family tests of its forms and paths. The spec keeps these tests out of scope. Examples are
`forward_single_ssd_split_matches_full` and its `_mimo` twin in
`src/mamba3/single_ssd/single_ssd/tests.rs`. They compare the gradients across a split
through the cache, on the `Minimal` path. The source tests also ran only the `Minimal`
path, whose backward is the autodiff of Burn.

The source tests read each field of the Mamba-3 cache, so that a backend hop that skips a
field shows. `RefCache` has one field. The contract `cache_traversal_moves_every_field`
checks that the traversal of a block visits each field of its cache. The backend hop of a
horizon goes through that traversal.

#### `RefBlock` tests that repeat a contract

None in `layers.rs`, `grad_horizon.rs` and `bidi.rs`. `layers_forward_equals_step_unrolled`
checks a `Layers` stack, not the bare block of `forward_equals_step_unrolled`.

### The red check

The ports test behavior that exists. So the red step of each port was a temporary break of
its mechanism in `src/modules/layers.rs` or `src/modules/layer.rs`, reverted after the run:

| Break | Ports that fail | Controls that pass |
|---|---|---|
| The step ignores the horizon | `step_matches_forward_under_horizon` (on the gradients only), `stepped_untracked_layers_get_no_gradient`, `prime_under_a_cut_with_a_latent_below_it` | `step_matches_forward` |
| The streams of Multi-Gate get no straight-through carry | `an_identity_untracked_span_is_exact_under_multi_gate_residuals` | `an_identity_untracked_span_is_exact_under_standard_residuals` |
| The step of a layer skips its feed-forward | `a_layer_with_a_feed_forward_keeps_forward_step_parity` | |
| An untracked layer reads its incoming cache on the autodiff backend | `nothing_tracked_reaches_the_untracked_caches` (only that test) | |
| A cut detaches the final cache of each tracked layer too | `split_forward_matches_single_forward_under_horizon` and `step_matches_forward_under_horizon`, on the gradients only | `split_forward_matches_single_forward`, `step_matches_forward` |
| The forward ignores the horizon | `untracked_layers_get_no_gradient`, `shared_weight_grad_counts_tracked_applications_only`, `nothing_tracked_reaches_the_untracked_caches`, and 4 older tests | |

Seven ports, and the new gradient check of `grad_horizon_cuts_the_prefix_but_not_the_input`,
got no break of their own.

### The memory probe

The probe runs 2 real layers of a `RefBlock` with `d_model` 256, a batch of 8 and 128
tokens. The commands are those of its doc (`tmp/probe-04/run.sh`, peaks in
`tmp/probe-04/peaks.txt`):

| virtual layers | no horizon | the last 2 layers |
|---|---|---|
| 8 | 110208 kB | 41316 kB |
| 32 | 388252 kB | 41516 kB |
| 64 | 761792 kB | 41632 kB |

At 64 virtual layers, the plain forward has a peak of 16836 kB. The untracked layers with
detached parameters (the last 2 layers tracked) have a peak of 545468 kB. So the
activations set the peak, and the inner backend keeps almost none of them. ADR-0003 and
`src/utils/detach.rs` now cite these numbers of the probe. They cited a probe on a
Mamba block that burn-mamba ticket 04 deletes.

### Coverage result

The base is the commit `a236fe3`. `scripts/coverage.py . --base a236fe3` lists no lost
item, and no covered item on a changed line. The report of this ticket is
`tmp/coverage-04/after.json`, and the compare is `tmp/coverage-04/compare.md`. The lib
tests of burn-stack cover the same items as at the base:

| measure | `a236fe3` | this ticket |
|---|---|---|
| lines | 3867 of 4415 | 3867 of 4415 |
| branches | 303 of 432 | 303 of 432 |
| conditions | 121 of 216 | 121 of 216 |

The ports run lines that the older tests of burn-stack already ran. The gate of
burn-mamba ticket 04 measures what the deletion of the source tests loses.

The nightly toolchain of the box had no `llvm-tools-preview`. The user chose that a run
of the script installs it through `cargo llvm-cov` (`CARGO_LLVM_COV_SETUP=yes`). This
run installed it.

### The review (`tmp/review-04-standards.md`, `tmp/review-04-spec.md`)

- Fixed:
  - The glossary terms: no "carry" alone and no "carrier" in the docs of the identity
    tests. The module header cites ADR-0003, and no longer contradicts its own checks.
  - The STE rules in the doc comments: the phrasal verbs, two long sentences, an ellipsis,
    and "probe" with two meanings (the closure is `grads_of_untracked_caches`).
  - The doc of the probe said that the steps of untracked ops keep no activation. It now
    states only what ADR-0003 states. Its reference curves run at 64 virtual layers.
  - The doc of the prime test: only the tracked layers are in the graph of its output.
  - `a_cut_changes_gradients_only` checks only values, so its port is
    `a_cut_keeps_the_outputs_and_the_caches`.
  - The shared code: `step_unrolled_with_caches` in `mod.rs`, `assert_same_caches` and
    `split_by_hand`. The names `label`, `x_earlier` and `weighted_loss`.
  - Each new message states the failure. `assert_only_tracked_layers_train` asserts its
    precondition.
  - Stale docs: the comment of `Layers::forward_rows` in `src/modules/layers.rs`, and
    the numbers of ADR-0003 and of `src/utils/detach.rs`. With the numbers of the
    `RefBlock` probe, a detach saves 28 % of the memory. So `detach.rs` and
    item 7 of `docs/burn/autodiff-semantics.md` say that a detach keeps most of the
    memory, not that it saves none.
  - The list names the contract that checks each field of a cache, and an example of
    burn-mamba that compares gradients across a split.
  - The peaks above come from the final code of the probe.
- Kept:
  - `src/utils/padding/tests.rs` has a copy of `param_grads` without the paths. A shared
    helper would change a test module outside this batch.
  - The new tensors have no shape suffix, as in the other files of `src/reference/tests/`.
  - §7 of `docs/burn/autodiff-semantics.md` measures about 1 KB for each untracked op,
    and `src/utils/detach.rs` says that the output of the op stays. This ticket does not
    settle which is true.
