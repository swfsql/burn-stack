# 05: Batch C, the tests of Multi-Gate Residuals and the optimizer

Status: resolved

Blocked by: `02-block-contract-suite.md`,
`../burn-mamba/.scratch/container-tests-home/issues/01-coverage-run-and-record.md`

Spec: `../burn-mamba/.scratch/container-tests-home/spec.md`, sections "The sort rule"
and "New homes in burn-stack". Run only one session on the two repos.

**What to build:** each test of this batch gets a home in burn-stack, or a block
contract. Then burn-mamba ticket 05 deletes the moved tests.

**Read:** these files of `src/family/tests/` in burn-mamba: `multi_gate.rs` (17 tests,
861 lines) and `optim.rs` (16 tests, 597 lines). Also read the matching `RefBlock`
tests.

**Steps:**

1. Under `## Comments`, write the sorted list first. It has one line for each test of
   the batch: the rule number, the new home, and the block settings that the test uses.
   When the sort rule does not place a test, stop and ask the user.
2. Rule 1: port each test to `RefBlock`, or to no block, in the `RefBlock` test file of
   its mechanism. Skip a test that repeats a test of burn-stack, and name the repeat in
   the list.
3. Rule 1, no block:
   - The 6 tests of `MultiGateResidual` alone go to the tests of the Multi-Gate module.
   - The 5 tests of `Segmented` alone go to the tests of the `Segmented` module. This
     gives the optimizer module its first tests. Update the "Muon plan" seam row of
     `docs/agents/testing.md`, because it says that the optimizer module has no tests.
4. Rule 2: add each new contract to `block_contract`, with its run on `RefBlock` and its
   negative test. The Muon plan of each family can match the Muon contract of ticket 02.
   Then mark it "replaced by contract" in the list.
5. Delete each `RefBlock` test that repeats a contract.

- [x] The sorted list has one line for each of the 33 tests.
- [x] Each test of rule 1 or 2 has its new home, or the name of the test that it repeats.
- [x] The optimizer module has its own tests, and the seam row says so.
- [x] The coverage compare of burn-stack against the base commit of this ticket lists no
      lost item.
- [x] `cargo check`, `cargo doc --all --no-deps` and `scripts/check.sh` pass with no
      warnings.

## Comments

### The sorted list

The base commit of this ticket is `d534e7d`. "Default path" is the default SSD path of
the family. A home in `src/reference/tests/` is a port to `RefBlock`.

`src/family/tests/multi_gate.rs` of burn-mamba (17 tests):

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `forward_step_parity` | 1, no block | `src/modules/multi_gate/tests.rs` | none |
| `init_is_convex_mean` | 1, no block | `src/modules/multi_gate/tests.rs` | none |
| `gradients_flow` | 1, no block | `src/modules/multi_gate/tests.rs` | none |
| `accumulate_appends_a_stream` | 1, no block | `src/modules/multi_gate/tests.rs` | none |
| `init_bias_offset_ramps_the_gates` | 1, no block | `src/modules/multi_gate/tests.rs` | none |
| `attn_pool_reproduces_a_row_present_in_every_stream` | 1, no block | `src/modules/multi_gate/tests.rs`, as `attn_pool_reproduces_a_vector_present_in_every_stream` (a glossary **Row** is a sequence) | none |
| `layers_standard_ignore_residuals_parity` | 1 | `src/reference/tests/layers.rs` | Mamba-2, default path |
| `layers_multi_gate_forward_step_parity` | 1, repeat | `multi_gate_forward_equals_step_and_stays_bounded`. Its 4 layers accumulate and mix. The 2 layers of this test only accumulate. | Mamba-2, default path |
| `layers_multi_gate_virtual_forward_step_parity` | 1 | `src/reference/tests/multi_gate.rs` | Mamba-3, default path, `mimo_rank` 1 |
| `layers_multi_gate_per_virtual_ignore_residuals_parity` | 1 | `src/reference/tests/multi_gate.rs` | Mamba-2, default path |
| `layers_multi_gate_streams_are_distinct` | 1 | `src/reference/tests/multi_gate.rs` | Mamba-2, default path |
| `bidi_multi_gate_forward_and_grads` | 1 | `src/reference/tests/multi_gate.rs` | Mamba-2, default path |
| `bidi_multi_gate_class_latents_forward` | 1 | `src/reference/tests/multi_gate.rs` | Mamba-2, default path |
| `layers_multi_gate_stack_class_latents_step_matches_forward` | 1 | `src/reference/tests/multi_gate.rs` | Mamba-2, default path |
| `layers_multi_gate_class_latents_split_forward_matches_single` | 1 | `src/reference/tests/multi_gate.rs` | Mamba-2, `Minimal` |
| `layers_multi_gate_prime_then_step_matches_step` | 1 | `src/reference/tests/multi_gate.rs` | Mamba-2, step only |
| `latent_network_multi_gate_class_markers_prime_step_matches_forward` | 1 | `src/reference/tests/multi_gate.rs` | Mamba-2, default path |

`src/family/tests/optim.rs` of burn-mamba (16 tests):

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `all_adamw_sub_projections_equal_plain_adamw` | 1, no block | `src/optim/segmented/tests.rs` | none |
| `one_muon_sub_projection_equals_plain_muon` | 1, no block | `src/optim/segmented/tests.rs` | none |
| `per_sub_projection_muon_differs_from_fused_muon` | 1, no block | `src/optim/segmented/tests.rs` | none |
| `segmented_state_round_trips` | 1, no block | `src/optim/segmented/tests.rs` | none |
| `sgd_fallback_sub_projections_are_stateless_sgd` | 1, no block | `src/optim/segmented/tests.rs` | none |
| `mamba3_plan_fits_the_model` | 2 | Replaced by contract. See "The plans of the families" below. | Mamba-3, `Complex`, in a `LatentNetwork` with an MLP |
| `mamba3_real1d_plan_fits_the_model` | 2 | Replaced by contract, on the Mamba-3 `Real` setting | Mamba-3, `Real` |
| `mamba2_plan_fits_the_model` | 2 | Replaced by contract | Mamba-2, in a `VocabNetwork` |
| `mamba1_plan_fits_the_model` | 2 | Replaced by contract | Mamba-1, in a `LatentNetwork` |
| `bidi_plan_fits_the_model` | 1 | `src/reference/tests/muon.rs`: the block specs select the block of each direction | Mamba-3, default path, `CatLinear` |
| `bidi_plan_holds_the_cat_linear_merge` | 1, repeat | `bidi_plan_holds_the_cat_linear_spec_only_for_cat_linear_merges` | Mamba-3 |
| `vocab_plan_holds_the_mlp` | 1 | `src/reference/tests/muon.rs`, with the count of the specs of the feed-forward, and the check that the plan of a network selects no I/O weight | Mamba-2, in a `VocabNetwork` with an MLP |
| `mamba3_scalar_channels_stay_on_adamw` | 3 | Stays in burn-mamba: the Δ/A/λ channels are Mamba-3 math | Mamba-3 config only |
| `build_assembles_groups_without_panicking` | 1 | `src/reference/tests/muon.rs`, with a fused spec, so that the plan builds a `Segmented` group | Mamba-3, default path |
| `module_optimizer_state_round_trips_through_a_record` | 1 | `src/reference/tests/muon.rs`, with a fused spec | Mamba-3, default path |
| `describe_reports_every_parameter` | 1 | `src/reference/tests/muon.rs`, with a fused spec in place of the `dt` channel of Mamba-3 | Mamba-3 |

#### The plans of the families

The four `*_plan_fits_the_model` tests make three checks on the plan of a network:

- Each spec selects rank-2 weights, and its widths sum to their columns. The Muon
  contract of ticket 02 states this.
- No spec selects an I/O weight of the network (the embedding, the LM head, the class
  marker tables, the `in_proj` and `out_proj` of a `LatentNetwork`). No contract states
  this. A new contract `muon_projections_select_only_weights_of_the_block` states it.
  A family that names a weight with a `ProjSpec::path` breaks it.
- The plan holds a fixed count of specs (Mamba-3: 2, Mamba-2: 2, Mamba-1: 3, plus the
  2 specs of an MLP). This pins the list of each family. No contract can state it.

Each spec of these tests selects at least 2 weights, one in each real layer. The match
in each real layer is a property of the plan, not of the family.
`muon_plan_matches_only_existing_rank_2_weights` checks it (3 specs × 2 real layers).
`bidi_plan_fits_the_model` also checks the I/O weights. A bidirectional stack has none,
so its port drops that check.

The ports of `build_assembles_groups_without_panicking`,
`module_optimizer_state_round_trips_through_a_record` and
`describe_reports_every_parameter` use a plan with a fused spec, written in the test.
`RefBlock` has no fused projection, so no container gives that plan. The test of the
plan of a container is `vocab_plan_holds_the_mlp`.

The user chose:

- The new contract, not a wider Muon contract of ticket 02.
- The count of the specs of each family is rule 3. burn-mamba ticket 05 moves these
  counts into one small rule-3 test, beside `mamba3_scalar_channels_stay_on_adamw`. The
  MLP part of the count goes with the port of `vocab_plan_holds_the_mlp`.

#### `RefBlock` tests that repeat a contract

None in `multi_gate.rs` and `muon.rs`. `muon_plan_matches_only_existing_rank_2_weights`
overlaps the Muon contract, but it also checks the 2 real layers and the decay on the
fallback optimizer. The two tests of `block.rs` that repeat contracts belong to batch D.

### Coverage result

The base is `tmp/coverage-record/burn-stack-by-burn-stack.json` of burn-mamba, the saved
report of the clean `d534e7d`. `scripts/coverage.py --base` lists no lost item. It lists
13 lines and 2 branches on changed lines: the lines of
`muon_projections_name_existing_rank_2_weights` that now call `muon_stack` and
`selected`. The new lines keep them covered, with both outcomes of `spec.tiled`. The lib
tests of burn-stack cover more:

| measure | `d534e7d` | this ticket |
|---|---|---|
| lines | 3639 of 4372 | 3867 of 4415 |
| branches | 257 of 430 | 303 of 432 |
| conditions | 93 of 215 | 121 of 216 |

Before this ticket, only the burn-mamba tests covered 418 lines and 67 branches of
burn-stack (`burn-stack-by-both.json` of the record). Now 233 lines and 23 branches remain
(`tmp/coverage-05/still-only-by-burn-mamba.md`). None of them is in
`src/modules/multi_gate.rs`. Each one that is in the code of this batch belongs to a test
of another batch:

- The class tests of batch A: the ignored residuals of a bidirectional stack, the `End`
  markers, and a prime from a cache.
- The `MuonPlan::build` of an untied spec (a branch of `ProjSpec::is_whole_muon`), and
  the init policy with an MLP: batch D.

### The review (`tmp/review-05-standards.md`, `tmp/review-05-spec.md`)

- Fixed:
  - Each tolerance goes through `dtype_tol`. A check that two optimizer updates differ
    compares them in units of the learning rate. Below 1, `max_rel_diff` is an
    absolute difference.
  - The second training step of the record round trip has its own input. With the
    same gradient twice, the update from the record differs from a first update by
    only about 1 %.
  - The round trip also checks that the reloaded optimizer and the live one give the
    same update.
  - The comment on the order of the markers in
    `latent_network_multi_gate_class_markers_prime_step_matches_forward`: the class
    latent of the stack comes first.
  - The glossary terms: optimizer update, row, I/O weight, feed-forward, and the passes
    of a bidirectional pair.
  - The test `latent_plan_selects_no_io_weight` repeated the new contract on
    `RefBlock`. The fix deleted it.
  - `vocab_plan_holds_the_mlp` checks the count of the specs of the feed-forward.
  - The shared code: `optim::report::selected`, one stack for the two Muon contracts,
    one bidirectional builder, and `step_unrolled` in the older forward/step tests.
- Kept:
  - `assert_plan_fits` of `muon.rs` has no branch for a tiled spec. The plans of these
    tests have none.
  - Three tests use a plan that the test writes, because `RefBlock` has no fused
    projection.
