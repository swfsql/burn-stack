# 06: Batch D, the tests of `MambaBlock`, the untied copies, packed rows, the H-Net, the captured step, the configs and the reset alignment

Status: resolved

Blocked by: `02-block-contract-suite.md`,
`../burn-mamba/.scratch/container-tests-home/issues/01-coverage-run-and-record.md`

Spec: `../burn-mamba/.scratch/container-tests-home/spec.md`, sections "The sort rule"
and "New homes in burn-stack". Run only one session on the two repos.

**What to build:** each test of this batch gets a home in burn-stack or a block
contract, or it stays in burn-mamba (rule 3). Then burn-mamba ticket 06 deletes the
moved tests.

**Read:** these files of `src/family/tests/` in burn-mamba:

| File | Tests | Lines |
|---|---:|---:|
| `block.rs` | 16 | 617 |
| `untied.rs` | 5 | 218 |
| `packed.rs` | 4 | 236 |
| `hnet.rs` | 4 | 223 |
| `capture.rs` | 4 | 160 |
| `config.rs` | 5 | 167 |
| `reset_align.rs` | 5 | 82 |

Also read the matching `RefBlock` tests and the H-Net tests of burn-stack.

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

**First guesses (the sort decides):**

- `record_round_trip` and the record tests of `MambaBlock`: rule 2, or a repeat of the
  record contract of ticket 02.
- `mamba1_assign_moves_every_field` and `mamba2_assign_moves_every_field`: rule 2, or a
  repeat of the cache-traversal contract of ticket 02.
- The JSON round trips of the configs: rule 2. Each family crate needs its configs to
  survive JSON.
- The reset alignment of each Mamba path: rule 3. The chunk length is Mamba math.
- `pathway_mismatch_panics` and `mamba3_layout_follows_the_untie_list`: rule 3.

- [x] The sorted list has one line for each of the 43 tests.
- [x] Each test of rule 1 or 2 has its new home, or the name of the test that it repeats.
- [x] The coverage compare of burn-stack against the base commit of this ticket lists no
      lost item. Its one lost line is a closing brace with no coverage region (see
      "Coverage result").
- [x] `cargo check`, `cargo doc --all --no-deps` and `scripts/check.sh` pass with no
      warnings.

## Comments

### The sorted list

The base commit of this ticket is `5416d97`. A home in `src/reference/tests/` is a port
to `RefBlock`. "Default path" is the default SSD path of the family. The Mamba-3 setting
of the source has `mimo_rank` 1 and `rotation_fraction` 0.5, when the line gives no other
value. "In `MambaBlock`" is the family enum of burn-mamba (its ADR-0009). A contract name
in bold is a new contract of this ticket.

`src/family/tests/block.rs` of burn-mamba (16 tests):

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `module_surface` | 3 | Stays: the `Module` derive of the enum (clone, fork, `to_device`, `no_grad`, a mapper, `Display`, `Debug`) | Each family, in `MambaBlock`, in a `LatentNetwork` |
| `param_paths_add_only_the_variant` | 3 | Stays: the path segment of a variant | Each family, in `MambaBlock` and alone |
| `record_round_trip` | 1, and 2, repeat | The record of a `LatentNetwork` moves to `src/reference/tests/layers.rs`. The port goes through bytes. The source also saves and loads a file, and the port drops that check. `Module::save_file` writes the same bytes through the file I/O of Burn, and it needs the `std` feature of Burn. No test of burn-stack loads the record of a network. The record of the block repeats `record_round_trip_keeps_the_block`. | Each family, in `MambaBlock`, default path |
| `record_of_another_family_panics` | 3 | Stays: the enum | Mamba-2 and Mamba-3, in `MambaBlock` |
| `block_output_is_exact` | 3 | Stays: the enum changes no math | Each family, in `MambaBlock` and alone, default path |
| `network_output_and_gradients_are_exact` | 3 | Stays: the enum changes no math | Mamba-3, in `MambaBlock` and alone, default path |
| `cache_stack_and_traversal` | 2, and 3 | **`cache_stack_moves_every_slot`**: the slots of `Block::Caches` survive `into_slots` and `from_slots`, and the cache traversal of the stack moves each slot. The detach repeats `detach_keeps_the_device_of_each_slot`. The step from the zero caches of a stack repeats `zero_cache_equals_no_cache`. Rule 3: the empty slots of `MambaCaches` and the single-SSD zero cache of Mamba-3 stay. | Each family, in `MambaBlock`, default path, autodiff |
| `captured_step_over_flat_caches` | 1, repeat, and 2 | `captured_latent_steps_are_the_eager_steps` of `src/utils/capture/tests.rs`. The write-back through the caches of the stack is **`cache_stack_moves_every_slot`**. | Mamba-3, in `MambaBlock`, step only |
| `stack_knobs` | 1, repeat, and 2 | `grad_horizon_mask_alternates_and_keeps_forward_step_parity` (virtual layers, a horizon, Multi-Gate) and `grad_horizon_keeps_a_class_latent_of_the_stack_training`. The run of the block on the inner backend under a cut is **`cut_keeps_the_outputs_and_the_caches`**. | Each family, in `MambaBlock`, default path, autodiff |
| `prime_and_class_tokens_through_the_enum` | 3 | Stays. Batch A deleted `family_net_config_*` as repeats of this test. Its container half repeats `latent_network_multi_gate_class_markers_prime_step_matches_forward`, and its block half repeats `zero_cache_equals_no_cache`. | Each family, in `MambaBlock`, default path |
| `untied_through_the_enum` | 2 | **`untied_stack_equals_the_stack_of_its_application_views`**, on a `MambaBlock` setting with untied parameters. The tiled shapes of the enum follow: a block of the wrong count of copies fails the contract. | Mamba-3, `DtBias` and `D` untied, in `MambaBlock`, default path |
| `hnet_mixes_stage_and_main_family` | 3 | Stays: one network type holds stages of one family and a main stack of another (ADR-0009 of burn-mamba). Its H-Net half repeats `forward_equals_step_unrolled` and `muon_plan_scopes_each_stack` of `src/modules/hnet/tests.rs`. | Mamba-2 stages (width 8), Mamba-3 main stack, in `MambaBlock` |
| `muon_plan_matches_the_family_type` | 2, and 3, repeat | The selection through the path segment of a variant is `muon_projections_name_existing_rank_2_weights` on `MambaBlock`. The count of the specs is `each_family_gives_a_fixed_count_of_muon_specs` (burn-mamba `optim.rs`, rule 3). The optimizer step repeats `build_assembles_groups_without_panicking`. | Mamba-3, in `MambaBlock` |
| `wrong_path_panics` | 3 | Stays: the enum | Mamba-2 path, Mamba-3 block |
| `wrong_cache_panics` | 3 | Stays: the enum | Mamba-2 cache, Mamba-3 block |
| `mixed_stack_rules` | 3 | Stays: the enum | Mamba-2 and Mamba-3 in one stack |

`src/family/tests/untied.rs` of burn-mamba (5 tests):

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `mamba3_untied_stack_equals_the_stack_of_its_application_views` | 2 | **`untied_stack_equals_the_stack_of_its_application_views`**. `an_untied_stack_equals_the_stack_of_its_application_views` (`src/reference/tests/untied.rs`) stays: it also unties the norms of each layer. | Mamba-3, `mimo_rank` 2, output norm, each untiable tensor except the learnable initial state, `Minimal(Some(4))` |
| `mamba2_untied_stack_equals_the_stack_of_its_application_views` | 2 | **`untied_stack_equals_the_stack_of_its_application_views`** | Mamba-2, each untiable tensor except the learnable initial state, `Minimal(Some(4))` |
| `mamba1_untied_stack_equals_the_stack_of_its_application_views` | 2 | **`untied_stack_equals_the_stack_of_its_application_views`** | Mamba-1, `state_rank` 4, each untiable tensor |
| `mamba3_layout_follows_the_untie_list` | 3 | Stays: the split of the `in_proj` tail of Mamba-3 | Mamba-3 config only |
| `mamba3_untied_tail_trains_one_copy_per_application` | 2, and 1, repeat | The tiled spec of the untied tail is `muon_projections_name_existing_rank_2_weights` on the untied Mamba-3 setting. The update of each copy repeats `muon_updates_an_untied_projection_one_copy_at_a_time`. | Mamba-3, `InProjTail` untied, 3 applications, `Minimal(Some(4))` |

`src/family/tests/packed.rs` of burn-mamba (4 tests). The four tests share one check:

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `packed_mamba3_u1_matches_alone` | 1, and 2, repeat | The vocab network half moves to `src/reference/tests/packed.rs`: the logits and the gradients of each parameter, with 2 class latents. No test of burn-stack compares the gradients of a packed network. The block half repeats `packed_segments_equal_each_segment_alone`. The gradients of the block are in `src/packing/tests.rs` of burn-mamba (a family test outside this batch). | Mamba-3, `SerialRecalculated(Some(4))` |
| `packed_mamba3_u2_vertical_multigate_matches_alone` | 1, and 2, repeat | The port above runs also over Multi-Gate. No packed test of burn-stack has Multi-Gate. The block half repeats `packed_segments_equal_each_segment_alone`. | Mamba-3, `micro_steps` 2, `Vertical`, `SerialRecalculated(Some(8))` |
| `packed_mamba3_serial_matches_alone` | 1, repeat, and 2, repeat | The port above, and `packed_segments_equal_each_segment_alone` | Mamba-3, `Serial(Some(4))` |
| `packed_mamba2_matches_alone` | 1, repeat, and 2, repeat | The port above, and `packed_segments_equal_each_segment_alone` | Mamba-2, `SerialRecalculated(Some(4))` |

`src/family/tests/hnet.rs` of burn-mamba (4 tests):

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `mamba2_stages_mamba3_main_forward_equals_step` | 1, repeat, and 2 | `forward_equals_step_unrolled` of `src/modules/hnet/tests.rs`: two stages, a wider main stack, both step modes. The row select of the H-Net step goes through the cache traversal of each stack: **`cache_stack_moves_every_slot`**. | Mamba-2 stages (width 8), Mamba-3 main stack (width 16), default paths |
| `mamba3_forward_and_step_have_the_same_gradients` | 1 | `forward_and_step_have_the_same_gradients` of `src/modules/hnet/tests.rs` gets the check of each float parameter. It checks 5 weights, the width vector and the input. | Mamba-3 stage (width 8) and main stack (width 16), default path, autodiff |
| `mamba3_padded_rows_match_each_row_alone` | 1, and 2 | `padded_rows_match_each_row_alone` of `src/modules/hnet/tests.rs` also runs from no cache. The source starts from no cache, and checks that the caches of the empty row stay finite. The port checks that the empty row keeps its row of `HNet::zero_caches`. A NaN fails that check. The block half: **`padded_rows_equal_each_row_alone` also runs from no cache**, and the empty row gives its row of the zero cache. | Mamba-3 stage (width 8) and main stack (width 16), default path |
| `the_muon_plan_fits_every_stack` | 1, repeat, and 2, repeat | `muon_plan_scopes_each_stack`. The width of each fused spec is `muon_projections_name_existing_rank_2_weights`. The Muon step of a fused spec repeats `build_assembles_groups_without_panicking`. | Mamba-2 stage (width 8), Mamba-3 main stack (width 16) |

`src/family/tests/capture.rs` of burn-mamba (4 tests):

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `owned_buffers_keep_values_and_assign_moves_them` | 3 | Stays: it compares each field of the Mamba-3 cache, with each optional field present, on both pathways. No contract reaches the double-SSD pathway, and `cache_traversal_moves_every_field` does not see a field that a step does not change. Its slotted and flat forms are also **`cache_stack_moves_every_slot`**. | Mamba-3, `ngroups` 2, `micro_steps` 2, `VerticalPlusHorizontalCarryOver`, Kalman gain, `MaxPlus`, `Quaternion4D`, single-SSD and double-SSD |
| `pathway_mismatch_panics` | 3 | Stays: the two pathways of Mamba-3 | As above |
| `mamba2_assign_moves_every_field` | 2, repeat | `cache_traversal_moves_every_field` | Mamba-2, step only |
| `mamba1_assign_moves_every_field` | 2, repeat | `cache_traversal_moves_every_field` | Mamba-1, step only |

`src/family/tests/config.rs` of burn-mamba (5 tests). Each JSON test checks three things:
the knobs of the shape, the config of the block, and the `{ shape, block }` struct of
burn-mamba. The user chose: the config of the block becomes a contract, and the four
JSON tests stay unchanged. So burn-stack gets no JSON test of its shapes.

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `latent_net_config_survives_a_json_round_trip` | 3, and 2 | Stays: `MambaLatentNetConfig`, its round trip and its build. The config of the block is also **`json_round_trip_keeps_the_config`**. | Each family, in `MambaBlock` |
| `vocab_net_config_survives_a_json_round_trip` | 3, and 2 | As above, for `MambaVocabNetConfig` | Each family, in `MambaBlock` |
| `bidi_layers_config_survives_a_json_round_trip` | 3, and 2 | As above, for `MambaBidiLayersConfig`. Batch A deleted `family_bidi_config_*` as repeats of the build of this config. | Each family, in `MambaBlock` |
| `hnet_vocab_net_config_survives_a_json_round_trip` | 3, and 2 | As above, for `MambaHNetVocabNetConfig` | Each pair of families, in `MambaBlock` |
| `default_ssd_path_fits_the_family_of_the_block` | 3 | Stays: `MambaBlockConfig::default_ssd_path`. Batch A deleted `family_net_config_*` as repeats of this test. | Each family, in `MambaBlock` |

`src/family/tests/reset_align.rs` of burn-mamba (5 tests). The reset alignment of a
Mamba block is its chunk length in tokens, which is Mamba math:

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `a_mamba2_block_aligns_to_its_chunk` | 3 | Stays | Mamba-2, `SerialRecalculated`, `Serial` |
| `a_mamba2_block_on_the_minimal_path_takes_no_packed_rows` | 3 | Stays | Mamba-2, `Minimal` |
| `a_mamba3_block_aligns_to_its_chunk_in_tokens` | 3 | Stays | Mamba-3, `micro_steps` 1–3, `Vertical`, `SerialRecalculated` |
| `a_mamba3_block_on_the_minimal_path_takes_no_packed_rows` | 3 | Stays | Mamba-3, `Minimal` |
| `a_mamba1_block_takes_no_packed_rows` | 3 | Stays | Mamba-1 |

#### The new contracts

- `untied_stack_equals_the_stack_of_its_application_views`: an untied block in a stack
  equals the stack of its application views, in the forward and in the step. Also,
  `Block::untied_params` names each tensor that grows with the count of applications,
  along the axis where it grows. It returns early when the block unties no parameter.
- `cache_stack_moves_every_slot`: the caches of a layer stack (`Block::Caches`) keep
  each slot through `into_slots` and `from_slots`. Their cache traversal moves each
  slot. The H-Net and the captured step need this traversal.
- `cut_keeps_the_outputs_and_the_caches`: a stack with a cut gives the outputs and the
  final caches of the same stack with no cut. The untracked layer runs `Module::valid`
  of its block on the inner backend, and its cache goes through the backend hop.
- `json_round_trip_keeps_the_config`: the config of the block survives a JSON round
  trip. The parsed config writes the same JSON. Its block loads the record of a block of
  the config, and gives the same outputs in the forward and in the step.
- `padded_rows_equal_each_row_alone` also runs from no cache. The empty row then gives
  its row of the zero cache.

The user chose to add `cut_keeps_the_outputs_and_the_caches`, and to extend
`padded_rows_equal_each_row_alone`.

#### `RefBlock` tests that repeat a contract

- `block_forward_equals_step_unrolled` repeats `forward_equals_step_unrolled`.
- `block_forward_splits_into_windows_through_the_cache` repeats
  `forward_splits_into_windows_through_the_cache`.
- `positions_before_the_first_reset_continue_from_the_cache` repeats
  `packed_segments_equal_each_segment_alone`: row 1 of the contract continues from the
  incoming cache before its first reset.

`a_cut_keeps_the_outputs_and_the_caches` of `src/reference/tests/grad_horizon.rs` stays.
It does not repeat the contract `cut_keeps_the_outputs_and_the_caches`. It tests the
straight-through carry of the container over 4 layers, with the horizons 4, 3, 1 and 0.
The horizon 0 ends the stack inside an untracked span. The contract tests the block on
the inner backend, with 1 horizon over 2 layers, in the forward and in the step.

#### For burn-mamba ticket 06

- The new untied settings above go into the block settings of `block_contract.rs`. So
  the untied contract applies, and the Muon contract checks a tiled spec.
- `Contract::run` of `block_contract.rs` gets the bound of the new contract
  `cache_stack_moves_every_slot`: `Block::Caches: CacheTensors`. The JSON contract needs
  no new bound, because `BlockConfig` requires `Config`.
- The `MambaBlock` setting runs on one family. This batch deletes tests that run each
  arm of the enum. The coverage gate shows whether the setting must run on each family.
- A probe of this ticket ran the 13 contracts on each Mamba setting, through the
  `[patch]`. It added the untied settings above, and the bounds. All the contracts pass.
  The probe is `tmp/probe-06/block_contract.probe.rs` of burn-mamba.

### The implementation

- `src/block_contract.rs` has the four new contracts, and the run of the padded contract
  from no cache. Each new contract runs on the three `RefBlock` settings. Its negative
  test breaks it with a new `Breach` of `BrokenBlock`:

  | Contract | Breach |
  |---|---|
  | `untied_stack_equals_the_stack_of_its_application_views` | `UntiedList`: `untied_params` names no parameter, and the decay still has one copy for each application |
  | `cache_stack_moves_every_slot` | `StackSlots`: `from_slots` reverses the slots. `StackTraversal`: the traversal of the stack skips its last slot. |
  | `cut_keeps_the_outputs_and_the_caches` | `InnerBackend`: the block adds 1 to each output off the autodiff backend |
  | `json_round_trip_keeps_the_config` | `Json`: the JSON writes the breach as `Step`. The JSON of the parsed config is the same, so only the outputs of the step show it. |
  | `padded_rows_equal_each_row_alone`, from no cache | `EmptyRowFromNoCache`: from no cache, an empty row starts from a state of ones |

- Each new contract had a red step: its tests did not compile before the function
  existed. The two negative tests of the padded contract failed before the run from no
  cache existed.
- The ports:
  - `src/reference/tests/layers.rs`: `a_latent_network_record_round_trip_keeps_the_network`.
    It goes through bytes, not through a file. `Module::save_file` writes the same
    bytes, and it needs the `std` feature of Burn.
  - `src/reference/tests/packed.rs`:
    `a_packed_vocab_network_has_the_logits_and_gradients_of_each_segment_alone`, over the
    Standard residuals and Multi-Gate. Under Multi-Gate, only the β gate of layer 0
    (`w_beta`, `b_beta`) gets no gradient, in both runs. So the test uses the new
    `assert_matching_grads`, which accepts a parameter with no gradient in both runs.
    `param_grads` and `assert_same_grads` moved from `grad_horizon.rs` to `mod.rs`.
  - `src/modules/hnet/tests.rs`: `forward_and_step_have_the_same_gradients` compares the
    gradient of each float parameter. Each parameter of the H-Net gets a gradient from
    both runs. `padded_rows_match_each_row_alone` also runs from no cache. The H-Net
    tests use the `cache_diff` of `block_contract.rs`, which keeps a NaN.
- Deleted: `src/reference/tests/block.rs` (its two tests repeat contracts), and
  `positions_before_the_first_reset_continue_from_the_cache`.

### The red check

The ports test behavior that exists. So the red step of each port was a temporary break
of its mechanism in `src/`, reverted after the run:

| Break | Ports that fail | Older tests that also fail |
|---|---|---|
| The record of a `LatentNetwork` skips its class-token table | `a_latent_network_record_round_trip_keeps_the_network` | none |
| `Layers::forward_packed` places a detached class-latent table | `a_packed_vocab_network_has_the_logits_and_gradients_of_each_segment_alone` | `layers_forward_packed_equals_each_segment_alone` |
| The H-Net step runs its decoder norm with no gradient | `forward_and_step_have_the_same_gradients` | none. Before this ticket, the test checked no parameter of a decoder. |
| `RouterCache::after_positions` marks an empty row as seen | `padded_rows_match_each_row_alone`, from no cache only | none |
| `RouterCache::after_positions` writes a NaN into the cache of an empty row that saw no token | `padded_rows_match_each_row_alone`, from no cache only: the difference is NaN | none |

### The review

The reports are `tmp/review-ct06-standards.md` and `tmp/review-ct06-spec.md`. The fixes:

- The docs of `block_contract.rs` said that the JSON contract needs `Config`. But
  `BlockConfig` requires `Config`, so the contract lost the bound `C: Config`.
- The docs of the cut contract use "cut" as a place, as the glossary does.
- The docs and the messages of the JSON contract use one term: the parsed config.
- The panic messages give the side of a missing slot or gradient, not a bool.
- `step_unrolled` and `layers_step_unrolled` of `block_contract.rs` share one loop.
- The H-Net tests have no copy of `cache_diff`.
- The docs of the stack contract and of the untied contract state only what the check
  checks. The stack has no virtual layers past its real layers. The count of slots of a
  stack with virtual layers is a container property, and the tests of `Layers` check it.
- The red check of a NaN in the cache (above).

Not changed: the tensor names of the new contracts have no shape suffix. The contracts
of the base also use such names (`y_fwd`, `y_got`).

### Coverage result

The base is the commit `5416d97`. `scripts/coverage.py . --base 5416d97` writes the
compare `tmp/ct06/compare.md` and the report `tmp/ct06/after.json`. The lib tests of
burn-stack cover more:

| measure | `5416d97` | this ticket |
|---|---|---|
| lines | 3905 of 4441 | 4088 of 4626 |
| branches | 312 of 432 | 316 of 438 |
| conditions | 128 of 216 | 129 of 219 |

The compare lists 1 lost line: line 232 of `src/block_contract.rs`. It is not lost code.
Line 232 is a closing brace of the two loops of `padded_rows_equal_each_row_alone`. The
compiler gives no coverage region to that brace and to the brace after it. Each line of
the loop body runs 26 times.

At the base, the one closing brace of the loop had a region, and the diff maps that brace
to line 232. `compare_items` counts a line with no region as a line that ran 0 times. The
user chose to record this here, with no change of the script.

The compare also lists 19 lines, 2 branches and 1 condition on changed lines. Of these,
14 lines, the branches and the condition are the old body of
`padded_rows_equal_each_row_alone`.
Its new body covers them again, with both outcomes of `len == 0`. The other 5 lines are
the signature of `cache_diff`, now `pub(crate)`, and the loop of `step_unrolled`, now in
`unrolled`. The new lines run in each contract that steps.

### From the gate of burn-mamba ticket 06

- The gate lost the branch `157:9 false` of `src/optim/spec.rs`, and its condition:
  `is_whole_muon` on a tiled spec. Only `MuonPlan::build` with an untied weight reaches
  it. The sorted list gives `muon_updates_an_untied_projection_one_copy_at_a_time` as a
  repeat. It repeats the update of each copy in
  `mamba3_untied_tail_trains_one_copy_per_application`. But that test called `Segmented`
  directly, and the deleted test ran the update through `MuonPlan::build`.
- The user chose rule 1. `muon_updates_an_untied_projection_one_copy_at_a_time`
  (`src/reference/tests/muon.rs`) also builds the plan of its untied stack, and runs one
  training step through its module optimizer:
  - Each copy must move as Muon moves that copy alone. One check does this for the
    `Segmented` step and for the training step.
  - The copies start tied, and they must part, as in the deleted test.
  - The training step has a learning rate of 1. The difference between the two updates
    grows with the learning rate. At `1e-2` it is 0.0036, which is below the fp16
    tolerance (about `1.8e-2`).
- The red check used an `is_whole_muon` that ignores `tiled`. The plan then gives the
  whole tensor to the stock Muon. The test fails at copy 0, in fp32 (difference 0.45,
  tolerance `1e-4`) and under `dev-f16` (difference 0.39). The red check reverted the
  break after the run. The test passes in fp32 and under `dev-f16`.
- The ticket deletes the seam row "Containers with real blocks" of
  `docs/agents/testing.md`, as the spec says: "**Block contracts** (new) replaces
  **Containers with real blocks**". The line of `CLAUDE.md` about the tests of
  `../burn-mamba` names the block contracts.
