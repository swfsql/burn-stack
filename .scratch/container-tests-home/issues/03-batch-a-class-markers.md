# 03: Batch A, the class-marker tests

Status: resolved

Blocked by: `02-block-contract-suite.md`,
`../burn-mamba/.scratch/container-tests-home/issues/01-coverage-run-and-record.md`

Spec: `../burn-mamba/.scratch/container-tests-home/spec.md`, sections "The sort rule"
and "New homes in burn-stack". Run only one session on the two repos.

**What to build:** each class-marker test of burn-mamba gets a home in burn-stack, or a
block contract. Then burn-mamba ticket 03 deletes the moved tests.

**Read:** `src/family/tests/class.rs` of burn-mamba (29 tests, 1471 lines), and the
class-marker tests of `RefBlock`.

**Steps:**

1. Under `## Comments`, write the sorted list first. It has one line for each test of
   the batch: the rule number, the new home, and the block settings that the test uses.
   When the sort rule does not place a test, stop and ask the user.
2. Rule 1: port each test to `RefBlock`, or to no block, in the `RefBlock` test file of
   its mechanism. Skip a test that repeats a test of burn-stack, and name the repeat in
   the list.
3. Rule 1, no block: 3 tests use no block (the class plans and the custom index). They
   go to the tests of the class module.
4. Rule 2: add each new contract to `block_contract`, with its run on `RefBlock` and its
   negative test.
5. Delete each `RefBlock` test that repeats a contract.

- [x] The sorted list has one line for each of the 29 tests.
- [x] Each test of rule 1 or 2 has its new home, or the name of the test that it repeats.
- [x] The coverage compare of burn-stack against the base commit of this ticket lists no
      lost item.
- [x] `cargo check`, `cargo doc --all --no-deps` and `scripts/check.sh` pass with no
      warnings.

## Comments

### The sorted list

The base commit of this ticket is `65334b6`. A home in `src/reference/tests/` is a port
to `RefBlock`. "Default path" is the default SSD path of the family. The Mamba-3 setting
of the source has `mimo_rank` 1 and `rotation_fraction` 0.5. "In `MambaBlock`" is the
family enum of burn-mamba (its ADR-0009), with the config types
`MambaLatentNetConfig`, `MambaBidiLayersConfig` and the path `MambaSsdPath`.

`src/family/tests/class.rs` of burn-mamba (29 tests):

| Test | Rule | New home, or the test that it repeats | Block settings |
|---|---|---|---|
| `latent_network_builder_mamba2` | 1 | `src/reference/tests/layers.rs`, as `latent_network_forward_equals_step_unrolled`. The source checks only the shapes. Its step with no cursors is the only test of that path of `LatentNetwork::step`. | Mamba-2, default path |
| `family_net_config_mamba2` | 3, repeat | `MambaLatentNetConfig` over the family enum. Two tests of burn-mamba make its checks on each family: `default_ssd_path_fits_the_family_of_the_block` (`config.rs`: the config and a forward) and `prime_and_class_tokens_through_the_enum` (`block.rs`: steps through the enum caches). The source also runs a forward from the returned caches and a step with no cursors. The block contracts on `MambaBlock` make these checks on one family, and `latent_network_forward_equals_step_unrolled` makes the step with no cursors. The coverage gate of burn-mamba ticket 03 shows an arm of the enum that only this test runs. | Mamba-2, in `MambaBlock` |
| `family_net_config_mamba3` | 3, repeat | As `family_net_config_mamba2` | Mamba-3, in `MambaBlock` |
| `family_net_config_mamba1` | 3, repeat | As `family_net_config_mamba2` | Mamba-1, in `MambaBlock` |
| `bidi_layers_mamba2` | 1, repeat | `padded_bidi_layers_are_each_row_alone`: real layers only, `CatLinear` merges | Mamba-2, default path |
| `bidi_layers_mamba3` | 1, repeat | `bidi_virtual_forward_is_deterministic`: `mean` merges | Mamba-3, default path |
| `bidi_layers_mamba1` | 1, repeat | `padded_bidi_layers_are_each_row_alone`. The block contracts on Mamba-1 state what the stack needs of the block. | Mamba-1 |
| `family_bidi_config_mamba2` | 3, repeat | `MambaBidiLayersConfig` over the family enum. Its rule-3 part is the build of that config: `bidi_layers_config_survives_a_json_round_trip` (`config.rs` of burn-mamba) builds it on each family. Its forward is a bidirectional stack over `MambaBlock`: the block contracts on `MambaBlock` and the bidi tests of burn-stack cover it. | Mamba-2, in `MambaBlock` |
| `bidi_forward_is_deterministic_mamba2` | 1 | `src/reference/tests/bidi.rs`, as `bidi_ignored_residuals_drop_the_input_skip`. This test and its two twins are the only runs of a bidirectional stack with ignored residuals. `bidi_virtual_forward_is_deterministic` checks the determinism. | Mamba-2, default path |
| `bidi_forward_is_deterministic_mamba1` | 1, repeat | The port of `bidi_forward_is_deterministic_mamba2` | Mamba-1 |
| `bidi_forward_is_deterministic_mamba3` | 1, repeat | The port of `bidi_forward_is_deterministic_mamba2` | Mamba-3, default path |
| `class_latents_lengthen_and_index` | 1 | `src/reference/tests/class.rs`, as `class_latents_lengthen_the_sequence_at_their_output_indices` | Mamba-2, default path |
| `class_latents_custom_index` | 1, no block | `src/utils/class/tests.rs`, as `class_marker_output_indices_shift_a_custom_index` | none |
| `class_tokens_on_latent_network` | 1, repeat | The port of `class_tokens_split_forward_matches_single_forward`. It checks the length of the output with an `End` class token. | Mamba-2, default path |
| `class_latents_step_panics` | 1 | `src/reference/tests/class.rs`, as `a_step_with_no_cursors_panics_on_a_middle_latent` | Mamba-2, default path |
| `class_latents_step_matches_forward` | 1, repeat | `layers_multi_gate_stack_class_latents_step_matches_forward`: the same markers and positions, over Multi-Gate. The ports below step class latents over the Standard residuals. | Mamba-2, default path |
| `class_prime_plan_emits_only_what_waits_for_the_next_token` | 1, no block | `src/utils/class/tests.rs` | none |
| `prime_emits_the_class_markers_a_step_would_drop` | 1 | `src/reference/tests/class.rs`. It is the only test that primes from a cache, and primes at the announced end. | Mamba-2, default path |
| `prime_on_a_network_covers_every_class_level` | 1, repeat | `latent_network_multi_gate_class_markers_prime_step_matches_forward`: the same markers and checks, over Multi-Gate | Mamba-2, default path |
| `prime_through_the_enum` | 1, and 3, repeat | The half on a `VocabNetwork` moves to `src/reference/tests/class.rs`, as `a_vocab_network_prime_and_steps_equal_its_forward`. The half on `MambaLatentNetConfig` repeats `prime_and_class_tokens_through_the_enum` (`block.rs` of burn-mamba). | Mamba-2, in `MambaBlock` |
| `class_window_plan_splits_a_sequence` | 1, no block | `src/utils/class/tests.rs` | none |
| `class_markers_split_forward_matches_single_forward` | 1 | `src/reference/tests/class.rs` | Mamba-2, default path |
| `class_markers_step_matches_forward_with_full_len` | 1 | `src/reference/tests/class.rs` | Mamba-2, default path |
| `class_tokens_split_forward_matches_single_forward` | 1 | `src/reference/tests/class.rs` | Mamba-2, default path |
| `class_markers_without_full_len_panic_in_forward` | 1 | `src/reference/tests/class.rs`, as `a_forward_with_no_length_hint_panics_on_an_end_latent` | Mamba-2, default path |
| `class_markers_forward_then_step` | 1 | `src/reference/tests/class.rs` | Mamba-2, default path |
| `class_tokens_step_matches_forward_with_full_len` | 1 | `src/reference/tests/class.rs`. It is the only test that steps an `End` class token. | Mamba-2, default path |
| `end_closes_the_sequence_custom_never_does` | 1 | `src/reference/tests/class.rs` | Mamba-2, default path |
| `step_output_and_state_follow_the_last_emitted_token` | 1 | `src/reference/tests/class.rs` | Mamba-2, default path |

#### The lines that only burn-mamba covers

`tmp/ct03-only-by-burn-mamba.md` of burn-mamba compares the two burn-stack reports of
`tmp/coverage-record/after-04/` there. It lists the items of burn-stack that only the
burn-mamba tests cover. Each item below is in the code of this batch, and the list above
names a port that runs it:

- `src/modules/bidi.rs` 355–363, 398: the ignored residuals of a pair.
- `src/modules/layers.rs` 138–140: `class_latent_output_indices`. 628: the step of a
  closing `End` latent. 894: a prime from a cache.
- `src/modules/network.rs` 138–140, 241: a step with no cursors. 162–164: a step of an
  `End` class token.
- `src/utils/class/mod.rs` 293: the panic with no length hint. 356: a prime at the
  announced end.

#### A property that no contract states

A container gives a block two forms of an empty cache. `Layers::forward` and
`Layers::step` start from `Block::zero_cache`. `Layers::prime` starts from empty slots,
so `Layer::step` gives `None` to `block_step`.

The prime tests compare a prime with a forward. So they rely on this: a block from its
zero cache equals the block from no cache. No block contract states it.
`prime_emits_the_class_markers_a_step_would_drop` and
`prime_on_a_network_covers_every_class_level` check it on Mamba-2 only.

So the sort adds one contract, `zero_cache_equals_no_cache` (rule 2). burn-mamba ticket 03
runs it on each block setting.

#### The rule-3 repeats

The user chose: each test of rule 3 in this batch repeats a test that burn-mamba keeps. So
burn-mamba ticket 03 deletes it as a repeat, under its coverage gate. Then `class.rs` of
burn-mamba has no test left.

#### `RefBlock` tests that repeat a contract

None in `class.rs`, `bidi.rs` and `layers.rs`. No `RefBlock` test compares a block from its
zero cache with the block from no cache. The two tests of `block.rs` that repeat contracts
belong to batch D.

### The red check

The contract `zero_cache_equals_no_cache` had a real red step: its tests failed to
compile before the function existed. It has two negative tests. A zero cache of ones
breaks its forward half. A step from no cache that starts from ones breaks only its step
half. The ports test behavior that exists. So the red step
of a port was a temporary break of its mechanism, reverted after the run
(`tmp/ct03-breaks/` keeps the logs):

| Break | Ports that fail |
|---|---|
| A prime flushes the markers at the announced end | `class_prime_plan_emits_only_what_waits_for_the_next_token` (its case of a `Custom` at the end) |
| `Layers::prime` ignores its incoming cache | `prime_emits_the_class_markers_a_step_would_drop` |
| The step of a network keeps the output of the user token after an `End` class token | `class_tokens_step_matches_forward_with_full_len` |
| A bidirectional stack ignores `ignore_first_residual` | `bidi_ignored_residuals_drop_the_input_skip` |
| `Layers::step` puts a closing `End` latent before the user token | `class_markers_step_matches_forward_with_full_len`, `class_markers_forward_then_step`, `end_closes_the_sequence_custom_never_does`, `step_output_and_state_follow_the_last_emitted_token`, `prime_emits_the_class_markers_a_step_would_drop` |
| `VocabNetwork::prime` skips the final norm | `a_vocab_network_prime_and_steps_equal_its_forward` |
| The check of the length hint does nothing | `a_step_with_no_cursors_panics_on_a_middle_latent`, `a_forward_with_no_length_hint_panics_on_an_end_latent` |

The last two breaks ran in one build. They change different code, and each failure
belongs to one of them. Six ports got no break of their own:
`class_latents_lengthen_the_sequence_at_their_output_indices`,
`class_markers_split_forward_matches_single_forward`,
`class_tokens_split_forward_matches_single_forward`,
`latent_network_forward_equals_step_unrolled`,
`class_marker_output_indices_shift_a_custom_index` and
`class_window_plan_splits_a_sequence`.

### Coverage result

The base is the commit `65334b6`. `scripts/coverage.py . --base 65334b6` lists no lost
item, and no covered item on a changed line. The report of this ticket is
`tmp/coverage-03/after.json`, and the compare is `tmp/coverage-03/compare.md`.
The lib tests of burn-stack cover more:

| measure | `65334b6` | this ticket |
|---|---|---|
| lines | 3867 of 4415 | 3905 of 4441 |
| branches | 303 of 432 | 312 of 432 |
| conditions | 121 of 216 | 128 of 216 |

The gate of burn-mamba ticket 03 measures what the deletion of the source tests loses.

### The review (`tmp/review-03-standards.md`, `tmp/review-03-spec.md`)

- Fixed:
  - The header of `class.rs` names its parity tests only. The doc of
    `class_window_plan_splits_a_sequence` names its four cases. No doc uses "stream" as a
    verb.
  - The shared code: `assert_same_outputs`, `latent_network` and `latent_input` in
    `mod.rs`, beside `assert_same_caches`. `class.rs` has one helper `position`, not
    `token` and `output_at`. The older test of `class.rs` uses the import of
    `ClassCursors`.
  - The contract has a second negative test, so its step half can fail.
  - The list: the two rows with a new port name give that name. The rows of
    `family_net_config_mamba2` and `family_bidi_config_mamba2` say which test makes each
    check. Line 628 of `layers.rs` is the step of a closing `End` latent. The section
    "`RefBlock` tests that repeat a contract" records step 5.
  - burn-mamba ticket 03 said to keep each test of rule 3. It now deletes each "3,
    repeat", as the user chose.
- Kept:
  - `bidi_ignored_residuals_drop_the_input_skip` has one pair, as its source. So a swap of
    the two flags does not show. Its doc says so.
  - The other contracts also have one negative test, which breaks their first check.
