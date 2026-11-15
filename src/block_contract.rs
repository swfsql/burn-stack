//! The block contracts: the properties that every block must have, so that
//! each container applies to it unchanged.
//!
//! ## Who proves what
//!
//! - burn-stack proves "block contracts ⇒ the containers work". Its container
//!   tests compose [`RefBlock`](crate::reference::RefBlock) (ADR-0005).
//! - Each family crate proves "my blocks ⇒ block contracts". It calls each
//!   function of this module on each block setting that it ships, through the
//!   `test-helpers` feature. It does not test the containers again.
//!
//! When a family crate needs a property that no contract states, the fix is a
//! new contract in this module.
//!
//! ## The form of a contract
//!
//! - Each contract is one public function, generic over the block config.
//!   Its inputs are the config, the block options and an fp32 tolerance. A
//!   contract that does not use an input still takes it, so a family crate
//!   can call each contract in the same way.
//! - A contract that needs more than [`BlockConfig`] and [`Block`] require
//!   states it in its bounds. For example, a contract that builds a layer
//!   stack needs `C: Clone`, and [`cache_stack_moves_every_slot`] needs the
//!   cache traversal of `Block::Caches`.
//! - The contract builds the block on [`test_device`], for one application,
//!   when its docs do not give another device or count. It draws random
//!   inputs.
//! - The tolerance goes through [`dtype_tol`]. The outputs and the caches of
//!   a real block can be large, so each check of values uses
//!   [`max_rel_diff`].
//! - A failure panics. The message starts with ``block contract `<name>`:``,
//!   where `<name>` is the name of the function. A panic inside the block
//!   also gets this start.
//! - A contract checks the bare block when it can. It puts the block in a
//!   container only when the property needs one.
//! - A contract that does not apply to a block returns early. The block gives
//!   the reason through the [`Block`] trait, for example a
//!   [`reset_align`](Block::reset_align) of `None`.
//!
//! A contract compares two caches, and reads one row of a cache, through the
//! cache traversal ([`CacheTensors`]). So it sees only the tensors that the
//! traversal visits. [`cache_traversal_moves_every_field`] checks that the
//! traversal visits each field.
//!
//! ## Chunks
//!
//! A window of a contract has 7 tokens. In a block that works in chunks, a
//! window holds a chunk boundary only when the chunk is shorter than the
//! window. So a family crate gives such a block a short chunk, for example 4
//! tokens, through its config or its options.

use crate::modules::{Block, BlockConfig, CacheStack, CacheTensors, Layers, LayersBuilder, TensorZip};
use crate::utils::test_helpers::{dtype_tol, max_rel_diff, test_device};
use crate::utils::{GradHorizon, Packed, Schedule};
use burn::config::config_to_json;
use burn::module::{ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::prelude::*;
use burn::store::ModuleRecord;
use burn::tensor::Distribution;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// The rows of the random inputs.
const BATCH: usize = 2;
/// The tokens of a window.
const SEQUENCE: usize = 7;
/// The tokens of the first window of a split. It is not a power of two, so
/// the split is inside a chunk for most chunk lengths.
const SPLIT: usize = 3;
/// The tokens that make a warm incoming cache.
const WARM: usize = 2;

/// `block_forward` over a window equals `block_step` unrolled over the same
/// window, on the outputs and on the final cache. Both start from no cache.
///
/// The containers rely on this: `forward` for training and prefill, `step`
/// for decode.
pub fn forward_equals_step_unrolled<C: BlockConfig>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) {
    run_contract("forward_equals_step_unrolled", || {
        let device = test_device();
        let tol = dtype_tol(f32_tol);
        let block = config.init_block(1, &device);
        let x_bsd = randn(BATCH, SEQUENCE, config.d_model(), &device);

        let (y_fwd, cache_fwd) = block.block_forward(x_bsd.clone(), None, options, None);
        let (y_step, cache_step) = step_unrolled(&block, x_bsd, None);

        check("the outputs of the forward and of the step unrolled", max_rel_diff(y_fwd, y_step), tol);
        let what = "the final caches of the forward and of the step unrolled";
        check(what, cache_diff(cache_fwd, cache_step), tol);
    });
}

/// A window split in two through the cache equals one window. The forward of
/// the first part gives a cache. The forward of the second part starts from
/// that cache. Together they give the outputs and the final cache of one
/// forward over the whole window.
///
/// The containers rely on this to continue a sequence in a later call, for
/// example a prefill and then a decode.
pub fn forward_splits_into_windows_through_the_cache<C: BlockConfig>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) where
    <C::Block as Block>::Options: Clone,
{
    run_contract("forward_splits_into_windows_through_the_cache", || {
        let device = test_device();
        let tol = dtype_tol(f32_tol);
        let block = config.init_block(1, &device);
        let x_bsd = randn(BATCH, SEQUENCE, config.d_model(), &device);

        let (y_all, cache_all) = block.block_forward(x_bsd.clone(), None, options.clone(), None);
        let first_bsd = x_bsd.clone().narrow(1, 0, SPLIT);
        let (y_first, cache_first) = block.block_forward(first_bsd, None, options.clone(), None);
        let second_bsd = x_bsd.narrow(1, SPLIT, SEQUENCE - SPLIT);
        let (y_second, cache_second) = block.block_forward(second_bsd, Some(cache_first), options, None);

        let y_split = Tensor::cat(vec![y_first, y_second], 1);
        check("the outputs of the two windows and of one window", max_rel_diff(y_split, y_all), tol);
        let what = "the final caches of the two windows and of one window";
        check(what, cache_diff(cache_second, cache_all), tol);
    });
}

/// The zero cache equals no cache. The forward and the step unrolled from
/// [`Block::zero_cache`] give the outputs and the final caches that they give
/// from `None`.
///
/// The containers rely on this, because they start a sequence in two ways.
/// [`Layers::forward`](crate::modules::Layers::forward) and
/// [`Layers::step`](crate::modules::Layers::step) start each slot from the
/// zero cache. [`Layers::prime`](crate::modules::Layers::prime) starts each
/// slot from no cache. A prime and the steps after it must give what the
/// forward gives.
pub fn zero_cache_equals_no_cache<C: BlockConfig>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) where
    <C::Block as Block>::Options: Clone,
{
    run_contract("zero_cache_equals_no_cache", || {
        let device = test_device();
        let tol = dtype_tol(f32_tol);
        let block = config.init_block(1, &device);
        let x_bsd = randn(BATCH, SEQUENCE, config.d_model(), &device);
        let zero = || Some(block.zero_cache(BATCH, &device));

        let (y_none, cache_none) = block.block_forward(x_bsd.clone(), None, options.clone(), None);
        let (y_zero, cache_zero) = block.block_forward(x_bsd.clone(), zero(), options, None);
        let what = "the outputs of the forward from the zero cache and from no cache";
        check(what, max_rel_diff(y_zero, y_none), tol);
        let what = "the final caches of the forward from the zero cache and from no cache";
        check(what, cache_diff(cache_zero, cache_none), tol);

        let (y_none, cache_none) = step_unrolled(&block, x_bsd.clone(), None);
        let (y_zero, cache_zero) = step_unrolled(&block, x_bsd, zero());
        let what = "the outputs of the step from the zero cache and from no cache";
        check(what, max_rel_diff(y_zero, y_none), tol);
        let what = "the final caches of the step from the zero cache and from no cache";
        check(what, cache_diff(cache_zero, cache_none), tol);
    });
}

/// In a right-padded batch, each row gives what it gives alone: the outputs
/// at its real tokens, and its row of the final cache. The batch runs two
/// times, from a warm cache and from no cache. Each row alone starts from the
/// same cache as the batch. An empty row (a row with no real token) has these
/// final caches:
///
/// - from a warm cache, its row of the incoming cache,
/// - from no cache, its row of the zero cache. [`zero_cache_equals_no_cache`]
///   states that the zero cache and no cache give the same values.
///
/// The containers rely on this to run rows of different lengths in one
/// batch. See the `pad` of [`Block::block_forward`].
pub fn padded_rows_equal_each_row_alone<C: BlockConfig>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) where
    <C::Block as Block>::Options: Clone,
{
    run_contract("padded_rows_equal_each_row_alone", || {
        let device = test_device();
        let tol = dtype_tol(f32_tol);
        let d_model = config.d_model();
        let block = config.init_block(1, &device);
        // A full row, a short row, and an empty row.
        let lens = [SEQUENCE, SPLIT, 0];
        let batch = lens.len();
        let x_bsd = randn(batch, SEQUENCE, d_model, &device);
        let (_, warm) = block.block_forward(randn(batch, WARM, d_model, &device), None, options.clone(), None);
        let lens_i64: Vec<i64> = lens.iter().map(|&len| len as i64).collect();
        let lens_bs = Tensor::<1, Int>::from_ints(lens_i64.as_slice(), &device)
            .reshape([batch, 1])
            .expand([batch, SEQUENCE]);
        let pad_bs = Tensor::<1, Int>::arange(0..SEQUENCE as i64, &device)
            .reshape([1, SEQUENCE])
            .expand([batch, SEQUENCE])
            .greater_equal(lens_bs);
        let zero = block.zero_cache(batch, &device);

        for (label, incoming) in [("from a warm cache", Some(warm)), ("from no cache", None)] {
            let (y_bsd, cache) = block.block_forward(x_bsd.clone(), incoming.clone(), options.clone(), Some(pad_bs.clone()));
            for (b, &len) in lens.iter().enumerate() {
                let incoming_b = incoming.clone().map(|cache| cache_row(cache, b));
                let cache_b = cache_row(cache.clone(), b);
                if len == 0 {
                    let (what, want_b) = match incoming_b {
                        Some(incoming_b) => (format!("{label}, the final and the incoming caches of the empty row {b}"), incoming_b),
                        None => (
                            format!("{label}, the final cache of the empty row {b} and its row of the zero cache"),
                            cache_row(zero.clone(), b),
                        ),
                    };
                    check(&what, cache_diff(cache_b, want_b), tol);
                    continue;
                }
                let row_bsd = x_bsd.clone().narrow(0, b, 1).narrow(1, 0, len);
                let (y_alone, cache_alone) = block.block_forward(row_bsd, incoming_b, options.clone(), None);
                let y_row = y_bsd.clone().narrow(0, b, 1).narrow(1, 0, len);
                let what = format!("{label}, the outputs of row {b} in the batch and alone");
                check(&what, max_rel_diff(y_row, y_alone), tol);
                let what = format!("{label}, the final caches of row {b} in the batch and alone");
                check(&what, cache_diff(cache_b, cache_alone), tol);
            }
        }
    });
}

/// Over packed rows, each segment gives the outputs that it gives alone from
/// no cache. The last segment of a row also gives that row of the final
/// cache. The positions before the first reset of a row continue from the
/// incoming cache.
///
/// The rows hold 4 units of positions. A unit is the smallest multiple of
/// the reset alignment that is at least 3. Row 0 holds three segments. Row 1
/// continues from the incoming cache for 2 units, then holds one segment.
///
/// It returns early when [`Block::reset_align`] gives `None`: the block takes
/// no packed rows. See [`Block::block_forward_packed`].
pub fn packed_segments_equal_each_segment_alone<C: BlockConfig>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) where
    <C::Block as Block>::Options: Clone,
{
    run_contract("packed_segments_equal_each_segment_alone", || {
        let device = test_device();
        let tol = dtype_tol(f32_tol);
        let d_model = config.d_model();
        let block = config.init_block(1, &device);
        let Some(align) = block.reset_align(&options) else {
            return;
        };
        let unit = 3usize.next_multiple_of(align);
        let sequence = 4 * unit;
        let starts = vec![vec![0, unit, 3 * unit], vec![2 * unit]];
        let reset_bs = Packed::from_starts(&starts, align, 0, sequence, &device).reset_bs;
        let x_bsd = randn(starts.len(), sequence, d_model, &device);
        let (_, warm) = block.block_forward(randn(starts.len(), WARM, d_model, &device), None, options.clone(), None);

        let (y_bsd, cache) = block.block_forward_packed(x_bsd.clone(), Some(warm.clone()), options.clone(), reset_bs);

        for (b, row) in starts.iter().enumerate() {
            // Each part `[start, end)` of the row, and the cache that it
            // starts from alone: the incoming cache before the first reset,
            // else none.
            let mut parts = Vec::new();
            if row[0] > 0 {
                parts.push((0, row[0], Some(cache_row(warm.clone(), b))));
            }
            for (i, &start) in row.iter().enumerate() {
                parts.push((start, row.get(i + 1).copied().unwrap_or(sequence), None));
            }
            for (start, end, incoming) in parts {
                let part_bsd = x_bsd.clone().narrow(0, b, 1).narrow(1, start, end - start);
                let (y_alone, cache_alone) = block.block_forward(part_bsd, incoming, options.clone(), None);
                let y_part = y_bsd.clone().narrow(0, b, 1).narrow(1, start, end - start);
                let what = format!("the outputs of row {b} at [{start}, {end}) when packed and alone");
                check(&what, max_rel_diff(y_part, y_alone), tol);
                if end == sequence {
                    let what = format!("the final caches of row {b} when packed and of its last segment alone");
                    check(&what, cache_diff(cache_row(cache.clone(), b), cache_alone), tol);
                }
            }
        }
    });
}

/// A record round trip keeps the block. A new block loads the record of a
/// block, and then gives the outputs of that block in the forward and in the
/// step. The record goes through bytes.
///
/// Before the record, noise moves each float parameter away from its initial
/// value. So a float parameter that the record does not keep changes the
/// outputs, also when the block gives it the same initial value each time.
///
/// The containers save and load a model through the records of its blocks.
pub fn record_round_trip_keeps_the_block<C: BlockConfig>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) where
    <C::Block as Block>::Options: Clone,
{
    run_contract("record_round_trip_keeps_the_block", || {
        let device = test_device();
        let tol = dtype_tol(f32_tol);
        let block = config.init_block(1, &device).map(&mut Jitter);
        let x_bsd = randn(BATCH, SEQUENCE, config.d_model(), &device);
        let (want_fwd, _) = block.block_forward(x_bsd.clone(), None, options.clone(), None);
        let (want_step, _) = step_unrolled(&block, x_bsd.clone(), None);

        let bytes = block
            .into_record()
            .into_bytes()
            .unwrap_or_else(|e| panic!("the record does not go to bytes: {e}"));
        let record = ModuleRecord::from_bytes(bytes).unwrap_or_else(|e| panic!("the bytes do not give a record: {e}"));
        let loaded = config
            .init_block(1, &device)
            .try_load_record(record)
            .unwrap_or_else(|e| panic!("the record does not load into a new block: {e}"));
        let (got_fwd, _) = loaded.block_forward(x_bsd.clone(), None, options, None);
        let (got_step, _) = step_unrolled(&loaded, x_bsd, None);

        let what = "the outputs of the forward of the block and of its record in a new block";
        check(what, max_rel_diff(got_fwd, want_fwd), tol);
        let what = "the outputs of the step of the block and of its record in a new block";
        check(what, max_rel_diff(got_step, want_step), tol);
    });
}

/// A JSON round trip keeps the config. The config writes its JSON, and the
/// JSON gives a parsed config. The check has these parts:
///
/// - the parsed config writes the same JSON,
/// - a block of the parsed config loads the record of a block of the config,
/// - the two blocks give the same outputs in the forward and in the step.
///
/// So the check finds a field that the JSON loses or changes, also when the
/// parsed config writes the same JSON. Noise first moves each float parameter
/// away from its initial value, as in [`record_round_trip_keeps_the_block`].
///
/// A family crate saves a model as the JSON of its config and the record of
/// its parameters. It loads the model in the same way.
pub fn json_round_trip_keeps_the_config<C: BlockConfig>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) where
    <C::Block as Block>::Options: Clone,
{
    run_contract("json_round_trip_keeps_the_config", || {
        let device = test_device();
        let tol = dtype_tol(f32_tol);
        let json = config_to_json(config);
        let parsed = C::load_binary(json.as_bytes()).unwrap_or_else(|e| panic!("the JSON of the config does not parse: {e}"));        let json_again = config_to_json(&parsed);
        assert!(
            json_again == json,
            "the parsed config writes another JSON.\nThe JSON of the config:\n{json}\nThe JSON of the parsed \
             config:\n{json_again}",
        );

        let block = config.init_block(1, &device).map(&mut Jitter);
        let x_bsd = randn(BATCH, SEQUENCE, config.d_model(), &device);
        let (want_fwd, _) = block.block_forward(x_bsd.clone(), None, options.clone(), None);
        let (want_step, _) = step_unrolled(&block, x_bsd.clone(), None);
        let loaded = parsed
            .init_block(1, &device)
            .try_load_record(block.into_record())
            .unwrap_or_else(|e| panic!("the record of a block of the config does not load into a block of the parsed config: {e}"));
        let (got_fwd, _) = loaded.block_forward(x_bsd.clone(), None, options, None);
        let (got_step, _) = step_unrolled(&loaded, x_bsd, None);

        let what = "the outputs of the forward of the block of the config and of the block of the parsed config";
        check(what, max_rel_diff(got_fwd, want_fwd), tol);
        let what = "the outputs of the step of the block of the config and of the block of the parsed config";
        check(what, max_rel_diff(got_step, want_step), tol);
    });
}

/// The cache traversal moves every field of the cache.
/// [`assign_in_place`](CacheTensors::assign_in_place) writes the values of a
/// second cache into a first cache, which then acts as the second cache. The
/// forward and the step unrolled from each give the same outputs and final
/// caches. Also, each row of the second cache, alone, gives the outputs of
/// that row.
///
/// The check reads the outputs, not the tensors that the traversal visits. So
/// a field that the traversal skips keeps its old value, and the outputs
/// show it. The two caches come from one step and from two steps. A field
/// that the second step does not change stays unseen.
///
/// The containers move a cache through the traversal in these operations:
///
/// - the backend hop of a gradient horizon,
/// - the detach,
/// - the cache write-back of a captured step,
/// - the row select of the H-Net step.
pub fn cache_traversal_moves_every_field<C: BlockConfig>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) where
    <C::Block as Block>::Options: Clone,
{
    run_contract("cache_traversal_moves_every_field", || {
        let device = test_device();
        let tol = dtype_tol(f32_tol);
        let d_model = config.d_model();
        let block = config.init_block(1, &device);
        let (_, first) = step_unrolled(&block, randn(BATCH, 1, d_model, &device), None);
        let (_, second) = step_unrolled(&block, randn(BATCH, 1, d_model, &device), Some(first.clone()));
        let moved = first.into_owned_buffers().assign_in_place(second.clone());
        let x_bsd = randn(BATCH, SEQUENCE, d_model, &device);

        let (y_fwd, cache_fwd) = block.block_forward(x_bsd.clone(), Some(second.clone()), options.clone(), None);
        let (y_got, cache_got) = block.block_forward(x_bsd.clone(), Some(moved.clone()), options.clone(), None);
        let what = "the outputs of the forward from the moved cache and from its source";
        check(what, max_rel_diff(y_got, y_fwd.clone()), tol);
        let what = "the final caches of the forward from the moved cache and from its source";
        check(what, cache_diff(cache_got, cache_fwd), tol);

        let (y_want, cache_want) = step_unrolled(&block, x_bsd.clone(), Some(second.clone()));
        let (y_got, cache_got) = step_unrolled(&block, x_bsd.clone(), Some(moved));
        let what = "the outputs of the step from the moved cache and from its source";
        check(what, max_rel_diff(y_got, y_want), tol);
        let what = "the final caches of the step from the moved cache and from its source";
        check(what, cache_diff(cache_got, cache_want), tol);

        for b in 0..BATCH {
            let row_bsd = x_bsd.clone().narrow(0, b, 1);
            let (y_row, _) = block.block_forward(row_bsd, Some(cache_row(second.clone(), b)), options.clone(), None);
            let what = format!("the outputs of row {b} from its row of the cache and in the batch");
            check(&what, max_rel_diff(y_row, y_fwd.clone().narrow(0, b, 1)), tol);
        }
    });
}

/// An untied block in a stack equals the stack of its application views.
/// The stack has 2 real layers under 5 virtual layers (`Cyclic`: 3
/// applications and 2). Virtual layer `i` runs application `index[i]` of its
/// real layer ([`Layer::application`](crate::modules::Layer::application)),
/// in the forward and in the step. Noise first moves each copy of each
/// parameter away from the others. So a read of the wrong copy changes the
/// outputs. The check has these parts:
///
/// - [`Block::untied_params`] names each tensor that grows with the count of
///   applications, with the axis along which it grows `n` times for `n`
///   applications. Each parameter that it names grows in this way.
/// - The forward of the untied stack equals the forward of the stack of its
///   application views.
/// - The step unrolled of the untied stack equals its forward.
/// - The untied stack differs from its tied view, which runs application 0
///   at each virtual layer. So the copies change the outputs.
///
/// It returns early when the block unties no parameter: no tensor grows with
/// the count of applications, and [`Block::untied_params`] is empty.
///
/// The containers rely on this to give each application its own copy.
pub fn untied_stack_equals_the_stack_of_its_application_views<C: BlockConfig + Clone>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) where
    <C::Block as Block>::Options: Clone,
{
    run_contract("untied_stack_equals_the_stack_of_its_application_views", || {
        let device = test_device();
        let tol = dtype_tol(f32_tol);
        if !untied_params_name_each_tiled_tensor(config, &device) {
            return;
        }

        let schedule = Schedule::Cyclic;
        let layers = LayersBuilder {
            n_virtual_layers: Some((UNTIED_VIRTUAL_LAYERS, schedule.clone())),
            ..LayersBuilder::new(UNTIED_REAL_LAYERS, config.clone())
        }
        .init(&device)
        .map(&mut Jitter);
        let applications = schedule.applications(UNTIED_VIRTUAL_LAYERS, UNTIED_REAL_LAYERS);
        let views = Layers {
            n_real_layers: UNTIED_VIRTUAL_LAYERS,
            n_virtual_layers: None,
            real_layers: (0..UNTIED_VIRTUAL_LAYERS)
                .map(|i| {
                    let real = schedule.real_idx(i, UNTIED_VIRTUAL_LAYERS, UNTIED_REAL_LAYERS);
                    layers.real_layers[real].application(applications.index[i]).into_owned()
                })
                .collect(),
            ..layers.clone()
        };
        let tied = Layers {
            real_layers: layers.real_layers.iter().map(|layer| layer.application(0).into_owned()).collect(),
            ..layers.clone()
        };
        let x_bsd = randn(BATCH, SEQUENCE, config.d_model(), &device);

        let (y_untied, _) = layers.forward(x_bsd.clone(), None, options.clone(), None, None);
        let (y_views, _) = views.forward(x_bsd.clone(), None, options.clone(), None, None);
        let what = "the outputs of the untied stack and of the stack of its application views";
        check(what, max_rel_diff(y_untied.clone(), y_views), tol);
        let (y_step, _) = layers_step_unrolled(&layers, x_bsd.clone());
        let what = "the outputs of the forward and of the step unrolled of the untied stack";
        check(what, max_rel_diff(y_step, y_untied.clone()), tol);
        let (y_tied, _) = tied.forward(x_bsd, None, options, None, None);
        let diff = max_rel_diff(y_untied, y_tied);
        assert!(
            diff > tol,
            "the outputs of the untied stack and of its tied view differ by {diff}, which is in the tolerance {tol}. \
             So the untied copies do not change the outputs.",
        );
    });
}

/// The caches of a layer stack ([`Block::Caches`]) move every slot. The
/// stack has 2 real layers, and it applies each real layer once. It gives two
/// caches: the first after one window, the second after one more window. The
/// check has these parts:
///
/// - the caches hold one slot for each layer,
/// - [`CacheStack::from_slots`] of the slots of [`CacheStack::into_slots`]
///   returns each slot to its place,
/// - [`assign_in_place`](CacheTensors::assign_in_place) on the caches of the
///   stack writes each slot of the second caches into the first caches.
///
/// The check compares each pair of slots through the cache traversal of the
/// block.
///
/// The containers rely on this. A layer stack splits its caches into slots,
/// and joins the slots again, in each call. The row select of
/// the H-Net step and the cache write-back of a captured step move the caches
/// of a stack through their cache traversal. So this contract needs that
/// traversal (`Block::Caches: CacheTensors`), which [`Block`] does not
/// require.
pub fn cache_stack_moves_every_slot<C: BlockConfig + Clone>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) where
    <C::Block as Block>::Options: Clone,
    <C::Block as Block>::Caches: CacheTensors,
{
    run_contract("cache_stack_moves_every_slot", || {
        let device = test_device();
        let tol = dtype_tol(f32_tol);
        let d_model = config.d_model();
        let layers = LayersBuilder::new(STACK_LAYERS, config.clone()).init(&device);
        let (_, first) = layers.forward(randn(BATCH, SEQUENCE, d_model, &device), None, options.clone(), None, None);
        let x_bsd = randn(BATCH, SEQUENCE, d_model, &device);
        let (_, second) = layers.forward(x_bsd, Some(first.clone()), options, None, None);
        assert!(
            first.slot_count() == STACK_LAYERS,
            "the caches of a stack of {STACK_LAYERS} layers have {} slots.",
            first.slot_count(),
        );

        let rebuilt = <C::Block as Block>::Caches::from_slots(first.clone().into_slots());
        check_slots("the caches from the slots and of the caches", rebuilt, first.clone(), tol);
        let moved = first.into_owned_buffers().assign_in_place(second.clone());
        check_slots("the moved caches and of their source", moved, second, tol);
    });
}

/// A stack with a cut gives the outputs and the final caches of the same
/// stack with no cut. The stack has 2 real layers. Its gradient horizon is
/// its last layer, so the cut is before that layer. The stack with no cut has
/// no gradient horizon, so each layer is a tracked layer. The check runs the
/// forward and the step unrolled. The stacks are on the autodiff backend of
/// the test device.
///
/// Layer 0 of the stack with a cut is an untracked layer. It runs the block
/// of [`Module::valid`] on the inner backend, and its cache goes through the
/// backend hop ([`cache_to_inner`](crate::modules::cache_to_inner) and
/// [`cache_from_inner`](crate::modules::cache_from_inner)). So a tensor of
/// the block that `valid` does not move, for example in a `#[module(skip)]`
/// field, changes the outputs or the caches.
///
/// The containers rely on this to run an untracked span with no graph.
pub fn cut_keeps_the_outputs_and_the_caches<C: BlockConfig + Clone>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) where
    <C::Block as Block>::Options: Clone,
{
    run_contract("cut_keeps_the_outputs_and_the_caches", || {
        let device = test_device().autodiff();
        let tol = dtype_tol(f32_tol);
        let no_cut = LayersBuilder::new(STACK_LAYERS, config.clone()).init(&device);
        let cut = Layers { grad_horizon: Some(GradHorizon::last(1, STACK_LAYERS)), ..no_cut.clone() };
        let x_bsd = randn(BATCH, SEQUENCE, config.d_model(), &device);

        let (y_want, caches_want) = no_cut.forward(x_bsd.clone(), None, options.clone(), None, None);
        let (y_cut, caches_cut) = cut.forward(x_bsd.clone(), None, options, None, None);
        check("the outputs of the forward with a cut and with no cut", max_rel_diff(y_cut, y_want), tol);
        check_slots("the final caches of the forward with a cut and with no cut", caches_cut, caches_want, tol);

        let (y_want, caches_want) = layers_step_unrolled(&no_cut, x_bsd.clone());
        let (y_cut, caches_cut) = layers_step_unrolled(&cut, x_bsd);
        check("the outputs of the step with a cut and with no cut", max_rel_diff(y_cut, y_want), tol);
        check_slots("the final caches of the step with a cut and with no cut", caches_cut, caches_want, tol);
    });
}

/// The count of real layers of the stack of the stack contract and of the
/// cut contract. The stack applies each real layer once.
const STACK_LAYERS: usize = 2;

/// Panic unless each slot of `got` matches its slot of `want`. `what` names
/// the two caches.
fn check_slots<S: CacheStack>(what: &str, got: S, want: S, tol: f32) {
    let (got, want) = (got.into_slots(), want.into_slots());
    assert!(got.len() == want.len(), "{what} differ in slot count: {} and {}.", got.len(), want.len());
    for (i, pair) in got.into_iter().zip(want).enumerate() {
        match pair {
            (Some(got), Some(want)) => check(&format!("slot {i} of {what}"), cache_diff(got, want), tol),
            (None, None) => {}
            (None, Some(_)) => panic!("slot {i} of {what}: the slot is empty in the first only."),
            (Some(_), None) => panic!("slot {i} of {what}: the slot is empty in the second only."),
        }
    }
}

/// The count of real layers of the stack of the untied contract.
const UNTIED_REAL_LAYERS: usize = 2;
/// The count of virtual layers of the stack of the untied contract.
const UNTIED_VIRTUAL_LAYERS: usize = 5;
/// The count of applications of the block that the untied contract compares
/// with a block of 1 application.
const UNTIED_APPLICATIONS: usize = 3;

/// Panic unless [`Block::untied_params`] names exactly the tensors that grow
/// with the count of applications, each with the axis along which it grows.
/// Return whether the block unties a parameter.
fn untied_params_name_each_tiled_tensor<C: BlockConfig>(config: &C, device: &Device) -> bool {
    let one = float_shapes(&config.init_block(1, device));
    let block = config.init_block(UNTIED_APPLICATIONS, device);
    let many = float_shapes(&block);
    let listed = block.untied_params();
    assert!(
        one.len() == many.len(),
        "the block has {} float parameters at {UNTIED_APPLICATIONS} applications and {} at 1 application.",
        many.len(),
        one.len(),
    );
    let mut unties = !listed.is_empty();
    for (one, many) in one.iter().zip(&many) {
        let grown: Vec<usize> = (0..many.dims.len()).filter(|&axis| many.dims[axis] != one.dims[axis]).collect();
        unties |= !grown.is_empty();
        let shapes = format!(
            "has the shape {:?} at {UNTIED_APPLICATIONS} applications and {:?} at 1 application",
            many.dims, one.dims,
        );
        match listed.iter().find(|param| param.id == many.id) {
            None => assert!(
                grown.is_empty(),
                "the parameter `{}` {shapes}. `Block::untied_params` does not name it.",
                many.path,
            ),
            Some(param) => {
                let axis = param.axis;
                assert!(
                    grown == [axis] && many.dims[axis] == UNTIED_APPLICATIONS * one.dims[axis],
                    "`Block::untied_params` names the parameter `{}` with its copies along axis {axis}. The \
                     parameter {shapes}.",
                    many.path,
                );
            }
        }
    }
    unties
}

/// One float parameter of a module: its id, its path and its shape.
struct FloatShape {
    id: ParamId,
    path: String,
    dims: Vec<usize>,
}

/// Each float parameter of `module`, in the order of the visit.
fn float_shapes(module: &impl Module) -> Vec<FloatShape> {
    #[derive(Default)]
    struct Collect {
        path: Vec<String>,
        params: Vec<FloatShape>,
    }
    impl ModuleVisitor for Collect {
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.path.push(name.to_string());
        }
        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let dims = param.val().shape().dims::<D>().to_vec();
            self.params.push(FloatShape { id: param.id, path: self.path.join("."), dims });
        }
    }
    let mut collect = Collect::default();
    module.visit(&mut collect);
    collect.params
}

/// The Muon projections of the config name rank-2 weights that exist. Each
/// spec of [`BlockConfig::muon_projections`] has these properties:
///
/// - it selects at least one weight of the block,
/// - each weight that it selects has rank 2,
/// - the widths of its sub-projections sum to the columns of that weight. For
///   a [tiled](crate::optim::ProjSpec::tiled) spec, they sum to the columns of
///   one copy.
///
/// A spec matches under a block container (see
/// [`BLOCK_CONTAINERS`](crate::optim::BLOCK_CONTAINERS)). So the check puts
/// the block in a [`Layers`] of 2 virtual layers
/// over 1 real layer, which gives each untied weight 2 copies.
///
/// Muon takes only rank-2 weights. A spec that selects nothing leaves its
/// weight on the fallback optimizer, with no error. The options and the
/// tolerance are not used.
#[cfg(feature = "optim")]
pub fn muon_projections_name_existing_rank_2_weights<C: BlockConfig + Clone>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) {
    use crate::optim::report::{float_params, selected};
    let _ = (options, f32_tol);
    run_contract("muon_projections_name_existing_rank_2_weights", || {
        let params = float_params(&muon_stack(config).init(&test_device()));

        for spec in config.muon_projections() {
            let hits = selected(&spec, &params);
            assert!(!hits.is_empty(), "the spec `{}` selects no weight of the block.", spec.path);
            let copies = if spec.tiled { MUON_APPLICATIONS } else { 1 };
            for param in hits {
                assert!(
                    param.dims.len() == 2,
                    "the spec `{}` selects `{}`, which has the shape {:?}. Muon takes only rank-2 weights.",
                    spec.path,
                    param.path,
                    param.dims,
                );
                assert!(
                    param.dims[1] == copies * spec.width(),
                    "the spec `{}` gives {copies} × {} columns. The weight `{}` has {} columns.",
                    spec.path,
                    spec.width(),
                    param.path,
                    param.dims[1],
                );
            }
        }
    });
}

/// The Muon projections of the config select only weights of the block. A
/// network around the block holds more rank-2 weights: its I/O weights (see
/// [`crate::optim`]) and the feed-forward of each layer. No spec of
/// [`BlockConfig::muon_projections`] selects one of them.
///
/// The check puts the block in two networks of 2 virtual layers over 1 real
/// layer:
///
/// - a [`LatentNetwork`](crate::modules::LatentNetwork) with a class token,
///   a class latent, a feed-forward and a final norm. Its `in_proj` and
///   `out_proj` have the names of usual block weights.
/// - a [`VocabNetwork`](crate::modules::VocabNetwork) with an embedding and
///   its own LM head.
///
/// A spec with a block scope ([`ProjSpec::block`](crate::optim::ProjSpec::block))
/// matches only under a block container. A spec with a plain path
/// ([`ProjSpec::path`](crate::optim::ProjSpec::path)) can also select a
/// weight of the network. Muon then owns an I/O weight, which the plan must
/// leave on the fallback optimizer. The options and the tolerance are not
/// used.
#[cfg(feature = "optim")]
pub fn muon_projections_select_only_weights_of_the_block<C: BlockConfig + Clone>(
    config: &C,
    options: <C::Block as Block>::Options,
    f32_tol: f32,
) {
    use crate::modules::{GatedMlpConfig, LatentNetworkBuilder, VocabNetworkBuilder};
    use crate::optim::report::{FloatParam, float_params, selected};
    use crate::utils::{ClassLatent, ClassToken};
    use std::collections::HashSet;
    let _ = (options, f32_tol);
    run_contract("muon_projections_select_only_weights_of_the_block", || {
        let device = test_device();
        let d_model = config.d_model();
        let latent = LatentNetworkBuilder {
            input_size: d_model,
            layers: LayersBuilder {
                class_latents: vec![ClassLatent::Start],
                mlp: Some(GatedMlpConfig::new(d_model, 2 * d_model)),
                ..muon_stack(config)
            },
            output_size: d_model,
            final_norm: true,
            class_tokens: vec![ClassToken::Start],
        }
        .init(&device);
        let vocab = VocabNetworkBuilder {
            vocab_size: d_model,
            pad_vocab_size_multiple: 1,
            layers: muon_stack(config),
            missing_lm_head: false,
        }
        .init(&device);
        // Each float parameter of a network, and the ids of those of its block.
        let block_ids = |block: &C::Block| float_params(block).into_iter().map(|param| param.id).collect();
        let networks: [(&str, Vec<FloatParam>, HashSet<_>); 2] = [
            ("LatentNetwork", float_params(&latent), block_ids(&latent.layers.real_layers[0].block)),
            ("VocabNetwork", float_params(&vocab), block_ids(&vocab.layers.real_layers[0].block)),
        ];

        for spec in config.muon_projections() {
            for (network, params, of_the_block) in &networks {
                let outside = selected(&spec, params).into_iter().find(|param| !of_the_block.contains(&param.id));
                if let Some(param) = outside {
                    panic!(
                        "the spec `{}` selects `{}` of a `{network}`, which is not a weight of the block.",
                        spec.path, param.path,
                    );
                }
            }
        }
    });
}

/// The applications of the block in the stack of a Muon contract. So each
/// untied weight has 2 copies.
#[cfg(feature = "optim")]
const MUON_APPLICATIONS: usize = 2;

/// The stack of a Muon contract: [`MUON_APPLICATIONS`] virtual layers over 1
/// real layer of the block.
#[cfg(feature = "optim")]
fn muon_stack<C: BlockConfig + Clone>(config: &C) -> crate::modules::LayersBuilder<C> {
    crate::modules::LayersBuilder {
        n_virtual_layers: Some((MUON_APPLICATIONS, crate::utils::Schedule::Cyclic)),
        ..crate::modules::LayersBuilder::new(1, config.clone())
    }
}

/// Run `checks`, the body of the contract `name`. Each panic in `checks`
/// panics again, with ``block contract `<name>`:`` in front of its message.
/// So a panic of the block or of a cache traversal also names the contract.
fn run_contract(name: &str, checks: impl FnOnce()) {
    let Err(payload) = catch_unwind(AssertUnwindSafe(checks)) else {
        return;
    };
    let message = match (payload.downcast_ref::<String>(), payload.downcast_ref::<&str>()) {
        (Some(message), _) => message.clone(),
        (None, Some(message)) => message.to_string(),
        (None, None) => "a check panicked with no text.".to_string(),
    };
    panic!("block contract `{name}`: {message}");
}

/// Panic unless `diff` is at most `tol`. A NaN `diff` panics.
fn check(what: &str, diff: f32, tol: f32) {
    assert!(diff <= tol, "{what} differ by {diff}. The tolerance is {tol}.");
}

/// Adds noise to each float parameter. So no float parameter keeps its
/// initial value.
struct Jitter;

impl ModuleMapper for Jitter {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|t| {
            let noise = Tensor::random(t.shape(), Distribution::Normal(0.0, 0.1), &t.device());
            t + noise
        })
    }
}

/// A random input `[batch, sequence, d_model]`.
fn randn(batch: usize, sequence: usize, d_model: usize, device: &Device) -> Tensor<3> {
    Tensor::random([batch, sequence, d_model], Distribution::Normal(0.0, 1.0), device)
}

/// The largest [`max_rel_diff`] over the matching tensors of two caches. Two
/// tensors with different shapes give infinity. A NaN stays. The container
/// tests also use it.
pub(crate) fn cache_diff<T: CacheTensors>(a: T, b: T) -> f32 {
    struct Diff(f32);
    impl TensorZip for Diff {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            let diff = match a.dims() == b.dims() {
                true => max_rel_diff(a.clone(), b),
                false => f32::INFINITY,
            };
            if diff.is_nan() || diff > self.0 {
                self.0 = diff;
            }
            a
        }
    }
    let mut diff = Diff(0.0);
    let _ = a.zip_tensors(b, &mut diff);
    diff.0
}

/// Row `b` of `cache`, with the batch axis kept. Axis 0 of each tensor of a
/// cache is the batch (see [`CacheTensors`]).
fn cache_row<T: CacheTensors>(cache: T, b: usize) -> T {
    struct Row(usize);
    impl TensorZip for Row {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            drop(b);
            a.narrow(0, self.0, 1)
        }
    }
    cache.clone().zip_tensors(cache, &mut Row(b))
}

/// [`Layers::step`] unrolled over each token of `x_bsd`, from no cache and
/// with no class cursors: the outputs `[batch, sequence, d_model]` and the
/// last caches.
fn layers_step_unrolled<M: Block>(layers: &Layers<M>, x_bsd: Tensor<3>) -> (Tensor<3>, M::Caches)
where
    M::Options: Clone,
{
    unrolled(x_bsd, None, |x_bd, caches| layers.step(x_bd, caches, None))
}

/// `block_step` unrolled over each token of `x_bsd`, from `cache`: the
/// outputs `[batch, sequence, d_model]` and the last cache.
fn step_unrolled<M: Block>(block: &M, x_bsd: Tensor<3>, cache: Option<M::Cache>) -> (Tensor<3>, M::Cache) {
    unrolled(x_bsd, cache, |x_bd, cache| block.block_step(x_bd, cache))
}

/// `step` unrolled over each token of `x_bsd`, from `state`: the outputs
/// `[batch, sequence, d_model]` and the last state.
fn unrolled<S>(
    x_bsd: Tensor<3>,
    state: Option<S>,
    step: impl Fn(Tensor<2>, Option<S>) -> (Tensor<2>, S),
) -> (Tensor<3>, S) {
    let [_batch, sequence, _d_model] = x_bsd.dims();
    assert!(sequence > 0, "a step needs at least one token");
    let mut state = state;
    let mut ys = Vec::with_capacity(sequence);
    for t in 0..sequence {
        let x_bd = x_bsd.clone().narrow(1, t, 1).squeeze_dim(1);
        let (y_bd, next) = step(x_bd, state);
        state = Some(next);
        ys.push(y_bd.unsqueeze_dim(1));
    }
    (Tensor::cat(ys, 1), state.expect("the sequence has at least one token"))
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
