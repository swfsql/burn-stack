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
//! - The contract builds the block on
//!   [`test_device`](crate::utils::test_helpers::test_device), for one
//!   application when its docs do not give another count. It draws random
//!   inputs.
//! - The tolerance goes through
//!   [`dtype_tol`](crate::utils::test_helpers::dtype_tol). The outputs and
//!   the caches of a real block can be large, so each check of values uses
//!   [`max_rel_diff`](crate::utils::test_helpers::max_rel_diff).
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

use crate::modules::{Block, BlockConfig, CacheTensors, TensorZip};
use crate::utils::Packed;
use crate::utils::test_helpers::{dtype_tol, max_rel_diff, test_device};
use burn::module::{ModuleMapper, Param};
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

/// In a right-padded batch, each row gives what it gives alone: the outputs
/// at its real tokens, and its row of the final cache. An empty row (a row
/// with no real token) keeps its row of the incoming cache. The batch starts
/// from a cache, so that the empty row has a cache to keep.
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

        let (y_bsd, cache) = block.block_forward(x_bsd.clone(), Some(warm.clone()), options.clone(), Some(pad_bs));

        for (b, &len) in lens.iter().enumerate() {
            let warm_b = cache_row(warm.clone(), b);
            let cache_b = cache_row(cache.clone(), b);
            if len == 0 {
                let what = format!("the final and the incoming caches of the empty row {b}");
                check(&what, cache_diff(cache_b, warm_b), tol);
                continue;
            }
            let row_bsd = x_bsd.clone().narrow(0, b, 1).narrow(1, 0, len);
            let (y_alone, cache_alone) = block.block_forward(row_bsd, Some(warm_b), options.clone(), None);
            let y_row = y_bsd.clone().narrow(0, b, 1).narrow(1, 0, len);
            check(&format!("the outputs of row {b} in the batch and alone"), max_rel_diff(y_row, y_alone), tol);
            let what = format!("the final caches of row {b} in the batch and alone");
            check(&what, cache_diff(cache_b, cache_alone), tol);
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
/// value. So a float parameter that the record does not keep shows, also
/// when the block gives it the same initial value each time.
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
/// the block in a [`Layers`](crate::modules::Layers) of 2 virtual layers
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
    use crate::modules::LayersBuilder;
    use crate::optim::report::float_params;
    use crate::utils::Schedule;
    const APPLICATIONS: usize = 2;
    let _ = (options, f32_tol);
    run_contract("muon_projections_name_existing_rank_2_weights", || {
        let layers = LayersBuilder {
            n_virtual_layers: Some((APPLICATIONS, Schedule::Cyclic)),
            ..LayersBuilder::new(1, config.clone())
        }
        .init(&test_device());
        let params = float_params(&layers);

        for spec in config.muon_projections() {
            let group = spec.param_group();
            let selected: Vec<_> = params
                .iter()
                .filter(|param| group.matches(&param.id, Some(param.path.as_str())))
                .collect();
            assert!(!selected.is_empty(), "the spec `{}` selects no weight of the block.", spec.path);
            let copies = if spec.tiled { APPLICATIONS } else { 1 };
            for param in selected {
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
/// tensors with different shapes give infinity. A NaN stays.
fn cache_diff<T: CacheTensors>(a: T, b: T) -> f32 {
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

/// `block_step` unrolled over each token of `x_bsd`, from `cache`: the
/// outputs `[batch, sequence, d_model]` and the last cache.
fn step_unrolled<M: Block>(block: &M, x_bsd: Tensor<3>, cache: Option<M::Cache>) -> (Tensor<3>, M::Cache) {
    let [_batch, sequence, _d_model] = x_bsd.dims();
    assert!(sequence > 0, "a step needs at least one token");
    let mut cache = cache;
    let mut ys = Vec::with_capacity(sequence);
    for t in 0..sequence {
        let x_bd = x_bsd.clone().narrow(1, t, 1).squeeze_dim(1);
        let (y_bd, next) = block.block_step(x_bd, cache);
        cache = Some(next);
        ys.push(y_bd.unsqueeze_dim(1));
    }
    (Tensor::cat(ys, 1), cache.expect("the sequence has at least one token"))
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
