//! Each contract passes on [`RefBlock`] with a near-zero tolerance. Each
//! contract panics on a [`BrokenBlock`] that breaks it.
//!
//! A [`BrokenBlock`] wraps a [`RefBlock`] and breaks at most one contract,
//! which its [`Breach`] selects. With no breach, it keeps every contract. So
//! the panic of a breach comes from that breach.

use super::*;
use crate::modules::CacheStack;
use crate::reference::{RefBlock, RefBlockConfig, RefCache, RefUntied};
use burn::config::Config;
use std::sync::atomic::{AtomicUsize, Ordering};

const D_MODEL: usize = 8;
/// Near zero: [`RefBlock`] applies the same recurrence on each path, so only
/// the rounding can differ.
const TOL: f32 = 1e-6;

/// The [`RefBlock`] settings that each contract runs on.
fn ref_configs() -> Vec<RefBlockConfig> {
    vec![
        RefBlockConfig::new(D_MODEL),
        RefBlockConfig::new(D_MODEL).with_reset_align(4),
        RefBlockConfig::new(D_MODEL).with_untied(vec![RefUntied::Decay, RefUntied::GateProj]),
    ]
}

/// The contract that a [`BrokenBlock`] breaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum Breach {
    /// The step adds 1 to each output.
    Step,
    /// The forward ignores its incoming cache.
    Window,
    /// The forward ignores its padding mask.
    Padding,
    /// The packed forward ignores its resets.
    Packing,
    /// Each new block adds its own constant to each output. The record does
    /// not hold the constant.
    Record,
    /// The cache traversal skips the state of the cache.
    Traversal,
    /// The Muon projections name the rank-1 decay.
    MuonRank1,
    /// The Muon projections name a weight that the block does not have.
    MuonMissing,
    /// The Muon projections give the input map one column too many.
    MuonWidth,
    /// The Muon projections name the input map by a plain path, so the spec
    /// also selects the `in_proj` of a network.
    MuonOutside,
    /// The forward panics. This breaks each contract that runs the forward.
    Panics,
}

/// The cache of a [`BrokenBlock`]: the cache of its [`RefBlock`].
#[derive(Clone, Debug)]
struct BrokenCache {
    inner: RefCache,
    /// [`Breach::Traversal`]: the traversal skips `inner`.
    skip_state: bool,
}

impl CacheTensors for BrokenCache {
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
        match self.skip_state {
            true => self,
            false => BrokenCache { inner: self.inner.zip_tensors(other.inner, z), ..self },
        }
    }
}

/// The cache stack of a [`BrokenBlock`]: one slot for each virtual layer.
struct BrokenCaches(Vec<BrokenCache>);

impl CacheStack for BrokenCaches {
    type Cache = BrokenCache;

    fn slot_count(&self) -> usize {
        self.0.len()
    }

    fn into_slots(self) -> Vec<Option<BrokenCache>> {
        self.0.into_iter().map(Some).collect()
    }

    fn from_slots(slots: Vec<Option<BrokenCache>>) -> Self {
        Self(slots.into_iter().map(Option::unwrap).collect())
    }
}

/// A [`RefBlock`] that breaks the contract of its [`Breach`].
#[derive(Module, Debug)]
struct BrokenBlock {
    inner: RefBlock,
    #[module(skip)]
    breach: Option<Breach>,
    /// `false`: the block takes no packed rows. This breaks no contract.
    #[module(skip)]
    takes_packed_rows: bool,
    /// The constant of [`Breach::Record`], `0` for each other breach.
    #[module(skip)]
    offset: f32,
}

impl BrokenBlock {
    fn breaks(&self, breach: Breach) -> bool {
        self.breach == Some(breach)
    }

    fn wrap(&self, inner: RefCache) -> BrokenCache {
        BrokenCache { inner, skip_state: self.breaks(Breach::Traversal) }
    }
}

impl Block for BrokenBlock {
    type Cache = BrokenCache;
    type Caches = BrokenCaches;
    type Options = ();

    fn block_forward(
        &self,
        x_bsd: Tensor<3>,
        cache: Option<BrokenCache>,
        _options: (),
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, BrokenCache) {
        assert!(!self.breaks(Breach::Panics), "the forward of this block panics");
        let cache = cache.filter(|_| !self.breaks(Breach::Window)).map(|c| c.inner);
        let pad = pad.filter(|_| !self.breaks(Breach::Padding));
        let (y_bsd, cache) = self.inner.block_forward(x_bsd, cache, (), pad);
        (y_bsd.add_scalar(self.offset), self.wrap(cache))
    }

    fn block_forward_packed(
        &self,
        x_bsd: Tensor<3>,
        cache: Option<BrokenCache>,
        _options: (),
        reset_bs: Tensor<2, Bool>,
    ) -> (Tensor<3>, BrokenCache) {
        assert!(self.takes_packed_rows, "this block takes no packed rows");
        let cache = cache.map(|c| c.inner);
        let (y_bsd, cache) = match self.breaks(Breach::Packing) {
            true => self.inner.block_forward(x_bsd, cache, (), None),
            false => self.inner.block_forward_packed(x_bsd, cache, (), reset_bs),
        };
        (y_bsd.add_scalar(self.offset), self.wrap(cache))
    }

    fn reset_align(&self, _options: &()) -> Option<usize> {
        self.takes_packed_rows.then(|| self.inner.reset_align(&())).flatten()
    }

    fn block_step(&self, x_bd: Tensor<2>, cache: Option<BrokenCache>) -> (Tensor<2>, BrokenCache) {
        let (y_bd, cache) = self.inner.block_step(x_bd, cache.map(|c| c.inner));
        let step_offset = if self.breaks(Breach::Step) { 1.0 } else { 0.0 };
        (y_bd.add_scalar(self.offset + step_offset), self.wrap(cache))
    }

    fn zero_cache(&self, batch: usize, device: &Device) -> BrokenCache {
        self.wrap(self.inner.zero_cache(batch, device))
    }

    fn untied_params(&self) -> Vec<crate::utils::UntiedParam> {
        self.inner.untied_params()
    }
}

/// Config for [`BrokenBlock`].
#[derive(Config, Debug)]
struct BrokenBlockConfig {
    inner: RefBlockConfig,
    #[config(default = "None")]
    breach: Option<Breach>,
    #[config(default = true)]
    takes_packed_rows: bool,
}

/// The count of the blocks of [`Breach::Record`] so far. Each one adds its
/// own count, so two such blocks differ.
static RECORD_BREACHES: AtomicUsize = AtomicUsize::new(0);

impl BlockConfig for BrokenBlockConfig {
    type Block = BrokenBlock;

    fn d_model(&self) -> usize {
        self.inner.d_model
    }

    fn init_block(&self, n_applications: usize, device: &Device) -> BrokenBlock {
        let offset = match self.breach {
            Some(Breach::Record) => (RECORD_BREACHES.fetch_add(1, Ordering::Relaxed) + 1) as f32,
            _ => 0.0,
        };
        BrokenBlock {
            inner: self.inner.init_block(n_applications, device),
            breach: self.breach,
            takes_packed_rows: self.takes_packed_rows,
            offset,
        }
    }

    #[cfg(feature = "optim")]
    fn muon_projections(&self) -> Vec<crate::optim::ProjSpec> {
        use crate::optim::ProjSpec;
        let mut specs = self.inner.muon_projections();
        match self.breach {
            Some(Breach::MuonRank1) => specs.push(ProjSpec::block_whole("decay_raw", D_MODEL)),
            Some(Breach::MuonMissing) => specs.push(ProjSpec::block_whole("missing_proj.weight", D_MODEL)),
            Some(Breach::MuonWidth) => specs.push(ProjSpec::block_whole("in_proj.weight", D_MODEL + 1)),
            Some(Breach::MuonOutside) => specs.push(ProjSpec::path_whole("in_proj.weight", D_MODEL)),
            _ => {}
        }
        specs
    }
}

/// A [`BrokenBlock`] over the default [`RefBlock`], with `breach`.
fn broken(breach: Option<Breach>) -> BrokenBlockConfig {
    BrokenBlockConfig::new(RefBlockConfig::new(D_MODEL)).with_breach(breach)
}

/// With no breach, the wrapper keeps every contract. So the panic of each
/// test below comes from its breach.
#[test]
fn a_broken_block_without_a_breach_keeps_every_contract() {
    let config = broken(None);
    forward_equals_step_unrolled(&config, (), TOL);
    forward_splits_into_windows_through_the_cache(&config, (), TOL);
    padded_rows_equal_each_row_alone(&config, (), TOL);
    packed_segments_equal_each_segment_alone(&config, (), TOL);
    record_round_trip_keeps_the_block(&config, (), TOL);
    cache_traversal_moves_every_field(&config, (), TOL);
    #[cfg(feature = "optim")]
    muon_projections_name_existing_rank_2_weights(&config, (), TOL);
    #[cfg(feature = "optim")]
    muon_projections_select_only_weights_of_the_block(&config, (), TOL);
}

/// A panic inside the block also names the contract.
#[test]
#[should_panic(expected = "block contract `forward_equals_step_unrolled`: the forward of this block panics")]
fn a_panic_of_the_block_names_the_contract() {
    forward_equals_step_unrolled(&broken(Some(Breach::Panics)), (), TOL);
}

// ---------------------------------------------------------------------------
// The forward equals the step unrolled
// ---------------------------------------------------------------------------

#[test]
fn ref_block_keeps_forward_equals_step_unrolled() {
    for config in ref_configs() {
        forward_equals_step_unrolled(&config, (), TOL);
    }
}

#[test]
#[should_panic(
    expected = "block contract `forward_equals_step_unrolled`: the outputs of the forward and of the step unrolled differ"
)]
fn a_step_that_differs_breaks_forward_equals_step_unrolled() {
    forward_equals_step_unrolled(&broken(Some(Breach::Step)), (), TOL);
}

// ---------------------------------------------------------------------------
// A window split through the cache equals one window
// ---------------------------------------------------------------------------

#[test]
fn ref_block_keeps_forward_splits_into_windows_through_the_cache() {
    for config in ref_configs() {
        forward_splits_into_windows_through_the_cache(&config, (), TOL);
    }
}

#[test]
#[should_panic(
    expected = "block contract `forward_splits_into_windows_through_the_cache`: the outputs of the two windows and of one window differ"
)]
fn a_forward_that_ignores_its_cache_breaks_forward_splits_into_windows_through_the_cache() {
    forward_splits_into_windows_through_the_cache(&broken(Some(Breach::Window)), (), TOL);
}

// ---------------------------------------------------------------------------
// Padded rows equal each row alone
// ---------------------------------------------------------------------------

#[test]
fn ref_block_keeps_padded_rows_equal_each_row_alone() {
    for config in ref_configs() {
        padded_rows_equal_each_row_alone(&config, (), TOL);
    }
}

#[test]
#[should_panic(
    expected = "block contract `padded_rows_equal_each_row_alone`: the final caches of row 1 in the batch and alone differ"
)]
fn a_forward_that_ignores_its_padding_breaks_padded_rows_equal_each_row_alone() {
    padded_rows_equal_each_row_alone(&broken(Some(Breach::Padding)), (), TOL);
}

// ---------------------------------------------------------------------------
// Packed segments equal each segment alone
// ---------------------------------------------------------------------------

#[test]
fn ref_block_keeps_packed_segments_equal_each_segment_alone() {
    for config in ref_configs() {
        packed_segments_equal_each_segment_alone(&config, (), TOL);
    }
}

#[test]
#[should_panic(
    expected = "block contract `packed_segments_equal_each_segment_alone`: the outputs of row 0 at [0, 3) when packed and alone differ"
)]
fn a_packed_forward_that_ignores_its_resets_breaks_packed_segments_equal_each_segment_alone() {
    packed_segments_equal_each_segment_alone(&broken(Some(Breach::Packing)), (), TOL);
}

/// The block takes no packed rows, so its packed forward panics. The contract
/// does not apply, and returns before the call.
#[test]
fn packed_segments_equal_each_segment_alone_skips_a_block_without_packed_rows() {
    let config = broken(None).with_takes_packed_rows(false);
    assert_eq!(config.init_block(1, &test_device()).reset_align(&()), None);
    packed_segments_equal_each_segment_alone(&config, (), TOL);
}

// ---------------------------------------------------------------------------
// A record round trip keeps the block
// ---------------------------------------------------------------------------

#[test]
fn ref_block_keeps_record_round_trip_keeps_the_block() {
    for config in ref_configs() {
        record_round_trip_keeps_the_block(&config, (), TOL);
    }
}

#[test]
#[should_panic(
    expected = "block contract `record_round_trip_keeps_the_block`: the outputs of the forward of the block and of its record in a new block differ"
)]
fn a_constant_outside_the_record_breaks_record_round_trip_keeps_the_block() {
    record_round_trip_keeps_the_block(&broken(Some(Breach::Record)), (), TOL);
}

// ---------------------------------------------------------------------------
// The cache traversal moves every field of the cache
// ---------------------------------------------------------------------------

#[test]
fn ref_block_keeps_cache_traversal_moves_every_field() {
    for config in ref_configs() {
        cache_traversal_moves_every_field(&config, (), TOL);
    }
}

#[test]
#[should_panic(
    expected = "block contract `cache_traversal_moves_every_field`: the outputs of the forward from the moved cache and from its source differ"
)]
fn a_traversal_that_skips_the_state_breaks_cache_traversal_moves_every_field() {
    cache_traversal_moves_every_field(&broken(Some(Breach::Traversal)), (), TOL);
}

// ---------------------------------------------------------------------------
// The Muon projections name rank-2 weights that exist
// ---------------------------------------------------------------------------

#[cfg(feature = "optim")]
#[test]
fn ref_block_keeps_muon_projections_name_existing_rank_2_weights() {
    for config in ref_configs() {
        muon_projections_name_existing_rank_2_weights(&config, (), TOL);
    }
}

#[cfg(feature = "optim")]
#[test]
#[should_panic(
    expected = "block contract `muon_projections_name_existing_rank_2_weights`: the spec `decay_raw` selects `real_layers.0.block.inner.decay_raw`, which has the shape [8]"
)]
fn a_rank_1_weight_breaks_muon_projections_name_existing_rank_2_weights() {
    muon_projections_name_existing_rank_2_weights(&broken(Some(Breach::MuonRank1)), (), TOL);
}

#[cfg(feature = "optim")]
#[test]
#[should_panic(
    expected = "block contract `muon_projections_name_existing_rank_2_weights`: the spec `missing_proj.weight` selects no weight"
)]
fn a_missing_weight_breaks_muon_projections_name_existing_rank_2_weights() {
    muon_projections_name_existing_rank_2_weights(&broken(Some(Breach::MuonMissing)), (), TOL);
}

#[cfg(feature = "optim")]
#[test]
#[should_panic(
    expected = "block contract `muon_projections_name_existing_rank_2_weights`: the spec `in_proj.weight` gives 1 × 9 columns"
)]
fn a_wrong_width_breaks_muon_projections_name_existing_rank_2_weights() {
    muon_projections_name_existing_rank_2_weights(&broken(Some(Breach::MuonWidth)), (), TOL);
}

// ---------------------------------------------------------------------------
// The Muon projections select only weights of the block
// ---------------------------------------------------------------------------

#[cfg(feature = "optim")]
#[test]
fn ref_block_keeps_muon_projections_select_only_weights_of_the_block() {
    for config in ref_configs() {
        muon_projections_select_only_weights_of_the_block(&config, (), TOL);
    }
}

#[cfg(feature = "optim")]
#[test]
#[should_panic(
    expected = "block contract `muon_projections_select_only_weights_of_the_block`: the spec `in_proj.weight` selects `in_proj.weight` of a `LatentNetwork`"
)]
fn a_spec_on_a_plain_path_breaks_muon_projections_select_only_weights_of_the_block() {
    muon_projections_select_only_weights_of_the_block(&broken(Some(Breach::MuonOutside)), (), TOL);
}
