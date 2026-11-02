//! The sub-projections of a fused projection: the data that the Muon
//! parameter groups are built from.
//!
//! Every family fuses several independent linear maps (the sub-projections)
//! into one `Linear`, a fused projection. So its weight is a *concatenation*
//! of matrices along its output (column) axis. Muon orthogonalises a whole
//! matrix at once, so it must know where those seams are. A [`ProjSpec`] is
//! that description. The
//! [`BlockConfig::muon_projections`](crate::modules::BlockConfig::muon_projections)
//! of each family builds one for each fused projection, right next to the
//! code that sizes them.

use burn::module::ParamGroup;

/// The field names that hold a [`Block`](crate::modules::Block).
///
/// [`Layer`](crate::modules::Layer) holds one (`block`).
/// [`BidiLayerPair`](crate::modules::bidi::BidiLayerPair) holds a pair. A
/// block weight that matches under any of them makes a [`ProjSpec`]
/// independent of the container. So one plan covers a plain stack, a
/// virtual-layer stack, and a bidirectional stack alike, including
/// hand-written models built from these pieces. Every entry ends in
/// `"block."`, which is what [`ProjSpec::predicates`] matches on.
pub const BLOCK_CONTAINERS: [&str; 3] = ["block.", "straight_block.", "reverse_block."];

/// Where in the module tree the path of a [`ProjSpec`] is anchored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjScope {
    /// A weight of the block: the path is matched under each of
    /// [`BLOCK_CONTAINERS`].
    Block,
    /// Any other weight: the path is matched as a plain substring of the
    /// parameter path (the layer MLP, a bidirectional merge, …).
    Path,
}

/// One sub-projection of a fused projection: a contiguous run of its columns.
///
/// `width` is the number of *columns* that the sub-projection owns (the
/// `Linear` weight of Burn is `[d_input, d_output]`, so the fused axis is
/// dim 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubProjection {
    /// Name of the sub-projection, for diagnostics (`"value"`, `"gate"`, …).
    pub name: &'static str,
    /// Number of columns that this sub-projection spans.
    pub width: usize,
    /// Whether Muon owns this sub-projection (`false` ⇒ it stays on the
    /// fallback optimizer, [`FallbackConfig`](super::FallbackConfig)).
    pub muon: bool,
}

impl SubProjection {
    /// A sub-projection that Muon orthogonalises on its own.
    pub fn muon(name: &'static str, width: usize) -> Self {
        Self { name, width, muon: true }
    }

    /// A sub-projection left to the fallback optimizer.
    ///
    /// For the channels that produce *per-head scalars*, not a feature vector
    /// (a step size, a decay rate). To orthogonalise a `[d_model, nheads]` slab
    /// would force the step-size directions of the heads to be mutually
    /// orthogonal. That is a constraint on a gain, not on a linear map. The
    /// reason is the same as for the usual "no Muon on biases, norm gains or
    /// embeddings" rule.
    pub fn adamw(name: &'static str, width: usize) -> Self {
        Self { name, width, muon: false }
    }
}

/// One 2-D weight tensor and how its columns split into sub-projections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjSpec {
    /// Trailing part of the parameter path that identifies the weight, e.g.
    /// `"in_proj.weight"`, anchored per [`Self::scope`].
    pub path: String,
    /// Where `path` is anchored.
    pub scope: ProjScope,
    /// The sub-projections, in order. Their widths must sum to the output
    /// width of the weight (or, when [`Self::tiled`], to one copy of it).
    pub sub_projections: Vec<SubProjection>,
    /// The weight is [untied](crate::utils::untied) along its fused axis: any
    /// whole number of copies of `sub_projections` side by side, one per application.
    /// The optimizer updates the sub-projections of each copy on their own.
    pub tiled: bool,
    /// More path substrings that the parameter must contain (empty ⇒ none).
    /// They scope the spec to one part of a model. A model that holds stacks
    /// of different widths needs this: the same weight name then has a
    /// different column layout in each stack (see [`Self::within`]).
    pub within: Vec<String>,
}

impl ProjSpec {
    /// A fused projection of the block.
    pub fn block(path: impl Into<String>, sub_projections: Vec<SubProjection>) -> Self {
        Self {
            path: path.into(),
            scope: ProjScope::Block,
            sub_projections,
            tiled: false,
            within: Vec::new(),
        }
    }

    /// An unfused weight of the block. Muon owns it in full.
    pub fn block_whole(path: impl Into<String>, width: usize) -> Self {
        Self::block(path, vec![SubProjection::muon("all", width)])
    }

    /// A fused projection matched by plain path substring.
    pub fn path(path: impl Into<String>, sub_projections: Vec<SubProjection>) -> Self {
        Self {
            path: path.into(),
            scope: ProjScope::Path,
            sub_projections,
            tiled: false,
            within: Vec::new(),
        }
    }

    /// This spec over an untied weight (see [`Self::tiled`]).
    pub fn tiled(mut self) -> Self {
        self.tiled = true;
        self
    }

    /// This spec, matched only under the path substring `scope` (for
    /// example, `"stages.0.encoder."`), in addition to its own predicates.
    /// Each call adds one more substring that the path must contain.
    pub fn within(mut self, scope: impl Into<String>) -> Self {
        self.within.push(scope.into());
        self
    }

    /// An unfused weight matched by plain path substring. Muon owns it in
    /// full.
    pub fn path_whole(path: impl Into<String>, width: usize) -> Self {
        Self::path(path, vec![SubProjection::muon("all", width)])
    }

    /// Total output width (the sum of the widths of the sub-projections).
    pub fn width(&self) -> usize {
        self.sub_projections.iter().map(|s| s.width).sum()
    }

    /// Whether Muon owns any sub-projection.
    pub fn has_muon(&self) -> bool {
        self.sub_projections.iter().any(|s| s.muon)
    }

    /// Whether Muon owns the whole tensor as a single sub-projection (so stock
    /// [`Muon`](burn::optim::Muon) applies directly, with no split). Never
    /// true for a [tiled](Self::tiled) weight, whose copies are separate
    /// matrices.
    pub fn is_whole_muon(&self) -> bool {
        !self.tiled && self.sub_projections.len() == 1 && self.sub_projections[0].muon
    }

    /// The path substrings that a parameter must **all** contain to belong to
    /// this spec: its own path and, under [`ProjScope::Block`], a block
    /// container.
    ///
    /// Two substrings, not one concatenation, because a block need not sit
    /// *directly* under the container field. A block that is an `enum` carries
    /// its variant name in between (`block.<Variant>.in_proj.weight`). A
    /// single `"block.in_proj.weight"` predicate would miss it, silently, and
    /// leave the weight on the fallback optimizer. `"block."` alone stands for
    /// every [`BLOCK_CONTAINERS`] entry, because each one ends with it.
    ///
    /// The [`Self::within`] scopes are more predicates of the same kind.
    pub fn predicates(&self) -> Vec<String> {
        let mut predicates = match self.scope {
            ProjScope::Block => vec!["block.".to_string(), self.path.clone()],
            ProjScope::Path => vec![self.path.clone()],
        };
        predicates.extend(self.within.iter().cloned());
        predicates
    }

    /// The parameter group selecting this weight (AND over [`Self::predicates`]).
    pub fn param_group(&self) -> ParamGroup {
        ParamGroup::from_predicates(self.predicates())
    }
}
