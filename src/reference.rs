//! A minimal reference [`Block`]: the smallest thing that satisfies the trait.
//!
//! It exists for two reasons:
//!
//! - The test suite of this crate composes it. So the tests exercise the
//!   containers without a dependency on any real mixer family.
//! - It is a worked example of the four things that a family must supply: a
//!   cache, a [`CacheStack`], [`Block`] and [`BlockConfig`] (plus
//!   [`CacheTensors`], which only a captured step needs).
//!
//! It unties its decay and its gate map on request ([`RefUntied`]), so the
//! tests exercise the untied path of the containers in the same way.
//!
//! The recurrence is a gated exponential moving average, one state vector per
//! token channel:
//!
//! ```text
//!   hₜ = σ(decay) ⊙ hₜ₋₁ + W_in xₜ
//!   yₜ = W_out (hₜ ⊙ silu(W_g xₜ))
//! ```
//!
//! It is stateful (so cache threading is observable), non-linear (so the
//! gradients are not degenerate), and cheap. `block_forward` unrolls the same
//! recurrence that `block_step` applies. So the forward/step parity that the
//! containers rely on is exact by construction. A real family gets it from a
//! chunkwise algorithm instead.
//!
//! The `test-helpers` feature enables it (the tests of this crate also have
//! it).

use crate::modules::{Block, BlockConfig, CacheStack, CacheTensors, Silu, TensorZip};
use crate::utils::untied::{self, UntiedParam};
use burn::config::Config;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// The streaming state of one (virtual) layer: the EMA accumulator
/// `[batch, d_model]`.
#[derive(Module, Debug)]
pub struct RefCache {
    /// The accumulator `hₜ`.
    pub state_bd: Tensor<2>,
}

/// One slot per (virtual) layer.
#[derive(Module, Debug)]
pub struct RefCaches {
    /// Per-layer caches, length = number of virtual layers.
    pub caches: Vec<RefCache>,
}

impl CacheStack for RefCaches {
    type Cache = RefCache;

    fn slot_count(&self) -> usize {
        self.caches.len()
    }

    fn into_slots(self) -> Vec<Option<RefCache>> {
        self.caches.into_iter().map(Some).collect()
    }

    fn from_slots(slots: Vec<Option<RefCache>>) -> Self {
        Self { caches: slots.into_iter().map(Option::unwrap).collect() }
    }

    fn cache_to_inner(c: RefCache) -> RefCache {
        RefCache { state_bd: c.state_bd.inner() }
    }

    fn cache_from_inner(c: RefCache) -> RefCache {
        RefCache { state_bd: Tensor::from_inner(c.state_bd) }
    }
}

impl CacheTensors for RefCache {
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
        RefCache { state_bd: z.zip(self.state_bd, other.state_bd) }
    }
}

impl CacheTensors for RefCaches {
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
        RefCaches { caches: self.caches.zip_tensors(other.caches, z) }
    }
}

/// A [`RefBlock`] parameter that can be held once per application, not tied
/// (see [`crate::utils::untied`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RefUntied {
    /// The decay [`RefBlock::decay_raw`], its copies along its only axis.
    Decay,
    /// The gate map [`RefBlock::gate_proj`], its copies along the output axis.
    /// This is the untied 2-D weight that Muon must step one copy at a time.
    GateProj,
}

/// The reference mixer block.
#[derive(Module, Debug)]
pub struct RefBlock {
    /// `d_model → d_model` input map feeding the accumulator.
    pub in_proj: Linear,
    /// `d_model → d_model` gate map (SiLU-activated).
    pub gate_proj: Linear,
    /// `d_model → d_model` readout.
    pub out_proj: Linear,
    /// Pre-sigmoid per-channel decay, shape `[d_model]`.
    pub decay_raw: Param<Tensor<1>>,
    /// The parameters held once per application.
    #[module(skip)]
    pub untied: Vec<RefUntied>,
    /// See [`RefBlockConfig::check_padding`].
    #[module(skip)]
    pub check_padding: bool,
}

impl RefBlock {
    /// One recurrence step: `(y, hₜ)` from `(xₜ, hₜ₋₁)`.
    fn recurrence(&self, x_bd: Tensor<2>, prev_bd: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let decay_1d: Tensor<2> = burn::tensor::activation::sigmoid(self.decay_raw.val()).unsqueeze();
        let state_bd = prev_bd * decay_1d + self.in_proj.forward(x_bd.clone());
        let gate_bd = Silu::new().forward(self.gate_proj.forward(x_bd));
        let y_bd = self.out_proj.forward(state_bd.clone() * gate_bd);
        (y_bd, state_bd)
    }

    fn zero_state(&self, batch: usize, device: &Device) -> Tensor<2> {
        let [d_model] = self.decay_raw.val().dims();
        Tensor::zeros([batch, d_model], device)
    }

    fn zero_caches(&self, batch: usize, n_virtual: usize, device: &Device) -> RefCaches {
        RefCaches {
            caches: (0..n_virtual)
                .map(|_| RefCache { state_bd: self.zero_state(batch, device) })
                .collect(),
        }
    }
}

impl Block for RefBlock {
    type Cache = RefCache;
    type Caches = RefCaches;
    /// Nothing to select: the block has one algorithm.
    type Options = ();

    /// A padded row keeps the state that it found: the recurrence skips it.
    fn block_forward(
        &self,
        x_bsd: Tensor<3>,
        cache: Option<RefCache>,
        _options: (),
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, RefCache) {
        let [batch, sequence, _d_model] = x_bsd.dims();
        let device = x_bsd.device();
        if let Some(pad_bs) = pad.as_ref().filter(|_| self.check_padding) {
            assert_right_padded(pad_bs.clone());
        }
        let mut state_bd = match cache {
            Some(c) => c.state_bd,
            None => self.zero_state(batch, &device),
        };
        let mut ys = Vec::with_capacity(sequence);
        for t in 0..sequence {
            let x_bd = x_bsd.clone().narrow(1, t, 1).squeeze_dim(1);
            let (y_bd, next_bd) = self.recurrence(x_bd, state_bd.clone());
            state_bd = match &pad {
                Some(pad_bs) => {
                    let pad_bd = pad_bs.clone().narrow(1, t, 1).expand(next_bd.dims());
                    next_bd.mask_where(pad_bd, state_bd)
                }
                None => next_bd,
            };
            ys.push(y_bd.unsqueeze_dim(1));
        }
        (Tensor::cat(ys, 1), RefCache { state_bd })
    }

    fn block_step(&self, x_bd: Tensor<2>, cache: Option<RefCache>) -> (Tensor<2>, RefCache) {
        let [batch, _d_model] = x_bd.dims();
        let prev_bd = match cache {
            Some(c) => c.state_bd,
            None => self.zero_state(batch, &x_bd.device()),
        };
        let (y_bd, state_bd) = self.recurrence(x_bd, prev_bd);
        (y_bd, RefCache { state_bd })
    }

    fn zero_caches_3d(&self, x_bsd: &Tensor<3>, n_virtual: usize) -> RefCaches {
        let [batch, _s, _d] = x_bsd.dims();
        self.zero_caches(batch, n_virtual, &x_bsd.device())
    }

    fn zero_caches_2d(&self, x_bd: &Tensor<2>, n_virtual: usize) -> RefCaches {
        let [batch, _d] = x_bd.dims();
        self.zero_caches(batch, n_virtual, &x_bd.device())
    }

    fn untied_params(&self) -> Vec<UntiedParam> {
        self.untied
            .iter()
            .map(|part| match part {
                RefUntied::Decay => UntiedParam::new(&self.decay_raw, 0),
                RefUntied::GateProj => UntiedParam::new(&self.gate_proj.weight, 1),
            })
            .collect()
    }
}

/// Panic unless the padding of every slot follows all of its real rows. The
/// reference block would skip padding anywhere. But a real block relies on
/// the right padding that [`Block::block_forward`] promises. So the reference
/// checks that the containers keep that promise, whatever they splice.
fn assert_right_padded(pad_bs: Tensor<2, Bool>) {
    let [_batch, sequence] = pad_bs.dims();
    // Read as ints: a backend may store a bool as a byte.
    let pad: Vec<bool> = pad_bs.int().into_data().iter::<i64>().map(|v| v != 0).collect();
    for (b, row) in pad.chunks(sequence).enumerate() {
        assert!(
            row.windows(2).all(|w| !w[0] || w[1]),
            "slot {b}: a real row follows padding ({row:?})",
        );
    }
}

/// Config for [`RefBlock`].
#[derive(Config, Debug)]
pub struct RefBlockConfig {
    /// Model width.
    pub d_model: usize,
    /// The parameters held once per application instead of tied.
    #[config(default = "Vec::new()")]
    pub untied: Vec<RefUntied>,
    /// Whether `forward` checks that its `pad` mask is right-padded. The check
    /// reads to the host, which a captured forward cannot do.
    #[config(default = true)]
    pub check_padding: bool,
}

impl RefBlockConfig {
    /// Allocate the block on `device`, for a single application.
    pub fn init(&self, device: &Device) -> RefBlock {
        self.init_applications(1, device)
    }

    /// Allocate the block on `device` for `n_applications` applications, with
    /// every [`Self::untied`] parameter tiled that many times.
    pub fn init_applications(&self, n_applications: usize, device: &Device) -> RefBlock {
        let lin = || LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device);
        let mut gate_proj = lin();
        if self.untied.contains(&RefUntied::GateProj) {
            gate_proj.weight = untied::tile(gate_proj.weight, 1, n_applications);
        }
        let mut decay_raw = Param::from_tensor(Tensor::zeros([self.d_model], device));
        if self.untied.contains(&RefUntied::Decay) {
            decay_raw = untied::tile(decay_raw, 0, n_applications);
        }
        RefBlock {
            in_proj: lin(),
            gate_proj,
            out_proj: lin(),
            decay_raw,
            untied: self.untied.clone(),
            check_padding: self.check_padding,
        }
    }
}

impl BlockConfig for RefBlockConfig {
    type Block = RefBlock;

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn init_block(&self, n_applications: usize, device: &Device) -> RefBlock {
        self.init_applications(n_applications, device)
    }

    /// Three plain (unfused) square maps. The gate map is one per application
    /// when untied. The `[d_model]` decay is rank 1, so it stays on the
    /// fallback optimizer.
    #[cfg(feature = "optim")]
    fn muon_projections(&self) -> Vec<crate::optim::ProjSpec> {
        use crate::optim::ProjSpec;
        let gate = ProjSpec::block_whole("gate_proj.weight", self.d_model);
        vec![
            ProjSpec::block_whole("in_proj.weight", self.d_model),
            match self.untied.contains(&RefUntied::GateProj) {
                true => gate.tiled(),
                false => gate,
            },
            ProjSpec::block_whole("out_proj.weight", self.d_model),
        ]
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
