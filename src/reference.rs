//! A minimal reference [`Block`] — the smallest thing that satisfies the trait.
//!
//! It exists for two reasons: it is what this crate's own test suite composes
//! (so the containers are exercised without depending on any real mixer
//! family), and it is a worked example of the four things a family has to
//! supply — a cache, a [`CacheStack`], [`Block`], and [`BlockConfig`].
//!
//! The recurrence is a gated exponential moving average, one state vector per
//! token channel:
//!
//! ```text
//!   hₜ = σ(decay) ⊙ hₜ₋₁ + W_in xₜ
//!   yₜ = W_out (hₜ ⊙ silu(W_g xₜ))
//! ```
//!
//! Stateful (so cache threading is observable), non-linear (so gradients are
//! not degenerate), and cheap. `block_forward` unrolls the same recurrence
//! `block_step` applies, which makes the forward/step parity the containers
//! rely on exact by construction — a real family earns it with a chunkwise
//! algorithm instead.
//!
//! Enabled by the `test-helpers` feature (or inside this crate's own tests).

use crate::modules::{Block, BlockConfig, CacheStack, Silu};
use burn::config::Config;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// One (virtual) layer's streaming state: the EMA accumulator `[batch, d_model]`.
#[derive(Module, Debug)]
pub struct RefCache {
    /// The accumulator `hₜ`.
    pub state_bd: Tensor<2>,
}

/// One slot per (virtual) layer.
#[derive(Module, Debug)]
pub struct RefCaches {
    /// Per-layer caches; length = number of virtual layers.
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
    /// Nothing to select — the block has one algorithm.
    type Options = ();

    fn block_forward(
        &self,
        x_bsd: Tensor<3>,
        cache: Option<RefCache>,
        _options: (),
    ) -> (Tensor<3>, RefCache) {
        let [batch, sequence, _d_model] = x_bsd.dims();
        let device = x_bsd.device();
        let mut state_bd = match cache {
            Some(c) => c.state_bd,
            None => self.zero_state(batch, &device),
        };
        let mut ys = Vec::with_capacity(sequence);
        for t in 0..sequence {
            let x_bd = x_bsd.clone().narrow(1, t, 1).squeeze_dim(1);
            let (y_bd, next_bd) = self.recurrence(x_bd, state_bd);
            state_bd = next_bd;
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
}

/// Config for [`RefBlock`].
#[derive(Config, Debug)]
pub struct RefBlockConfig {
    /// Model width.
    pub d_model: usize,
}

impl RefBlockConfig {
    /// Allocate the block on `device`.
    pub fn init(&self, device: &Device) -> RefBlock {
        let lin = || LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device);
        RefBlock {
            in_proj: lin(),
            gate_proj: lin(),
            out_proj: lin(),
            decay_raw: Param::from_tensor(Tensor::zeros([self.d_model], device)),
        }
    }
}

impl BlockConfig for RefBlockConfig {
    type Block = RefBlock;

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn init_block(&self, device: &Device) -> RefBlock {
        self.init(device)
    }

    /// Three plain (unfused) square maps; the `[d_model]` decay is rank 1 and so
    /// stays on the fallback optimizer.
    #[cfg(feature = "optim")]
    fn muon_projections(&self) -> Vec<crate::optim::ProjSpec> {
        use crate::optim::ProjSpec;
        vec![
            ProjSpec::block_whole("in_proj.weight", self.d_model),
            ProjSpec::block_whole("gate_proj.weight", self.d_model),
            ProjSpec::block_whole("out_proj.weight", self.d_model),
        ]
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
