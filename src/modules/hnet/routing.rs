//! The routing module of H-Net: it decides where a chunk starts.
//!
//! ```text
//!   pₜ = ½·(1 − cos(W_q x̂ₜ₋₁, W_k x̂ₜ)) ∈ [0, 1]      bₜ = [pₜ > ½]
//! ```
//!
//! `x̂` is the output of the encoder. A row starts a new chunk (`bₜ = 1`) when
//! it turns away from the row before it. The first row of a sequence has no
//! row before it, so `p₀ = 1`. The paper writes `q` on the current row and `k`
//! on the previous row. The reference code (and this module) projects the
//! previous row with `W_q` and the current row with `W_k`. Both maps are
//! learnable and start as the identity, so the two forms span the same
//! function class. The reference also decides `p > ½`, not `p ≥ ½`, in both
//! `forward` and `step`.
//!
//! The router has a state: the `x̂` of the last row, and whether the slot saw a
//! row at all ([`RouterCache`]). So a chunked `forward` and a `step` both
//! continue a sequence where the previous call stopped.

use crate::utils::{div_eps, downcast, upcast};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// The routing module: two `d_model × d_model` maps, no bias, identity at
/// init. So at init, `pₜ` is the cosine distance of two adjacent encoder
/// outputs.
#[derive(Module, Debug)]
pub struct Router {
    /// The map of the previous row (`W_q`).
    pub q_proj: Linear,
    /// The map of the current row (`W_k`).
    pub k_proj: Linear,
}

impl Router {
    /// A router of width `d_model` with identity maps.
    pub fn init(d_model: usize, device: &Device) -> Self {
        let identity = || {
            let mut linear = LinearConfig::new(d_model, d_model).with_bias(false).init(device);
            // `Param::map` reads the flag again from the tensor that it gets.
            linear.weight = linear
                .weight
                .map(|w| Tensor::eye(d_model, &w.device()).set_require_grad(w.is_require_grad()));
            linear
        };
        Self { q_proj: identity(), k_proj: identity() }
    }

    /// `pₜ` for every row of `x_bsd`, continued from `cache`. A slot that saw
    /// no row before this call gets `p₀ = 1`.
    pub fn probs(&self, x_bsd: Tensor<3>, cache: &RouterCache) -> Tensor<2> {
        let [batch, sequence, _d_model] = x_bsd.dims();
        // The previous row of every row: the cached row, then all rows but the
        // last.
        let cached_b1d = cache.last_bd.clone().unsqueeze_dim(1);
        let prev_bsd = match sequence {
            1 => cached_b1d,
            _ => Tensor::cat(vec![cached_b1d, x_bsd.clone().narrow(1, 0, sequence - 1)], 1),
        };
        let p_bs = cos_distance(self.q_proj.forward(prev_bsd), self.k_proj.forward(x_bsd));
        // `p₀ = 1` where the slot is new. Only column 0 can be the first row.
        let fresh_b1 = cache.seen_b.clone().lower_elem(0.5).reshape([batch, 1]);
        let first_bs = Tensor::<1, Int>::arange(0..sequence as i64, &p_bs.device())
            .reshape([1, sequence])
            .equal_elem(0)
            .expand([batch, sequence]);
        let force_bs = first_bs.bool_and(fresh_b1.expand([batch, sequence]));
        p_bs.mask_fill(force_bs, 1.0)
    }

    /// `p` of one new row `x_bd`, continued from `cache`.
    pub fn step_prob(&self, x_bd: Tensor<2>, cache: &RouterCache) -> Tensor<1> {
        let p_bs = cos_distance(
            self.q_proj.forward(cache.last_bd.clone()).unsqueeze_dim(1),
            self.k_proj.forward(x_bd).unsqueeze_dim(1),
        );
        let [batch, 1] = p_bs.dims() else { unreachable!() };
        let fresh_b = cache.seen_b.clone().lower_elem(0.5);
        p_bs.reshape([batch]).mask_fill(fresh_b, 1.0)
    }
}

/// `½·(1 − cos(q, k))` along the last axis, clamped to `[0, 1]`. Each
/// vector is divided by `max(‖v‖, ε)`, as `F.normalize` does. It computes in
/// f32 when the inputs are f16/bf16.
///
/// The clamp is on `‖v‖²`, before the square root. A zero vector (the cached
/// row of a new slot) then gets a finite derivative. The derivative of `√0`
/// is infinite, and `0·∞` would put a NaN into the gradient.
fn cos_distance(q_bsd: Tensor<3>, k_bsd: Tensor<3>) -> Tensor<2> {
    let (q_bsd, dtype) = upcast(q_bsd);
    let (k_bsd, _) = upcast(k_bsd);
    let eps = div_eps(q_bsd.dtype());
    let norm = |v: Tensor<3>| v.clone() / v.square().sum_dim(2).clamp_min(eps * eps).sqrt();
    let cos_bs1 = (norm(q_bsd) * norm(k_bsd)).sum_dim(2);
    let [batch, sequence, 1] = cos_bs1.dims() else { unreachable!() };
    let p_bs = ((cos_bs1.neg() + 1.0) * 0.5).clamp(0.0, 1.0).reshape([batch, sequence]);
    downcast(p_bs, dtype)
}

/// The state of a [`Router`] between calls.
#[derive(Clone, Debug)]
pub struct RouterCache {
    /// The encoder output `x̂` of the last real row of each slot,
    /// `[batch, d_model]`.
    pub last_bd: Tensor<2>,
    /// `1` where the slot saw a real row, else `0`, `[batch]`. A float (not a
    /// bool), so that a captured step can write it back in place.
    pub seen_b: Tensor<1>,
}

impl RouterCache {
    /// The state of a slot that saw no row.
    pub fn zeros(batch: usize, d_model: usize, device: &Device) -> Self {
        Self {
            last_bd: Tensor::zeros([batch, d_model], device),
            seen_b: Tensor::zeros([batch], device),
        }
    }

    /// The state after the rows `x_bsd` of a `forward`. `pad_bs` (right
    /// padding, `None` ⇒ none) marks the rows that are absent. A slot with no
    /// real row keeps its state.
    pub(crate) fn after_rows(self, x_bsd: Tensor<3>, pad_bs: Option<&Tensor<2, Bool>>) -> Self {
        let [batch, sequence, d_model] = x_bsd.dims();
        let Some(pad_bs) = pad_bs else {
            let last_bd = x_bsd.narrow(1, sequence - 1, 1).reshape([batch, d_model]);
            return Self { seen_b: self.seen_b.ones_like(), last_bd };
        };
        let len_b = pad_bs.clone().bool_not().int().sum_dim(1).reshape([batch]);
        let has_b = len_b.clone().greater_elem(0);
        let idx_b11 = (len_b - 1).clamp_min(0).reshape([batch, 1, 1]);
        let row_bd = x_bsd
            .gather(1, idx_b11.expand([batch, 1, d_model]))
            .reshape([batch, d_model]);
        let has_bd = has_b.clone().reshape([batch, 1]).expand([batch, d_model]);
        Self {
            last_bd: self.last_bd.mask_where(has_bd, row_bd),
            seen_b: self.seen_b.mask_fill(has_b, 1.0),
        }
    }
}

/// What the router of one stage decided in one call.
///
/// A training loop reads it for the ratio loss ([`Self::ratio_loss`]), and a
/// sampler reads it to show where the chunks start.
#[derive(Clone, Debug)]
pub struct Routing {
    /// `pₜ`, `[batch, sequence]`.
    pub prob_bs: Tensor<2>,
    /// `bₜ = [pₜ > ½]`, `[batch, sequence]`. It is `false` on every absent
    /// row.
    pub boundary_bs: Tensor<2, Bool>,
    /// The absent rows of this stage, `[batch, sequence]` (`None` ⇒ none):
    /// the padding of the call. In a `step`, a row that did not reach this
    /// stage (its outer stage started no chunk) is also absent.
    pub pad_bs: Option<Tensor<2, Bool>>,
}

impl Routing {
    /// A stage that no row reached in a `step` (every row absent).
    pub(crate) fn absent(batch: usize, device: &Device) -> Self {
        Self {
            prob_bs: Tensor::zeros([batch, 1], device),
            boundary_bs: Tensor::<2, Int>::zeros([batch, 1], device).equal_elem(1),
            pad_bs: Some(Tensor::<2, Int>::zeros([batch, 1], device).equal_elem(0)),
        }
    }

    /// `(F, G)`, each `[1]`: the share of the real rows that start a chunk, and
    /// the mean `p` over the real rows. A call with no real row gives `0`.
    pub fn fractions(&self) -> (Tensor<1>, Tensor<1>) {
        let boundary_bs = self.boundary_bs.clone().float();
        match &self.pad_bs {
            None => (boundary_bs.mean(), self.prob_bs.clone().mean()),
            Some(pad_bs) => {
                let real_bs = pad_bs.clone().bool_not().float();
                let n = real_bs.clone().sum().clamp_min(1.0);
                let f = boundary_bs.sum() / n.clone();
                let g = (self.prob_bs.clone() * real_bs).sum() / n;
                (f, g)
            }
        }
    }

    /// The ratio loss of the paper, toward a mean chunk of `target` rows
    /// (`N > 1`), `[1]`:
    ///
    /// ```text
    ///   L = N/(N−1) · ((N−1)·F·G + (1−F)·(1−G))
    /// ```
    ///
    /// `F` has no gradient (it counts decisions). `G` carries the gradient to
    /// the router. At `F = G = 1/N`, `L = 1`. The reference weights it by
    /// `α = 0.03` per stage, beside the language-model loss.
    pub fn ratio_loss(&self, target: f64) -> Tensor<1> {
        assert!(target > 1.0, "the target chunk length must be above 1 row, got {target}");
        let (f, g) = self.fractions();
        let a = f.clone() * g.clone() * (target - 1.0);
        let b = (f.neg() + 1.0) * (g.neg() + 1.0);
        (a + b) * (target / (target - 1.0))
    }
}
