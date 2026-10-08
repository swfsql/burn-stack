//! # H-Net: dynamic chunking between levels of block stacks
//!
//! An H-Net (Hwang, Wang & Gu, *Dynamic Chunking for End-to-End Hierarchical
//! Sequence Modeling*, 2025) is a U-Net over the sequence axis. Each level
//! (a **stage**) runs a stack at the resolution of its input. A learned router
//! decides where a new chunk starts. The stage keeps only those rows, gives
//! that shorter sequence to the next level, and expands the result back:
//!
//! ```text
//!   x̂ˢ      = Eˢ(xˢ)                                encoder (a stack + RMSNorm)
//!   pˢ, bˢ  = Router(x̂ˢ)                            pₜ = ½(1 − cos(W_q x̂ₜ₋₁, W_k x̂ₜ)), bₜ = [pₜ > ½]
//!   xˢ⁺¹    = x̂ˢ[bˢ]                                chunk: keep the boundary rows
//!   ẑˢ⁺¹    = the inner network on xˢ⁺¹             the next stage, or the main network M
//!   z̄ₖ      = Pₖ ẑₖ + (1 − Pₖ) z̄ₖ₋₁                 smoothing (EMA over the chunks)
//!   zₜˢ     = STE(cₜ) · z̄_{upto(t)} + W_r x̂ₜˢ       dechunk: repeat, confidence, residual
//!   ẑˢ      = Dˢ(zˢ)                                decoder (a stack + RMSNorm)
//! ```
//!
//! - `cₜ = pₜ` where `bₜ = 1`, else `1 − pₜ`: the confidence of the decision
//!   at row `t`. `STE(c) = c + sg(1 − c)` is `1` in the forward pass, with the
//!   gradient of `c`. So the router gets a gradient through every row, and
//!   through the smoothing.
//! - `W_r` (`residual_proj`) starts at zero, so at init each stage passes the
//!   coarse path only. The reference computes it in fp32.
//! - Every network ends in an RMSNorm (the *network normalization* of the
//!   paper). Without it, the deep main network would drown the residual of
//!   the encoder.
//! - A stage can be wider than its input. It then appends a learnable vector
//!   (shared by all rows) to each input row, and returns the first columns of
//!   its output. So the widths must not decrease inward.
//!
//! The submodules hold the details:
//!
//! - [`routing`](crate::modules::hnet::routing): the router and the ratio
//!   loss.
//! - `chunk`: the index math.
//! - [`smooth`](crate::modules::hnet::smooth): the EMA as a chunked scan.
//! - [`cache`](crate::modules::hnet::cache): the caches and the per-row cache
//!   operations.
//!
//! ## Two block families
//!
//! `HNet<E, M>`: every encoder and decoder is a [`Layers<E>`], and the main
//! network is a [`Layers<M>`]. The reference uses Mamba-2 outside (it handles
//! the fine-grained rows well) and attention inside. Each stack has its own
//! [`BlockConfig`] (so its own width) and its own
//! [`NetworkShape`](crate::modules::NetworkShape) (virtual layers, MLP,
//! residuals, …). See [`HNetShape`].
//!
//! ## Execution
//!
//! - `forward` takes a right-padded batch, like every container of this
//!   crate. The chunk count differs from slot to slot. So the inner sequence
//!   gets right padding up to the largest count, and the inner network runs
//!   under the padding contract of [`Block`]. That count is a shape, so
//!   `forward` reads it to the host once per stage. A `forward` thus does not
//!   replay as a captured graph.
//! - `step` runs one row. The inner network must step only on the rows that
//!   start a chunk. [`StepMode::Masked`] steps it on every row and keeps the
//!   old inner cache on the other rows. Every call runs the same kernels, so a
//!   captured step replays it. [`StepMode::Gathered`] reads the decisions to
//!   the host and steps only the rows that start a chunk (no inner work when
//!   none does).
//! - `forward` from any cache equals `step` unrolled from that cache, on the
//!   outputs, the caches and the gradients. The router and the smoothing keep
//!   their own state ([`HNetStageCache`](crate::modules::hnet::HNetStageCache)),
//!   so a sequence can split into any number of calls.
//!
//! ## Not supported
//!
//! - Class markers in a stack of an H-Net: they would move the rows that the
//!   chunk indices point at. [`HNetShape`] refuses them.
//! - Packed rows: a reset must reach the inner network at a row that the
//!   block accepts (a chunk start of a chunked SSD). Dynamic chunking can put
//!   it at any row.

/// The caches of an H-Net and the per-row cache operations of its `step`.
pub mod cache;
pub(crate) mod chunk;
/// The vocab network over an H-Net ([`HNetVocabNetwork`]).
pub mod network;
/// The routing module and the ratio loss.
pub mod routing;
/// The serializable shapes of an H-Net ([`HNetShape`], [`HNetVocabShape`]).
pub mod shape;
/// The smoothing module (the EMA over the chunks).
pub mod smooth;

#[cfg(all(test, feature = "_dev-test"))]
mod tests;

pub use cache::{HNetCaches, HNetStageCache};
pub use network::HNetVocabNetwork;
pub use routing::{Router, RouterCache, Routing};
pub use shape::{HNetShape, HNetStageShape, HNetVocabShape};

use crate::modules::{Block, CacheTensors, Layers, RmsNorm};
use burn::module::Param;
use burn::nn::Linear;
use burn::prelude::*;
use cache::{gather_rows, merge_rows, select_rows};
use smooth::{smooth, smooth_step};

/// How [`HNet::step`] runs the network inside a stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StepMode {
    /// Step the inner network on every row, then keep the old inner cache on
    /// the rows that start no chunk. Every step runs the same launches, so a
    /// captured step can replay it. No compute is saved.
    #[default]
    Masked,
    /// Read the decisions to the host, and step the inner network only on the
    /// rows that start a chunk. When no row does, the inner network does not
    /// run. This saves the inner compute, but it is not capturable.
    Gathered,
}

/// One stage of an [`HNet`]: an encoder, a router, the dechunking residual
/// and a decoder, around the inner network.
#[derive(Module, Debug)]
pub struct HNetStage<E: Module> {
    /// The vector appended to each input row when this stage is wider than
    /// its input (`None` at equal width, and at stage 0).
    pub pad_dimension: Option<Param<Tensor<1>>>,
    /// The encoder stack.
    pub encoder: Layers<E>,
    /// The final RMSNorm of the encoder.
    pub encoder_norm: RmsNorm,
    /// The routing module.
    pub router: Router,
    /// `W_r`: the residual from the encoder output to the decoder input. Zero
    /// at init.
    pub residual_proj: Linear,
    /// The decoder stack.
    pub decoder: Layers<E>,
    /// The final RMSNorm of the decoder.
    pub decoder_norm: RmsNorm,
}

/// The main network of an [`HNet`]: the innermost stack, on the most
/// compressed sequence.
#[derive(Module, Debug)]
pub struct HNetMain<M: Module> {
    /// The vector appended to each input row when this network is wider than
    /// its input. `None` at equal width, and when the H-Net has no stage.
    pub pad_dimension: Option<Param<Tensor<1>>>,
    /// The stack.
    pub layers: Layers<M>,
    /// The final RMSNorm.
    pub norm: RmsNorm,
}

/// An H-Net over `[batch, sequence, d₀]`: the stages (outermost first) and
/// the main network. See the module header.
#[derive(Module, Debug)]
pub struct HNet<E: Module, M: Module> {
    /// The stages, outermost first. Empty ⇒ the main network alone (an
    /// isotropic stack with a final norm).
    pub stages: Vec<HNetStage<E>>,
    /// The main network.
    pub main: HNetMain<M>,
    /// The inner chunk count of a `forward` is rounded up to a multiple of
    /// this. A larger value gives fewer distinct inner shapes in a run.
    #[module(skip)]
    pub chunk_multiple: usize,
    /// The block length of the smoothing scan ([`smooth()`]).
    #[module(skip)]
    pub smooth_block: usize,
}

/// `x` with the vector `pad` appended to each row (on the last axis), or `x`
/// when there is none.
fn widen<const D: usize>(x: Tensor<D>, pad: &Option<Param<Tensor<1>>>) -> Tensor<D> {
    let Some(pad) = pad else { return x };
    let pad = pad.val();
    let [extra] = pad.dims();
    let mut dims = x.dims();
    dims[D - 1] = extra;
    let mut shape = [1usize; D];
    shape[D - 1] = extra;
    Tensor::cat(vec![x, pad.reshape(shape).expand(dims)], D - 1)
}

/// The first `width` columns of `x` (on the last axis).
fn narrow_last<const D: usize>(x: Tensor<D>, width: usize) -> Tensor<D> {
    match x.dims()[D - 1] == width {
        true => x,
        false => x.narrow(D - 1, 0, width),
    }
}

/// The dechunked rows: `STE(c) · up + residual`, with `c = p` where `b`, else
/// `1 − p`. `STE(c)` is `1` in value and carries the gradient of `c`.
fn confident(up_bsd: Tensor<3>, p_bs: Tensor<2>, b_bs: Tensor<2, Bool>, residual_bsd: Tensor<3>) -> Tensor<3> {
    let c_bs = (p_bs.clone().neg() + 1.0).mask_where(b_bs, p_bs);
    let ste_bs = (c_bs.clone() - c_bs.detach()) + 1.0;
    up_bsd * ste_bs.unsqueeze_dim(2) + residual_bsd
}

impl<E: Module> HNetStage<E> {
    /// The width of this stage.
    pub fn width(&self) -> usize {
        self.encoder_norm.gamma.dims()[0]
    }
}

impl<M: Module> HNetMain<M> {
    /// The width of the main network.
    pub fn width(&self) -> usize {
        self.norm.gamma.dims()[0]
    }
}

impl<M: Block> HNetMain<M>
where
    M::Options: Clone,
{
    fn forward(
        &self,
        x_bsd: Tensor<3>,
        caches: Option<M::Caches>,
        options: M::Options,
        pad_bs: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, M::Caches) {
        let [_batch, _sequence, d_in] = x_bsd.dims();
        let x_bsd = widen(x_bsd, &self.pad_dimension);
        let (y_bsd, caches) = self.layers.forward(x_bsd, caches, options, None, pad_bs);
        (narrow_last(self.norm.forward(y_bsd), d_in), caches)
    }

    fn step(&self, x_bd: Tensor<2>, caches: M::Caches) -> (Tensor<2>, M::Caches) {
        let [_batch, d_in] = x_bd.dims();
        let x_bd = widen(x_bd, &self.pad_dimension);
        let (y_bd, caches) = self.layers.step(x_bd, Some(caches), None);
        (narrow_last(self.norm.forward(y_bd), d_in), caches)
    }
}

impl<E: Module, M: Module> HNet<E, M> {
    /// The width of the input (and of the output): that of stage 0, or that
    /// of the main network when there is no stage.
    pub fn width(&self) -> usize {
        match self.stages.first() {
            Some(stage) => stage.width(),
            None => self.main.width(),
        }
    }
}

impl<E: Block, M: Block> HNet<E, M>
where
    E::Options: Clone,
    M::Options: Clone,
{
    /// The full-sequence pass: `[batch, sequence, d₀]` → the same shape.
    ///
    /// `options` goes to every encoder and decoder (`.0`) and to the main
    /// network (`.1`). `pad` (`[batch, sequence]`, `true` at padding, `None` ⇒
    /// none) is right padding. The outputs of the real rows and the caches of
    /// each slot are those of that slot run alone. The output of a padded row
    /// is unspecified.
    ///
    /// Returns the output, the caches, and one [`Routing`] per stage
    /// (outermost first). A stage that no row reached (every slot of its
    /// outer stage started no chunk in this call) reports all of its rows
    /// absent.
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<HNetCaches<E, M>>,
        options: (E::Options, M::Options),
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, HNetCaches<E, M>, Vec<Routing>) {
        if let Some(caches) = &caches {
            assert_eq!(caches.stages.len(), self.stages.len(), "one cache per stage");
        }
        let mut routing = Vec::with_capacity(self.stages.len());
        let (y, caches) = self.forward_from(0, x, caches, &options, pad, &mut routing);
        (y, caches, routing)
    }

    /// The single-row step: `[batch, d₀]` → the same shape. `mode` selects how
    /// the inner network of each stage runs ([`StepMode`]). Returns the
    /// output, the caches, and one [`Routing`] per stage (each `[batch, 1]`).
    pub fn step(
        &self,
        x: Tensor<2>,
        caches: Option<HNetCaches<E, M>>,
        mode: StepMode,
    ) -> (Tensor<2>, HNetCaches<E, M>, Vec<Routing>)
    where
        E::Caches: CacheTensors,
        M::Caches: CacheTensors,
    {
        let [batch, _d] = x.dims();
        let caches = caches.unwrap_or_else(|| self.zero_caches_2d(0, batch, &x.device()));
        assert_eq!(caches.stages.len(), self.stages.len(), "one cache per stage");
        let mut routing = Vec::with_capacity(self.stages.len());
        let (y, caches) = self.step_from(0, x, caches, mode, &mut routing);
        (y, caches, routing)
    }

    /// The zero caches of the stages from `s` inward (and of the main
    /// network), sized for single-row steps.
    pub fn zero_caches_2d(&self, s: usize, batch: usize, device: &Device) -> HNetCaches<E, M> {
        let stages = self.stages[s..]
            .iter()
            .map(|stage| {
                let d = stage.width();
                let x_bd = Tensor::<2>::zeros([batch, d], device);
                HNetStageCache {
                    encoder: stage.encoder.zero_caches_2d(&x_bd),
                    router: RouterCache::zeros(batch, d, device),
                    smooth_bd: x_bd.clone(),
                    decoder: stage.decoder.zero_caches_2d(&x_bd),
                }
            })
            .collect();
        let x_bd = Tensor::<2>::zeros([batch, self.main.width()], device);
        HNetCaches { stages, main: self.main.layers.zero_caches_2d(&x_bd) }
    }

    /// The zero caches of the stages from `s` inward (and of the main
    /// network), sized for a full-sequence pass.
    pub fn zero_caches_3d(&self, s: usize, batch: usize, device: &Device) -> HNetCaches<E, M> {
        let stages = self.stages[s..]
            .iter()
            .map(|stage| {
                let d = stage.width();
                let x_b1d = Tensor::<3>::zeros([batch, 1, d], device);
                HNetStageCache {
                    encoder: stage.encoder.zero_caches_3d(&x_b1d),
                    router: RouterCache::zeros(batch, d, device),
                    smooth_bd: Tensor::zeros([batch, d], device),
                    decoder: stage.decoder.zero_caches_3d(&x_b1d),
                }
            })
            .collect();
        let x_b1d = Tensor::<3>::zeros([batch, 1, self.main.width()], device);
        HNetCaches { stages, main: self.main.layers.zero_caches_3d(&x_b1d) }
    }

    /// [`Self::forward`] from stage `s` inward. `caches` holds the stages
    /// from `s` inward (its stage `0` is stage `s`).
    fn forward_from(
        &self,
        s: usize,
        x_bsd: Tensor<3>,
        caches: Option<HNetCaches<E, M>>,
        options: &(E::Options, M::Options),
        pad_bs: Option<Tensor<2, Bool>>,
        routing: &mut Vec<Routing>,
    ) -> (Tensor<3>, HNetCaches<E, M>) {
        let Some(stage) = self.stages.get(s) else {
            let main = caches.map(|c| c.main);
            let (y_bsd, main) = self.main.forward(x_bsd, main, options.1.clone(), pad_bs);
            return (y_bsd, HNetCaches { stages: Vec::new(), main });
        };
        let [batch, _sequence, d_in] = x_bsd.dims();
        let device = x_bsd.device();
        let d = stage.width();
        let (head, rest) = match caches {
            Some(mut c) => {
                let head = c.stages.remove(0);
                (Some(head), Some(c))
            }
            None => (None, None),
        };
        let (enc, router, carry_bd, dec) = match head {
            Some(h) => (Some(h.encoder), h.router, h.smooth_bd, Some(h.decoder)),
            None => (None, RouterCache::zeros(batch, d, &device), Tensor::zeros([batch, d], &device), None),
        };

        // Encode.
        let x_bsd = widen(x_bsd, &stage.pad_dimension);
        let (xh_bsd, enc) = stage.encoder.forward(x_bsd, enc, options.0.clone(), None, pad_bs.clone());
        let xh_bsd = stage.encoder_norm.forward(xh_bsd);
        let residual_bsd = stage.residual_proj.forward(xh_bsd.clone());

        // Route. No absent row starts a chunk.
        let p_bs = stage.router.probs(xh_bsd.clone(), &router);
        let b_bs = p_bs.clone().greater_elem(0.5);
        let b_bs = match &pad_bs {
            Some(pad_bs) => b_bs.bool_and(pad_bs.clone().bool_not()),
            None => b_bs,
        };
        routing.push(Routing { prob_bs: p_bs.clone(), boundary_bs: b_bs.clone(), pad_bs: pad_bs.clone() });
        let router = router.after_rows(xh_bsd.clone(), pad_bs.as_ref());

        // Chunk, run the inner network, and smooth. `z_all` is `[carry, z̄…]`.
        let plan = chunk::plan(b_bs.clone(), self.chunk_multiple);
        let (z_all_bkd, rest) = if plan.len == 0 {
            // No row starts a chunk, so the inner network does not run.
            let rest = rest.unwrap_or_else(|| self.zero_caches_3d(s + 1, batch, &device));
            routing.extend((s + 1..self.stages.len()).map(|_| Routing::absent(batch, &device)));
            (carry_bd.unsqueeze_dim(1), rest)
        } else {
            let x_inner_bkd = chunk::gather_rows(xh_bsd, plan.src_bk.clone());
            let (z_inner_bkd, rest) =
                self.forward_from(s + 1, x_inner_bkd, rest, options, plan.pad_bk.clone(), routing);
            let p_bk = p_bs.clone().gather(1, plan.src_bk.clone());
            let zbar_bkd = smooth(z_inner_bkd, p_bk, plan.pad_bk.clone(), carry_bd.clone(), self.smooth_block);
            (Tensor::cat(vec![carry_bd.unsqueeze_dim(1), zbar_bkd], 1), rest)
        };
        // The inner padding holds the state. So the last column is the last
        // real chunk of each slot, or the carry for a slot with none.
        let smooth_bd = z_all_bkd.clone().narrow(1, plan.len, 1).reshape([batch, d]);
        let up_bsd = chunk::upsample(z_all_bkd, plan.upto_bs);

        // Dechunk, then decode.
        let z_bsd = confident(up_bsd, p_bs, b_bs, residual_bsd);
        let (y_bsd, dec) = stage.decoder.forward(z_bsd, dec, options.0.clone(), None, pad_bs);
        let y_bsd = narrow_last(stage.decoder_norm.forward(y_bsd), d_in);

        let mut caches = rest;
        caches.stages.insert(0, HNetStageCache { encoder: enc, router, smooth_bd, decoder: dec });
        (y_bsd, caches)
    }

    /// [`Self::step`] from stage `s` inward. `caches` holds the stages from
    /// `s` inward.
    fn step_from(
        &self,
        s: usize,
        x_bd: Tensor<2>,
        caches: HNetCaches<E, M>,
        mode: StepMode,
        routing: &mut Vec<Routing>,
    ) -> (Tensor<2>, HNetCaches<E, M>)
    where
        E::Caches: CacheTensors,
        M::Caches: CacheTensors,
    {
        let Some(stage) = self.stages.get(s) else {
            let (y_bd, main) = self.main.step(x_bd, caches.main);
            return (y_bd, HNetCaches { stages: Vec::new(), main });
        };
        let [batch, d_in] = x_bd.dims();
        let device = x_bd.device();
        let mut rest = caches;
        let head = rest.stages.remove(0);

        // Encode.
        let x_bd = widen(x_bd, &stage.pad_dimension);
        let (xh_bd, enc) = stage.encoder.step(x_bd, Some(head.encoder), None);
        let xh_bd = stage.encoder_norm.forward(xh_bd);
        let residual_bd = stage.residual_proj.forward(xh_bd.clone());

        // Route.
        let p_b = stage.router.step_prob(xh_bd.clone(), &head.router);
        let b_b = p_b.clone().greater_elem(0.5);
        routing.push(Routing {
            prob_bs: p_b.clone().reshape([batch, 1]),
            boundary_bs: b_b.clone().reshape([batch, 1]),
            pad_bs: None,
        });
        let router = RouterCache { last_bd: xh_bd.clone(), seen_b: head.router.seen_b.ones_like() };
        let carry_bd = head.smooth_bd;

        // The inner network, on the rows that start a chunk.
        let first = routing.len();
        let (z_bd, mut rest) = match mode {
            StepMode::Masked => {
                let (z_bd, new) = self.step_from(s + 1, xh_bd, rest.clone(), mode, routing);
                (z_bd, select_rows(rest, new, &b_b))
            }
            StepMode::Gathered => {
                let take: Vec<bool> = b_b.clone().int().into_data().iter::<i64>().map(|v| v != 0).collect();
                let idx: Vec<i64> = (0..batch as i64).filter(|&i| take[i as usize]).collect();
                if idx.is_empty() {
                    routing.extend((s + 1..self.stages.len()).map(|_| Routing::absent(batch, &device)));
                    (carry_bd.clone(), rest)
                } else {
                    let r = idx.len();
                    let idx_r = Tensor::<1, Int>::from_data(TensorData::new(idx, [r]), &device);
                    // Row i of the full batch is row i of `[full; sub]`, or the
                    // row of `sub` that holds it.
                    let mut next = batch as i64;
                    let map: Vec<i64> = (0..batch)
                        .map(|i| match take[i] {
                            true => {
                                next += 1;
                                next - 1
                            }
                            false => i as i64,
                        })
                        .collect();
                    let map_b = Tensor::<1, Int>::from_data(TensorData::new(map, [batch]), &device);
                    let x_sub_rd = xh_bd.select(0, idx_r.clone());
                    let rest_sub = gather_rows(rest.clone(), &idx_r);
                    let mut sub_routing = Vec::new();
                    let (z_sub_rd, rest_sub) = self.step_from(s + 1, x_sub_rd, rest_sub, mode, &mut sub_routing);
                    routing.extend(sub_routing.into_iter().map(|r| r.merged(&map_b)));
                    let z_bd = Tensor::cat(vec![carry_bd.clone(), z_sub_rd], 0).select(0, map_b.clone());
                    (z_bd, merge_rows(rest, rest_sub, &map_b))
                }
            }
        };
        // A row that started no chunk did not reach the stages inward.
        for r in &mut routing[first..] {
            *r = r.clone().reached(&b_b);
        }

        // Smooth, dechunk, decode.
        let zbar_bd = smooth_step(z_bd, p_b.clone(), b_b.clone(), carry_bd);
        let z_bd = confident(
            zbar_bd.clone().unsqueeze_dim(1),
            p_b.reshape([batch, 1]),
            b_b.reshape([batch, 1]),
            residual_bd.unsqueeze_dim(1),
        )
        .squeeze_dim(1);
        let (y_bd, dec) = stage.decoder.step(z_bd, Some(head.decoder), None);
        let y_bd = narrow_last(stage.decoder_norm.forward(y_bd), d_in);

        rest.stages.insert(0, HNetStageCache { encoder: enc, router, smooth_bd: zbar_bd, decoder: dec });
        (y_bd, rest)
    }
}

impl Routing {
    /// This routing of a gathered sub-batch, written back into the full
    /// batch: row `i` reads row `map_b[i]` of `[absent rows; sub-batch]`.
    fn merged(self, map_b: &Tensor<1, Int>) -> Self {
        let [batch] = map_b.dims();
        let [rows, sequence] = self.prob_bs.dims();
        let device = self.prob_bs.device();
        let merge_f = |sub: Tensor<2>, fill: f32| {
            Tensor::cat(vec![Tensor::full([batch, sequence], fill, &device), sub], 0).select(0, map_b.clone())
        };
        let merge_b = |sub: Tensor<2, Bool>, fill: f32| merge_f(sub.float(), fill).greater_elem(0.5);
        let pad_sub = self.pad_bs.unwrap_or_else(|| Tensor::<2>::zeros([rows, sequence], &device).greater_elem(0.5));
        Self {
            prob_bs: merge_f(self.prob_bs, 0.0),
            boundary_bs: merge_b(self.boundary_bs, 0.0),
            pad_bs: Some(merge_b(pad_sub, 1.0)),
        }
    }

    /// This routing with every row where `reached_b` is `false` marked absent
    /// (and starting no chunk).
    fn reached(self, reached_b: &Tensor<1, Bool>) -> Self {
        let [batch, sequence] = self.prob_bs.dims();
        let reached_bs = reached_b.clone().reshape([batch, 1]).expand([batch, sequence]);
        let absent_bs = reached_bs.clone().bool_not();
        Self {
            prob_bs: self.prob_bs,
            boundary_bs: self.boundary_bs.bool_and(reached_bs),
            pad_bs: Some(match self.pad_bs {
                Some(pad_bs) => pad_bs.bool_or(absent_bs),
                None => absent_bs,
            }),
        }
    }
}
