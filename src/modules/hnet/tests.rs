//! H-Net over [`RefBlock`]. The tests check the parts against plain loops.
//! They check the whole network against the contract of the containers:
//! forward = step unrolled, chunked calls and right padding. The checks
//! compare the outputs, the caches and the gradients.

use super::chunk::plan;
use super::smooth::{P_MAX, P_MIN, smooth, smooth_step};
use super::*;
use crate::modules::{CacheTensors, NetworkShape, TensorZip};
use crate::reference::{RefBlock, RefBlockConfig};
use crate::utils::test_helpers::{dtype_tol, max_rel_diff, test_device};
use burn::tensor::Distribution;

type Net = HNet<RefBlock, RefBlock>;

const TOL: f32 = 1e-4;

/// Stage widths (outermost first) and the main width. Two layers per stack.
fn shape(n_stages: usize) -> HNetShape {
    let stack = || NetworkShape::new(2);
    HNetShape::new(
        (0..n_stages).map(|_| HNetStageShape::new(stack(), stack())).collect(),
        stack(),
    )
    .with_smooth_block(4)
}

/// An H-Net whose stages have the widths `widths` and whose main network has
/// the width `main`. The residual of each stage is redrawn, so that the
/// tests also cover it (it is zero at init).
fn net(widths: &[usize], main: usize, device: &Device) -> Net {
    let blocks: Vec<_> = widths.iter().map(|&d| RefBlockConfig::new(d)).collect();
    let mut net = shape(widths.len()).init(&blocks, &RefBlockConfig::new(main), device);
    for stage in &mut net.stages {
        stage.residual_proj.weight = stage.residual_proj.weight.clone().map(|w| {
            Tensor::random(w.shape(), Distribution::Normal(0.0, 0.3), &w.device())
                .set_require_grad(w.is_require_grad())
        });
    }
    net
}

fn randn<const D: usize>(dims: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::random(dims, Distribution::Normal(0.0, 1.0), device)
}

/// The largest relative difference over every tensor pair of two caches.
fn cache_diff<C: CacheTensors>(a: C, b: C) -> f32 {
    struct MaxDiff(f32);
    impl TensorZip for MaxDiff {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            self.0 = self.0.max(max_rel_diff(a.clone(), b));
            a
        }
    }
    let mut z = MaxDiff(0.0);
    a.zip_tensors(b, &mut z);
    z.0
}

fn bools<const D: usize>(t: Tensor<D, Bool>) -> Vec<bool> {
    t.int().into_data().iter::<i64>().map(|v| v != 0).collect()
}

/// `step` unrolled over the rows of `x` (`[batch, sequence, d]`).
fn step_unrolled(
    net: &Net,
    x: Tensor<3>,
    caches: Option<HNetCaches<RefBlock, RefBlock>>,
    mode: StepMode,
) -> (Tensor<3>, HNetCaches<RefBlock, RefBlock>, Vec<Tensor<2, Bool>>) {
    let [_batch, sequence, _d] = x.dims();
    let mut caches = caches;
    let mut ys = Vec::new();
    let mut boundaries = Vec::new();
    for t in 0..sequence {
        let (y, c, routing) = net.step(x.clone().narrow(1, t, 1).squeeze_dim(1), caches, mode);
        caches = Some(c);
        ys.push(y.unsqueeze_dim(1));
        boundaries.push(routing[0].boundary_bs.clone());
    }
    (Tensor::cat(ys, 1), caches.unwrap(), boundaries)
}

// ---------------------------------------------------------------------------
// The parts
// ---------------------------------------------------------------------------

/// The chunked scan of the smoothing module equals the recurrence, over
/// several blocks, from a carry, with absent rows that hold the state. The
/// step form equals it too.
#[test]
fn smooth_equals_the_sequential_ema() {
    let device = test_device();
    let [batch, k, d] = [2, 23, 3];
    let x = randn([batch, k, d], &device);
    let p = Tensor::<2>::random([batch, k], Distribution::Uniform(0.0, 1.0), &device);
    let carry = randn([batch, d], &device);
    // Slot 1 has 17 real chunks.
    let pad = Tensor::<1, Int>::arange(0..k as i64, &device)
        .reshape([1, k])
        .expand([batch, k])
        .greater_equal(Tensor::<2, Int>::from_data([[k as i64], [17]], &device).expand([batch, k]));

    let got = smooth(x.clone(), p.clone(), Some(pad.clone()), carry.clone(), 5);

    let mut z = carry.clone();
    let mut z_step = carry;
    let mut want = Vec::new();
    for i in 0..k {
        let x_i = x.clone().narrow(1, i, 1).squeeze_dim::<2>(1);
        let p_i = p.clone().narrow(1, i, 1).squeeze_dim::<1>(1);
        let real_i = pad.clone().narrow(1, i, 1).squeeze_dim::<1>(1).bool_not();
        let w = p_i.clone().clamp(P_MIN, P_MAX).mask_fill(real_i.clone().bool_not(), 0.0).unsqueeze_dim(1);
        z = w.clone() * x_i.clone() + (w.neg() + 1.0) * z;
        z_step = smooth_step(x_i, p_i, real_i, z_step);
        want.push(z.clone().unsqueeze_dim(1));
    }
    let want = Tensor::cat(want, 1);
    assert!(max_rel_diff(got.clone(), want) < dtype_tol(TOL));
    let last = got.narrow(1, k - 1, 1).squeeze_dim::<2>(1);
    assert!(max_rel_diff(last, z_step) < dtype_tol(TOL), "the absent rows hold the last chunk");
}

/// The plan of a hand-made mask: the boundary rows, packed left, the count
/// before each row, and the inner padding (rounded up to the multiple).
#[test]
fn chunk_plan_points_at_the_boundary_rows() {
    let device = test_device();
    let b = Tensor::<2, Int>::from_data([[1, 0, 1, 1, 0], [1, 0, 0, 0, 0]], &device).equal_elem(1);
    let p = plan(b.clone(), 1);
    assert_eq!(p.len, 3);
    let src: Vec<i64> = p.src_bk.into_data().iter::<i64>().collect();
    assert_eq!(src, vec![0, 2, 3, 0, 0, 0]);
    let upto: Vec<i64> = p.upto_bs.into_data().iter::<i64>().collect();
    assert_eq!(upto, vec![1, 1, 2, 3, 3, 1, 1, 1, 1, 1]);
    assert_eq!(bools(p.pad_bk.unwrap()), vec![false, false, false, false, true, true]);

    let p4 = plan(b, 4);
    assert_eq!(p4.len, 4, "the count rounds up to the multiple");
    assert_eq!(bools(p4.pad_bk.unwrap()), vec![false, false, false, true, false, true, true, true]);
}

/// The ratio loss is 1 where `F = G = 1/N`, and the mean is over the real
/// rows only.
#[test]
fn ratio_loss_is_one_at_the_target() {
    let device = test_device();
    let n = 4.0;
    // 8 real rows, 2 boundaries (F = 1/4), p = 1/4 everywhere (G = 1/4), and
    // 4 padded rows with p = 1 that must not count.
    let b = Tensor::<2, Int>::from_data([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], &device).equal_elem(1);
    let pad = Tensor::<2, Int>::from_data([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]], &device).equal_elem(1);
    let p = Tensor::<2>::full([1, 12], 0.25, &device).mask_fill(pad.clone(), 1.0);
    let routing = Routing { prob_bs: p, boundary_bs: b, pad_bs: Some(pad) };
    let loss = routing.ratio_loss(n).into_scalar::<f32>();
    assert!((loss - 1.0).abs() < dtype_tol(1e-5), "loss = {loss}");
}

/// The router starts a chunk at the first row of a new sequence, and only
/// there. A call that continues a cache has no forced boundary.
#[test]
fn only_a_new_sequence_forces_its_first_boundary() {
    let device = test_device();
    let mut net = net(&[8], 8, &device);
    // W_q = 0 gives cos = 0, so p = ½ and no row starts a chunk on its own.
    let q = &mut net.stages[0].router.q_proj;
    q.weight = q.weight.clone().map(|w| w.zeros_like());
    let x = randn([2, 6, 8], &device);

    let (_, caches, routing) = net.forward(x.clone(), None, ((), ()), None);
    let first_only = [true, false, false, false, false, false];
    assert_eq!(bools(routing[0].boundary_bs.clone()), first_only.repeat(2));

    let (_, _, routing) = net.forward(x, Some(caches), ((), ()), None);
    assert!(bools(routing[0].boundary_bs.clone()).iter().all(|b| !b));
}

// ---------------------------------------------------------------------------
// The whole network: the contract of the containers
// ---------------------------------------------------------------------------

/// `forward` equals `step` unrolled on the outputs, the caches and the
/// decisions. The cases are one and two stages (with a wider main network),
/// in both step modes.
#[test]
fn forward_equals_step_unrolled() {
    let device = test_device();
    for (widths, main) in [(vec![8], 12), (vec![8, 8], 12), (vec![6, 10], 10)] {
        let net = net(&widths, main, &device);
        let x = randn([3, 12, widths[0]], &device);
        let (y_fwd, c_fwd, routing) = net.forward(x.clone(), None, ((), ()), None);
        for mode in [StepMode::Masked, StepMode::Gathered] {
            let (y_step, c_step, boundaries) = step_unrolled(&net, x.clone(), None, mode);
            let tag = format!("{widths:?} → {main}, {mode:?}");
            assert_eq!(
                bools(routing[0].boundary_bs.clone()),
                bools(Tensor::cat(boundaries, 1)),
                "{tag}: the decisions differ"
            );
            assert!(max_rel_diff(y_fwd.clone(), y_step) < dtype_tol(TOL), "{tag}: outputs");
            assert!(cache_diff(c_fwd.clone(), c_step) < dtype_tol(TOL), "{tag}: caches");
        }
    }
}

/// A `forward` split into two calls (the cache in between) equals one
/// `forward`. The second call can open with rows before its first boundary,
/// which read the last chunk of the first call.
#[test]
fn forward_is_chunkable_through_the_cache() {
    let device = test_device();
    let net = net(&[8, 8], 12, &device);
    let x = randn([2, 13, 8], &device);
    let (y_all, c_all, _) = net.forward(x.clone(), None, ((), ()), None);
    let (y_a, c, _) = net.forward(x.clone().narrow(1, 0, 6), None, ((), ()), None);
    let (y_b, c, _) = net.forward(x.narrow(1, 6, 7), Some(c), ((), ()), None);
    assert!(max_rel_diff(y_all, Tensor::cat(vec![y_a, y_b], 1)) < dtype_tol(TOL));
    assert!(cache_diff(c_all, c) < dtype_tol(TOL));
}

/// In a right-padded batch, each slot gives what it gives alone: the outputs
/// of its real rows, and its caches. A slot with no real row keeps its
/// incoming cache.
#[test]
fn padded_slots_match_each_slot_alone() {
    let device = test_device();
    let net = net(&[8, 8], 12, &device);
    let lens = [11usize, 4, 0];
    let x = randn([3, 11, 8], &device);
    let pad = Tensor::<1, Int>::arange(0..11, &device)
        .reshape([1, 11])
        .expand([3, 11])
        .greater_equal(Tensor::<2, Int>::from_data([[11], [4], [0]], &device).expand([3, 11]));
    // A warm cache, so that the empty slot has something to keep.
    let (_, warm, _) = net.forward(randn([3, 5, 8], &device), None, ((), ()), None);
    let (y, caches, _) = net.forward(x.clone(), Some(warm.clone()), ((), ()), Some(pad));

    let row = |c: HNetCaches<RefBlock, RefBlock>, b: usize| {
        let idx = Tensor::<1, Int>::from_data([b as i64], &device);
        cache::gather_rows(c, &idx)
    };
    for (b, &len) in lens.iter().enumerate() {
        let warm_b = row(warm.clone(), b);
        if len == 0 {
            assert!(cache_diff(row(caches.clone(), b), warm_b) < dtype_tol(TOL), "slot {b}: the cache moved");
            continue;
        }
        let x_b = x.clone().narrow(0, b, 1).narrow(1, 0, len);
        let (y_b, c_b, _) = net.forward(x_b, Some(warm_b), ((), ()), None);
        let y_real = y.clone().narrow(0, b, 1).narrow(1, 0, len);
        assert!(max_rel_diff(y_real, y_b) < dtype_tol(TOL), "slot {b}: outputs");
        assert!(cache_diff(row(caches.clone(), b), c_b) < dtype_tol(TOL), "slot {b}: caches");
    }
}

/// A call in which no row starts a chunk skips the inner network. `forward`
/// and both step modes still agree (the gathered step takes its empty path).
#[test]
fn a_call_without_boundaries_skips_the_inner_network() {
    let device = test_device();
    let mut net = net(&[8, 8], 12, &device);
    let q = &mut net.stages[1].router.q_proj;
    q.weight = q.weight.clone().map(|w| w.zeros_like());
    let q = &mut net.stages[0].router.q_proj;
    q.weight = q.weight.clone().map(|w| w.zeros_like());
    let x = randn([2, 9, 8], &device);
    let (_, warm, _) = net.forward(x.clone().narrow(1, 0, 3), None, ((), ()), None);
    let (y_fwd, c_fwd, routing) = net.forward(x.clone().narrow(1, 3, 6), Some(warm.clone()), ((), ()), None);
    assert!(bools(routing[0].boundary_bs.clone()).iter().all(|b| !b));
    for mode in [StepMode::Masked, StepMode::Gathered] {
        let (y_step, c_step, _) = step_unrolled(&net, x.clone().narrow(1, 3, 6), Some(warm.clone()), mode);
        assert!(max_rel_diff(y_fwd.clone(), y_step) < dtype_tol(TOL), "{mode:?}: outputs");
        assert!(cache_diff(c_fwd.clone(), c_step) < dtype_tol(TOL), "{mode:?}: caches");
    }
}

/// The gradients of `forward` equal those of `step` unrolled: at the router,
/// the encoder, the residual, the main network, the width vector and the
/// input.
#[test]
fn forward_and_step_have_the_same_gradients() {
    let device = test_device().autodiff();
    let net = net(&[6, 10], 10, &device);
    let x = randn([2, 10, 6], &device).require_grad();
    let w = randn([2, 10, 6], &device);

    let grads_of = |y: Tensor<3>| (y * w.clone()).sum().backward();
    let (y_fwd, _, _) = net.forward(x.clone(), None, ((), ()), None);
    let g_fwd = grads_of(y_fwd);
    let (y_step, _, _) = step_unrolled(&net, x.clone(), None, StepMode::Masked);
    let g_step = grads_of(y_step);

    let s0 = &net.stages[0];
    let s1 = &net.stages[1];
    let pairs: Vec<(&str, Tensor<2>, Tensor<2>)> = vec![
        ("router q", s0.router.q_proj.weight.val().grad(&g_fwd).unwrap(), s0.router.q_proj.weight.val().grad(&g_step).unwrap()),
        ("router k", s1.router.k_proj.weight.val().grad(&g_fwd).unwrap(), s1.router.k_proj.weight.val().grad(&g_step).unwrap()),
        ("encoder", s0.encoder.real_layers[0].block.in_proj.weight.val().grad(&g_fwd).unwrap(), s0.encoder.real_layers[0].block.in_proj.weight.val().grad(&g_step).unwrap()),
        ("residual", s1.residual_proj.weight.val().grad(&g_fwd).unwrap(), s1.residual_proj.weight.val().grad(&g_step).unwrap()),
        ("main", net.main.layers.real_layers[1].block.out_proj.weight.val().grad(&g_fwd).unwrap(), net.main.layers.real_layers[1].block.out_proj.weight.val().grad(&g_step).unwrap()),
    ];
    for (name, a, b) in pairs {
        let diff = max_rel_diff(a, b);
        assert!(diff < dtype_tol(1e-3), "grad of {name}: max rel diff {diff}");
    }
    let pad_dim = s1.pad_dimension.as_ref().expect("stage 1 is wider than stage 0").val();
    assert!(max_rel_diff(pad_dim.grad(&g_fwd).unwrap(), pad_dim.grad(&g_step).unwrap()) < dtype_tol(1e-3));
    assert!(max_rel_diff(x.grad(&g_fwd).unwrap(), x.grad(&g_step).unwrap()) < dtype_tol(1e-3));
}

/// The vocab network: `forward` equals `step` unrolled on the logits.
#[test]
fn vocab_network_forward_equals_step_unrolled() {
    let device = test_device();
    let blocks = [RefBlockConfig::new(8)];
    let net = HNetVocabShape::new(shape(1), 11)
        .with_missing_lm_head(true)
        .init(&blocks, &RefBlockConfig::new(12), &device);
    let ids = Tensor::<2, Int>::random([2, 9], Distribution::Uniform(0.0, 11.0), &device);
    let flags = Tensor::<2, Int>::random([2, 9], Distribution::Uniform(0.0, 2.0), &device);
    let x = Tensor::stack::<3>(vec![ids, flags], 2);
    let (logits, _, _) = net.forward(x.clone(), None, ((), ()), None);
    assert_eq!(logits.dims(), [2, 9, 12]);
    let mut caches = None;
    let mut steps = Vec::new();
    for t in 0..9 {
        let (l, c, _) = net.step(x.clone().narrow(1, t, 1).squeeze_dim(1), caches, StepMode::Masked);
        caches = Some(c);
        steps.push(l.unsqueeze_dim(1));
    }
    assert!(max_rel_diff(logits, Tensor::cat(steps, 1)) < dtype_tol(TOL));
}

/// The start rows: `forward_opened` equals `prime` then `step` unrolled. The
/// output of the last start row is the logits that `prime` returns.
#[test]
fn start_rows_forward_equals_prime_then_steps() {
    let device = test_device();
    let blocks = [RefBlockConfig::new(8)];
    let net = HNetVocabShape::new(shape(1), 11)
        .with_n_start_tokens(3)
        .init(&blocks, &RefBlockConfig::new(8), &device);
    assert_eq!(net.n_start(), 3);
    let ids = Tensor::<2, Int>::random([2, 7], Distribution::Uniform(0.0, 11.0), &device);
    let flags = Tensor::<2, Int>::zeros([2, 7], &device);
    let x = Tensor::stack::<3>(vec![ids, flags], 2);
    let (logits, c_fwd, _) = net.forward_opened(x.clone(), ((), ()), None);
    assert_eq!(logits.dims(), [2, 10, 12]);

    let (first, caches) = net.prime(2, StepMode::Gathered).expect("the network has start rows");
    let mut caches = Some(caches);
    let mut steps = vec![first.unsqueeze_dim(1)];
    for t in 0..7 {
        let (l, c, _) = net.step(x.clone().narrow(1, t, 1).squeeze_dim(1), caches, StepMode::Gathered);
        caches = Some(c);
        steps.push(l.unsqueeze_dim(1));
    }
    assert!(max_rel_diff(logits.narrow(1, 2, 8), Tensor::cat(steps, 1)) < dtype_tol(TOL));
    assert!(cache_diff(c_fwd, caches.unwrap()) < dtype_tol(TOL));
}

/// The Muon plan scopes the specs of each stack. The `in_proj` of stage 0
/// (width 6) and that of the main network (width 10) each match only their
/// own stack.
#[cfg(feature = "optim")]
#[test]
fn muon_plan_scopes_each_stack() {
    let device = test_device();
    let blocks = [RefBlockConfig::new(6)];
    let main = RefBlockConfig::new(10);
    let shape = shape(1);
    let net = shape.init(&blocks, &main, &device);
    let plan = shape.muon_plan(&blocks, &main);
    let report = plan.describe(&net);
    let line = |path: &str| {
        report
            .lines()
            .find(|l| l.contains(path))
            .unwrap_or_else(|| panic!("no line for {path} in\n{report}"))
            .to_string()
    };
    assert!(line("stages.0.encoder.real_layers.0.block.in_proj.weight").ends_with("muon[all:6]"));
    assert!(line("stages.0.decoder.real_layers.1.block.out_proj.weight").ends_with("muon[all:6]"));
    assert!(line("main.layers.real_layers.0.block.in_proj.weight").ends_with("muon[all:10]"));
    assert!(line("stages.0.router.q_proj.weight").ends_with("fallback"));
    assert!(line("stages.0.residual_proj.weight").ends_with("fallback"));
}
