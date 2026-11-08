//! A padded batch is each of its rows run alone. A row owns its user tokens
//! and the class markers placed against its own length. Each of these
//! positions comes out as the forward of that row alone gives it, with the
//! same caches and gradients. The tests compose [`RefBlock`], so they pin the
//! half of the contract that belongs to the containers.

use crate::modules::bidi::{BidiLayersBuilder, OutputMergeConfig};
use crate::modules::{LatentNetworkBuilder, Layers, LayersBuilder, ResidualsConfig};
use crate::reference::{RefBlock, RefBlockConfig, RefCaches};
use crate::utils::class::{ClassMarker, class_window_plan};
use crate::utils::test_helpers::{dtype_tol, max_rel_diff};
use crate::utils::{ClassCursor, ClassCursors, ClassLatent, ClassToken, GradHorizon};
use burn::module::{ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::{Distribution, Gradients};
use crate::utils::test_helpers::test_device;

const D_MODEL: usize = 8;
const TOL: f32 = 1e-4;
/// The gradient tolerance for f16 and bf16 gradients. The padded batch and the
/// rows alone add in different orders, and each rounds on its own. In f16,
/// the multi-gate layers differ by `3·10⁻³` to `2.2·10⁻²` (30 runs), above
/// `dtype_tol(TOL)` (`1.8·10⁻²`). In f64, they agree to `6.5·10⁻¹⁶`. A padding
/// error gives a difference of order 1, so this tolerance still finds it.
const HALF_GRAD_TOL: f32 = 5e-2;

/// One position of the output of a container: user token `t`, or marker `i`
/// of the `level`-th splice. The splices are the class tokens of the network,
/// then the class latents of the stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Position {
    User(usize),
    Marker(usize, usize),
}

fn users(len: usize) -> Vec<Position> {
    (0..len).map(Position::User).collect()
}

/// Splice the markers of one marker level into `positions`, as a
/// whole-sequence call places them.
fn place<M: ClassMarker>(positions: Vec<Position>, markers: &[M], level: usize) -> Vec<Position> {
    let mut cursor = ClassCursor::whole(positions.len());
    let plan = class_window_plan(markers, positions.len(), &mut cursor, "test");
    let mut out = Vec::with_capacity(positions.len() + plan.len());
    let mut taken = 0;
    for (at, i) in plan {
        out.extend_from_slice(&positions[taken..at]);
        taken = at;
        out.push(Position::Marker(level, i));
    }
    out.extend_from_slice(&positions[taken..]);
    out
}

/// `[batch, padded]`, `true` past the length of each row.
fn pad_mask(lens: &[usize], padded: usize, device: &Device) -> Tensor<2, Bool> {
    let batch = lens.len();
    let lens: Vec<i32> = lens.iter().map(|&l| l as i32).collect();
    let lens_bs = Tensor::<1, Int>::from_ints(lens.as_slice(), device)
        .reshape([batch, 1])
        .expand([batch, padded]);
    Tensor::<1, Int>::arange(0..padded as i64, device)
        .reshape([1, padded])
        .expand([batch, padded])
        .greater_equal(lens_bs)
}

/// Where each of `owned` sits in `positions`.
fn indices(positions: &[Position], owned: &[Position], device: &Device) -> Tensor<1, Int> {
    let idx: Vec<i32> = owned
        .iter()
        .map(|r| {
            positions.iter()
                .position(|p| p == r)
                .unwrap_or_else(|| panic!("{r:?}, owned by a row, is missing from the padded batch"))
                as i32
        })
        .collect();
    Tensor::<1, Int>::from_ints(idx.as_slice(), device)
}

/// Every float parameter's gradient, in visiting order.
struct Collect<'a> {
    grads: &'a Gradients,
    out: Vec<Option<Tensor<1>>>,
}

impl ModuleVisitor for Collect<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let n = param.val().shape().num_elements();
        self.out
            .push(param.val().grad(self.grads).map(|g| g.reshape([n])));
    }
}

fn param_grads<M: Module>(module: &M, grads: &Gradients) -> Vec<Option<Tensor<1>>> {
    let mut collect = Collect {
        grads,
        out: Vec::new(),
    };
    module.visit(&mut collect);
    collect.out
}

fn assert_grads_match(padded: Vec<Option<Tensor<1>>>, solo: Vec<Option<Tensor<1>>>) {
    assert_eq!(padded.len(), solo.len());
    for (i, (p, s)) in padded.into_iter().zip(solo).enumerate() {
        match (p, s) {
            (Some(p), Some(s)) => {
                let tol = match p.dtype() {
                    burn::tensor::DType::F16 | burn::tensor::DType::BF16 => HALF_GRAD_TOL,
                    _ => TOL,
                };
                let diff = max_rel_diff(p, s);
                assert!(diff < tol, "parameter {i}: gradient differs by {diff}");
            }
            (None, None) => {}
            (p, s) => panic!(
                "parameter {i}: gradient present padded {} / alone {}",
                p.is_some(),
                s.is_some()
            ),
        }
    }
}

/// Run `forward` on the padded batch and on each row alone. Compare every
/// position that each row owns, every cache slot, and the gradient of every
/// parameter for a loss over those positions. `positions(len)` names the
/// positions that a `len`-token input comes out as.
fn check_each_row_alone<M: Module>(
    module: &M,
    lens: &[usize],
    input_width: usize,
    positions: impl Fn(usize) -> Vec<Position>,
    forward: impl Fn(Tensor<3>, Option<Tensor<2, Bool>>) -> (Tensor<3>, RefCaches),
) {
    let device = test_device().autodiff();
    let padded = *lens.iter().max().unwrap();
    let batch = lens.len();
    let x = Tensor::<3>::random([batch, padded, input_width], Distribution::Normal(0.0, 1.0), &device);
    let (y, caches) = forward(x.clone(), Some(pad_mask(lens, padded, &device)));
    let padded_positions = positions(padded);
    let width = y.dims()[2];
    assert_eq!(y.dims(), [batch, padded_positions.len(), width]);
    let weight =
        Tensor::<2>::random([padded_positions.len(), width], Distribution::Normal(0.0, 1.0), &device);

    let mut loss_padded = Tensor::<1>::zeros([1], &device);
    let mut loss_solo = Tensor::<1>::zeros([1], &device);
    for (b, &len) in lens.iter().enumerate() {
        let (y_solo, caches_solo) = forward(x.clone().narrow(0, b, 1).narrow(1, 0, len), None);
        let owned = positions(len);
        let idx = indices(&padded_positions, &owned, &device);
        let y_owned = y.clone().narrow(0, b, 1).select(1, idx.clone());
        let diff = max_rel_diff(y_owned.clone(), y_solo.clone());
        assert!(diff < dtype_tol(TOL), "row {b} (length {len}): its positions differ by {diff}");
        for (l, (c, c_solo)) in caches.caches.iter().zip(&caches_solo.caches).enumerate() {
            let diff = max_rel_diff(c.state_bd.clone().narrow(0, b, 1), c_solo.state_bd.clone());
            assert!(diff < dtype_tol(TOL), "row {b} (length {len}): cache {l} differs by {diff}");
        }
        let weight = weight.clone().select(0, idx).unsqueeze::<3>();
        loss_padded = loss_padded + (y_owned * weight.clone()).sum();
        loss_solo = loss_solo + (y_solo * weight).sum();
    }
    assert_grads_match(
        param_grads(module, &loss_padded.backward()),
        param_grads(module, &loss_solo.backward()),
    );
}

/// Three layers with every marker kind, over rows down to a single token.
/// `Middle`/`End` land at the length of each row. A `Custom` past the end of a
/// short row is absent there. Several markers share a place.
fn marked_layers(residuals: ResidualsConfig, device: &Device) -> Layers<RefBlock> {
    LayersBuilder {
        class_latents: vec![
            ClassLatent::End,
            ClassLatent::Custom(2),
            ClassLatent::Start,
            ClassLatent::Middle,
            ClassLatent::Custom(5),
        ],
        residuals,
        ..LayersBuilder::new(3, RefBlockConfig::new(D_MODEL))
    }
    .init(device)
}

const LENS: [usize; 4] = [7, 3, 5, 1];

#[test]
fn padded_layers_are_each_row_alone() {
    let device = test_device().autodiff();
    let layers = marked_layers(ResidualsConfig::Standard, &device);
    check_each_row_alone(
        &layers,
        &LENS,
        D_MODEL,
        |len| place(users(len), &layers.class_latents, 0),
        |x, pad| layers.forward(x, None, (), None, pad),
    );
}

#[test]
fn padded_multi_gate_layers_are_each_row_alone() {
    let device = test_device().autodiff();
    let residuals = ResidualsConfig::MultiGate {
        n_stream: 2,
        init_bias: 1.0,
        init_bias_offset: 0.0,
        per_virtual_layer: false,
    };
    let layers = marked_layers(residuals, &device);
    check_each_row_alone(
        &layers,
        &LENS,
        D_MODEL,
        |len| place(users(len), &layers.class_latents, 0),
        |x, pad| layers.forward(x, None, (), None, pad),
    );
}

/// The untracked layers run on the inner backend, and the padding hops with
/// them.
#[test]
fn padded_layers_under_a_grad_horizon_are_each_row_alone() {
    let device = test_device().autodiff();
    let mut layers = marked_layers(ResidualsConfig::Standard, &device);
    layers.grad_horizon = Some(GradHorizon::Mask(vec![false, true, false]));
    check_each_row_alone(
        &layers,
        &LENS,
        D_MODEL,
        |len| place(users(len), &layers.class_latents, 0),
        |x, pad| layers.forward(x, None, (), None, pad),
    );
}

/// A network also places its own class tokens for each row, one marker level
/// below the class latents of the stack.
#[test]
fn padded_latent_network_is_each_row_alone() {
    let device = test_device().autodiff();
    let net = LatentNetworkBuilder {
        input_size: 3,
        layers: LayersBuilder {
            class_latents: vec![ClassLatent::End],
            ..LayersBuilder::new(2, RefBlockConfig::new(D_MODEL))
        },
        output_size: 2,
        final_norm: true,
        class_tokens: vec![ClassToken::Middle, ClassToken::Start, ClassToken::End],
    }
    .init(&device);
    check_each_row_alone(
        &net,
        &LENS,
        3,
        |len| place(place(users(len), &net.class_tokens, 0), &net.layers.class_latents, 1),
        |x, pad| net.forward(x, None, (), None, pad),
    );
}

/// The reversed pass reads each row from its own last real token. It does
/// not start at the last position of the batch, because the padding would
/// then come first.
#[test]
fn padded_bidi_layers_are_each_row_alone() {
    let device = test_device().autodiff();
    let layers = BidiLayersBuilder {
        n_real_layers: 4,
        n_virtual_layers: None,
        block: RefBlockConfig::new(D_MODEL),
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: vec![OutputMergeConfig::CatLinear; 2],
        class_latents: vec![ClassLatent::Start, ClassLatent::End, ClassLatent::Middle],
        residuals: ResidualsConfig::Standard,
        untied: Vec::new(),
    }
    .init(&device);
    check_each_row_alone(
        &layers,
        &LENS,
        D_MODEL,
        |len| place(users(len), &layers.class_latents, 0),
        |x, pad| layers.forward(x, None, (), None, pad),
    );
}

/// Split into windows, a padded batch is still each row alone. The row that
/// ends in the first window carries its state through the second window
/// untouched. Its `End` lands there, in the window that closes the batch.
#[test]
fn padded_layers_split_into_windows_are_each_row_alone() {
    let device = test_device().autodiff();
    let layers = LayersBuilder {
        class_latents: vec![ClassLatent::Start, ClassLatent::End, ClassLatent::Custom(4)],
        ..LayersBuilder::new(2, RefBlockConfig::new(D_MODEL))
    }
    .init(&device);

    let lens = [9, 3, 6];
    let split = 5;
    check_each_row_alone(
        &layers,
        &lens,
        D_MODEL,
        |len| place(users(len), &layers.class_latents, 0),
        |x, pad| {
            let len = x.dims()[1];
            let Some(pad) = pad else {
                return layers.forward(x, None, (), None, None);
            };
            let mut class = ClassCursors::new(len);
            let (y_a, caches) = layers.forward(
                x.clone().narrow(1, 0, split),
                None,
                (),
                Some(&mut class),
                Some(pad.clone().narrow(1, 0, split)),
            );
            let (y_b, caches) = layers.forward(
                x.narrow(1, split, len - split),
                Some(caches),
                (),
                Some(&mut class),
                Some(pad.narrow(1, split, len - split)),
            );
            (Tensor::cat(vec![y_a, y_b], 1), caches)
        },
    );
}

#[test]
#[should_panic(expected = "needs the whole padded sequence")]
fn a_middle_marker_needs_the_whole_padded_sequence() {
    let device = test_device();
    let layers = LayersBuilder {
        class_latents: vec![ClassLatent::Middle],
        ..LayersBuilder::new(1, RefBlockConfig::new(D_MODEL))
    }
    .init(&device);
    let x = Tensor::random([2, 6, D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    let pad = pad_mask(&[6, 2], 6, &device);
    let mut class = ClassCursors::new(6);
    layers.forward(
        x.narrow(1, 0, 4),
        None,
        (),
        Some(&mut class),
        Some(pad.narrow(1, 0, 4)),
    );
}
