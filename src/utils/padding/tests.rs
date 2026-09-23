//! A padded batch is each of its slots run alone. Every row that a slot owns
//! (its user tokens, and the class markers placed against its own length)
//! comes out as the forward of that slot alone gives it, with the same caches
//! and gradients. The tests compose [`RefBlock`], so they pin the half of the
//! contract that belongs to the containers.

use crate::modules::bidi::{BidiLayersBuilder, OutputMergeConfig};
use crate::modules::{LatentNetworkBuilder, Layers, LayersBuilder, ResidualsConfig};
use crate::reference::{RefBlock, RefBlockConfig, RefCaches};
use crate::utils::class::{ClassMarker, class_chunk_plan, init_class_emb};
use crate::utils::test_helpers::max_abs_diff;
use crate::utils::{ClassCursor, ClassCursors, ClassLatent, ClassToken, GradHorizon};
use burn::module::{ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::{Distribution, Gradients};

const D_MODEL: usize = 8;
const TOL: f32 = 1e-4;

/// One row of a container's output: user token `t`, or marker `i` of the
/// `level`-th splice (network tokens, stack latents, then each layer's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Row {
    User(usize),
    Marker(usize, usize),
}

fn users(len: usize) -> Vec<Row> {
    (0..len).map(Row::User).collect()
}

/// Splice one level's markers into `rows`, as a whole-sequence call places them.
fn place<M: ClassMarker>(rows: Vec<Row>, markers: &[M], level: usize) -> Vec<Row> {
    let mut cursor = ClassCursor::whole(rows.len());
    let plan = class_chunk_plan(markers, rows.len(), &mut cursor, "test");
    let mut out = Vec::with_capacity(rows.len() + plan.len());
    let mut taken = 0;
    for (at, i) in plan {
        out.extend_from_slice(&rows[taken..at]);
        taken = at;
        out.push(Row::Marker(level, i));
    }
    out.extend_from_slice(&rows[taken..]);
    out
}

/// The rows a stack (no virtual layers) outputs from `rows`: its own latents,
/// then each layer's.
fn stack_rows(layers: &Layers<RefBlock>, rows: Vec<Row>, level: usize) -> Vec<Row> {
    let mut rows = place(rows, &layers.class_latents, level);
    for (i, layer) in layers.real_layers.iter().enumerate() {
        rows = place(rows, &layer.class_latents, level + 1 + i);
    }
    rows
}

/// `[batch, padded]`, `true` past each slot's length.
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

/// Where each of `owned` sits in `rows`.
fn indices(rows: &[Row], owned: &[Row], device: &Device) -> Tensor<1, Int> {
    let idx: Vec<i32> = owned
        .iter()
        .map(|r| {
            rows.iter()
                .position(|p| p == r)
                .unwrap_or_else(|| panic!("{r:?}, owned by a slot, is missing from the padded batch"))
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
                let diff = max_abs_diff(p, s);
                assert!(diff < TOL, "parameter {i}: gradient differs by {diff}");
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

/// Run `forward` on the padded batch and on each slot alone. Compare every row
/// that each slot owns, every cache slot, and the gradient of every parameter
/// for a loss over those rows. `rows(len)` names the rows that a `len`-token
/// input comes out as.
fn check_each_slot_alone<M: Module>(
    module: &M,
    lens: &[usize],
    input_width: usize,
    rows: impl Fn(usize) -> Vec<Row>,
    forward: impl Fn(Tensor<3>, Option<Tensor<2, Bool>>) -> (Tensor<3>, RefCaches),
) {
    let device = Device::default().autodiff();
    let padded = *lens.iter().max().unwrap();
    let batch = lens.len();
    let x = Tensor::<3>::random([batch, padded, input_width], Distribution::Normal(0.0, 1.0), &device);
    let (y, caches) = forward(x.clone(), Some(pad_mask(lens, padded, &device)));
    let padded_rows = rows(padded);
    let width = y.dims()[2];
    assert_eq!(y.dims(), [batch, padded_rows.len(), width]);
    let weight =
        Tensor::<2>::random([padded_rows.len(), width], Distribution::Normal(0.0, 1.0), &device);

    let mut loss_padded = Tensor::<1>::zeros([1], &device);
    let mut loss_solo = Tensor::<1>::zeros([1], &device);
    for (b, &len) in lens.iter().enumerate() {
        let (y_solo, caches_solo) = forward(x.clone().narrow(0, b, 1).narrow(1, 0, len), None);
        let owned = rows(len);
        let idx = indices(&padded_rows, &owned, &device);
        let y_owned = y.clone().narrow(0, b, 1).select(1, idx.clone());
        let diff = max_abs_diff(y_owned.clone(), y_solo.clone());
        assert!(diff < TOL, "slot {b} (length {len}): its rows differ by {diff}");
        for (l, (c, c_solo)) in caches.caches.iter().zip(&caches_solo.caches).enumerate() {
            let diff = max_abs_diff(c.state_bd.clone().narrow(0, b, 1), c_solo.state_bd.clone());
            assert!(diff < TOL, "slot {b} (length {len}): cache {l} differs by {diff}");
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

/// Three layers with every marker kind at both levels, over slots down to a
/// single token. So `Middle`/`End` land at the length of each slot, a `Custom`
/// past the end of a short slot is absent there, and several markers share a
/// place.
fn marked_layers(residuals: ResidualsConfig, device: &Device) -> Layers<RefBlock> {
    let mut layers = LayersBuilder {
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
    .init(device);
    let per_layer = [
        vec![ClassLatent::End, ClassLatent::Start],
        vec![],
        vec![ClassLatent::Custom(3), ClassLatent::Middle, ClassLatent::End],
    ];
    for (layer, latents) in layers.real_layers.iter_mut().zip(per_layer) {
        layer.class_latents_emb = init_class_emb(latents.len(), D_MODEL, device);
        layer.class_latents = latents;
    }
    layers
}

const LENS: [usize; 4] = [7, 3, 5, 1];

#[test]
fn padded_layers_are_each_slot_alone() {
    let device = Device::default().autodiff();
    let layers = marked_layers(ResidualsConfig::Standard, &device);
    check_each_slot_alone(
        &layers,
        &LENS,
        D_MODEL,
        |len| stack_rows(&layers, users(len), 0),
        |x, pad| layers.forward(x, None, (), None, pad),
    );
}

#[test]
fn padded_multi_gate_layers_are_each_slot_alone() {
    let device = Device::default().autodiff();
    let residuals = ResidualsConfig::MultiGate {
        n_stream: 2,
        init_bias: 1.0,
        init_bias_step: 0.0,
        per_virtual_layer: false,
    };
    let layers = marked_layers(residuals, &device);
    check_each_slot_alone(
        &layers,
        &LENS,
        D_MODEL,
        |len| stack_rows(&layers, users(len), 0),
        |x, pad| layers.forward(x, None, (), None, pad),
    );
}

/// The untracked layers run on the inner backend, and the padding hops with
/// them.
#[test]
fn padded_layers_under_a_grad_horizon_are_each_slot_alone() {
    let device = Device::default().autodiff();
    let mut layers = marked_layers(ResidualsConfig::Standard, &device);
    layers.grad_horizon = Some(GradHorizon::Mask(vec![false, true, false]));
    check_each_slot_alone(
        &layers,
        &LENS,
        D_MODEL,
        |len| stack_rows(&layers, users(len), 0),
        |x, pad| layers.forward(x, None, (), None, pad),
    );
}

/// A network's own class tokens are placed per slot too, one level below the
/// stack's latents.
#[test]
fn padded_latent_network_is_each_slot_alone() {
    let device = Device::default().autodiff();
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
    check_each_slot_alone(
        &net,
        &LENS,
        3,
        |len| stack_rows(&net.layers, place(users(len), &net.class_tokens, 0), 1),
        |x, pad| net.forward(x, None, (), None, pad),
    );
}

/// The reversed direction reads each slot from its own last row — not from the
/// batch's, whose padding would then lead.
#[test]
fn padded_bidi_layers_are_each_slot_alone() {
    let device = Device::default().autodiff();
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
    check_each_slot_alone(
        &layers,
        &LENS,
        D_MODEL,
        |len| place(users(len), &layers.class_latents, 0),
        |x, pad| layers.forward(x, None, (), None, pad),
    );
}

/// Split into chunks, a padded batch is still each slot alone. The slot that
/// ends in the first chunk carries its state through the second chunk
/// untouched. Its `End` lands there, in the chunk that closes the batch.
#[test]
fn chunked_padded_layers_are_each_slot_alone() {
    let device = Device::default().autodiff();
    let mut layers = LayersBuilder {
        class_latents: vec![ClassLatent::Start, ClassLatent::End, ClassLatent::Custom(4)],
        ..LayersBuilder::new(2, RefBlockConfig::new(D_MODEL))
    }
    .init(&device);
    layers.real_layers[1].class_latents = vec![ClassLatent::End];
    layers.real_layers[1].class_latents_emb = init_class_emb(1, D_MODEL, &device);

    let lens = [9, 3, 6];
    let split = 5;
    check_each_slot_alone(
        &layers,
        &lens,
        D_MODEL,
        |len| stack_rows(&layers, users(len), 0),
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
    let device = Device::default();
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
