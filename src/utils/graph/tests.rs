//! A captured step is the eager step: same outputs, same final caches, for a
//! host-fed (token ids) and a device-fed (latent) input, and a state set in
//! place is continued from; a stateless forward (caches `()`) is the eager one.
//!
//! Off a hardware-graph build (flex, the default) the captured step falls back
//! to stepping eagerly, which still runs `capture`'s closure and so pins the
//! snapshot/restore around it; under `backend-cuda` the same suite replays a
//! real graph, which [`expects_graph`] asserts it got.

use super::CapturedStep;
use crate::modules::{Layers, LayersBuilder, VocabNetwork, VocabNetworkBuilder};
use crate::reference::{RefBlock, RefBlockConfig, RefCaches};
use crate::utils::test_helpers::max_abs_diff;
use crate::utils::{ClassCursors, ClassLatent};
use burn::prelude::*;
use burn::tensor::{Distribution, TensorData};

const D_MODEL: usize = 8;
const VOCAB: usize = 11;
const BATCH: usize = 3;
const STEPS: usize = 12;

/// Whether this build and device give a hardware graph (cubecl without
/// fusion, on a CUDA/HIP device).
fn expects_graph(device: &Device) -> bool {
    let name = format!("{device:?}");
    cfg!(all(feature = "cubecl", not(feature = "fusion")))
        && (name.contains("Cuda") || name.contains("Hip"))
}

fn assert_caches(label: &str, a: &RefCaches, b: &RefCaches) {
    assert_eq!(a.caches.len(), b.caches.len());
    for (i, (a, b)) in a.caches.iter().zip(&b.caches).enumerate() {
        let d = max_abs_diff(a.state_bd.clone(), b.state_bd.clone());
        assert_eq!(d, 0.0, "{label}: slot {i} differs by {d}");
    }
}

fn tokens(k: usize) -> Vec<i32> {
    (0..BATCH).map(|b| ((k * 7 + b * 3 + 1) % VOCAB) as i32).collect()
}

/// Two real layers, opened by two `Start` latents: after `prime`, a step with
/// no cursors is the step with them.
fn vocab_net(device: &Device) -> VocabNetwork<RefBlock> {
    VocabNetworkBuilder {
        vocab_size: VOCAB,
        pad_vocab_size_multiple: 1,
        layers: LayersBuilder {
            class_latents: vec![ClassLatent::Start, ClassLatent::Start],
            ..LayersBuilder::new(2, RefBlockConfig::new(D_MODEL))
        },
        missing_lm_head: false,
    }
    .init(device)
}

/// Eager: every step with the cursors, from the opening.
fn eager_run(net: &VocabNetwork<RefBlock>, device: &Device) -> (Vec<Tensor<2>>, RefCaches) {
    let mut class = ClassCursors::stream();
    let (_, caches) = net.prime(BATCH, None, Some(&mut class));
    let mut caches = caches.expect("two Start latents leave a cache");
    let mut outs = Vec::new();
    for k in 0..STEPS {
        let x = Tensor::<1, Int>::from_ints(tokens(k).as_slice(), device);
        let (y, c) = net.step(x, Some(caches), Some(&mut class));
        outs.push(y);
        caches = c;
    }
    (outs, caches)
}

#[test]
fn captured_token_steps_are_the_eager_steps() {
    let device = Device::default();
    let net = vocab_net(&device);
    assert!(net.layers.only_start_latents());
    let (eager, eager_caches) = eager_run(&net, &device);

    let mut class = ClassCursors::stream();
    let (_, caches) = net.prime(BATCH, None, Some(&mut class));
    let x = Tensor::<1, Int>::from_ints(tokens(0).as_slice(), &device);
    let (y0, caches) = net.step(x, caches, Some(&mut class));
    let opening = caches.clone();
    let x = Tensor::<1, Int>::from_ints(tokens(1).as_slice(), &device);
    // Safety: the step reads nothing but its arguments and `net`, borrowed.
    let mut captured =
        unsafe { CapturedStep::capture(&device, x, caches, |x, c| net.step(x, Some(c), None)) };
    assert_eq!(captured.is_captured(), expects_graph(&device));

    let d = max_abs_diff(y0, eager[0].clone());
    assert_eq!(d, 0.0, "step 0 differs by {d}");
    for k in 1..STEPS {
        let data = TensorData::new(tokens(k), [BATCH]);
        // Compared at once: the next replay overwrites it.
        let y = captured.step_data(data).clone();
        let d = max_abs_diff(y, eager[k].clone());
        assert_eq!(d, 0.0, "step {k} differs by {d}");
    }
    assert_caches("final caches", &captured.caches(), &eager_caches);

    // Setting the state in place restarts from it.
    captured.set_caches(opening);
    for k in 1..STEPS {
        let data = TensorData::new(tokens(k), [BATCH]);
        let y = captured.step_data(data).clone();
        let d = max_abs_diff(y, eager[k].clone());
        assert_eq!(d, 0.0, "restarted step {k} differs by {d}");
    }
    assert_caches("restarted caches", &captured.into_caches(), &eager_caches);
}

#[test]
fn captured_latent_steps_are_the_eager_steps() {
    let device = Device::default();
    let layers: Layers<RefBlock> = LayersBuilder::new(2, RefBlockConfig::new(D_MODEL)).init(&device);
    let xs: Vec<Tensor<2>> = (0..STEPS)
        .map(|_| Tensor::random([BATCH, D_MODEL], Distribution::Normal(0.0, 1.0), &device))
        .collect();

    let mut caches = None;
    let mut eager = Vec::new();
    for x in &xs {
        let (y, c) = layers.step(x.clone(), caches, None);
        eager.push(y);
        caches = Some(c);
    }
    let eager_caches = caches.unwrap();

    let (y0, caches) = layers.step(xs[0].clone(), None, None);
    // Safety: the step reads nothing but its arguments and `layers`, borrowed.
    let mut captured = unsafe {
        CapturedStep::capture(&device, xs[1].clone(), caches, |x, c| layers.step(x, Some(c), None))
    };
    assert_eq!(captured.is_captured(), expects_graph(&device));
    let d = max_abs_diff(y0, eager[0].clone());
    assert_eq!(d, 0.0, "step 0 differs by {d}");
    for k in 1..STEPS {
        let y = captured.step(xs[k].clone()).clone();
        let d = max_abs_diff(y, eager[k].clone());
        assert_eq!(d, 0.0, "step {k} differs by {d}");
    }
    assert_caches("final caches", &captured.into_caches(), &eager_caches);
}

#[test]
fn captured_forward_is_the_eager_forward() {
    const LEN: usize = 5;
    let device = Device::default();
    let layers: Layers<RefBlock> = LayersBuilder::new(2, RefBlockConfig::new(D_MODEL)).init(&device);
    let forward = |x| layers.forward(x, None, (), None, None).0;
    let xs: Vec<Tensor<3>> = (0..STEPS)
        .map(|_| Tensor::random([BATCH, LEN, D_MODEL], Distribution::Normal(0.0, 1.0), &device))
        .collect();

    // Captured cold, before any eager call has compiled the forward's kernels.
    // Safety: the forward reads nothing but its argument and `layers`, borrowed.
    let mut captured =
        unsafe { CapturedStep::capture(&device, xs[0].clone(), (), |x, ()| (forward(x), ())) };
    assert_eq!(captured.is_captured(), expects_graph(&device));
    for (k, x) in xs.iter().enumerate() {
        let y = captured.step(x.clone()).clone();
        let d = max_abs_diff(y, forward(x.clone()));
        assert_eq!(d, 0.0, "call {k} differs by {d}");
    }
}

#[test]
fn only_start_latents_sees_both_levels() {
    let device = Device::default();
    let mut net = vocab_net(&device);
    assert!(net.layers.only_start_latents());
    net.layers.real_layers[1].class_latents = vec![ClassLatent::Custom(4)];
    assert!(!net.layers.only_start_latents());
}
