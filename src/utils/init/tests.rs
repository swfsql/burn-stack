//! What the policy redraws, what it leaves alone, and what survives the redraw.

use super::*;
use crate::modules::{GatedMlpConfig, LatentNetwork, LatentNetworkBuilder, LayersBuilder};
use crate::reference::{RefBlock, RefBlockConfig};
use burn::module::{ModuleVisitor, ParamId};

const D_MODEL: usize = 16;
const STD: f64 = 0.5;
/// Two layers, each a mixer plus a feed-forward.
const RESIDUAL_DEPTH: usize = 4;

type Device = burn::prelude::Device;

fn net(device: &Device) -> LatentNetwork<RefBlock> {
    LatentNetworkBuilder {
        input_size: 4,
        layers: LayersBuilder::new(2, RefBlockConfig::new(D_MODEL)).with_mlp(Some(
            GatedMlpConfig::new(D_MODEL, 2 * D_MODEL).with_multiple_of(D_MODEL),
        )),
        output_size: 3,
        final_norm: true,
        class_tokens: Vec::new(),
    }
    .init(device)
}

fn policy() -> InitPolicy {
    InitPolicy::new()
        .with_std(STD)
        .with_residual_paths(InitPolicy::default_residual_paths())
        .with_residual_depth(Some(RESIDUAL_DEPTH))
}

fn values<const D: usize>(t: &burn::module::Param<Tensor<D>>) -> Vec<f32> {
    t.val().into_data().try_to_vec::<f32>().unwrap()
}

fn std_of(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32).sqrt()
}

/// Every 2-D `weight` is redrawn at the policy's `std`, and every `bias` zeroed.
/// Burn's own default is Kaiming-uniform off `fan_in`, so a width change would
/// otherwise change the initial scale — which is the thing a global rule fixes.
#[test]
fn weights_are_redrawn_and_biases_zeroed() {
    let device: Device = Default::default();
    let net = policy().apply(net(&device));

    let block = &net.layers.real_layers[0].block;
    let sampled = std_of(&values(&block.in_proj.weight));
    assert!(
        (sampled - STD as f32).abs() < 0.15,
        "block in_proj drawn at std {sampled}, expected {STD}",
    );

    // `LatentNetwork`'s boundary projections carry a bias; it is zeroed.
    let bias = net.in_proj.bias.as_ref().expect("in_proj has a bias");
    assert!(values(bias).iter().all(|v| *v == 0.0));
}

/// A weight that writes into the residual stream is drawn narrower, by
/// `√(residual branches)` — the GPT-2 scheme, which is about the stream's
/// variance at depth and so cannot be a per-module decision.
#[test]
fn residual_weights_are_rescaled_by_depth() {
    let device: Device = Default::default();
    let net = policy().apply(net(&device));
    let layer = &net.layers.real_layers[0];

    let expected = STD as f32 / (RESIDUAL_DEPTH as f32).sqrt();
    for (name, sampled) in [
        ("block out_proj", std_of(&values(&layer.block.out_proj.weight))),
        (
            "mlp fc2",
            std_of(&values(&layer.mlp.as_ref().unwrap().fc2.weight)),
        ),
    ] {
        assert!(
            (sampled - expected).abs() < 0.1,
            "{name} drawn at std {sampled}, expected {expected}",
        );
    }

    // The mixer's *input* map is not on the residual path, so it keeps `std`.
    let entry = std_of(&values(&layer.block.in_proj.weight));
    assert!(entry > expected * 1.5, "in_proj was rescaled too: {entry}");
}

/// A block's own parameters mean something — a decay, a spread of timescales, a
/// norm gain — and a global rule must not touch them. Only a matrix stored as
/// `weight` and a 1-D `bias` are its business.
#[test]
fn bespoke_parameters_are_left_alone() {
    let device: Device = Default::default();
    let net = policy().apply(net(&device));
    let layer = &net.layers.real_layers[0];

    assert!(
        values(&layer.block.decay_raw).iter().all(|v| *v == 0.0),
        "the block's decay was redrawn",
    );
    assert!(
        values(&layer.norm.gamma).iter().all(|v| *v == 1.0),
        "a norm gain was redrawn",
    );
}

/// The redraw replaces values, not parameters: ids are what an optimizer's
/// state and a saved record are keyed by.
#[test]
fn parameter_ids_survive_the_redraw() {
    #[derive(Default)]
    struct Ids(Vec<ParamId>);
    impl ModuleVisitor for Ids {
        fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<D>>) {
            self.0.push(param.id);
        }
    }

    let device: Device = Default::default();
    let net = net(&device);
    let mut before = Ids::default();
    net.visit(&mut before);

    let net = policy().apply(net);
    let mut after = Ids::default();
    net.visit(&mut after);

    assert_eq!(before.0, after.0);
}

/// A redrawn weight must still train. `Param::map` reads `require_grad` off the
/// tensor it is handed, and a freshly drawn one carries none — so a redraw that
/// forgets to restore the flag detaches the parameter silently, with every
/// forward value still correct.
#[test]
fn redrawn_weights_still_receive_gradient() {
    let device = Device::default().autodiff();
    let net = policy().apply(net(&device));

    let x = Tensor::<3>::random([2, 3, 4], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let (y, _) = net.forward(x, None, (), None);
    let grads = y.sum().backward();

    let layer = &net.layers.real_layers[0];
    for (name, grad) in [
        ("block in_proj", layer.block.in_proj.weight.val().grad(&grads)),
        ("block out_proj", layer.block.out_proj.weight.val().grad(&grads)),
        (
            "mlp fc2",
            layer.mlp.as_ref().unwrap().fc2.weight.val().grad(&grads),
        ),
        ("network in_proj", net.in_proj.weight.val().grad(&grads)),
    ] {
        assert!(grad.is_some(), "{name} lost its gradient in the redraw");
    }
}
