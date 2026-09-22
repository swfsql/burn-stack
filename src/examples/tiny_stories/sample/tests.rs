//! The device draw is the host loop's inverse-CDF draw — at every running total
//! and one step either side, where the two could part — and a story decodes to
//! the same text captured, eagerly, and sampled on the host a step at a time.
//!
//! Off a hardware-graph build (flex, the default) the captured decode falls back
//! to stepping eagerly; under `backend-cuda` it replays a real graph.

use super::{generate, sample_token};
use crate::examples::tiny_stories::dataset::{VOCAB, VOCAB_SIZE};
use crate::modules::{LayersBuilder, VocabNetwork, VocabNetworkBuilder};
use crate::reference::{RefBlock, RefBlockConfig};
use crate::utils::{ClassCursors, ClassLatent};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation::softmax;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Inverse-CDF sampling on the host: the first token whose running total of
/// `probs` reaches `threshold`, the last one if rounding leaves it short.
fn host_draw(probs: &[f32], threshold: f32) -> u8 {
    let mut cumulative = 0.0;
    for (token, p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= threshold {
            return token as u8;
        }
    }
    (VOCAB_SIZE - 1) as u8
}

fn host_probs(logits: Tensor<2>, temperature: f64) -> Vec<f32> {
    softmax(logits / temperature, 1).into_data().iter::<f32>().collect()
}

fn host_argmax(logits: Tensor<2>) -> u8 {
    logits.argmax(1).into_data().iter::<i64>().next().unwrap() as u8
}

fn device_draw(logits: Tensor<2>, temperature: f64, threshold: f32, device: &Device) -> u8 {
    let draw = Tensor::<1>::from_floats([threshold], device);
    let token = sample_token(logits, temperature, draw).into_data();
    token.iter::<i64>().next().unwrap() as u8
}

#[test]
fn device_draw_is_the_host_draw() {
    let device = Device::default();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    for temperature in [0.5, 1.0, 2.0] {
        let logits = Tensor::<2>::random([1, VOCAB_SIZE], Distribution::Normal(0.0, 3.0), &device);
        let probs = host_probs(logits.clone(), temperature);
        let mut thresholds = vec![0.0, 1.0f32.next_down()];
        let mut cumulative = 0.0f32;
        for p in &probs {
            cumulative += p;
            thresholds.extend([cumulative.next_down(), cumulative, cumulative.next_up()]);
        }
        thresholds.extend((0..64).map(|_| rng.random_range(0.0f32..1.0)));
        for t in thresholds.into_iter().filter(|t| (0.0..1.0).contains(t)) {
            let host = host_draw(&probs, t);
            let device_token = device_draw(logits.clone(), temperature, t, &device);
            assert_eq!(device_token, host, "temperature {temperature}, draw {t}");
        }
        let greedy = device_draw(logits.clone(), 0.0, 0.5, &device);
        assert_eq!(greedy, host_argmax(logits), "greedy");
    }
}

/// Two real layers over the story alphabet, opened by two `Start` latents (so
/// `generate` primes, and may capture).
fn story_net(device: &Device) -> VocabNetwork<RefBlock> {
    VocabNetworkBuilder {
        vocab_size: VOCAB_SIZE,
        pad_vocab_size_multiple: 1,
        layers: LayersBuilder {
            class_latents: vec![ClassLatent::Start, ClassLatent::Start],
            ..LayersBuilder::new(2, RefBlockConfig::new(8))
        },
        missing_lm_head: false,
    }
    .init(device)
}

/// The story a host sampler writes: every step's probabilities read back, one
/// draw per character from the seed's stream (none when greedy).
fn host_story(
    net: &VocabNetwork<RefBlock>,
    device: &Device,
    n_chars: usize,
    temperature: f64,
    seed: u64,
) -> String {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut class = ClassCursors::stream();
    let (logits, caches) = net.prime(1, None, Some(&mut class));
    let (mut logits, mut caches) = (logits.unwrap(), caches.unwrap());
    let mut out = String::new();
    for _ in 0..n_chars {
        let id = if temperature > 0.0 {
            host_draw(&host_probs(logits.clone(), temperature), rng.random_range(0.0..1.0))
        } else {
            host_argmax(logits.clone())
        };
        out.push(VOCAB.character(id));
        let x = Tensor::<1, Int>::from_ints([id as i32], device);
        (logits, caches) = net.step(x, Some(caches), Some(&mut class));
    }
    out
}

#[test]
fn a_story_is_the_same_captured_eager_and_host_sampled() {
    let device = Device::default();
    let net = story_net(&device);
    for (seed, temperature) in [(0, 0.8), (1, 1.0), (2, 0.0)] {
        let host = host_story(&net, &device, 40, temperature, seed);
        for capture in [false, true] {
            let story = generate(&net, &device, (), None, 40, temperature, seed, capture);
            assert_eq!(story, host, "capture {capture}, temperature {temperature}");
        }
    }
}
