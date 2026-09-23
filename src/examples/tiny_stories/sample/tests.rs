//! These tests assert three things:
//!
//! - The device draw is the inverse-CDF draw of the host loop. The tests check
//!   every running total and one step to each side, where the two could
//!   differ.
//! - A story decodes to the same text captured, eagerly, and sampled on the
//!   host one step at a time.
//! - A chunked prefill is the same captured and eagerly, across prompts that
//!   share one graph. It is within float noise of the opening and the prompt
//!   in one pass.
//!
//! Off a hardware-graph build (flex, the default), the captured decode runs
//! eager steps. Under `backend-cuda`, it replays a real graph.

use super::{Prefill, generate, sample_token};
use crate::examples::tiny_stories::dataset::{VOCAB, VOCAB_SIZE};
use crate::modules::{LayersBuilder, VocabNetwork, VocabNetworkBuilder};
use crate::reference::{RefBlock, RefBlockConfig, RefCaches};
use crate::utils::test_helpers::max_abs_diff;
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
/// `generate` primes, and can capture). The block does not check its padding: a
/// captured chunk cannot read its mask back.
fn story_net(device: &Device) -> VocabNetwork<RefBlock> {
    VocabNetworkBuilder {
        vocab_size: VOCAB_SIZE,
        pad_vocab_size_multiple: 1,
        layers: LayersBuilder {
            class_latents: vec![ClassLatent::Start, ClassLatent::Start],
            ..LayersBuilder::new(2, RefBlockConfig::new(8).with_check_padding(false))
        },
        missing_lm_head: false,
    }
    .init(device)
}

/// Whether this build and device give a hardware graph (cubecl without
/// fusion, on a CUDA/HIP device).
fn expects_graph(device: &Device) -> bool {
    let name = format!("{device:?}");
    cfg!(all(feature = "cubecl", not(feature = "fusion")))
        && (name.contains("Cuda") || name.contains("Hip"))
}

/// The story that a host sampler writes. The probabilities of every step are
/// read back, with one draw per character from the stream of the seed (none
/// when greedy).
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
            let story = generate(&net, &device, (), None, 40, temperature, seed, capture, None);
            assert_eq!(story, host, "capture {capture}, temperature {temperature}");
        }
    }
}

const CHUNK: usize = 8;

/// A [`Prefill`] of `net`, in chunks of [`CHUNK`].
fn prefill_of<'a>(
    net: &'a VocabNetwork<RefBlock>,
    device: &Device,
    capture: bool,
) -> Prefill<'a, RefCaches> {
    // Safety: the chunk reads nothing but its arguments and `net`, borrowed.
    unsafe {
        Prefill::new(device, CHUNK, capture, |x, c, pad, class| {
            net.forward(x, Some(c), (), Some(class), Some(pad))
        })
    }
}

/// A prompt of `len` ids, different for every length.
fn prompt(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 5 + len * 3) % VOCAB_SIZE) as u8).collect()
}

/// The cache of the opening and the cursors that it leaves.
fn opening(net: &VocabNetwork<RefBlock>) -> (RefCaches, ClassCursors) {
    let mut class = ClassCursors::stream();
    let (_, caches) = net.prime(1, None, Some(&mut class));
    (caches.expect("two Start latents leave a cache"), class)
}

#[test]
fn a_prefill_is_the_same_captured_and_eager_and_is_the_prompt_in_one_pass() {
    let device = Device::default();
    let net = story_net(&device);
    let mut eager = prefill_of(&net, &device, false);
    // One capture, reused by every prompt after the first.
    let mut captured = prefill_of(&net, &device, true);
    for len in [CHUNK + 1, 1, CHUNK - 1, CHUNK, 3 * CHUNK] {
        let prompt = prompt(len);
        let (logits, prefilled, _) = eager.run(&prompt, || Some(opening(&net))).unwrap();
        let (logits_captured, prefilled_captured, _) =
            captured.run(&prompt, || Some(opening(&net))).unwrap();
        assert_eq!(captured.is_captured(), expects_graph(&device));
        let d = max_abs_diff(logits_captured, logits.clone());
        assert_eq!(d, 0.0, "length {len}: captured logits differ by {d}");
        for (a, b) in prefilled_captured.caches.iter().zip(&prefilled.caches) {
            let d = max_abs_diff(a.state_bd.clone(), b.state_bd.clone());
            assert_eq!(d, 0.0, "length {len}: captured cache differs by {d}");
        }

        // The opening and the whole prompt in one forward.
        let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let x = Tensor::<1, Int>::from_ints(ids.as_slice(), &device).reshape([1, len]);
        let (whole, whole_caches) = net.forward(x, None, (), Some(&mut ClassCursors::stream()), None);
        let rows = whole.dims()[1];
        let d = max_abs_diff(logits, whole.narrow(1, rows - 1, 1).squeeze_dim(1));
        assert!(d < 1e-4, "length {len}: logits differ from one pass by {d}");
        for (a, b) in prefilled.caches.iter().zip(&whole_caches.caches) {
            let d = max_abs_diff(a.state_bd.clone(), b.state_bd.clone());
            assert!(d < 1e-4, "length {len}: cache differs from one pass by {d}");
        }
    }
}

#[test]
fn a_prompted_story_is_the_same_captured_and_eager() {
    let device = Device::default();
    let net = story_net(&device);
    let mut eager = prefill_of(&net, &device, false);
    let mut captured = prefill_of(&net, &device, true);
    for (seed, prompt) in [(0, "once upon a time"), (1, "a"), (2, "the little dog ran home.")] {
        let run = |capture: bool, prefill: &mut Prefill<'_, RefCaches>| {
            generate(&net, &device, (), Some(prompt), 40, 0.8, seed, capture, Some(prefill))
        };
        assert_eq!(run(true, &mut captured), run(false, &mut eager), "prompt {prompt:?}");
    }
}
