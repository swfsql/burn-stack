//! These tests assert four things:
//!
//! - The device draw of the id is the inverse-CDF draw of the host loop. The
//!   tests check every running total and one step to each side, where the two
//!   could differ.
//! - The device draw of the case flag follows its logit, and a symbol never
//!   gets a flag.
//! - A story decodes to the same text captured, eagerly, and sampled on the
//!   host one step at a time.
//! - A chunked prefill is the same captured and eagerly, across prompts that
//!   share one graph. It is within float noise of the opening and the prompt
//!   in one pass.
//!
//! Off a hardware-graph build (flex, the default), the captured decode runs
//! eager steps. Under `backend-cuda`, it replays a real graph.

use super::{Prefill, generate, sample_token};
use crate::examples::tiny_stories::dataset::{FIRST_LETTER, UPPER, VOCAB, VOCAB_SIZE, pair, unpair};
use crate::modules::{LayersBuilder, VocabNetwork, VocabNetworkBuilder};
use crate::reference::{RefBlock, RefBlockConfig, RefCaches};
use crate::utils::test_helpers::{dtype_tol, max_abs_diff, max_rel_diff};
use crate::utils::{ClassCursors, ClassLatent};
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::{DType, Distribution, FloatDType};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use crate::utils::test_helpers::test_device;

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

/// The probabilities of the ids (the vocab logits of `[1, VOCAB_SIZE + 1]`),
/// in f32, as [`sample_token`] computes them for f16, bf16 and f32 `logits`.
fn host_probs(logits: Tensor<2>, temperature: f64) -> Vec<f32> {
    let vocab = logits.narrow(1, 0, VOCAB_SIZE).cast(FloatDType::F32);
    softmax(vocab / temperature, 1).into_data().iter::<f32>().collect()
}

fn host_argmax(logits: Tensor<2>) -> u8 {
    let vocab = logits.narrow(1, 0, VOCAB_SIZE);
    vocab.argmax(1).into_data().iter::<i64>().next().unwrap() as u8
}

/// The case logit of `logits` (`[1, VOCAB_SIZE + 1]`), in f32.
fn host_case(logits: Tensor<2>) -> f32 {
    let case = logits.narrow(1, VOCAB_SIZE, 1).cast(FloatDType::F32);
    case.into_data().iter::<f32>().next().unwrap()
}

/// The device draw `(id, flag)`, with the uniforms `[threshold, case]`.
fn device_draw(logits: Tensor<2>, temperature: f64, threshold: f32, case: f32, device: &Device) -> (u8, bool) {
    let draw = Tensor::<1>::from_data([threshold, case], (device, DType::F32));
    let token: Vec<i64> = sample_token(logits, temperature, draw).into_data().iter::<i64>().collect();
    (token[0] as u8, token[1] == 1)
}

#[test]
fn device_draw_is_the_host_draw() {
    let device = test_device();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    for temperature in [0.5, 1.0, 2.0] {
        let logits = Tensor::<2>::random([1, VOCAB_SIZE + 1], Distribution::Normal(0.0, 3.0), &device);
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
            let (device_token, _) = device_draw(logits.clone(), temperature, t, 0.5, &device);
            assert_eq!(device_token, host, "temperature {temperature}, draw {t}");
        }
        let (greedy, _) = device_draw(logits.clone(), 0.0, 0.5, 0.5, &device);
        assert_eq!(greedy, host_argmax(logits), "greedy");
    }
}

/// The logits `[1, VOCAB_SIZE + 1]` of a sure `id` and the case logit `z`.
fn sure(id: usize, z: f32, device: &Device) -> Tensor<2> {
    let mut values = vec![-1e4f32; VOCAB_SIZE + 1];
    values[id] = 0.0;
    values[VOCAB_SIZE] = z;
    Tensor::<1>::from_floats(values.as_slice(), device).reshape([1, VOCAB_SIZE + 1])
}

#[test]
fn the_case_flag_follows_its_logit_and_skips_symbols() {
    let device = test_device();
    let letter = FIRST_LETTER + 3;
    let symbol = FIRST_LETTER - 1;
    // σ(1) ≈ 0.731 at T 1, σ(0.5) ≈ 0.622 at T 2.
    for (temperature, p) in [(1.0, 0.731f32), (2.0, 0.622)] {
        for (draw, flag) in [(p - 0.01, true), (p + 0.01, false)] {
            let token = device_draw(sure(letter, 1.0, &device), temperature, 0.5, draw, &device);
            assert_eq!(token, (letter as u8, flag), "temperature {temperature}, draw {draw}");
            let token = device_draw(sure(symbol, 1.0, &device), temperature, 0.5, draw, &device);
            assert_eq!(token, (symbol as u8, false), "a symbol has no case");
        }
    }
    for (z, flag) in [(0.3, true), (-0.3, false)] {
        let token = device_draw(sure(letter, z, &device), 0.0, 0.5, 0.5, &device);
        assert_eq!(token, (letter as u8, flag), "greedy, case logit {z}");
    }
    let token = device_draw(sure(symbol, 5.0, &device), 0.0, 0.5, 0.5, &device);
    assert_eq!(token, (symbol as u8, false), "greedy: a symbol has no case");
}

/// An f64 model draws in f64. The first two tokens share the probability
/// mass, with `p₀ = ½ + 10⁻¹²`, and the draw is `½ + 2·10⁻¹²`. So the draw is
/// in the interval of token 1. In f32, both round to `½`, and the draw would
/// give token 0.
#[test]
fn an_f64_model_draws_in_f64() {
    let device = test_device();
    let mut values = vec![-1e4f64; VOCAB_SIZE + 1];
    values[0] = 4e-12;
    values[1] = 0.0;
    values[VOCAB_SIZE] = 0.0;
    let logits = Tensor::<1>::from_data(
        burn::tensor::TensorData::new(values, [VOCAB_SIZE + 1]),
        (&device, DType::F64),
    )
    .reshape([1, VOCAB_SIZE + 1]);
    let draw = Tensor::<1>::from_data([0.5 + 2e-12f64, 0.5], (&device, DType::F64));
    let token = sample_token(logits, 1.0, draw).into_data().iter::<i64>().next().unwrap();
    assert_eq!(token, 1, "the draw is above the total of token 0");
}

/// Two real layers over the story alphabet, opened by two `Start` latents (so
/// `generate` primes, and can capture). The block does not check its padding: a
/// captured chunk cannot read its mask back. The flag input is random (it is
/// zero at init), so the flags of the inputs change the outputs.
fn story_net(device: &Device) -> VocabNetwork<RefBlock> {
    let mut net = VocabNetworkBuilder {
        vocab_size: VOCAB_SIZE,
        pad_vocab_size_multiple: 1,
        layers: LayersBuilder {
            class_latents: vec![ClassLatent::Start, ClassLatent::Start],
            ..LayersBuilder::new(2, RefBlockConfig::new(8).with_check_padding(false))
        },
        missing_lm_head: false,
    }
    .init(device);
    net.flag_embedding = Param::from_tensor(Tensor::random([8], Distribution::Normal(0.0, 1.0), device));
    net
}

#[test]
fn the_flag_of_a_token_changes_the_next_logits() {
    let device = test_device();
    let net = story_net(&device);
    let (caches, _) = opening(&net);
    let step = |flag: i32| {
        let x = Tensor::<1, Int>::from_ints([(FIRST_LETTER + 2) as i32, flag], &device).reshape([1, 2]);
        net.step(x, Some(caches.clone()), None).0
    };
    let d = max_abs_diff(step(0), step(1));
    assert!(d > 1e-3, "the flag input is read: the logits differ by {d}");
}

/// Whether this build and device give a hardware graph (cubecl without
/// fusion, on a CUDA/HIP device).
fn expects_graph(device: &Device) -> bool {
    let name = format!("{device:?}");
    cfg!(all(feature = "cubecl", not(feature = "fusion")))
        && (name.contains("Cuda") || name.contains("Hip"))
}

/// The story that a host sampler writes. The probabilities of every step are
/// read back, with two draws per character (the id, then the case) from the
/// stream of the seed (none when greedy).
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
        let z = host_case(logits.clone());
        let (id, upper) = if temperature > 0.0 {
            let id = host_draw(&host_probs(logits.clone(), temperature), rng.random_range(0.0..1.0));
            let p = 1.0 / (1.0 + (-z / temperature as f32).exp());
            (id, rng.random_range(0.0f32..1.0) < p)
        } else {
            (host_argmax(logits.clone()), z > 0.0)
        };
        let token = unpair(id, upper);
        out.push(VOCAB.character(token));
        let x = Tensor::<1, Int>::from_ints(pair(token), device).reshape([1, 2]);
        (logits, caches) = net.step(x, Some(caches), Some(&mut class));
    }
    out
}

#[test]
fn a_story_is_the_same_captured_eager_and_host_sampled() {
    let device = test_device();
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

/// A prompt of `len` cased tokens, different for every length. Every third
/// letter is upper case.
fn prompt(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            let id = ((i * 5 + len * 3) % VOCAB_SIZE) as u8;
            if i % 3 == 0 && id as usize >= FIRST_LETTER { id | UPPER } else { id }
        })
        .collect()
}

/// The cache of the opening and the cursors that it leaves.
fn opening(net: &VocabNetwork<RefBlock>) -> (RefCaches, ClassCursors) {
    let mut class = ClassCursors::stream();
    let (_, caches) = net.prime(1, None, Some(&mut class));
    (caches.expect("two Start latents leave a cache"), class)
}

#[test]
fn a_prefill_is_the_same_captured_and_eager_and_is_the_prompt_in_one_pass() {
    let device = test_device();
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
        let pairs: Vec<i32> = prompt.iter().flat_map(|&t| pair(t)).collect();
        let x = Tensor::<1, Int>::from_ints(pairs.as_slice(), &device).reshape([1, len, 2]);
        let (whole, whole_caches) = net.forward(x, None, (), Some(&mut ClassCursors::stream()), None);
        let rows = whole.dims()[1];
        let d = max_rel_diff(logits, whole.narrow(1, rows - 1, 1).squeeze_dim(1));
        assert!(d < dtype_tol(1e-4), "length {len}: logits differ from one pass by {d}");
        for (a, b) in prefilled.caches.iter().zip(&whole_caches.caches) {
            let d = max_abs_diff(a.state_bd.clone(), b.state_bd.clone());
            assert!(d < dtype_tol(1e-4), "length {len}: cache differs from one pass by {d}");
        }
    }
}

#[test]
fn a_prompted_story_is_the_same_captured_and_eager() {
    let device = test_device();
    let net = story_net(&device);
    let mut eager = prefill_of(&net, &device, false);
    let mut captured = prefill_of(&net, &device, true);
    for (seed, prompt) in [(0, "Once upon a time"), (1, "A"), (2, "The little dog, Max, ran home.")] {
        let run = |capture: bool, prefill: &mut Prefill<'_, RefCaches>| {
            generate(&net, &device, (), Some(prompt), 40, 0.8, seed, capture, Some(prefill))
        };
        assert_eq!(run(true, &mut captured), run(false, &mut eager), "prompt {prompt:?}");
    }
}
