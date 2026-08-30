//! Sampling characters from a trained character LM.
//!
//! [`generate`] is the whole sampler for any [`VocabNetwork`], and shows a
//! library's two execution modes back to back: the prompt is consumed by one
//! chunkwise [`forward`](VocabNetwork::forward) (prefill), and every generated
//! character then costs one [`step`](VocabNetwork::step) against the same cache
//! — O(state) per token, with no growing KV cache.
//!
//! A consumer whose network is an *enum* over families (rather than the generic
//! container) cannot call [`generate`]; it writes the same loop over its own
//! dispatch and reuses [`sample_token`].

use crate::examples::device::FloatElement;
use crate::examples::tiny_stories::dataset::{VOCAB, VOCAB_SIZE};
use crate::modules::{Block, VocabNetwork};
use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn::tensor::activation::softmax;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Continue `prompt` with `n_chars` sampled characters.
///
/// The prompt is case-folded and filtered through the alphabet (see [`VOCAB`])
/// and must not come out empty. `temperature` scales the logits before the
/// softmax; `<= 0` samples greedily (argmax). Returns only the generated
/// continuation, not the prompt.
pub fn generate<M: Block>(
    model: &VocabNetwork<M>,
    device: &Device,
    options: M::Options,
    prompt: &str,
    n_chars: usize,
    temperature: f64,
    seed: u64,
) -> String
where
    M::Options: Clone,
{
    let tokens = VOCAB.encode(prompt);
    assert!(
        !tokens.is_empty(),
        "the prompt has no character inside the alphabet: {prompt:?}"
    );
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Prefill: one chunkwise pass over the whole prompt, keeping its cache and
    // the logits of its last character (what the next character is drawn from).
    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let prompt_len = ids.len();
    let input = Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([1, prompt_len]);
    let (logits, caches) = model.forward(input, None, options, None);
    let mut logits = logits.narrow(1, prompt_len - 1, 1).squeeze_dim::<2>(1); // [1, VOCAB_SIZE]
    let mut caches = Some(caches);

    // Decode: one `step` per character, against that same cache.
    let mut out = String::with_capacity(n_chars);
    for _ in 0..n_chars {
        let token = sample_token(logits, temperature, &mut rng);
        out.push(VOCAB.character(token));
        let next = Tensor::<1, Int>::from_ints([token as i32], device);
        let (next_logits, next_caches) = model.step(next, caches.take(), None);
        logits = next_logits;
        caches = Some(next_caches);
    }
    out
}

/// Draw one token from `logits` (`[1, VOCAB_SIZE]`): temperature-scaled
/// multinomial sampling, or argmax when `temperature <= 0`.
pub fn sample_token(logits: Tensor<2>, temperature: f64, rng: &mut ChaCha8Rng) -> u8 {
    assert_eq!([1, VOCAB_SIZE], logits.dims());
    if temperature <= 0.0 {
        let best = logits.argmax(1).into_data().try_to_vec::<i32>().unwrap();
        return best[0] as u8;
    }
    let probs = to_host(softmax(logits / temperature, 1));
    let threshold: f32 = rng.random_range(0.0..1.0);
    let mut cumulative = 0.0;
    for (token, p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= threshold {
            return token as u8;
        }
    }
    // Only reachable when the probabilities sum to slightly under 1 (rounding).
    (VOCAB_SIZE - 1) as u8
}

/// Read a float tensor back to a host `Vec<f32>` (dtype-agnostic).
pub fn to_host<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data()
        .try_to_vec::<FloatElement>()
        .unwrap()
        .into_iter()
        .map(|x| x.elem::<f32>())
        .collect()
}
