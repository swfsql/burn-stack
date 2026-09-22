//! Sampling characters from a trained character LM.
//!
//! [`generate`] is *a* sampler for any [`VocabNetwork`] — one policy, not the
//! contract — and shows a library's three execution modes back to back: whatever
//! the model splices in front of a sequence is replayed by one
//! [`prime`](VocabNetwork::prime) (which needs no input token, and answers with
//! the first character's distribution when there was anything to replay), a
//! prompt — when there is one — is consumed by one chunkwise
//! [`forward`](VocabNetwork::forward) (prefill), and every generated character
//! then costs one [`step`](VocabNetwork::step) against the same cache — O(state)
//! per token, with no growing KV cache.
//!
//! A model that opens sequences differently — or not at all — wants a different
//! opening, and is free to write one: the loops in [`lm`](super::lm) ask only for
//! an unprompted sample, never for this particular way of producing it. A model
//! with no class markers has no seedless opening here and must be prompted.
//!
//! One call generates **one** story: it opens the sequence, so the
//! [`ClassCursors`] it threads are used up. A second story wants a second call,
//! against a **reset** (zero) cache, since a story never followed another in
//! training.
//!
//! A consumer whose network is an *enum* over families (rather than the generic
//! container) cannot call [`generate`]; it writes the opening over its own
//! dispatch and hands the rest to [`decode`], which is where the decode steps
//! are captured into one replayed graph (see
//! [`CapturedStep`]).
//!
//! Characters are drawn on the device ([`sample_token`]), so decoding waits for
//! it only to read the story back, a chunk at a time ([`READBACK`]).

#[cfg(test)]
mod tests;

use crate::examples::tiny_stories::dataset::{VOCAB, VOCAB_SIZE};
use crate::modules::{Block, CacheTensors, VocabNetwork};
use crate::utils::ClassCursors;
use crate::utils::graph::{CapturedStep, WARMUP_STEPS};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::activation::softmax;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Sample `n_chars` characters of one story, continuing `prompt` when there is
/// one.
///
/// With `prompt: None` the model writes from its own opening: `prime` replays
/// whatever it splices in front of a sequence and hands back the distribution of
/// the first character, so nothing has to be fed in — and nothing
/// out-of-distribution is, which a seed character taken from the corpus would be.
/// A model that splices nothing has no such opening and panics here; prompt it,
/// or write the sampler its opening calls for.
///
/// A prompt is case-folded and filtered through the alphabet (see [`VOCAB`]) and
/// must not come out empty; anything the model opens with is spliced in front of
/// it by the same cursors, exactly as in training. `temperature` scales the
/// logits before the softmax; `<= 0` samples greedily (argmax). `capture`
/// replays the decode steps from a captured graph where the model allows it
/// (see [`decode`]); it changes the speed, never the text. Returns only the
/// generated characters, not the prompt.
#[allow(clippy::too_many_arguments)]
pub fn generate<M: Block>(
    model: &VocabNetwork<M>,
    device: &Device,
    options: M::Options,
    prompt: Option<&str>,
    n_chars: usize,
    temperature: f64,
    seed: u64,
    capture: bool,
) -> String
where
    M::Options: Clone,
    M::Caches: CacheTensors,
{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // One story: the cursors open the sequence here and are threaded through
    // every call below, so the opening is emitted once.
    let mut class = ClassCursors::stream();

    let (logits, caches) = match prompt {
        // Prefill: one chunkwise pass over the opening and the whole prompt,
        // keeping its cache and the logits of its last character (what the next
        // character is drawn from).
        Some(prompt) => {
            let tokens = VOCAB.encode(prompt);
            assert!(
                !tokens.is_empty(),
                "the prompt has no character inside the alphabet: {prompt:?}"
            );
            let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let input = Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([1, ids.len()]);
            let (logits, caches) = model.forward(input, None, options, Some(&mut class), None);
            let last = logits.dims()[1] - 1;
            (logits.narrow(1, last, 1).squeeze_dim::<2>(1), Some(caches))
        }
        // Seedless: the opening alone, which already predicts the first
        // character.
        None => {
            let (logits, caches) = model.prime(1, None, Some(&mut class));
            (
                logits.expect(
                    "the model has no class latents to prime from; pass a prompt instead",
                ),
                caches,
            )
        }
    };

    // Decode: one `step` per character, against that same cache.
    let caches = caches.expect("the opening leaves a cache");
    let capture = capture && model.layers.only_start_latents();
    // Safety: the step reads nothing but its arguments and `model`, which it
    // borrows for the whole call.
    unsafe {
        decode(
            device,
            logits,
            caches,
            class,
            n_chars,
            temperature,
            &mut rng,
            capture,
            |x, caches, class| model.step(x, Some(caches), class),
        )
    }
}

/// Decode `n_chars` characters of one story from its opening — the opening's
/// `logits`, `caches` and `class` cursors — the loop [`generate`] and a
/// consumer's own sampler share.
///
/// The opening's logits give the first character, and every later one costs a
/// `step` on the one before, drawn from its logits on the device
/// ([`sample_token`]): the token is state, like the cache, and the only thing a
/// step takes from the host is its uniform, all of which `rng` draws up front.
/// The characters are read back [`READBACK`] at a time.
///
/// With `capture`, the first [`WARMUP_STEPS`] steps run eagerly and the rest
/// replay one [`CapturedStep`] of the step and its draw, without cursors. `step`
/// must then run the same launches at every call — no class marker left to land
/// ([`Layers::only_start_latents`](crate::modules::Layers::only_start_latents)).
/// Where no hardware graph is available the captured step runs eagerly, so
/// `capture` changes the speed, never the text.
///
/// # Safety
///
/// With `capture`, that of [`CapturedStep::capture`]: every tensor `step`
/// reads other than through its arguments stays the same device buffer until
/// this returns — true of a model it borrows.
#[allow(clippy::too_many_arguments)]
pub unsafe fn decode<C: CacheTensors>(
    device: &Device,
    logits: Tensor<2>,
    caches: C,
    mut class: ClassCursors,
    n_chars: usize,
    temperature: f64,
    rng: &mut ChaCha8Rng,
    capture: bool,
    mut step: impl FnMut(Tensor<1, Int>, C, Option<&mut ClassCursors>) -> (Tensor<2>, C),
) -> String {
    if n_chars == 0 {
        return String::new();
    }
    // One uniform per character (none are used when greedy).
    let draws: Vec<f32> = (0..n_chars)
        .map(|_| if temperature > 0.0 { rng.random_range(0.0..1.0) } else { 0.0 })
        .collect();
    let draw = |i: usize| Tensor::<1>::from_floats([draws[i]], device);

    // One decode step, from the last token to the next. The token rides as a
    // float id (exact), the kind a captured state holds.
    let advance = move |u: Tensor<1>,
                        (caches, token): (C, Tensor<1>),
                        class: Option<&mut ClassCursors>| {
        let (logits, caches) = step(token.int(), caches, class);
        let token = sample_token(logits, temperature, u);
        (token.clone(), (caches, token.float()))
    };

    let first = sample_token(logits, temperature, draw(0));
    let mut tokens = Vec::with_capacity(READBACK);
    tokens.push(first.clone());
    let (mut state, mut advance) = (Some((caches, first.float())), Some(advance));
    let mut captured = None;
    let mut ids = Vec::with_capacity(n_chars);
    for i in 1..n_chars {
        if capture && i == 1 + WARMUP_STEPS {
            let mut advance = advance.take().expect("captured once");
            let state = state.take().expect("stepped eagerly until captured");
            // Safety: forwarded from this function's own contract.
            captured = Some(unsafe {
                CapturedStep::capture(device, draw(i), state, move |u, s| advance(u, s, None))
            });
        }
        let token = match captured.as_mut() {
            Some(captured) => {
                // Copied out of the graph's output buffer, which the next
                // replay overwrites.
                let token = captured.step_data(TensorData::from([draws[i]]));
                token.empty_like().slice_assign([0..1], token.clone())
            }
            None => {
                let advance = advance.as_mut().expect("stepped eagerly until captured");
                let current = state.take().expect("stepped eagerly until captured");
                let (token, next) = advance(draw(i), current, Some(&mut class));
                state = Some(next);
                token
            }
        };
        tokens.push(token);
        if tokens.len() == READBACK {
            read_back(&mut tokens, &mut ids);
        }
    }
    read_back(&mut tokens, &mut ids);
    ids.into_iter().map(|id| VOCAB.character(id as u8)).collect()
}

/// Characters [`decode`] reads back per sync. Until then the tokens stay on the
/// device, and every live one slows cubecl's allocator, which an eager step
/// calls hundreds of times; a replay, bound by the device, barely feels a sync
/// per chunk.
pub const READBACK: usize = 32;

/// Move `tokens` to the host, onto `ids`.
fn read_back(tokens: &mut Vec<Tensor<1, Int>>, ids: &mut Vec<i64>) {
    if !tokens.is_empty() {
        ids.extend(Tensor::cat(std::mem::take(tokens), 0).into_data().iter::<i64>());
    }
}

/// Draw one token from `logits` (`[1, VOCAB_SIZE]`) on the device: the first
/// whose cumulative temperature-scaled probability reaches the uniform `draw`
/// (`[1]`, in `[0, 1)`), or the argmax when `temperature <= 0` (`draw` unused).
///
/// The running total never decreases, so that first token is the count of
/// totals below `draw` — clamped, since rounding can leave the last total short
/// of 1, and the last token then takes the remainder. Nothing is read back.
pub fn sample_token(logits: Tensor<2>, temperature: f64, draw: Tensor<1>) -> Tensor<1, Int> {
    assert_eq!([1, VOCAB_SIZE], logits.dims());
    if temperature <= 0.0 {
        return logits.argmax(1).reshape([1]);
    }
    let cumulative = softmax(logits / temperature, 1).cumsum(1);
    let draw = draw.reshape([1, 1]).expand([1, VOCAB_SIZE]);
    let below = cumulative.lower(draw).int().sum_dim(1);
    below.clamp_max(VOCAB_SIZE as i64 - 1).reshape([1])
}
