//! Sampling characters from a trained character LM.
//!
//! [`generate`] is *a* sampler for any [`VocabNetwork`]: one policy, not the
//! contract. It shows the three execution modes of a library in sequence:
//!
//! 1. One [`prime`](VocabNetwork::prime) replays what the model splices in
//!    front of a sequence. It needs no input token. When there was something
//!    to replay, it answers with the distribution of the first character.
//! 2. A prompt, when there is one, goes chunkwise through
//!    [`forward`](VocabNetwork::forward) (prefill: in one pass, or in
//!    fixed-shape chunks by a [`Prefill`]).
//! 3. Every generated character then costs one [`step`](VocabNetwork::step)
//!    against the same cache: O(state) per token, with no growing KV cache.
//!
//! A model that opens sequences differently (or not at all) needs a different
//! opening, and can write one. The loops in [`lm`](super::lm) ask only for an
//! unprompted sample, never for this particular way to produce it. A model
//! with no class markers has no seedless opening here, and needs a prompt.
//!
//! One call generates **one** story. It opens the sequence, so it uses up the
//! [`ClassCursors`] that it threads. For a second story, make a second call
//! against a **reset** (zero) cache, because a story never followed another
//! story in training.
//!
//! A consumer whose network is an *enum* over families (not the generic
//! container) cannot call [`generate`]. It writes the opening over its own
//! dispatch and gives the rest to [`decode`]. [`decode`] captures the decode
//! steps into one replayed graph (see [`CapturedStep`]). A [`Prefill`] is also
//! family-agnostic.
//!
//! Characters are drawn on the device ([`sample_token`]). So decoding waits
//! for the device only to read the story back, one chunk at a time
//! ([`READBACK`]).

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
use std::cell::RefCell;
use std::rc::Rc;

/// Sample `n_chars` characters of one story. When there is a `prompt`,
/// continue it.
///
/// With `prompt: None`, the model writes from its own opening. `prime`
/// replays what the model splices in front of a sequence, and gives back the
/// distribution of the first character. So no input is necessary, and no
/// out-of-distribution input goes in (a seed character from the corpus would
/// be one). A model that splices nothing has no such opening and panics here.
/// Prompt it, or write the sampler that its opening needs.
///
/// A prompt is case-folded and filtered through the alphabet (see [`VOCAB`]),
/// and must not come out empty. The same cursors splice the opening of the
/// model in front of it, exactly as in training.
///
/// - `temperature` scales the logits before the softmax. `<= 0` samples
///   greedily (argmax).
/// - `capture` replays the decode steps from a captured graph where the model
///   allows it (see [`decode`]). It changes the speed, never the text.
/// - With a `prefill` (see [`Prefill`], worth a hold across calls), the prompt
///   follows a `prime`d opening in fixed-shape chunks, when every marker is a
///   `Start`. Without one, or with any other marker, one `forward` takes the
///   opening and the prompt.
///
/// Returns only the generated characters, not the prompt.
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
    prefill: Option<&mut Prefill<'_, M::Caches>>,
) -> String
where
    M::Options: Clone,
    M::Caches: CacheTensors,
{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // One story: the cursors open the sequence here, and go through every call
    // below. So the opening is emitted once.
    let mut class = ClassCursors::stream();

    let (logits, caches) = match prompt {
        // Prefill. Keep the cache and the logits of the last prompt character
        // (the next character is drawn from them).
        Some(prompt) => {
            let tokens = VOCAB.encode(prompt);
            assert!(
                !tokens.is_empty(),
                "the prompt has no character inside the alphabet: {prompt:?}"
            );
            // In chunks after the opening, when the opening holds every marker.
            let prefilled = match prefill {
                Some(prefill) if model.layers.only_start_latents() => prefill.run(&tokens, || {
                    let mut class = ClassCursors::stream();
                    let (_, caches) = model.prime(1, None, Some(&mut class));
                    caches.map(|caches| (caches, class))
                }),
                _ => None,
            };
            match prefilled {
                Some((logits, caches, opened)) => {
                    class = opened;
                    (logits, Some(caches))
                }
                // One chunkwise pass over the opening and the whole prompt.
                None => {
                    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
                    let input =
                        Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([1, ids.len()]);
                    let (logits, caches) =
                        model.forward(input, None, options, Some(&mut class), None);
                    let last = logits.dims()[1] - 1;
                    (logits.narrow(1, last, 1).squeeze_dim::<2>(1), Some(caches))
                }
            }
        }
        // Seedless: the opening alone, which already predicts the first
        // character.
        None => {
            let (logits, caches) = model.prime(1, None, Some(&mut class));
            (
                logits.expect(
                    "the model has no class latents to prime from. Pass a prompt.",
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

/// Decode `n_chars` characters of one story from its opening: the `logits`,
/// `caches` and `class` cursors of the opening. [`generate`] and the sampler
/// of a consumer share this loop.
///
/// The logits of the opening give the first character. Every later character
/// costs a `step` on the character before it, and is drawn from its logits on
/// the device ([`sample_token`]). The token is state, like the cache. The only
/// thing that a step takes from the host is its uniform, and `rng` draws all
/// of them at the start. The characters are read back [`READBACK`] at a time.
///
/// With `capture`, the first [`WARMUP_STEPS`] steps run eagerly. The other
/// steps replay one [`CapturedStep`] of the step and its draw, without
/// cursors. `step` must then run the same launches at every call, with no
/// class marker left to land
/// ([`Layers::only_start_latents`](crate::modules::Layers::only_start_latents)).
/// Where no hardware graph is available, the captured step runs eagerly. So
/// `capture` changes the speed, never the text.
///
/// # Safety
///
/// With `capture`, the contract of [`CapturedStep::capture`] applies: every
/// tensor that `step` reads, other than through its arguments, stays the same
/// device buffer until this function returns. This is true of a model that
/// `step` borrows.
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

    // One decode step, from the last token to the next. The token travels as a
    // float id (exact), the kind that a captured state holds.
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
            // Safety: the contract of this function covers it.
            captured = Some(unsafe {
                CapturedStep::capture(device, draw(i), state, move |u, s| advance(u, s, None))
            });
        }
        let token = match captured.as_mut() {
            Some(captured) => {
                // Copied out of the output buffer of the graph, because the
                // next replay overwrites it.
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

/// A prompt consumed into a cache `chunk` tokens at a time, with the last
/// chunk right-padded. Every chunk of every prompt has one shape. So with
/// `capture`, one graph of a chunk is recorded at the first chunk (after
/// [`WARMUP_STEPS`] eager chunks) and replayed for all later chunks, while the
/// value lives. Hold one across prompts: a capture costs a few forwards, and a
/// short prompt is a single chunk.
///
/// A prompt continues the opening that `prime` runs, and a chunk places no
/// class marker. So each chunk runs the same launches. This holds only where
/// every marker is a `Start`
/// ([`Layers::only_start_latents`](crate::modules::Layers::only_start_latents)).
/// That opening takes no input, so it is the same for every story. It runs
/// once, at the first prompt, and is kept. Each chunk comes out as its real
/// rows alone would come out, the cache included: this is the `pad` contract
/// of [`Layers::forward`](crate::modules::Layers::forward).
pub struct Prefill<'a, C: CacheTensors> {
    device: Device,
    chunk: usize,
    capture: bool,
    /// The cache of the opening and the cursors that it leaves (`None`: the
    /// model has no opening), after the first prompt ran it.
    opening: Option<Option<(C, ClassCursors)>>,
    /// The cursors that the opening left. Every chunk starts from them.
    opened: Rc<RefCell<ClassCursors>>,
    /// The chunk, until it is captured.
    run: Option<Box<ChunkFn<'a, C>>>,
    captured: Option<CapturedStep<'a, Tensor<2, Int>, Tensor<2>, C>>,
    /// The chunks run eagerly so far. [`WARMUP_STEPS`] of them precede a
    /// capture.
    eager_chunks: usize,
}

/// One chunk: its ids (`-1` at the padding) and the cache before it → the
/// logits of its last real row and the cache after it.
type ChunkFn<'a, C> = dyn FnMut(Tensor<2, Int>, C) -> (Tensor<2>, C) + 'a;

impl<'a, C: CacheTensors + 'a> Prefill<'a, C> {
    /// `forward` runs one chunk from a cache. It takes the `[1, chunk]` ids,
    /// the cache, the `pad` mask of the chunk (`true` at padding) and the
    /// cursors that the opening left. It returns the logits `[1, chunk, vocab]`
    /// and the cache after the chunk. Where no hardware graph is available,
    /// the captured chunk runs eagerly. So `capture` changes the speed, never
    /// the text.
    ///
    /// # Safety
    ///
    /// With `capture`, the contract of [`CapturedStep::capture`] applies while
    /// the value lives: every tensor that `forward` reads, other than through
    /// its arguments, stays the same device buffer. This is true of a model
    /// that `forward` borrows.
    pub unsafe fn new(
        device: &Device,
        chunk: usize,
        capture: bool,
        mut forward: impl FnMut(Tensor<2, Int>, C, Tensor<2, Bool>, &mut ClassCursors) -> (Tensor<3>, C)
        + 'a,
    ) -> Self {
        assert!(chunk > 0, "a prefill chunk holds at least one token");
        let opened = Rc::new(RefCell::new(ClassCursors::stream()));
        let run = {
            let opened = opened.clone();
            move |x: Tensor<2, Int>, caches: C| {
                // The mask, the ids and the last real row all come from `x`, on
                // the device.
                let pad = x.clone().lower_elem(0);
                let last = pad.clone().bool_not().int().sum_dim(1).sub_scalar(1).reshape([1]);
                let mut class = opened.borrow().clone();
                let (logits, caches) = forward(x.clamp_min(0), caches, pad, &mut class);
                (logits.select(1, last).squeeze_dim(1), caches)
            }
        };
        Self {
            device: device.clone(),
            chunk,
            capture,
            opening: None,
            opened,
            run: Some(Box::new(run)),
            captured: None,
            eager_chunks: 0,
        }
    }

    /// Whether chunks replay a hardware graph (`false` before the first capture,
    /// without `capture`, or where none is available).
    pub fn is_captured(&self) -> bool {
        self.captured.as_ref().is_some_and(|c| c.is_captured())
    }

    /// Consume `prompt` (token ids, not empty) after the opening. Returns the
    /// logits of its last token (`[1, vocab]`, the next token is drawn from
    /// them), the cache after it, and the cursors to continue with. `open` runs
    /// the opening at the first call only: `prime` from a zero cache and new
    /// cursors, or `None` when there is nothing to prime. With no opening,
    /// there is nothing to continue, and this returns `None`.
    pub fn run(
        &mut self,
        prompt: &[u8],
        open: impl FnOnce() -> Option<(C, ClassCursors)>,
    ) -> Option<(Tensor<2>, C, ClassCursors)> {
        assert!(!prompt.is_empty(), "an empty prompt has nothing to prefill");
        let (caches, class) = self.opening.get_or_insert_with(open).clone()?;
        *self.opened.borrow_mut() = class.clone();
        let chunk = self.chunk;
        let ids = |k: usize| {
            let mut ids = vec![-1i32; chunk];
            for (id, &token) in ids.iter_mut().zip(&prompt[k * chunk..]) {
                *id = token as i32;
            }
            TensorData::new(ids, [1, chunk])
        };
        let mut caches = Some(caches);
        let mut logits = None;
        for k in 0..prompt.len().div_ceil(chunk) {
            if self.capture && self.captured.is_none() && self.eager_chunks >= WARMUP_STEPS {
                let run = self.run.take().expect("captured once");
                let x = Tensor::from_data(ids(k), &self.device);
                let caches = caches.take().expect("run eagerly until captured");
                // Safety: the contract of `new` covers it.
                self.captured = Some(unsafe { CapturedStep::capture(&self.device, x, caches, run) });
            } else if let Some(captured) = self.captured.as_mut() {
                // The opening of a new prompt, into the buffers of the graph.
                if let Some(caches) = caches.take() {
                    captured.set_caches(caches);
                }
            }
            logits = Some(match self.captured.as_mut() {
                Some(captured) => captured.step_data(ids(k)).clone(),
                None => {
                    let run = self.run.as_mut().expect("run eagerly until captured");
                    let x = Tensor::from_data(ids(k), &self.device);
                    let (logits, next) = run(x, caches.take().expect("run eagerly until captured"));
                    caches = Some(next);
                    self.eager_chunks += 1;
                    logits
                }
            });
        }
        let logits = logits.expect("a prompt fills at least one chunk");
        Some(match self.captured.as_ref() {
            // Copied out of the buffers of the graph, because the next prompt
            // overwrites them.
            Some(captured) => {
                let whole = logits.dims().map(|d| 0..d);
                (logits.empty_like().slice_assign(whole, logits), captured.caches(), class)
            }
            None => (logits, caches.expect("run eagerly until captured"), class),
        })
    }
}

/// The characters that [`decode`] reads back per sync. Until the sync, the
/// tokens stay on the device. Every live token slows the allocator of cubecl,
/// and an eager step calls that allocator hundreds of times. A replay is bound
/// by the device, so a sync per chunk costs it almost nothing.
pub const READBACK: usize = 32;

/// Move `tokens` to the host, onto `ids`.
fn read_back(tokens: &mut Vec<Tensor<1, Int>>, ids: &mut Vec<i64>) {
    if !tokens.is_empty() {
        ids.extend(Tensor::cat(std::mem::take(tokens), 0).into_data().iter::<i64>());
    }
}

/// Draw one token from `logits` (`[1, VOCAB_SIZE]`) on the device. The token
/// is the first one whose cumulative temperature-scaled probability reaches
/// the uniform `draw` (`[1]`, in `[0, 1)`). When `temperature <= 0`, it is the
/// argmax (`draw` is not used).
///
/// The running total never decreases, so that first token is the count of
/// totals below `draw`. The count is clamped, because rounding can leave the
/// last total short of 1. The last token then takes the remainder. Nothing is
/// read back.
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
