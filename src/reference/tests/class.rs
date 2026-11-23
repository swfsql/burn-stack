//! The class markers: the class tokens of a network and the class latents of
//! a stack.
//!
//! A parity test compares the outputs and the final caches of the markers
//! with one of two references. One reference is a single forward over the
//! whole sequence. The other is a sequence that the test splices by hand, run
//! with the same weights and with no markers.

use super::*;
use crate::utils::ClassCursors;
use Splice::{Latent, Token};

/// One position of a sequence that a test splices by hand.
#[derive(Clone, Copy)]
enum Splice {
    /// The class latent `i` of the stack: row `i` of its embedding table.
    Latent(usize),
    /// The user token `t` of the input.
    Token(usize),
}

/// The sequence `order`, spliced from the class latents of `layers` and the
/// tokens of `x` `[batch, sequence, d_model]`.
fn spliced(layers: &Layers<RefBlock>, x: &Tensor<3>, order: &[Splice]) -> Tensor<3> {
    let [batch, _sequence, d_model] = x.dims();
    let emb = layers.class_latents_emb.as_ref().expect("the stack has class latents").val();
    let parts = order
        .iter()
        .map(|&part| match part {
            Latent(i) => emb.clone().narrow(0, i, 1).unsqueeze_dim::<3>(0).expand([batch, 1, d_model]),
            Token(t) => x.clone().narrow(1, t, 1),
        })
        .collect();
    Tensor::cat(parts, 1)
}

/// The position `p` of `x` `[batch, sequence, width]`, as `[batch, width]`:
/// a user token of an input, or an output.
fn position(x: &Tensor<3>, p: usize) -> Tensor<2> {
    x.clone().narrow(1, p, 1).squeeze_dim::<2>(1)
}

/// A stack of `n_real` layers with the class latents `markers`.
fn marked_layers(n_real: usize, markers: Vec<ClassLatent>, device: &Device) -> Layers<RefBlock> {
    LayersBuilder { class_latents: markers, ..layers_builder(n_real) }.init(device)
}

/// A `Start` class token makes the sequence one longer, and reports its own
/// output index. The same placement must hold whether the sequence arrives
/// through `forward` or one token at a time.
#[test]
fn class_token_placement_is_the_same_for_forward_and_step() {
    let device = test_device();
    let net = latent_network(vec![ClassToken::Start], Vec::new(), &device);

    let sequence = 4;
    let x = latent_input(1, sequence, &device);
    let (y_fwd, _) = net.forward(x.clone(), None, (), None, None);
    assert_eq!(
        y_fwd.dims(),
        [1, sequence + 1, 2],
        "the network emits the Start token before the first user token",
    );

    let mut cursors = ClassCursors::new(sequence);
    let mut caches = None;
    let mut ys = Vec::new();
    for t in 0..sequence {
        let (y_t, c) = net.step(position(&x, t), caches, Some(&mut cursors));
        caches = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }
    // `step` returns the last token that it emitted. So the first step folds
    // the marker into its state. The marker does not appear as an extra
    // output position.
    assert_eq!(Tensor::cat(ys, 1).dims(), [1, sequence, 2]);
}

/// A class latent of the stack is a learnable input: it must train.
#[test]
fn class_latents_receive_gradients() {
    let device = test_device().autodiff();
    let layers = LayersBuilder {
        class_latents: vec![ClassLatent::Start],
        ..layers_builder(2)
    }
    .init(&device);

    let x = Tensor::random([2, 4, D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    let (y, _) = layers.forward(x, None, (), None, None);
    let grads = y.sum().backward();

    let emb = layers.class_latents_emb.as_ref().expect("latent table");
    assert!(emb.val().grad(&grads).is_some());
}

/// `Start`, `Middle` and `End` class latents make the sequence longer. Each
/// one lands at the output index that the stack reports for it.
#[test]
fn class_latents_lengthen_the_sequence_at_their_output_indices() {
    let device = test_device();
    let mut layers = marked_layers(1, vec![ClassLatent::Start, ClassLatent::Middle, ClassLatent::End], &device);
    let x = randn3(2, 4, &device);
    // L = 4 ⇒ S u0 u1 M u2 u3 E.
    assert_eq!(layers.class_latent_output_indices(4), vec![0, 3, 6]);
    let reference = spliced(&layers, &x, &[Latent(0), Token(0), Token(1), Latent(1), Token(2), Token(3), Latent(2)]);

    let (y, _) = layers.forward(x, None, (), None, None);

    assert_eq!(y.dims(), [2, 7, D_MODEL]);
    layers.class_latents = Vec::new();
    let (y_ref, _) = layers.forward(reference, None, (), None, None);
    assert_same_outputs(y_ref, y, "the forward and the spliced sequence");
}

/// With no cursors, a step has no length hint. So it cannot place a `Middle`
/// latent, and it panics.
#[test]
#[should_panic(expected = "need a full-length hint")]
fn a_step_with_no_cursors_panics_on_a_middle_latent() {
    let device = test_device();
    let layers = marked_layers(1, vec![ClassLatent::Middle], &device);
    let _ = layers.step(Tensor::zeros([2, D_MODEL], &device), None, None);
}

/// An open-ended stream has no length hint. So a forward cannot place an
/// `End` latent either, and it panics.
#[test]
#[should_panic(expected = "need a full-length hint")]
fn a_forward_with_no_length_hint_panics_on_an_end_latent() {
    let device = test_device();
    let layers = marked_layers(1, vec![ClassLatent::End], &device);
    let mut class = ClassCursors::stream();
    let _ = layers.forward(randn3(2, 4, &device), None, (), Some(&mut class), None);
}

/// A prime is the opening half of a step. A prime before each step runs the
/// sequence that the steps alone run. It also returns the class latents that
/// those steps do not return. The reference is the sequence spliced by hand:
///
/// ```text
///   markers   Start, Custom(2), End                  L = 3 user tokens
///   sequence  S u0 u1 C u2 E
///   prime 0   S      → returns S            step 0   u0 → returns u0
///   prime 1   (none) → returns None         step 1   u1 → returns u1
///   prime 2   C      → returns C            step 2   u2 E → returns E
///   prime 3   (none) → returns None, because the sequence is closed
/// ```
///
/// The first prime starts from no cache. Each prime after it starts from the
/// cache of a step.
#[test]
fn prime_emits_the_class_markers_a_step_would_drop() {
    let device = test_device();
    let (batch, sequence) = (2, 3);
    let mut layers = marked_layers(2, vec![ClassLatent::Start, ClassLatent::Custom(2), ClassLatent::End], &device);
    let x = randn3(batch, sequence, &device);
    assert_eq!(layers.class_latent_output_indices(sequence), vec![0, 3, 5]);
    let reference = spliced(&layers, &x, &[Latent(0), Token(0), Token(1), Latent(1), Token(2), Latent(2)]);

    let mut class = ClassCursors::new(sequence);
    let mut caches = None;
    let mut got = Vec::new();
    for t in 0..sequence {
        let (primed, c) = layers.prime(batch, caches, Some(&mut class));
        let (y, c) = layers.step(position(&x, t), c, Some(&mut class));
        caches = Some(c);
        got.push((primed, y));
    }
    let (last, caches) = layers.prime(batch, caches, Some(&mut class));
    assert!(last.is_none(), "a prime after the closed sequence emits a latent");
    let caches = caches.expect("the steps give caches");
    // S u0 u1 C u2 E: each marker lands once.
    assert_eq!(class.stack, sequence + 3);

    layers.class_latents = Vec::new();
    let (y_ref, caches_ref) = layers.forward(reference, None, (), None, None);
    let primed_at = [Some(0), None, Some(3)]; // S, none, C
    let stepped_at = [1, 2, 5]; // u0, u1, E
    for (t, (primed, y)) in got.into_iter().enumerate() {
        match (primed, primed_at[t]) {
            (Some(primed), Some(p)) => assert_same_outputs(position(&y_ref, p), primed, &format!("prime {t}")),
            (None, None) => {}
            (primed, _) => panic!("prime {t} returns a latent: {}. The reference expects the opposite.", primed.is_some()),
        }
        assert_same_outputs(position(&y_ref, stepped_at[t]), y, &format!("step {t} after the prime {t}"));
    }
    // The primes put the markers into the recurrence, and change nothing else.
    assert_same_caches(&caches_ref, &caches, "the primes and the steps against the spliced sequence");
}

/// A vocab network primes the class latents of its stack. The prime returns
/// the logits of the last latent, through the final norm and the heads. A
/// generation with no seed picks its first token from these logits. A second
/// prime finds nothing. The prime and the steps after it give the logits of
/// the forward.
#[test]
fn a_vocab_network_prime_and_steps_equal_its_forward() {
    const VOCAB: i64 = 8;
    let device = test_device();
    let (batch, sequence) = (2, 3);
    let net = VocabNetworkBuilder {
        vocab_size: VOCAB as usize,
        pad_vocab_size_multiple: 1,
        layers: LayersBuilder { class_latents: vec![ClassLatent::Start], ..layers_builder(1) },
        missing_lm_head: true,
    }
    .init(&device);
    let pairs: Vec<i64> = (0..(batch * sequence) as i64).flat_map(|p| [(p * 5) % VOCAB, p % 2]).collect();
    let x = Tensor::<1, Int>::from_ints(pairs.as_slice(), &device).reshape([batch, sequence, 2]);

    // S u0 u1 u2: the vocab logits, then the flag logit.
    let (logits, caches_fwd) = net.forward(x.clone(), None, (), None, None);
    assert_eq!(logits.dims(), [batch, sequence + 1, VOCAB as usize + 1]);

    let mut class = ClassCursors::new(sequence);
    let (primed, caches) = net.prime(batch, None, Some(&mut class));
    assert_same_outputs(position(&logits, 0), primed.expect("the Start latent waits"), "the prime");
    let (again, mut caches) = net.prime(batch, caches, Some(&mut class));
    assert!(again.is_none(), "a second prime emits the Start latent again");
    for t in 0..sequence {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y, c) = net.step(x_t, caches, Some(&mut class));
        caches = Some(c);
        assert_same_outputs(position(&logits, t + 1), y, &format!("step {t} after the prime"));
    }
    let caches = caches.expect("the steps give caches");
    assert_same_caches(&caches_fwd, &caches, "the prime and the steps against the forward");
}

/// A forward split in two windows places each class latent where one forward
/// over the whole sequence places it. The cursors decide, not the length of
/// the window. The four kinds land on both sides of the split:
///
/// - `Start` is behind the cursor of the second window,
/// - `Custom` and `Middle` are inside the first window,
/// - `End` lands only in the window that closes the sequence.
///
/// The outputs and the final caches agree.
#[test]
fn class_markers_split_forward_matches_single_forward() {
    let device = test_device();
    let markers = vec![ClassLatent::Start, ClassLatent::Middle, ClassLatent::End, ClassLatent::Custom(1)];
    let layers = marked_layers(2, markers, &device);
    let (batch, sequence, split) = (2, 6, 4);
    let x = randn3(batch, sequence, &device);

    // With no cursors, a forward takes its input as the whole sequence.
    let (y_one, caches_one) = layers.forward(x.clone(), None, (), None, None);
    assert_eq!(y_one.dims(), [batch, sequence + 4, D_MODEL]);

    let mut class = ClassCursors::new(sequence);
    let (y_a, caches) = layers.forward(x.clone().narrow(1, 0, split), None, (), Some(&mut class), None);
    let x_b = x.narrow(1, split, sequence - split);
    let (y_b, caches_split) = layers.forward(x_b, Some(caches), (), Some(&mut class), None);

    // The first window holds Start, Custom and Middle. The second window
    // holds End.
    assert_eq!(y_a.dims(), [batch, split + 3, D_MODEL]);
    assert_eq!(y_b.dims(), [batch, sequence - split + 1, D_MODEL]);
    assert_same_outputs(y_one, Tensor::cat(vec![y_a, y_b], 1), "two windows and one forward");
    assert_same_caches(&caches_one, &caches_split, "two windows and one forward");
    // Each marker lands once.
    assert_eq!(class.stack, sequence + 4);
}

/// With a length hint, a step also places `Middle` and `End`. A `Middle`
/// opens the step of the token that it precedes. `End` follows the last user
/// token in its step, and it is the last token of the sequence. So that step
/// returns the output of `End`. Each step returns the output of the forward
/// at the last token that the step emits. The final caches agree.
#[test]
fn class_markers_step_matches_forward_with_full_len() {
    let device = test_device();
    let layers = marked_layers(2, vec![ClassLatent::Middle, ClassLatent::End], &device);
    let (batch, sequence) = (2, 4);
    let x = randn3(batch, sequence, &device);

    // The forward gives u0 u1 M u2 u3 E. The steps emit u0, then u1, then
    // M u2, then u3 E.
    let (y_fwd, caches_fwd) = layers.forward(x.clone(), None, (), None, None);
    assert_eq!(y_fwd.dims(), [batch, sequence + 2, D_MODEL]);
    let last_at = [0, 1, 3, 5];

    let mut class = ClassCursors::new(sequence);
    let mut caches = None;
    for (t, p) in last_at.into_iter().enumerate() {
        let (y, c) = layers.step(position(&x, t), caches, Some(&mut class));
        caches = Some(c);
        assert_same_outputs(position(&y_fwd, p), y, &format!("step {t} at its last token"));
    }
    assert_eq!(class.stack, sequence + 2);
    // The step of u3 also stepped `End`. So the final caches are those of the
    // whole sequence of the forward.
    assert_same_caches(&caches_fwd, &caches.expect("the steps give caches"), "the steps and the forward");
}

/// A forward and a step share the cursors. So a prefill forward can give the
/// rest of the sequence to the steps. The markers of the rest land on the
/// steps after the prefill. Each step returns the output of the one forward
/// at the last token that the step emits.
#[test]
fn class_markers_forward_then_step() {
    let device = test_device();
    let layers = marked_layers(2, vec![ClassLatent::Start, ClassLatent::Middle, ClassLatent::End], &device);
    let (batch, sequence, prefill) = (2, 4, 2);
    let x = randn3(batch, sequence, &device);

    // The forward gives S u0 u1 M u2 u3 E. The step of u2 emits M u2, and the
    // step of u3 emits u3 E.
    let (y_fwd, caches_fwd) = layers.forward(x.clone(), None, (), None, None);
    assert_eq!(y_fwd.dims(), [batch, sequence + 3, D_MODEL]);
    let last_at = [1, 2, 4, 6];

    // The prefill of u0 and u1 also emits Start.
    let mut class = ClassCursors::new(sequence);
    let (y_pre, mut caches) = layers.forward(x.clone().narrow(1, 0, prefill), None, (), Some(&mut class), None);
    assert_same_outputs(y_fwd.clone().narrow(1, 0, prefill + 1), y_pre, "the prefill");
    assert_eq!(class.stack, prefill + 1);

    for (t, p) in last_at.into_iter().enumerate().skip(prefill) {
        let (y, c) = layers.step(position(&x, t), Some(caches), Some(&mut class));
        caches = c;
        assert_same_outputs(position(&y_fwd, p), y, &format!("step {t} at its last token"));
    }
    // Middle opens the step of u2, and End closes the step of u3.
    assert_eq!(class.stack, sequence + 3);
    assert_same_caches(&caches_fwd, &caches, "the prefill and the steps against the forward");
}

/// A network places its class tokens over windows in the same way as a stack
/// places its class latents. The network splices them at its input width,
/// before `in_proj`. Two windows give one forward.
#[test]
fn class_tokens_split_forward_matches_single_forward() {
    let device = test_device();
    let net = latent_network(vec![ClassToken::Start, ClassToken::End], vec![ClassLatent::End], &device);
    let (batch, sequence, split) = (2, 5, 3);
    let x = latent_input(batch, sequence, &device);

    // The network gives S u0 u1 u2 u3 u4 E. Its stack adds its own End latent
    // after them.
    let (y_one, caches_one) = net.forward(x.clone(), None, (), None, None);
    assert_eq!(y_one.dims(), [batch, sequence + 3, 2]);

    let mut class = ClassCursors::new(sequence);
    let (y_a, caches) = net.forward(x.clone().narrow(1, 0, split), None, (), Some(&mut class), None);
    let x_b = x.narrow(1, split, sequence - split);
    let (y_b, caches_split) = net.forward(x_b, Some(caches), (), Some(&mut class), None);

    assert_same_outputs(y_one, Tensor::cat(vec![y_a, y_b], 1), "two windows and one forward");
    assert_same_caches(&caches_one, &caches_split, "two windows and one forward");
    // The cursor of the network counts its 2 class tokens. The cursor of the
    // stack also counts its End latent.
    assert_eq!(class.network, sequence + 2);
    assert_eq!(class.stack, sequence + 3);
}

/// The class tokens of a network also step, at both ends. The step of the
/// first user token first steps `Start`, as a full pass of the network. So
/// the stack places its own latents around it. The step of the last user
/// token then steps `End`. That step returns the output of `End`, because
/// `End` closes the sequence.
#[test]
fn class_tokens_step_matches_forward_with_full_len() {
    let device = test_device();
    let net = latent_network(vec![ClassToken::Start, ClassToken::End], vec![ClassLatent::Start], &device);
    let (batch, sequence) = (2, 3);
    let x = latent_input(batch, sequence, &device);

    // The forward gives L S u0 u1 u2 E. The stack places its latent `L` before
    // the first token that it gets, and that token is `S`. The steps end on
    // u0, on u1, and on E.
    let (y_fwd, caches_fwd) = net.forward(x.clone(), None, (), None, None);
    assert_eq!(y_fwd.dims(), [batch, sequence + 3, 2]);
    let last_at = [2, 3, 5];

    let mut class = ClassCursors::new(sequence);
    let mut caches = None;
    for (t, p) in last_at.into_iter().enumerate() {
        let (y, c) = net.step(position(&x, t), caches, Some(&mut class));
        caches = Some(c);
        assert_same_outputs(position(&y_fwd, p), y, &format!("step {t} at its last token"));
    }
    assert_eq!(class.network, sequence + 2);
    assert_eq!(class.stack, sequence + 3);
    assert_same_caches(&caches_fwd, &caches.expect("the steps give caches"), "the steps and the forward");
}

/// Only `End` closes a sequence. A `Custom` always precedes the token that it
/// names, at each index. The stack has two `End` latents and a `Custom(L)`
/// over `L = 3` announced tokens. The reference is the sequence spliced by
/// hand:
///
/// ```text
///   sequence   u0 u1 u2 E1 E2 [C u3]     ← C and u3 only when the caller feeds u3
///   step 0     u0            → returns u0
///   step 1     u1            → returns u1
///   step 2     u2 E1 E2      → returns E2: both End latents ran, and C did not
///   step 3     C u3          → returns u3: past the announced L, C has a token
///                               to precede, and it precedes it
/// ```
#[test]
fn end_closes_the_sequence_custom_never_does() {
    let device = test_device();
    let (batch, sequence) = (2, 3);
    // Both `End` latents are at the index `sequence`, in `Vec` order. The
    // `Custom` names the same index, but no token there follows it.
    let markers = vec![ClassLatent::End, ClassLatent::End, ClassLatent::Custom(sequence)];
    let mut layers = marked_layers(2, markers, &device);
    // One token more than the announced length. The last step feeds it.
    let x = randn3(batch, sequence + 1, &device);
    let order = [Token(0), Token(1), Token(2), Latent(0), Latent(1), Latent(2), Token(3)];
    let reference = spliced(&layers, &x, &order);

    // `Custom(sequence)` reports a position past the emitted sequence. So it
    // does not land, and the forward over the announced tokens gives
    // u0 u1 u2 E1 E2.
    assert_eq!(layers.class_latent_output_indices(sequence), vec![3, 4, 5]);
    let (y_marked, _) = layers.forward(x.clone().narrow(1, 0, sequence), None, (), None, None);
    assert_eq!(y_marked.dims(), [batch, sequence + 2, D_MODEL]);

    // Step all four tokens. The last one is past the announced length.
    let mut class = ClassCursors::new(sequence);
    let mut caches = None;
    let mut got = Vec::new();
    for t in 0..sequence + 1 {
        let (y, c) = layers.step(position(&x, t), caches, Some(&mut class));
        got.push((y, class.stack));
        caches = Some(c);
    }
    // u0 u1 u2 E1 E2 C u3: each marker lands once, and `Custom` lands last.
    assert_eq!(class.stack, sequence + 4);

    layers.class_latents = Vec::new();
    let (y_ref, caches_ref) = layers.forward(reference, None, (), None, None);
    assert_same_outputs(y_ref.clone().narrow(1, 0, sequence + 2), y_marked, "the forward and the spliced sequence");

    let last_at = [0, 1, 4, 6]; // u0, u1, E2, u3
    let dropped_at = [0, 1, 3, 5]; // u0, u1, E1, C
    for (t, (y, cursor)) in got.into_iter().enumerate() {
        assert_same_outputs(position(&y_ref, last_at[t]), y.clone(), &format!("step {t} at its last token"));
        if dropped_at[t] != last_at[t] {
            let diff = max_abs_diff(y, position(&y_ref, dropped_at[t]));
            assert!(diff > dtype_tol(1e-3), "step {t} returns the output of a token before its last token");
        }
        assert_eq!(cursor, last_at[t] + 1, "the cursor after step {t}");
    }
    // Each token of the reference ran through the recurrence.
    assert_same_caches(&caches_ref, &caches.expect("the steps give caches"), "the steps and the spliced sequence");
}

/// The output and the cache of a step, for each marker kind, against the
/// sequence spliced by hand. The test splices the class rows into the input
/// where the four kinds land. Then it runs the same weights with no markers
/// over that sequence:
///
/// ```text
///   markers   Start, Custom(1), Middle, End         L = 4 user tokens
///   sequence  S u0 C u1 M u2 u3 E
///   step 0    S u0   → returns u0, the cache after 2 tokens of the reference
///   step 1    C u1   → returns u1, the cache after 4
///   step 2    M u2   → returns u2, the cache after 6
///   step 3    u3 E   → returns E (not u3), the cache after 8
/// ```
///
/// `Start`, `Middle` and a `Custom(k)` with `k < L` open the step of the token
/// that they precede, so the user token ends that step. `End` closes the
/// sequence and ends its step. So a step that emits `End` returns the output
/// of `End`, and the cache after it.
#[test]
fn step_output_and_state_follow_the_last_emitted_token() {
    let device = test_device();
    let (batch, sequence) = (2, 4);
    let markers = vec![ClassLatent::Start, ClassLatent::Middle, ClassLatent::End, ClassLatent::Custom(1)];
    let mut layers = marked_layers(2, markers, &device);
    let x = randn3(batch, sequence, &device);
    // S u0 C u1 M u2 u3 E. The latents are in `Vec` order: Start 0, Middle 1,
    // End 2, Custom 3.
    let order = [Latent(0), Token(0), Latent(3), Token(1), Latent(1), Token(2), Token(3), Latent(2)];
    let reference = spliced(&layers, &x, &order);
    assert_eq!(reference.dims(), [batch, sequence + 4, D_MODEL]);

    // Step the four user tokens. Keep the output, the cache and the cursor of
    // each step.
    let mut class = ClassCursors::new(sequence);
    let mut caches = None;
    let mut got = Vec::new();
    for t in 0..sequence {
        let (y, c) = layers.step(position(&x, t), caches, Some(&mut class));
        got.push((y, c.clone(), class.stack));
        caches = Some(c);
    }

    // The forward with the markers gives the spliced sequence.
    let (y_marked, caches_marked) = layers.forward(x, None, (), None, None);
    layers.class_latents = Vec::new();
    let (y_ref, caches_ref) = layers.forward(reference.clone(), None, (), None, None);
    assert_same_outputs(y_ref.clone(), y_marked, "the forward and the spliced sequence");
    assert_same_caches(&caches_ref, &caches_marked, "the forward and the spliced sequence");

    let last_at = [1, 3, 5, 7]; // u0, u1, u2, End
    let dropped_at = [0, 2, 4, 6]; // Start, Custom, Middle, u3
    let consumed = [2, 4, 6, 8]; // the tokens of the reference after each step
    for (t, (y, step_caches, cursor)) in got.into_iter().enumerate() {
        assert_eq!(cursor, consumed[t], "the cursor after step {t}");
        assert_same_outputs(position(&y_ref, last_at[t]), y.clone(), &format!("step {t} at its last token"));
        let diff = max_abs_diff(y, position(&y_ref, dropped_at[t]));
        assert!(diff > dtype_tol(1e-3), "step {t} returns the output of a token before its last token");
        // The cache is that of the reference after these tokens. So the
        // closing `End` of step 3 ran through the recurrence.
        let (_, want) = layers.forward(reference.clone().narrow(1, 0, consumed[t]), None, (), None, None);
        assert_same_caches(&want, &step_caches, &format!("step {t}"));
    }
}
