//! Packed rows: each segment against the segment alone, and the checks of
//! the layout.

use super::*;

/// The block gives the reset alignment of its config, which is 1 by default.
#[test]
fn the_block_gives_the_reset_alignment_of_its_config() {
    let device = test_device();
    assert_eq!(block_config().init(&device).reset_align(&()), Some(1));
    assert_eq!(block_config().with_reset_align(4).init(&device).reset_align(&()), Some(4));
}

/// A start that the block does not accept is a caller error. The check reads
/// only host integers, so it runs in every build.
#[test]
#[should_panic(expected = "not a multiple of the reset alignment")]
fn a_start_off_the_reset_alignment_panics() {
    let _ = Packed::from_starts(&[vec![0, 3]], 2, 0, 8, &test_device());
}

/// A table with one row per class latent fills exactly the `lead` opening
/// slots of a segment. Any other row count is a caller error.
#[test]
#[should_panic(expected = "one row per opening slot")]
fn a_lead_that_differs_from_the_table_panics() {
    let device = test_device();
    let layers = LayersBuilder {
        class_latents: vec![ClassLatent::Start],
        ..layers_builder(1)
    }
    .init(&device);
    let packed = Packed::from_starts(&[vec![0]], 1, 2, 4, &device);
    let _ = layers.forward_packed(randn3(1, 4, &device), None, (), &packed);
}

/// A stack with no class latents has no table to put into the opening slots.
/// So a layout that reserves them is a caller error.
#[test]
#[should_panic(expected = "no class latents")]
fn a_lead_on_a_stack_with_no_class_latents_panics() {
    let device = test_device();
    let packed = Packed::from_starts(&[vec![0]], 1, 1, 4, &device);
    let _ = layers(1, &device).forward_packed(randn3(1, 4, &device), None, (), &packed);
}

/// The packed layout of two rows for `block`, with its starts and the width of
/// the rows. Each start is a multiple of the reset alignment of `block`, and
/// each segment has at least 3 positions.
fn packed_layout(block: &RefBlock, lead: usize, device: &Device) -> (Packed, Vec<Vec<usize>>, usize) {
    let align = block.reset_align(&()).expect("this block must take packed rows");
    let starts = vec![vec![0, 3 * align, 8 * align], vec![0, 5 * align]];
    let width = 11 * align;
    (Packed::from_starts(&starts, align, lead, width, device), starts, width)
}

/// Each segment of `starts` as `(row, start, end)`. A segment ends at the next
/// start or at the end of the row. So the gap after a segment runs as a part
/// of that segment, and the tests compare every position.
fn packed_segments(starts: &[Vec<usize>], width: usize) -> Vec<(usize, usize, usize)> {
    let mut segments = Vec::new();
    for (b, row) in starts.iter().enumerate() {
        for (i, &start) in row.iter().enumerate() {
            segments.push((b, start, row.get(i + 1).copied().unwrap_or(width)));
        }
    }
    segments
}

/// The part `[start, end)` of row `b`, with the batch axis kept.
fn segment_of(tensor: Tensor<3>, b: usize, start: usize, end: usize) -> Tensor<3> {
    tensor.narrow(0, b, 1).narrow(1, start, end - start)
}

/// A layer over packed rows gives each segment the outputs that the segment
/// gives alone from a zero cache. The last segment of a row gives the cache.
#[test]
fn layer_forward_packed_equals_each_segment_alone() {
    let device = test_device();
    let layers = LayersBuilder {
        block: block_config().with_reset_align(3),
        mlp: Some(GatedMlpConfig::new(D_MODEL, D_MODEL * 2)),
        ..layers_builder(1)
    }
    .init(&device);
    let layer = &layers.real_layers[0];
    let (packed, starts, width) = packed_layout(&layer.block, 0, &device);
    let x = randn3(2, width, &device);

    let (y, cache) = layer.forward_packed(x.clone(), None, (), &packed);

    for (b, start, end) in packed_segments(&starts, width) {
        let (alone, alone_cache) = layer.forward(segment_of(x.clone(), b, start, end), None, (), None);
        assert!(max_abs_diff(segment_of(y.clone(), b, start, end), alone) < dtype_tol(TOL));
        if end == width {
            let last = cache.state_bd.clone().narrow(0, b, 1);
            assert!(max_abs_diff(last, alone_cache.state_bd) < dtype_tol(TOL));
        }
    }
}

/// A stack over packed rows gives each segment the outputs that the segment
/// gives alone, with the class latents in its opening slots. Alone, `forward`
/// splices the same latents in front of the segment. The layout also crosses
/// the cuts of a gradient horizon, and a class latent still trains.
#[test]
fn layers_forward_packed_equals_each_segment_alone() {
    let device = test_device().autodiff();
    let lead = 2;
    let layers = LayersBuilder {
        block: block_config().with_reset_align(3),
        class_latents: vec![ClassLatent::Start; lead],
        grad_horizon: Some(GradHorizon::Mask(vec![true, false, true])),
        ..layers_builder(3)
    }
    .init(&device);
    let (packed, starts, width) = packed_layout(&layers.real_layers[0].block, lead, &device);
    let x = randn3(2, width, &device);

    let (y, caches) = layers.forward_packed(x.clone(), None, (), &packed);

    for (b, start, end) in packed_segments(&starts, width) {
        let tokens = segment_of(x.clone(), b, start + lead, end);
        let (alone, alone_caches) = layers.forward(tokens, None, (), None, None);
        assert!(max_abs_diff(segment_of(y.clone(), b, start, end), alone) < dtype_tol(TOL));
        if end == width {
            for (cache, alone_cache) in caches.caches.iter().zip(alone_caches.caches) {
                let last = cache.state_bd.clone().narrow(0, b, 1);
                assert!(max_abs_diff(last, alone_cache.state_bd) < dtype_tol(TOL));
            }
        }
    }

    let grads = y.sum().backward();
    let emb = layers.class_latents_emb.as_ref().expect("latent table");
    assert!(emb.val().grad(&grads).is_some(), "a class latent trains from its slots");
}

/// A vocab network over packed rows gives each segment the logits that the
/// segment gives alone. The stack does not read the token of an opening slot.
#[test]
fn vocab_network_forward_packed_equals_each_segment_alone() {
    const VOCAB: i64 = 11;
    let device = test_device();
    let lead = 2;
    let net = VocabNetworkBuilder {
        vocab_size: VOCAB as usize,
        pad_vocab_size_multiple: 1,
        layers: LayersBuilder {
            block: block_config().with_reset_align(3),
            class_latents: vec![ClassLatent::Start; lead],
            ..layers_builder(2)
        },
        missing_lm_head: true,
    }
    .init(&device);
    let (packed, starts, width) = packed_layout(&net.layers.real_layers[0].block, lead, &device);
    let pairs: Vec<i64> = (0..2 * width as i64).flat_map(|p| [(p * 7) % VOCAB, p % 2]).collect();
    let x = Tensor::<1, Int>::from_ints(pairs.as_slice(), &device).reshape([2, width, 2]);

    let (logits, _) = net.forward_packed(x.clone(), None, (), &packed);

    for (b, start, end) in packed_segments(&starts, width) {
        let tokens = x.clone().narrow(0, b, 1).narrow(1, start + lead, end - start - lead);
        let (alone, _) = net.forward(tokens, None, (), None, None);
        assert!(max_rel_diff(segment_of(logits.clone(), b, start, end), alone) < dtype_tol(TOL));
    }
}

/// The real tokens of each segment of each row of
/// [`a_packed_vocab_network_has_the_logits_and_gradients_of_each_segment_alone`].
const SEGMENT_LENS: [&[usize]; 2] = [&[5, 9, 1], &[12, 3]];

/// A packed vocab network gives each segment the logits and the gradients
/// that the segment gives alone. Each segment starts at a multiple of the
/// reset alignment (4), with 2 class latents in its opening slots. A segment
/// can be shorter or longer than the alignment, and a gap follows each one.
/// The check compares:
///
/// - the logits of each opening slot and of each real token of each segment,
/// - the gradients of each parameter, of one loss over those logits.
///
/// It runs over the Standard residuals and over Multi-Gate Residuals.
#[test]
fn a_packed_vocab_network_has_the_logits_and_gradients_of_each_segment_alone() {
    const VOCAB: usize = 11;
    let (align, lead, batch) = (4, 2, SEGMENT_LENS.len());
    let device = test_device().autodiff();
    let multi_gate = ResidualsConfig::MultiGate {
        n_stream: 2,
        init_bias: 0.0,
        init_bias_offset: 0.0,
        per_virtual_layer: false,
    };

    // The first position of each segment, and the width of the rows. A free
    // tail of `align` positions follows the last segment of the longest row.
    let mut width = 0;
    let starts: Vec<Vec<usize>> = SEGMENT_LENS
        .iter()
        .map(|lens| {
            let mut at = 0;
            let starts = lens
                .iter()
                .map(|&len| {
                    let start = at;
                    at = (start + lead + len).next_multiple_of(align);
                    start
                })
                .collect();
            width = width.max(at + align);
            starts
        })
        .collect();
    // Token `j` of segment `s` of row `b` is the pair (id, `j mod 2`).
    let pairs = |b: usize, s: usize, len: usize| -> Vec<i64> {
        (0..len).flat_map(|j| [((7 * (31 * b + 13 * s + j) + 3) % VOCAB) as i64, (j % 2) as i64]).collect()
    };
    // Token 0 fills the opening slots and the gaps.
    let mut ids = vec![0i64; batch * width * 2];
    for (b, lens) in SEGMENT_LENS.iter().enumerate() {
        for (s, (&len, &start)) in lens.iter().zip(&starts[b]).enumerate() {
            let at = (b * width + start + lead) * 2;
            ids[at..at + 2 * len].copy_from_slice(&pairs(b, s, len));
        }
    }
    let ids = Tensor::<1, Int>::from_ints(ids.as_slice(), &device).reshape([batch, width, 2]);
    let packed = Packed::from_starts(&starts, align, lead, width, &device);
    // A fixed weight for each logit, so that the loss reaches each one.
    let weight = Tensor::<3>::random([batch, width, VOCAB + 1], Distribution::Normal(0.0, 1.0), &device);

    for residuals in [ResidualsConfig::Standard, multi_gate] {
        let label = match residuals {
            ResidualsConfig::Standard => "Standard",
            _ => "Multi-Gate",
        };
        let net = VocabNetworkBuilder {
            vocab_size: VOCAB,
            pad_vocab_size_multiple: 1,
            layers: LayersBuilder {
                block: block_config().with_reset_align(align),
                class_latents: vec![ClassLatent::Start; lead],
                residuals,
                ..layers_builder(2)
            },
            missing_lm_head: false,
        }
        .init(&device);

        let (logits, _) = net.forward_packed(ids.clone(), None, (), &packed);
        let mut loss_packed = Tensor::<1>::zeros([1], &device);
        let mut loss_alone = Tensor::<1>::zeros([1], &device);
        for (b, lens) in SEGMENT_LENS.iter().enumerate() {
            for (s, (&len, &start)) in lens.iter().zip(&starts[b]).enumerate() {
                let segment = Tensor::<1, Int>::from_ints(pairs(b, s, len).as_slice(), &device).reshape([1, len, 2]);
                let (alone, _) = net.forward(segment, None, (), None, None);
                let in_row = segment_of(logits.clone(), b, start, start + lead + len);
                assert_same_outputs(alone.clone(), in_row.clone(), &format!("{label}, row {b}, the segment at {start}"));
                let w = segment_of(weight.clone(), b, start, start + lead + len);
                loss_packed = loss_packed + (in_row * w.clone()).sum();
                loss_alone = loss_alone + (alone * w).sum();
            }
        }
        // Under Multi-Gate, the β gate of the first layer can get no gradient.
        assert_matching_grads(&net, &loss_alone.backward(), &loss_packed.backward(), label);
    }
}

/// `place` puts latent `k` into slot `k` of each segment, and it keeps every
/// other position.
#[test]
fn place_puts_each_class_latent_into_its_slots() {
    let device = test_device();
    let packed = Packed::from_starts(&[vec![0, 3], vec![2]], 1, 2, 6, &device);
    let x = Tensor::<3>::ones([2, 6, D_MODEL], &device);
    let table = Tensor::<2>::from_floats([[2.0; D_MODEL], [3.0; D_MODEL]], &device);
    let want = Tensor::<2>::from_floats(
        [[2.0, 3.0, 1.0, 2.0, 3.0, 1.0], [1.0, 1.0, 2.0, 3.0, 1.0, 1.0]],
        &device,
    )
    .unsqueeze_dim::<3>(2)
    .expand([2, 6, D_MODEL]);

    assert_eq!(max_abs_diff(packed.place(x, table), want), 0.0);
}
