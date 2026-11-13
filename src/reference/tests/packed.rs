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

/// The positions before the first reset of a row continue from the incoming
/// cache. So a stream of packed calls can continue a segment.
#[test]
fn positions_before_the_first_reset_continue_from_the_cache() {
    let device = test_device();
    let block = block_config().init(&device);
    let (_, cache) = block.block_forward(randn3(1, 2, &device), None, (), None);
    let packed = Packed::from_starts(&[vec![3]], 1, 0, 6, &device);
    let x = randn3(1, 6, &device);

    let (y, _) = block.block_forward_packed(x.clone(), Some(cache.clone()), (), packed.reset_bs);
    let (prefix, _) = block.block_forward(x.narrow(1, 0, 3), Some(cache), (), None);

    assert!(max_abs_diff(y.narrow(1, 0, 3), prefix) < dtype_tol(TOL));
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
