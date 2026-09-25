//! The packer and the packed batcher, on the host:
//!
//! - [`pack_rows`] places every story exactly once, at aligned starts, and no
//!   row overflows. One open row keeps the order of the stories.
//! - [`PackedStoriesBatcher`] lays out the slots, the inputs, the targets and
//!   the scored positions of each story where the packer put it.

use super::{PackLayout, PackedItem, PackedStoriesBatcher, pack_rows};
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

/// Story lengths (tokens) from a fixed linear congruential sequence.
fn lens(n: usize, max: usize) -> Vec<usize> {
    let mut x: u64 = 12345;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            2 + (x >> 33) as usize % (max - 1)
        })
        .collect()
}

#[test]
fn pack_rows_places_every_story_once_and_fits() {
    let layout = PackLayout { align: 4, lead: 2 };
    let width = 64;
    let lens = lens(300, width - layout.lead + 1);
    let order: Vec<usize> = (0..lens.len()).rev().collect();
    for open_rows in [1, 3, 8] {
        let rows = pack_rows(&lens, &order, width, layout, open_rows);
        let mut seen = vec![0; lens.len()];
        for row in &rows {
            assert!(!row.is_empty(), "no empty row");
            let mut used = 0;
            for &story in row {
                seen[story] += 1;
                let end = used + layout.positions(lens[story]);
                assert!(end <= width, "a row overflows: {end} > {width}");
                used = end.next_multiple_of(layout.align);
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "every story in exactly one row");
        if open_rows == 1 {
            let flat: Vec<usize> = rows.concat();
            assert_eq!(flat, order, "one open row keeps the order");
        }
    }
}

#[test]
fn packed_batcher_lays_out_each_story() {
    let layout = PackLayout { align: 4, lead: 2 };
    let width = 16;
    // Row 0: a story of 3 tokens at 0 (slots 0..2, inputs 2..4, next at 4).
    // Then a story of 5 tokens at 4 (slots 4..6, inputs 6..10). Row 1: one
    // story of 2 tokens at 0.
    let items = vec![
        PackedItem {
            stories: vec![vec![10, 11, 12], vec![20, 21, 22, 23, 24]],
        },
        PackedItem {
            stories: vec![vec![30, 31]],
        },
    ];
    let device = Device::default();
    let batch = PackedStoriesBatcher::new(width, layout).batch(items, &device);
    let packed = batch.packed.expect("a packed batch");
    let ints = |t: Tensor<2, Int>| t.into_data().convert::<i64>().try_to_vec::<i64>().unwrap();
    let bools = |t: Tensor<2, Bool>| t.into_data().try_to_vec::<bool>().unwrap();

    let mut inputs = vec![0i64; 2 * width];
    let mut targets = vec![0i64; 2 * width];
    let mut score = vec![false; 2 * width];
    let mut reset = vec![false; 2 * width];
    let mut latent = vec![-1i64; 2 * width];
    // (row, start, tokens)
    for (b, start, tokens) in [(0, 0, &[10, 11, 12][..]), (0, 4, &[20, 21, 22, 23, 24]), (1, 0, &[30, 31])] {
        let at = b * width + start;
        reset[at] = true;
        latent[at] = 0;
        latent[at + 1] = 1;
        targets[at + 1] = tokens[0];
        score[at + 1] = true;
        for j in 0..tokens.len() - 1 {
            inputs[at + 2 + j] = tokens[j];
            targets[at + 2 + j] = tokens[j + 1];
            score[at + 2 + j] = true;
        }
    }
    assert_eq!(ints(batch.inputs), inputs);
    assert_eq!(ints(batch.targets), targets);
    assert_eq!(bools(packed.score_bs), score);
    assert_eq!(bools(packed.layout.reset_bs), reset);
    assert_eq!(ints(packed.layout.latent_bs), latent);
    assert_eq!(batch.scored, vec![3 + 5, 2]);
    assert_eq!(batch.seq_len, width);
}
