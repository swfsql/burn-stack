//! The loss of [`lm_output`] against a host reference: per real position, the
//! cross-entropy of the id, plus the binary cross-entropy of the case flag on
//! a letter, summed and divided by the real positions. The padding and the
//! flags of the symbols do not count. [`case_nll`] stays finite for a large
//! logit.

use super::{PAD_TARGET, case_nll, lm_output};
use crate::examples::tiny_stories::dataset::{FIRST_LETTER, VOCAB_SIZE};
use crate::utils::test_helpers::{dtype_tol, test_device};
use burn::prelude::*;
use burn::tensor::Distribution;

#[test]
fn the_loss_adds_the_case_of_the_letters() {
    let device = test_device();
    let (batch, seq) = (2, 5);
    let logits = Tensor::<3>::random([batch, seq, VOCAB_SIZE + 1], Distribution::Normal(0.0, 2.0), &device);
    // Letters and symbols, both flags. A flag on a symbol must not count.
    let pairs: Vec<[i32; 2]> = (0..batch * seq)
        .map(|k| {
            let id = if k % 3 == 0 { k % FIRST_LETTER } else { FIRST_LETTER + k % 26 };
            [id as i32, (k % 2) as i32]
        })
        .collect();
    let targets = Tensor::<1, Int>::from_ints(pairs.concat().as_slice(), &device).reshape([batch, seq, 2]);
    let inputs = targets.zeros_like();
    // Slot 1 has 2 real positions, then padding.
    let scored = [seq, 2];
    let out = lm_output(logits.clone(), inputs, targets, &scored);

    let values: Vec<f32> = logits.into_data().try_into_vec_as().unwrap();
    let (mut sum, mut real) = (0.0f64, 0.0f64);
    for b in 0..batch {
        for s in 0..scored[b] {
            let row = &values[(b * seq + s) * (VOCAB_SIZE + 1)..(b * seq + s + 1) * (VOCAB_SIZE + 1)];
            let [id, flag] = pairs[b * seq + s];
            let max = row[..VOCAB_SIZE].iter().fold(f32::MIN, |a, &x| a.max(x)) as f64;
            let log_z = max + row[..VOCAB_SIZE].iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln();
            sum += log_z - row[id as usize] as f64;
            if id as usize >= FIRST_LETTER {
                let z = row[VOCAB_SIZE] as f64;
                let p_upper = 1.0 / (1.0 + (-z).exp());
                sum -= if flag == 1 { p_upper.ln() } else { (1.0 - p_upper).ln() };
            }
            real += 1.0;
        }
    }
    let want = sum / real;
    let got = out.loss.into_scalar::<f64>();
    assert!((got - want).abs() < f64::from(dtype_tol(1e-4)) * want, "loss {got} against {want}");

    // The accuracy reads the vocab logits, and the padding has its own target.
    assert_eq!(out.output.dims(), [batch * seq, VOCAB_SIZE]);
    let ids: Vec<i64> = out.targets.into_data().try_into_vec_as().unwrap();
    let pad: Vec<bool> = ids.iter().map(|&t| t == PAD_TARGET as i64).collect();
    let want_pad: Vec<bool> = (0..batch * seq).map(|k| k % seq >= scored[k / seq]).collect();
    assert_eq!(pad, want_pad);
}

#[test]
fn case_nll_is_finite_for_a_large_logit() {
    let device = test_device();
    let z = Tensor::<1>::from_floats([-60.0, -1.0, 0.0, 1.0, 60.0], &device);
    for flag in [0, 1] {
        let y = Tensor::<1, Int>::from_ints([flag; 5], &device);
        let nll: Vec<f32> = case_nll(z.clone(), y).into_data().try_into_vec_as().unwrap();
        assert!(nll.iter().all(|x| x.is_finite() && *x >= 0.0), "flag {flag}: {nll:?}");
        // At z = 0, both flags cost ln 2.
        assert!((nll[2] - std::f32::consts::LN_2).abs() < dtype_tol(1e-4), "flag {flag}: {nll:?}");
    }
}
