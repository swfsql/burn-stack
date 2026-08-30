//! Rendering a classifier's output: one digit beside its 10-bin class
//! distribution, as terminal text or as a PNG.
//!
//! Nothing here touches a model — every entry point takes the probabilities
//! already computed, so the same rendering serves any example's classifier.

use crate::examples::device::FloatElement;
use crate::examples::mnist::dataset::{HEIGHT, WIDTH};
use burn::prelude::*;
use burn::tensor::ElementConversion;
use image::{GrayImage, Luma};
use std::path::Path;

/// Number of classes (digits 0–9).
pub const NUM_CLASSES: usize = 10;

/// Nearest-neighbour upscale factor for the saved PNGs.
const SCALE: u32 = 8;

/// Width of the gray separator column between the digit and the bar chart.
const SEP: usize = 2;

/// Write one "digit + 10-bin probability bar chart" PNG per sample into
/// `out_dir` (created if missing). The true label and the prediction are encoded
/// in each file name.
///
/// `images_norm` is `[n, H, W, 1]` in `[0, 1]`; `probs` is `[n, 10]`.
pub fn save_predictions(probs: Tensor<2>, images_norm: Tensor<4>, labels: &[u8], out_dir: &Path) {
    let [n, _h, _w, _c] = images_norm.dims();
    let probs_host = to_host(probs);
    let digits_host = to_host(images_norm.reshape([n, HEIGHT * WIDTH]));

    std::fs::create_dir_all(out_dir).expect("failed to create sample directory");
    for i in 0..n {
        let off = i * HEIGHT * WIDTH;
        let p = &probs_host[i * NUM_CLASSES..i * NUM_CLASSES + NUM_CLASSES];
        let true_label = labels.get(i).copied().unwrap_or(0) as usize;
        let pred = argmax(p);
        let img = digit_with_bars_png(&digits_host[off..off + HEIGHT * WIDTH], p, true_label, pred);
        let path = out_dir.join(format!("sample-{i:02}-label-{true_label}-pred-{pred}.png"));
        img.save(&path).expect("failed to save prediction PNG");
    }
}

/// Print each digit as ASCII art beside a text bar chart of its 10 class
/// probabilities. Same shapes as [`save_predictions`].
pub fn print_predictions(probs: Tensor<2>, images_norm: Tensor<4>, labels: &[u8]) {
    let [n, _h, _w, _c] = images_norm.dims();
    let probs_host = to_host(probs);
    let digits_host = to_host(images_norm.reshape([n, HEIGHT * WIDTH]));
    for i in 0..n {
        let off = i * HEIGHT * WIDTH;
        let p = &probs_host[i * NUM_CLASSES..i * NUM_CLASSES + NUM_CLASSES];
        let pred = argmax(p);
        println!(
            "\n--- digit {i} (true label {}, predicted {pred}) ---",
            labels[i]
        );
        println!(
            "{}",
            render_digit_ascii(&digits_host[off..off + HEIGHT * WIDTH])
        );
        println!("{}", render_prediction(p, labels[i] as usize, pred));
    }
}

/// Build a nearest-upscaled grayscale image: the digit on the left, a separator,
/// then a 10-bar probability chart (bars left→right are classes 0–9). The
/// predicted bar is full-white; the true class is marked by a faint full-height
/// column behind its bar.
pub fn digit_with_bars_png(
    digit: &[f32],
    probs: &[f32],
    true_label: usize,
    pred: usize,
) -> GrayImage {
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let (bar_w, gap, lead) = (2usize, 1usize, 1usize);
    let chart_w = lead + NUM_CLASSES * (bar_w + gap);
    let native_w = (WIDTH + SEP + chart_w) as u32;
    let h = HEIGHT as u32;
    let mut img = GrayImage::new(native_w, h);

    // Digit panel.
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            img.put_pixel(
                col as u32,
                row as u32,
                Luma([to_u8(digit[row * WIDTH + col])]),
            );
        }
        for s in 0..SEP {
            img.put_pixel((WIDTH + s) as u32, row as u32, Luma([90])); // separator
        }
    }

    // Bar chart panel.
    let base_x = WIDTH + SEP + lead;
    // Faint full-height marker behind the true class.
    let true_x = base_x + true_label.min(NUM_CLASSES - 1) * (bar_w + gap);
    for xx in 0..bar_w {
        for row in 0..HEIGHT {
            img.put_pixel((true_x + xx) as u32, row as u32, Luma([60]));
        }
    }
    // Bars: height ∝ probability; predicted class brightest.
    for (c, p) in probs.iter().enumerate().take(NUM_CLASSES) {
        let x0 = base_x + c * (bar_w + gap);
        let hbar = (p.clamp(0.0, 1.0) * (HEIGHT as f32 - 1.0)).round() as usize;
        let shade = if c == pred { 255 } else { 140 };
        for xx in 0..bar_w {
            for row in (HEIGHT - hbar)..HEIGHT {
                img.put_pixel((x0 + xx) as u32, row as u32, Luma([shade]));
            }
        }
    }

    image::imageops::resize(
        &img,
        native_w * SCALE,
        h * SCALE,
        image::imageops::FilterType::Nearest,
    )
}

/// Index of the maximum value (the predicted class).
pub fn argmax(probs: &[f32]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
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

/// Render a `[HEIGHT * WIDTH]` intensity buffer as 28×28 ASCII.
pub fn render_digit_ascii(digit: &[f32]) -> String {
    const RAMP: &[u8] = b" .:-=+*#%@";
    let pixel = |v: f32| -> char {
        let v = v.clamp(0.0, 1.0);
        RAMP[(v * (RAMP.len() - 1) as f32).round() as usize] as char
    };
    let mut out = String::new();
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            out.push(pixel(digit[row * WIDTH + col]));
        }
        out.push('\n');
    }
    out
}

/// Render the 10 class probabilities as a text bar chart, marking the predicted
/// and true classes.
pub fn render_prediction(probs: &[f32], true_label: usize, pred: usize) -> String {
    let mut out = String::from("  class  prob\n");
    for (c, &p) in probs.iter().enumerate() {
        let bar = "#".repeat((p * 20.0).round() as usize);
        let mark = match (c == pred, c == true_label) {
            (true, true) => " <- pred ✓",
            (true, false) => " <- pred",
            (false, true) => " (true)",
            (false, false) => "",
        };
        out.push_str(&format!(
            "  {c:>5}  {:>5.1}% |{bar:<20}|{mark}\n",
            p * 100.0
        ));
    }
    out
}
