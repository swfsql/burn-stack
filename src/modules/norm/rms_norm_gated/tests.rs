use super::*;
use burn::module::Param;
use burn::tensor::Distribution;
use crate::utils::test_helpers::test_device;

/// A half-precision input gives the value of the f64 formula, in both orders
/// of the gate, to two steps of its dtype. The inputs have the size of the
/// output norm of a Mamba-2 block at init (gate first): `mean((x·SiLU(z))²)`
/// is about `4·10⁻³`. There, an `ε` of `7.1·10⁻⁴` (the [`div_eps`] of f16)
/// changes the output by 6–9%.
///
/// [`div_eps`]: crate::utils::div_eps
#[test]
#[cfg_attr(
    not(feature = "dev-f16"),
    ignore = "f16 build only: a half-precision check (f16 and bf16 inputs)"
)]
fn a_half_input_gives_the_formula_at_small_activations() {
    let device = test_device();
    let eps = f64::from(crate::utils::div_eps(burn::tensor::DType::F32));
    let (rows, width) = (6, 64);
    let host = |t: Tensor<2>| -> Vec<f64> {
        let v: Vec<f32> = t.into_data().try_into_vec_as().unwrap();
        v.into_iter().map(f64::from).collect()
    };
    let silu = |z: f64| z / (1.0 + (-z).exp());
    for norm_before_gate in [false, true] {
        let norm = RmsNormGatedConfig::new(width)
            .with_norm_before_gate(norm_before_gate)
            .init(&device);
        for dtype in [FloatDType::F16, FloatDType::BF16] {
            let draw = |std: f64| {
                Tensor::<2>::random([rows, width], Distribution::Normal(0.0, std), &device).cast(dtype)
            };
            let (x, z) = (draw(0.1), draw(1.0));
            let (xs, zs) = (host(x.clone()), host(z.clone()));
            let got = host(norm.forward(x, z));
            let (mut err, mut size) = (0f64, 0f64);
            for r in 0..rows {
                let row = r * width..(r + 1) * width;
                let gate: Vec<f64> = zs[row.clone()].iter().map(|&z| silu(z)).collect();
                let u: Vec<f64> = if norm_before_gate {
                    xs[row.clone()].to_vec()
                } else {
                    xs[row.clone()].iter().zip(&gate).map(|(x, g)| x * g).collect()
                };
                let rms = (u.iter().map(|v| v * v).sum::<f64>() / width as f64 + eps).sqrt();
                for (c, i) in row.enumerate() {
                    let want = if norm_before_gate { u[c] / rms * gate[c] } else { u[c] / rms };
                    err = err.max((got[i] - want).abs());
                    size = size.max(want.abs());
                }
            }
            let step = dtype.finfo().epsilon;
            assert!(
                err <= 2.0 * step * size,
                "{dtype:?}, norm_before_gate {norm_before_gate}: off by {err:.2e} (size {size:.2e})"
            );
        }
    }
}

/// The backward of gated RMSNorm must stay finite when the normalised input
/// (the output `y` of a mixer) collapses to zero norm on a slice. This is the
/// exact NaN localised in a training run (`d_y` arrived NaN at the backward of
/// the mixer). The root cause is the same as for the ungated
/// [`RmsNorm`](crate::modules::norm::rms_norm): `div_eps` guards the forward
/// division, but not the `1/(2√·)` backward of the `sqrt` node.
#[test]
fn rms_norm_gated_gradient_finite_on_collapsed_slice() {
    let device = test_device();
    let (batch, seq, d_model) = (2, 3, 8);
    let norm = RmsNormGatedConfig::new(d_model).init(&device.clone().autodiff());

    // The normalised input (mixer output) has one token collapsed to zero
    // norm. The gate `z` stays healthy.
    let normal = Tensor::<3>::random(
        [batch, seq - 1, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let collapsed = Tensor::<3>::zeros([batch, 1, d_model], &device);
    let base = Tensor::cat(vec![collapsed, normal], 1);
    let z = Tensor::from_inner(Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));

    let x = Param::from_tensor(Tensor::from_inner(base));
    let grads = norm.forward(x.val(), z).sum().backward();
    let g = x.val().grad(&grads).expect("grad exists");
    let gvec = g.into_data().try_into_vec_as::<f32>().unwrap();
    assert!(
        gvec.iter().all(|v| v.is_finite()),
        "gated RMSNorm gradient must stay finite for a zero-norm slice"
    );
}
