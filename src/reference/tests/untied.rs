//! The untied copies of a parameter: one copy for each application of a real
//! layer.

use super::*;

/// Two real layers over five virtual ones (`Cyclic`: three applications and
/// two). Everything untiable is untied: both pre-norms and both untiable
/// tensors of `RefBlock`.
fn untied_builder() -> LayersBuilder<RefBlockConfig> {
    LayersBuilder {
        n_virtual_layers: Some((5, Schedule::Cyclic)),
        block: block_config().with_untied(vec![RefUntied::Decay, RefUntied::GateProj]),
        mlp: Some(GatedMlpConfig::new(D_MODEL, D_MODEL * 2)),
        untied: vec![LayerUntied::Norm, LayerUntied::Norm2],
        ..layers_builder(2)
    }
}

/// `layers` with every real layer replaced by its application-0 view: the tied
/// stack an untied one starts as.
fn tied_view(layers: &Layers<RefBlock>) -> Layers<RefBlock> {
    Layers {
        real_layers: layers.real_layers.iter().map(|l| l.application(0).into_owned()).collect(),
        ..layers.clone()
    }
}

/// Adds noise to every parameter, so the copies of an untied stack stop
/// agreeing.
struct Jitter;

impl ModuleMapper for Jitter {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|t| {
            let noise = Tensor::random(t.shape(), Distribution::Normal(0.0, 0.5), &t.device());
            t + noise
        })
    }
}

/// An untied stack equals the stack of its application views, with no weight
/// sharing. Virtual layer `i` runs application `index[i]` of its real layer,
/// in `forward` and in `step`. The test first adds noise to the copies, so a
/// read of the wrong copy shows.
#[test]
fn an_untied_stack_equals_the_stack_of_its_application_views() {
    let device = test_device();
    let layers = untied_builder().init(&device).map(&mut Jitter);
    let n = layers.n_virtual_count();
    let apps = Schedule::Cyclic.applications(n, 2);
    let view_stack = Layers {
        n_real_layers: n,
        n_virtual_layers: None,
        real_layers: (0..n)
            .map(|i| {
                let real = Schedule::Cyclic.real_idx(i, n, 2);
                layers.real_layers[real].application(apps.index[i]).into_owned()
            })
            .collect(),
        ..layers.clone()
    };
    let x = randn3(2, 4, &device);

    let (y, _) = layers.forward(x.clone(), None, (), None, None);
    let (want, _) = view_stack.forward(x.clone(), None, (), None, None);
    assert!(max_abs_diff(y.clone(), want) < dtype_tol(TOL));

    let mut caches = None;
    let mut ys = Vec::new();
    for t in 0..4 {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = layers.step(x_t, caches, None);
        caches = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }
    assert!(max_rel_diff(y.clone(), Tensor::cat(ys, 1)) < dtype_tol(TOL));

    // Both comparisons prove nothing unless the copies really differ.
    let (y_tied, _) = tied_view(&layers).forward(x, None, (), None, None);
    assert!(max_abs_diff(y, y_tied) > dtype_tol(TOL));
}

/// Every copy of `g` (the untied gradient, copies along `axis`) is non-zero.
/// Together they sum to copy 0 of `g_tied`: the gradient that the tied weight
/// collects, which lands on copy 0 alone.
fn assert_copies_split<const D: usize>(g: Tensor<D>, g_tied: Tensor<D>, axis: usize, n: usize) {
    let copies = g.chunk(n, axis);
    for copy in &copies {
        let peak = copy.clone().abs().max().into_scalar::<f32>();
        assert!(peak > 0.0, "every application trains its own copy");
    }
    let sum = copies.into_iter().reduce(|a, b| a + b).expect("n >= 1");
    let len = sum.dims()[axis];
    assert!(max_abs_diff(sum, g_tied.narrow(axis, 0, len)) < dtype_tol(TOL));
}

/// Untied copies start as the tied weight. The untied stack computes exactly
/// its application-0 view, and the gradient of the tied weight is the sum of
/// the gradients of its copies. Only training makes the copies differ.
#[test]
fn untied_copies_start_tied_and_split_the_tied_gradient() {
    let device = test_device().autodiff();
    let layers = untied_builder().init(&device);
    let tied = tied_view(&layers);
    let x = randn3(2, 4, &device);

    let (y, _) = layers.forward(x.clone(), None, (), None, None);
    let (y_tied, _) = tied.forward(x, None, (), None, None);
    assert!(max_abs_diff(y.clone(), y_tied.clone()) < dtype_tol(TOL));

    // A view reads the stored parameters, so both gradients land on them.
    let grads = y.sum().backward();
    let grads_tied = y_tied.sum().backward();
    for layer in &layers.real_layers {
        let n = layer.n_applications;
        let norm2 = layer.norm2.as_ref().expect("the builder sets an mlp");
        for (param, axis) in [
            (&layer.norm.gamma, 0),
            (&norm2.gamma, 0),
            (&layer.block.decay_raw, 0),
        ] {
            let g = param.val().grad(&grads).expect("untied gradient");
            let g_tied = param.val().grad(&grads_tied).expect("tied gradient");
            assert_copies_split(g, g_tied, axis, n);
        }
        let w = &layer.block.gate_proj.weight;
        let g = w.val().grad(&grads).expect("untied gradient");
        let g_tied = w.val().grad(&grads_tied).expect("tied gradient");
        assert_copies_split(g, g_tied, 1, n);
    }
}

/// Only its own application reads a copy. So a cut through a layer that
/// unties anything would leave copies that never train. It panics instead.
#[test]
#[should_panic(expected = "unties parameters across its applications")]
fn grad_horizon_refuses_to_cut_an_untied_layer() {
    let device = test_device().autodiff();
    let layers = LayersBuilder {
        grad_horizon: Some(GradHorizon::Depth(1)),
        ..untied_builder()
    }
    .init(&device);
    let _ = layers.forward(randn3(1, 3, &device), None, (), None, None);
}

/// A post-build `InitPolicy` redraws every 2-D `weight` element by element.
/// `NetworkShape` then ties an untied weight back to copies of its first
/// application.
#[test]
fn an_init_policy_keeps_untied_copies_tied() {
    use crate::modules::NetworkShape;
    use crate::utils::InitPolicy;

    let device = test_device();
    let layers = NetworkShape::new(1)
        .with_n_virtual_layers(Some((3, Schedule::Cyclic)))
        .with_init(Some(InitPolicy::new()))
        .init(block_config().with_untied(vec![RefUntied::GateProj]), &device);
    let w = layers.real_layers[0].block.gate_proj.weight.val();
    assert_eq!(w.dims(), [D_MODEL, 3 * D_MODEL]);
    let copies = w.chunk(3, 1);
    for copy in &copies[1..] {
        assert_eq!(max_abs_diff(copies[0].clone(), copy.clone()), 0.0);
    }
}

/// On a plain device, the untied copies stay plain (`tile` and `retie`). A
/// copy with autodiff makes each pass record a graph that keeps the saved
/// state of its ops. Only a backward that reaches the copy frees that graph,
/// and a pass without autodiff (a decode, an evaluation) never runs one.
#[test]
fn untied_copies_stay_plain_on_a_plain_device() {
    use crate::modules::NetworkShape;
    use crate::utils::InitPolicy;

    let device = test_device();
    assert!(!device.is_autodiff());
    let layers = untied_builder().init(&device);
    for layer in &layers.real_layers {
        let norm2 = layer.norm2.as_ref().expect("the builder sets an mlp");
        assert!(!layer.norm.gamma.val().is_autodiff());
        assert!(!norm2.gamma.val().is_autodiff());
        assert!(!layer.block.decay_raw.val().is_autodiff());
        assert!(!layer.block.gate_proj.weight.val().is_autodiff());
    }

    let layers = NetworkShape::new(1)
        .with_n_virtual_layers(Some((3, Schedule::Cyclic)))
        .with_init(Some(InitPolicy::new()))
        .init(block_config().with_untied(vec![RefUntied::GateProj]), &device);
    assert!(!layers.real_layers[0].block.gate_proj.weight.val().is_autodiff());
}

/// Both directions of a bidirectional pair are applications of the real layer
/// that they index. Every copy that an untied bidi stack holds is read, and
/// trains.
#[test]
fn bidi_untied_copies_each_train() {
    let device = test_device().autodiff();
    let layers = BidiLayersBuilder {
        n_real_layers: 2,
        n_virtual_layers: Some((6, BidiSchedule::SymmetricCyclic)),
        block: block_config().with_untied(vec![RefUntied::Decay]),
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: vec![OutputMergeConfig::Mean],
        class_latents: Vec::new(),
        residuals: ResidualsConfig::Standard,
        untied: vec![LayerUntied::Norm],
    }
    .init(&device);
    // `SymmetricCyclic` runs 0 0 1 1 0 0.
    let counts: Vec<_> = layers.real_layers.iter().map(|l| l.n_applications).collect();
    assert_eq!(counts, vec![4, 2]);

    let (y, _) = layers.forward(randn3(2, 4, &device), None, (), None, None);
    let grads = y.sum().backward();
    for layer in &layers.real_layers {
        for param in [&layer.norm.gamma, &layer.block.decay_raw] {
            let g = param.val().grad(&grads).expect("untied gradient");
            for copy in g.chunk(layer.n_applications, 0) {
                let peak = copy.abs().max().into_scalar::<f32>();
                assert!(peak > 0.0, "every application trains its own copy");
            }
        }
    }
}
