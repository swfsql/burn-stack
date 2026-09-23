//! A whole-model initialisation policy: the reference LM init, applied to a
//! module after its build.
//!
//! Burn initialises each module from its own config (Kaiming-uniform for a
//! `Linear`, ones for a norm gain, and so on). This is the right *local*
//! choice, but not the one that the reference language models train under.
//! They use a single global rule instead: every projection and embedding
//! drawn from `N(0, initializer_range²)`, and biases zeroed. The reason: at
//! depth, the scale of the residual stream is a property of the whole stack,
//! not of one layer. This module walks a built module and applies that rule.
//! It keeps the id and the `require_grad` flag of each parameter.
//!
//! ```text
//!   weight (2-D)   ← N(0, std²)         Linear + Embedding
//!   bias   (1-D)   ← 0
//!   everything else                     left as its own module built it
//! ```
//!
//! **What it leaves alone, on purpose.** It touches only a parameter whose own
//! field is `weight` (and is a matrix) or `bias`. Consider the bespoke
//! parameters of a block: a log-decay, a step-size bias, the `γ` of a norm,
//! the 3-D kernel of a depthwise convolution, a class-token table. Their
//! initialisations *mean* something (a spread of timescales, a decay that
//! cannot amplify). A global rule would silently erase them.
//!
//! **The residual rescale** ([`InitPolicy::residual_paths`]) is the GPT-2
//! scheme: a weight that writes into the residual stream is divided by
//! `√(residual branches in the stack)`, so the variance of the stream does
//! not grow with depth. The reference exposes it as
//! `prenorm_residual_strategy='rescale'` and ships it **off**. It is also off
//! here (an empty path list), and [`InitPolicy::default_residual_paths`] names
//! the two weights that it applies to.

use burn::module::{Module, ModuleMapper, Param};
use burn::prelude::*;
use burn::tensor::Distribution;

/// How to re-initialise a built model. See the module header.
#[derive(Config, Debug)]
pub struct InitPolicy {
    /// Standard deviation of the normal every 2-D `weight` is drawn from
    /// (`initializer_range` in the reference configs).
    #[config(default = 0.02)]
    pub std: f64,

    /// Zero every 1-D `bias`.
    #[config(default = true)]
    pub zero_bias: bool,

    /// Path fragments that identify the weights that write into the residual
    /// stream. Each one is drawn with `std / √residual_depth`, not `std`.
    /// Empty (the default) ⇒ no rescale. See
    /// [`Self::default_residual_paths`].
    #[config(default = "Vec::new()")]
    pub residual_paths: Vec<String>,

    /// The total number of residual branches of the stack: layers × branches
    /// per layer (a mixer, plus a feed-forward when there is one). `None` with
    /// a non-empty [`Self::residual_paths`] is a caller error and panics: the
    /// rescale has no meaning without a depth to count.
    #[config(default = "None")]
    pub residual_depth: Option<usize>,
}

impl InitPolicy {
    /// The weights that write into the residual stream in the containers of
    /// this crate: the output projection of a block, and the down-projection
    /// of the feed-forward (the `o_proj` and `down_proj` of the reference).
    pub fn default_residual_paths() -> Vec<String> {
        vec!["out_proj.weight".to_string(), "mlp.fc2.weight".to_string()]
    }

    /// Set [`Self::residual_depth`] when the caller did not state one. A
    /// network config uses this to supply the depth that only it knows.
    pub fn with_default_residual_depth(mut self, depth: usize) -> Self {
        self.residual_depth = self.residual_depth.or(Some(depth));
        self
    }

    /// The standard deviation a weight at `path` is drawn from.
    fn std_for(&self, path: &str) -> f64 {
        if !self.residual_paths.iter().any(|p| path.contains(p.as_str())) {
            return self.std;
        }
        let depth = self
            .residual_depth
            .expect("InitPolicy::residual_paths is set but residual_depth is not");
        self.std / (depth as f64).sqrt()
    }

    /// Apply this policy to a built module.
    pub fn apply<M: Module>(&self, module: M) -> M {
        let mut mapper = Reinit {
            policy: self,
            path: Vec::new(),
        };
        module.map(&mut mapper)
    }
}

/// Walks the module tree, and keeps the current parameter path.
struct Reinit<'a> {
    policy: &'a InitPolicy,
    path: Vec<String>,
}

impl Reinit<'_> {
    /// The field that holds the parameter under the map.
    fn field(&self) -> &str {
        self.path.last().map(String::as_str).unwrap_or("")
    }
}

impl ModuleMapper for Reinit<'_> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        // A matrix stored as `weight` is a `Linear` or an `Embedding`, and a
        // 1-D `bias` is theirs too. Anything else is a parameter of a block,
        // whose initialisation carries a meaning that this rule does not know.
        match self.field() {
            "weight" if D == 2 => {
                let std = self.policy.std_for(&self.path.join("."));
                redraw(param, |shape, device| {
                    Tensor::random(shape, Distribution::Normal(0.0, std), device)
                })
            }
            "bias" if D == 1 && self.policy.zero_bias => {
                redraw(param, |shape, device| Tensor::zeros(shape, device))
            }
            _ => param,
        }
    }
}

/// Replace the value of a parameter, and keep its id and `require_grad` flag.
fn redraw<const D: usize>(
    param: Param<Tensor<D>>,
    value: impl FnOnce(Shape, &Device) -> Tensor<D>,
) -> Param<Tensor<D>> {
    param.map(|tensor| {
        // `Param::map` reads the flag again from the tensor that it gets. So a
        // new draw must get the flag explicitly.
        let require_grad = tensor.is_require_grad();
        let device = tensor.device();
        value(tensor.shape(), &device).set_require_grad(require_grad)
    })
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
