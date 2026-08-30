//! A whole-model initialisation policy: the reference LM init, applied to a
//! module that has already been built.
//!
//! Burn initialises each module from its own config (Kaiming-uniform for a
//! `Linear`, ones for a norm gain, and so on), which is the right *local*
//! choice and not the one the reference language models train under. Those use
//! a single global rule instead — every projection and embedding drawn from
//! `N(0, initializer_range²)`, biases zeroed — because at depth the residual
//! stream's scale is a property of the whole stack, not of one layer. This
//! walks a built module and applies that rule, keeping each parameter's id and
//! `require_grad` flag.
//!
//! ```text
//!   weight (2-D)   ← N(0, std²)         Linear + Embedding
//!   bias   (1-D)   ← 0
//!   everything else                     left as its own module built it
//! ```
//!
//! **What it deliberately leaves alone.** Only a parameter whose own field is
//! `weight` (and is a matrix) or `bias` is touched. A block's bespoke
//! parameters — `A_log`, `dt_bias`, a norm's `γ`, a depthwise convolution's
//! 3-D kernel, a class-token table — carry initialisations that *mean*
//! something (a spread of timescales, a decay that cannot amplify), and a
//! global rule would silently erase them.
//!
//! **The residual rescale** ([`InitPolicy::residual_paths`]) is the GPT-2
//! scheme: a weight that writes into the residual stream is divided by
//! `√(residual branches in the stack)`, so the stream's variance does not grow
//! with depth. The reference exposes it as `prenorm_residual_strategy='rescale'`
//! and ships it **off**; it is off here too (an empty path list), with
//! [`InitPolicy::default_residual_paths`] naming the two weights it applies to.

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

    /// Path fragments identifying the weights that write into the residual
    /// stream; each is drawn with `std / √residual_depth` instead of `std`.
    /// Empty (the default) ⇒ no rescale. See
    /// [`Self::default_residual_paths`].
    #[config(default = "Vec::new()")]
    pub residual_paths: Vec<String>,

    /// How many residual branches the stack has in total — layers × branches
    /// per layer (a mixer, plus a feed-forward when there is one). `None` with
    /// a non-empty [`Self::residual_paths`] is a caller error and panics: the
    /// rescale is meaningless without a depth to count.
    #[config(default = "None")]
    pub residual_depth: Option<usize>,
}

impl InitPolicy {
    /// The weights that write into the residual stream in this crate's
    /// containers: a block's output projection and the feed-forward's
    /// down-projection — the reference's `o_proj` and `down_proj`.
    pub fn default_residual_paths() -> Vec<String> {
        vec!["out_proj.weight".to_string(), "mlp.fc2.weight".to_string()]
    }

    /// Fill in [`Self::residual_depth`] when the caller did not state one — how
    /// a network config supplies the depth it alone knows.
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

/// Walks the module tree keeping the current parameter path.
struct Reinit<'a> {
    policy: &'a InitPolicy,
    path: Vec<String>,
}

impl Reinit<'_> {
    /// The field the parameter being mapped is stored under.
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
        // A matrix stored as `weight` is a `Linear` or an `Embedding`; a 1-D
        // `bias` is theirs too. Anything else is a block's own parameter, whose
        // initialisation carries meaning this rule does not know about.
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

/// Replace a parameter's value, keeping its id and `require_grad` flag.
fn redraw<const D: usize>(
    param: Param<Tensor<D>>,
    value: impl FnOnce(Shape, &Device) -> Tensor<D>,
) -> Param<Tensor<D>> {
    param.map(|tensor| {
        // `Param::map` re-reads the flag off the tensor it is handed, so a
        // freshly drawn one has to be told.
        let require_grad = tensor.is_require_grad();
        let device = tensor.device();
        value(tensor.shape(), &device).set_require_grad(require_grad)
    })
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
