//! [`MuonPlan::describe`]: the optimizer of each parameter of a model. This is
//! the quickest way to check a plan against a real model.

use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::prelude::*;
use core::fmt::Write as _;

use super::MuonPlan;

/// One float parameter of a module: its path, its shape and its id.
pub(crate) struct FloatParam {
    pub(crate) path: String,
    pub(crate) dims: Vec<usize>,
    pub(crate) id: ParamId,
}

/// Each float parameter of `module`, in the order of the visit.
pub(crate) fn float_params(module: &impl Module) -> Vec<FloatParam> {
    let mut collect = Collect::default();
    module.visit(&mut collect);
    collect.params
}

#[derive(Default)]
struct Collect {
    path: Vec<String>,
    params: Vec<FloatParam>,
}

impl ModuleVisitor for Collect {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.params.push(FloatParam {
            path: self.path.join("."),
            dims: param.val().shape().dims::<D>().to_vec(),
            id: param.id,
        });
    }
}

impl MuonPlan {
    /// A report of this plan applied to `module`, with one line for each
    /// parameter: its path, its shape and the optimizer that owns it. The line
    /// of a fused projection also lists its sub-projections, with `*` on those
    /// left on the fallback. The report ends with the share of parameters on
    /// Muon.
    ///
    /// Purely diagnostic: print it once to confirm that the plan matches the
    /// model that you built.
    pub fn describe(&self, module: &impl Module) -> String {
        let mut out = String::new();
        let (mut total, mut on_muon) = (0usize, 0usize);

        for row in &float_params(module) {
            let count: usize = row.dims.iter().product();
            total += count;

            let spec = self
                .specs
                .iter()
                .rev()
                .find(|spec| spec.param_group().matches(&row.id, Some(row.path.as_str())));

            let owner = match spec {
                None => "fallback".to_string(),
                Some(spec) => {
                    let muon_width: usize =
                        spec.sub_projections.iter().filter(|s| s.muon).map(|s| s.width).sum();
                    // An untied weight holds one copy of the sub-projections
                    // per application.
                    let copies = if spec.tiled { row.dims[1] / spec.width() } else { 1 };
                    on_muon += muon_width * copies * row.dims[0];
                    let sub_projections = spec
                        .sub_projections
                        .iter()
                        .map(|s| {
                            let mark = if s.muon { "" } else { "*" };
                            format!("{}:{}{mark}", s.name, s.width)
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    let tiling = if spec.tiled { format!("×{copies}") } else { String::new() };
                    format!("muon[{sub_projections}]{tiling}")
                }
            };

            let _ = writeln!(
                out,
                "{:>2}D {:<62} {:?} ({count})  {owner}",
                row.dims.len(),
                row.path,
                row.dims
            );
        }

        let share = if total == 0 {
            0.0
        } else {
            100.0 * on_muon as f64 / total as f64
        };
        let _ = writeln!(
            out,
            "total params: {total}; on muon: {on_muon} ({share:.1}%)  (* = sub-projection left on the fallback)"
        );
        out
    }
}
