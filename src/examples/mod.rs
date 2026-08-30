//! Example-support scaffolding shared by this crate's consumers.
//!
//! None of this is part of the composition layer: it is the plumbing every
//! `examples/` directory otherwise rewrites — CLI + artifact handling ([`cli`]),
//! runtime dtype selection ([`device`]), the [`training`] config (AdamW,
//! optionally Muon), and the two datasets with their epoch loops — the
//! sequential-[`mnist`] classifier and the character-level [`tiny_stories`]
//! language model. It lives here so `burn-mamba` and `burn-deltanet` share one
//! copy.
//!
//! Gated behind the off-by-default `examples-common` feature, which is what
//! pulls `burn/train`, `burn/dataset` and the download/CLI crates; a consumer
//! enables it in its **dev**-dependencies only.
//!
//! The `config → module` seam these use,
//! [`ModelConfigExt`](crate::modules::ModelConfigExt), is *not* here: consumers
//! implement it on their own network configs, so it sits in [`crate::modules`]
//! with the rest of the plug-in surface.

pub mod cli;
pub mod device;
pub mod mnist;
pub mod tiny_stories;
pub mod training;
