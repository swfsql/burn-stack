//! Example-support scaffolding shared by the consumers of this crate.
//!
//! None of this is part of the composition layer. It is the plumbing that
//! every `examples/` directory would otherwise write again:
//!
//! - CLI + artifact handling ([`cli`]),
//! - runtime dtype selection ([`device`]),
//! - the [`training`] config (AdamW or SGD, optionally with Muon),
//! - the [`trainer`] that steps a module under it (from a captured graph under
//!   plain SGD),
//! - the per-invocation [`session`] that every epoch loop threads (resume
//!   position, budget, cadence, metrics log),
//! - the two datasets with their epoch loops: the sequential-[`mnist`]
//!   classifier and the character-level [`tiny_stories`] language model.
//!
//! It lives here so that the consumer crates share one copy.
//!
//! The off-by-default `examples-common` feature gates it. That feature pulls
//! `burn/train`, `burn/dataset` and the download/CLI crates. A consumer
//! enables it in its **dev**-dependencies only.
//!
//! The `config → module` interface that these use,
//! [`ModelConfigExt`](crate::modules::ModelConfigExt), is *not* here.
//! Consumers implement it on their own network configs, so it is in
//! [`crate::modules`] with the rest of the plug-in surface.

pub mod cli;
pub mod device;
pub mod mnist;
pub mod session;
pub mod tiny_stories;
pub mod trainer;
pub mod training;
