//! Rust-only scenario loading - the independent Rust implementation of
//! `python/malsim/scenario/` + `python/malsim/config/` used by an embedding
//! Rust program with no Python runtime (`PORTING_NOTES.md` §2.6, §7). The
//! PyO3 path never calls into this module: Python keeps its own `Scenario`
//! and flattens settings itself (§2.4).

pub mod node_property_rule;

pub use node_property_rule::{NodePropertyRule, NodePropertyRuleError, RuleValue, StepValues};
