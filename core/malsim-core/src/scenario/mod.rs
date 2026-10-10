//! Rust-only scenario loading - the independent Rust implementation of
//! `python/malsim/scenario/` + `python/malsim/config/` used by an embedding
//! Rust program with no Python runtime (`PORTING_NOTES.md` §2.6, §7). The
//! PyO3 path never calls into this module: Python keeps its own `Scenario`
//! and flattens settings itself (§2.4).

pub mod agent_settings;
pub mod flatten;
pub mod loading;
pub mod node_property_rule;

pub use agent_settings::{
    agent_settings_from_dict, AgentSettings, AgentSettingsError, AgentType, AttackerSettings,
    DefenderSettings, EntryPoints,
};
pub use flatten::{flatten_attacker_settings, flatten_defender_settings, get_entry_points};
pub use loading::{
    load_scenario_dict, recursive_update, validate_scenario_dict, ScenarioFileError,
};
pub use node_property_rule::{NodePropertyRule, NodePropertyRuleError, RuleValue, StepValues};
