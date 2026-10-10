//! Rust port of `python/malsim/config/agent_settings.py`'s
//! `AttackerSettings`/`DefenderSettings` and of
//! `agent_settings_factories.py::agent_settings_from_dict` - see
//! `PORTING_NOTES.md` §7 C3.
//!
//! Like the Python generic `AttackerSettings[T]`, `AttackerSettings<N>` is
//! parsed with `N = String` (entry points/goals as full names) and turned
//! into `N = AttackGraphNodeId` once a graph exists
//! (`convert_to_attack_graph_nodes`). The rules stay unresolved
//! `NodePropertyRule`s, as in Python; `scenario::flatten` resolves them.
//!
//! `policy` is carried as an opaque string and never instantiated (§2.6):
//! there are no Rust policies, and an embedding caller does its own action
//! selection.

use std::collections::BTreeSet;
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use serde_json::{Map, Value};

use crate::scenario::node_property_rule::{NodePropertyRule, NodePropertyRuleError};
use crate::settings::RewardMode;
use crate::ttc::TtcDist;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentSettingsError {
    /// The agent's settings are not a mapping.
    NotAMapping { agent: String },
    /// Python's `ValueError("Illegal keys in agent scenario settings: ...")`.
    IllegalKeys { agent: String, keys: Vec<String> },
    /// `type` is missing (Python: `KeyError: 'type'`).
    MissingType { agent: String },
    /// `type` is not `attacker`/`defender` (Python: `AgentType(...)`'s
    /// `ValueError`).
    InvalidType { agent: String, value: String },
    /// A field has a value of the wrong shape.
    InvalidField {
        agent: String,
        field: &'static str,
        reason: String,
    },
    /// A `NodePropertyRule` field could not be parsed.
    Rule {
        agent: String,
        field: &'static str,
        source: NodePropertyRuleError,
    },
    /// An entry point or goal names no node in the attack graph (Python:
    /// `full_name_or_node_to_node`'s lookup error).
    UnknownNode { agent: String, message: String },
}

impl fmt::Display for AgentSettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentSettingsError::NotAMapping { agent } => {
                write!(f, "Settings of agent '{agent}' must be a mapping")
            }
            AgentSettingsError::IllegalKeys { agent, keys } => write!(
                f,
                "Illegal keys in agent scenario settings of '{agent}': {keys:?}"
            ),
            AgentSettingsError::MissingType { agent } => {
                write!(f, "Agent '{agent}' has no 'type'")
            }
            AgentSettingsError::InvalidType { agent, value } => {
                write!(f, "'{value}' is not a valid AgentType (agent '{agent}')")
            }
            AgentSettingsError::InvalidField {
                agent,
                field,
                reason,
            } => {
                write!(f, "Invalid '{field}' for agent '{agent}': {reason}")
            }
            AgentSettingsError::Rule {
                agent,
                field,
                source,
            } => {
                write!(f, "Invalid '{field}' for agent '{agent}': {source}")
            }
            AgentSettingsError::UnknownNode { agent, message } => {
                write!(f, "Agent '{agent}': {message}")
            }
        }
    }
}

impl std::error::Error for AgentSettingsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AgentSettingsError::Rule { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Port of `AgentType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentType {
    Attacker,
    Defender,
}

impl AgentType {
    /// Parses the Python enum *value* (`AgentType('attacker')`).
    pub fn from_value(value: &str) -> Option<AgentType> {
        match value {
            "attacker" => Some(AgentType::Attacker),
            "defender" => Some(AgentType::Defender),
            _ => None,
        }
    }

    pub fn value(self) -> &'static str {
        match self {
            AgentType::Attacker => "attacker",
            AgentType::Defender => "defender",
        }
    }
}

/// `AttackerSettings.entry_points`'s two shapes: one set of entry points,
/// or several alternative sets of which one is sampled per reset (Python's
/// `Set[T]` vs `tuple[Set[T], ...]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryPoints<N> {
    Single(BTreeSet<N>),
    Multiple(Vec<BTreeSet<N>>),
}

impl<N> Default for EntryPoints<N> {
    fn default() -> Self {
        EntryPoints::Single(BTreeSet::new())
    }
}

/// Port of `AttackerSettings`.
#[derive(Debug, Clone, PartialEq)]
pub struct AttackerSettings<N> {
    pub name: String,
    pub entry_points: EntryPoints<N>,
    /// Name of the Python policy class; opaque in Rust (§2.6).
    pub policy: Option<String>,
    pub actionable_steps: Option<NodePropertyRule<bool>>,
    pub rewards: Option<NodePropertyRule<f64>>,
    pub config: Map<String, Value>,
    pub reward_mode: RewardMode,
    /// Goals affect simulation termination but are optional.
    pub goals: BTreeSet<N>,
    /// TTC distributions that override TTCs set in the language.
    pub ttc_dists: Option<NodePropertyRule<TtcDist>>,
}

/// Port of `DefenderSettings`.
#[derive(Debug, Clone, PartialEq)]
pub struct DefenderSettings {
    pub name: String,
    /// Name of the Python policy class; opaque in Rust (§2.6).
    pub policy: Option<String>,
    pub observable_steps: Option<NodePropertyRule<bool>>,
    pub actionable_steps: Option<NodePropertyRule<bool>>,
    pub rewards: Option<NodePropertyRule<f64>>,
    pub false_positive_rates: Option<NodePropertyRule<f64>>,
    pub false_negative_rates: Option<NodePropertyRule<f64>>,
    pub config: Map<String, Value>,
    pub reward_mode: RewardMode,
}

/// `AttackerSettings[T] | DefenderSettings`.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentSettings<N> {
    Attacker(AttackerSettings<N>),
    Defender(DefenderSettings),
}

impl<N> AgentSettings<N> {
    pub fn name(&self) -> &str {
        match self {
            AgentSettings::Attacker(a) => &a.name,
            AgentSettings::Defender(d) => &d.name,
        }
    }

    pub fn agent_type(&self) -> AgentType {
        match self {
            AgentSettings::Attacker(_) => AgentType::Attacker,
            AgentSettings::Defender(_) => AgentType::Defender,
        }
    }
}

impl AttackerSettings<String> {
    /// Port of `convert_to_attack_graph_nodes`: resolves entry points and
    /// goals from full names to node ids.
    pub fn convert_to_attack_graph_nodes(
        &self,
        attack_graph: &AttackGraph,
    ) -> Result<AttackerSettings<AttackGraphNodeId>, AgentSettingsError> {
        let to_nodes = |names: &BTreeSet<String>| {
            names
                .iter()
                .map(|full_name| {
                    attack_graph.get_node_by_full_name(full_name).map_err(|e| {
                        AgentSettingsError::UnknownNode {
                            agent: self.name.clone(),
                            message: e.to_string(),
                        }
                    })
                })
                .collect::<Result<BTreeSet<_>, _>>()
        };

        let entry_points = match &self.entry_points {
            EntryPoints::Single(eps) => EntryPoints::Single(to_nodes(eps)?),
            EntryPoints::Multiple(sets) => {
                EntryPoints::Multiple(sets.iter().map(to_nodes).collect::<Result<Vec<_>, _>>()?)
            }
        };

        Ok(AttackerSettings {
            name: self.name.clone(),
            entry_points,
            policy: self.policy.clone(),
            actionable_steps: self.actionable_steps.clone(),
            rewards: self.rewards.clone(),
            config: self.config.clone(),
            reward_mode: self.reward_mode,
            goals: to_nodes(&self.goals)?,
            ttc_dists: self.ttc_dists.clone(),
        })
    }
}

/// Port of `_validate_agent_dict`'s `allowed_keys`.
pub const ALLOWED_AGENT_KEYS: [&str; 13] = [
    "type",
    "policy",
    "agent_class",
    "config",
    "rewards",
    "false_positive_rates",
    "false_negative_rates",
    "observable_steps",
    "actionable_steps",
    "entry_points",
    "goals",
    "reward_mode",
    "ttc_overrides",
];

/// Port of `_validate_agent_dict`. Like Python, only illegal keys are
/// rejected: Python's missing-key check tests `illegal_keys` a second
/// time instead of `missing_keys`, so a missing `policy` is accepted (and
/// a missing `type` only fails later, when it's read).
fn validate_agent_dict<'a>(
    name: &str,
    d: &'a Value,
) -> Result<&'a Map<String, Value>, AgentSettingsError> {
    let obj = d
        .as_object()
        .ok_or_else(|| AgentSettingsError::NotAMapping {
            agent: name.to_string(),
        })?;
    let mut illegal_keys: Vec<String> = obj
        .keys()
        .filter(|k| !ALLOWED_AGENT_KEYS.contains(&k.as_str()))
        .cloned()
        .collect();
    if !illegal_keys.is_empty() {
        illegal_keys.sort();
        return Err(AgentSettingsError::IllegalKeys {
            agent: name.to_string(),
            keys: illegal_keys,
        });
    }
    Ok(obj)
}

fn string_set(
    name: &str,
    field: &'static str,
    items: &[Value],
) -> Result<BTreeSet<String>, AgentSettingsError> {
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .ok_or_else(|| AgentSettingsError::InvalidField {
                    agent: name.to_string(),
                    field,
                    reason: format!("expected a full name string, got {item}"),
                })
        })
        .collect()
}

/// Port of `_load_entry_points`: absent/`null` is an empty set, a list of
/// strings is one set, a list of lists is several alternative sets.
fn load_entry_points(
    name: &str,
    d: Option<&Value>,
) -> Result<EntryPoints<String>, AgentSettingsError> {
    let invalid = |reason: String| AgentSettingsError::InvalidField {
        agent: name.to_string(),
        field: "entry_points",
        reason,
    };
    let items = match d {
        None | Some(Value::Null) => return Ok(EntryPoints::default()),
        Some(Value::Array(items)) => items,
        Some(other) => {
            return Err(invalid(format!(
                "entry_points must be a set or list of strings, got {other}"
            )))
        }
    };

    // `all(...)` over an empty list is true, so `[]` is an empty single set.
    if items.iter().all(Value::is_string) {
        Ok(EntryPoints::Single(string_set(
            name,
            "entry_points",
            items,
        )?))
    } else if items.iter().all(Value::is_array) {
        items
            .iter()
            .map(|set| {
                string_set(
                    name,
                    "entry_points",
                    set.as_array().expect("checked is_array"),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(EntryPoints::Multiple)
    } else {
        Err(invalid(
            "entry_points list must contain either all strings or all sets/lists of strings"
                .to_string(),
        ))
    }
}

fn load_goals(name: &str, d: Option<&Value>) -> Result<BTreeSet<String>, AgentSettingsError> {
    match d {
        None => Ok(BTreeSet::new()),
        Some(Value::Array(items)) => string_set(name, "goals", items),
        Some(other) => Err(AgentSettingsError::InvalidField {
            agent: name.to_string(),
            field: "goals",
            reason: format!("expected a list of full names, got {other}"),
        }),
    }
}

fn load_rule<T: crate::scenario::node_property_rule::RuleValue>(
    name: &str,
    d: &Map<String, Value>,
    field: &'static str,
) -> Result<Option<NodePropertyRule<T>>, AgentSettingsError> {
    NodePropertyRule::from_optional_value(d.get(field)).map_err(|source| AgentSettingsError::Rule {
        agent: name.to_string(),
        field,
        source,
    })
}

/// `d.get('policy') or d.get('agent_class')` - a falsy (empty/null)
/// `policy` falls through to `agent_class`.
fn load_policy(name: &str, d: &Map<String, Value>) -> Result<Option<String>, AgentSettingsError> {
    for field in ["policy", "agent_class"] {
        match d.get(field) {
            None | Some(Value::Null) => {}
            Some(Value::String(s)) if s.is_empty() => {}
            Some(Value::String(s)) => return Ok(Some(s.clone())),
            Some(other) => {
                return Err(AgentSettingsError::InvalidField {
                    agent: name.to_string(),
                    field,
                    reason: format!("expected a policy class name, got {other}"),
                })
            }
        }
    }
    Ok(None)
}

/// `d.get('config', {})`; `null` is treated as empty.
fn load_config(
    name: &str,
    d: &Map<String, Value>,
) -> Result<Map<String, Value>, AgentSettingsError> {
    match d.get("config") {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(config)) => Ok(config.clone()),
        Some(other) => Err(AgentSettingsError::InvalidField {
            agent: name.to_string(),
            field: "config",
            reason: format!("expected a mapping, got {other}"),
        }),
    }
}

/// `RewardMode[d.get('reward_mode', 'CUMULATIVE')]`.
fn load_reward_mode(name: &str, d: &Map<String, Value>) -> Result<RewardMode, AgentSettingsError> {
    let invalid = |reason: String| AgentSettingsError::InvalidField {
        agent: name.to_string(),
        field: "reward_mode",
        reason,
    };
    match d.get("reward_mode") {
        None => Ok(RewardMode::Cumulative),
        Some(Value::String(s)) => {
            RewardMode::from_name(s).ok_or_else(|| invalid(format!("unknown RewardMode '{s}'")))
        }
        Some(other) => Err(invalid(format!("expected a RewardMode name, got {other}"))),
    }
}

/// Port of `agent_settings_from_dict`, minus policy-class resolution
/// (§2.6: `policy` is kept as its name).
pub fn agent_settings_from_dict(
    name: &str,
    d: &Value,
) -> Result<AgentSettings<String>, AgentSettingsError> {
    let d = validate_agent_dict(name, d)?;
    let agent_type = match d.get("type") {
        None => {
            return Err(AgentSettingsError::MissingType {
                agent: name.to_string(),
            })
        }
        Some(v) => {
            let value = v.as_str().unwrap_or_default();
            AgentType::from_value(value).ok_or_else(|| AgentSettingsError::InvalidType {
                agent: name.to_string(),
                value: v.to_string(),
            })?
        }
    };

    let policy = load_policy(name, d)?;
    let config = load_config(name, d)?;
    let reward_mode = load_reward_mode(name, d)?;

    Ok(match agent_type {
        AgentType::Attacker => AgentSettings::Attacker(AttackerSettings {
            name: name.to_string(),
            entry_points: load_entry_points(name, d.get("entry_points"))?,
            goals: load_goals(name, d.get("goals"))?,
            ttc_dists: load_rule(name, d, "ttc_overrides")?,
            policy,
            actionable_steps: load_rule(name, d, "actionable_steps")?,
            rewards: load_rule(name, d, "rewards")?,
            config,
            reward_mode,
        }),
        AgentType::Defender => AgentSettings::Defender(DefenderSettings {
            name: name.to_string(),
            policy,
            observable_steps: load_rule(name, d, "observable_steps")?,
            actionable_steps: load_rule(name, d, "actionable_steps")?,
            rewards: load_rule(name, d, "rewards")?,
            false_positive_rates: load_rule(name, d, "false_positive_rates")?,
            false_negative_rates: load_rule(name, d, "false_negative_rates")?,
            config,
            reward_mode,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::wiper_attack_graph;
    use serde_json::json;

    fn attacker(settings: AgentSettings<String>) -> AttackerSettings<String> {
        match settings {
            AgentSettings::Attacker(a) => a,
            AgentSettings::Defender(_) => panic!("expected an attacker"),
        }
    }

    fn defender(settings: AgentSettings<String>) -> DefenderSettings {
        match settings {
            AgentSettings::Defender(d) => d,
            AgentSettings::Attacker(_) => panic!("expected a defender"),
        }
    }

    fn names(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn attacker_from_dict() {
        let a = attacker(
            agent_settings_from_dict(
                "Attacker1",
                &json!({
                    "type": "attacker",
                    "agent_class": "BreadthFirstAttacker",
                    "entry_points": ["User:3:phishing", "Host:0:connect"],
                    "goals": ["Data:2:read"],
                    "actionable_steps": {"by_asset_type": {"Host": ["authenticate", "connect"]}},
                    "rewards": {"by_asset_name": {"Host:0": {"access": 4}}},
                    "ttc_overrides": {"by_asset_type": {"Host": {"connect": "HardAndUncertain"}}},
                    "config": {"seed": 1},
                }),
            )
            .unwrap(),
        );
        assert_eq!(a.name, "Attacker1");
        assert_eq!(a.policy.as_deref(), Some("BreadthFirstAttacker"));
        assert_eq!(
            a.entry_points,
            EntryPoints::Single(names(&["User:3:phishing", "Host:0:connect"]))
        );
        assert_eq!(a.goals, names(&["Data:2:read"]));
        assert!(a.actionable_steps.is_some());
        assert!(a.rewards.is_some());
        assert!(a.ttc_dists.is_some());
        assert_eq!(a.config["seed"], json!(1));
        // `agent_settings_from_dict`'s default, not the dataclass's ONE_OFF.
        assert_eq!(a.reward_mode, RewardMode::Cumulative);
    }

    #[test]
    fn defender_from_dict() {
        let d = defender(
            agent_settings_from_dict(
                "Defender1",
                &json!({
                    "type": "defender",
                    "policy": "PassiveAgent",
                    "reward_mode": "ONE_OFF",
                    "observable_steps": {"by_asset_type": {"Host": ["access"]}},
                    "false_positive_rates": {"by_asset_type": {"Host": {"connect": 0.5}}},
                    "false_negative_rates": {"by_asset_type": {"Host": {"access": 0.5}}},
                }),
            )
            .unwrap(),
        );
        assert_eq!(d.policy.as_deref(), Some("PassiveAgent"));
        assert_eq!(d.reward_mode, RewardMode::OneOff);
        assert!(d.observable_steps.is_some());
        assert!(d.actionable_steps.is_none());
        assert!(d.false_positive_rates.is_some());
        assert!(d.false_negative_rates.is_some());
        assert!(d.config.is_empty());
    }

    #[test]
    fn policy_takes_precedence_over_agent_class_and_is_opaque() {
        let a = attacker(
            agent_settings_from_dict(
                "a",
                &json!({"type": "attacker", "policy": "NotARealPolicy", "agent_class": "Other"}),
            )
            .unwrap(),
        );
        assert_eq!(a.policy.as_deref(), Some("NotARealPolicy"));

        let a = attacker(agent_settings_from_dict("a", &json!({"type": "attacker"})).unwrap());
        assert_eq!(a.policy, None);
    }

    #[test]
    fn entry_points_shapes() {
        let load = |v: Value| {
            attacker(
                agent_settings_from_dict("a", &json!({"type": "attacker", "entry_points": v}))
                    .unwrap(),
            )
            .entry_points
        };
        assert_eq!(load(Value::Null), EntryPoints::Single(BTreeSet::new()));
        assert_eq!(load(json!([])), EntryPoints::Single(BTreeSet::new()));
        assert_eq!(
            load(json!([["A:x", "B:y"], ["C:z"]])),
            EntryPoints::Multiple(vec![names(&["A:x", "B:y"]), names(&["C:z"])])
        );
        assert!(matches!(
            agent_settings_from_dict(
                "a",
                &json!({"type": "attacker", "entry_points": ["A:x", ["B:y"]]})
            ),
            Err(AgentSettingsError::InvalidField {
                field: "entry_points",
                ..
            })
        ));
        assert!(matches!(
            agent_settings_from_dict("a", &json!({"type": "attacker", "entry_points": "A:x"})),
            Err(AgentSettingsError::InvalidField {
                field: "entry_points",
                ..
            })
        ));
    }

    #[test]
    fn invalid_agent_dicts_are_rejected() {
        assert!(matches!(
            agent_settings_from_dict("a", &json!({"type": "attacker", "bogus": 1})),
            Err(AgentSettingsError::IllegalKeys { keys, .. }) if keys == vec!["bogus".to_string()]
        ));
        assert!(matches!(
            agent_settings_from_dict("a", &json!({"policy": "PassiveAgent"})),
            Err(AgentSettingsError::MissingType { .. })
        ));
        assert!(matches!(
            agent_settings_from_dict("a", &json!({"type": "observer"})),
            Err(AgentSettingsError::InvalidType { .. })
        ));
        assert!(matches!(
            agent_settings_from_dict(
                "a",
                &json!({"type": "attacker", "reward_mode": "SOMETIMES"})
            ),
            Err(AgentSettingsError::InvalidField {
                field: "reward_mode",
                ..
            })
        ));
        assert!(matches!(
            agent_settings_from_dict(
                "a",
                &json!({"type": "defender", "rewards": {"Host:0:access": 1}})
            ),
            Err(AgentSettingsError::Rule {
                field: "rewards",
                ..
            })
        ));
    }

    #[test]
    fn convert_to_attack_graph_nodes_resolves_full_names() {
        let (graph, _model) = wiper_attack_graph();
        let a = attacker(
            agent_settings_from_dict(
                "a",
                &json!({
                    "type": "attacker",
                    "entry_points": [["InfectedDevice:infect"], ["VulnerableDevice:infect"]],
                    "goals": ["InfectedData:read"],
                }),
            )
            .unwrap(),
        );
        let converted = a.convert_to_attack_graph_nodes(&graph).unwrap();
        let id = |n: &str| graph.get_node_by_full_name(n).unwrap();
        assert_eq!(
            converted.entry_points,
            EntryPoints::Multiple(vec![
                BTreeSet::from([id("InfectedDevice:infect")]),
                BTreeSet::from([id("VulnerableDevice:infect")]),
            ])
        );
        assert_eq!(converted.goals, BTreeSet::from([id("InfectedData:read")]));

        let bad = attacker(
            agent_settings_from_dict("a", &json!({"type": "attacker", "goals": ["Nope:read"]}))
                .unwrap(),
        );
        assert!(matches!(
            bad.convert_to_attack_graph_nodes(&graph),
            Err(AgentSettingsError::UnknownNode { .. })
        ));
    }
}
