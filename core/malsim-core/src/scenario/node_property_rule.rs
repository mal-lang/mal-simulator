//! Rust port of `python/malsim/config/node_property_rule.py` for the
//! Rust-only scenario path - see `PORTING_NOTES.md` §2.6, §7 C1.
//!
//! This is a *second, independent* implementation of the
//! `by_asset_type`/`by_asset_name` matching: the PyO3 path never uses it
//! (Python resolves its own `NodePropertyRule` and hands flat id maps
//! across, §2.4). Both implementations target the same dict shape, the one
//! `NodePropertyRule.from_dict()`/`.to_dict()` read and write:
//!
//! ```yaml
//! by_asset_type:
//!   Host: {access: 0.5}          # step name -> value
//!   User: [compromise]           # list form: listed steps get `T::listed()`
//! by_asset_name:
//!   Host:0: {access: 0.9}
//! ```
//!
//! `value()` keeps Python's `by_asset_val or asset_type_val or default`
//! precedence *including* its truthiness: a falsy by-name value (`0`,
//! `false`) falls through to the by-type value instead of winning (see
//! `RuleValue::is_truthy`).

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNode, AttackGraphNodeId};
use maltoolbox_model::Model;
use serde_json::{Map, Value};

use crate::ttc::{named_ttc_dist, TtcDist};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodePropertyRuleError {
    /// The rule is not a mapping.
    NotAMapping,
    /// Neither `by_asset_type` nor `by_asset_name` was given (Python's
    /// "Node property dict need at least 'by_asset_type' or 'by_asset_name'").
    MissingFields,
    /// Keys other than `by_asset_type`/`by_asset_name` were given.
    ForbiddenFields(Vec<String>),
    /// A `by_asset_*` entry has the wrong shape (not a mapping of asset ->
    /// list/mapping of steps).
    InvalidShape(String),
    /// A step value could not be parsed as the rule's value type.
    InvalidValue {
        asset: String,
        step: String,
        reason: String,
    },
    /// List form was used for a value type that has no list-form meaning
    /// (e.g. TTC overrides).
    ListFormNotAllowed { asset: String },
}

impl fmt::Display for NodePropertyRuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodePropertyRuleError::NotAMapping => {
                write!(f, "Node property rule must be a mapping")
            }
            NodePropertyRuleError::MissingFields => write!(
                f,
                "Node property dict need at least 'by_asset_type' or 'by_asset_name'"
            ),
            NodePropertyRuleError::ForbiddenFields(fields) => {
                write!(f, "Node property fields not allowed: {fields:?}")
            }
            NodePropertyRuleError::InvalidShape(msg) => {
                write!(f, "Invalid node property rule shape: {msg}")
            }
            NodePropertyRuleError::InvalidValue {
                asset,
                step,
                reason,
            } => {
                write!(
                    f,
                    "Invalid node property value for '{asset}'/'{step}': {reason}"
                )
            }
            NodePropertyRuleError::ListFormNotAllowed { asset } => write!(
                f,
                "List form is not allowed for this node property (asset '{asset}')"
            ),
        }
    }
}

impl std::error::Error for NodePropertyRuleError {}

/// A value type a `NodePropertyRule` can map nodes to. Captures the three
/// places Python's dynamic typing leaks into `NodePropertyRule`'s
/// behavior: how a YAML value is read, what a list-form entry means
/// (`_lookup` returns `True` for list membership), and Python truthiness
/// (which decides precedence fall-through in `value()`).
pub trait RuleValue: Sized + Clone {
    /// The value a list-form entry stands for, or `None` if this type has
    /// no list form.
    fn listed() -> Option<Self>;

    /// Parse one step value from the scenario dict.
    fn from_rule_value(value: &Value) -> Result<Self, String>;

    /// Serialize back to the scenario dict shape (`to_dict()`).
    fn to_rule_value(&self) -> Value;

    /// Python truthiness of the value.
    fn is_truthy(&self) -> bool;
}

impl RuleValue for bool {
    fn listed() -> Option<Self> {
        Some(true)
    }

    fn from_rule_value(value: &Value) -> Result<Self, String> {
        value
            .as_bool()
            .ok_or_else(|| format!("expected a boolean, got {value}"))
    }

    fn to_rule_value(&self) -> Value {
        Value::Bool(*self)
    }

    fn is_truthy(&self) -> bool {
        *self
    }
}

impl RuleValue for f64 {
    /// Python's list form yields `True`, which is `1` numerically.
    fn listed() -> Option<Self> {
        Some(1.0)
    }

    /// Accepts ints, floats and (like Python, where `True == 1`) booleans.
    fn from_rule_value(value: &Value) -> Result<Self, String> {
        match value {
            Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
            _ => value
                .as_f64()
                .ok_or_else(|| format!("expected a number, got {value}")),
        }
    }

    fn to_rule_value(&self) -> Value {
        serde_json::json!(*self)
    }

    fn is_truthy(&self) -> bool {
        *self != 0.0
    }
}

/// TTC overrides (`ttc_overrides`) name a predefined distribution
/// (`TTCDist.from_name`), the only form the Python path accepts - see
/// `native_settings.py::_flatten_ttc_dists`.
impl RuleValue for TtcDist {
    fn listed() -> Option<Self> {
        None
    }

    fn from_rule_value(value: &Value) -> Result<Self, String> {
        let name = value
            .as_str()
            .ok_or_else(|| format!("expected a TTC distribution name, got {value}"))?;
        named_ttc_dist(name).ok_or_else(|| format!("unknown TTC distribution name '{name}'"))
    }

    fn to_rule_value(&self) -> Value {
        self.to_dict()
    }

    /// A Python `TTCDist` object (and a non-empty name) is always truthy.
    fn is_truthy(&self) -> bool {
        true
    }
}

/// The steps configured for one asset type/name: either the list form
/// (`[read, write]`) or the mapping form (`{read: 0.5}`).
#[derive(Debug, Clone, PartialEq)]
pub enum StepValues<T> {
    Listed(Vec<String>),
    Valued(BTreeMap<String, T>),
}

impl<T: RuleValue> StepValues<T> {
    /// Port of `_lookup`'s per-asset half: `True` (here `T::listed()`) for
    /// list membership, the mapped value otherwise.
    fn lookup(&self, step_name: &str) -> Option<T> {
        match self {
            StepValues::Listed(steps) => {
                if steps.iter().any(|s| s == step_name) {
                    T::listed()
                } else {
                    None
                }
            }
            StepValues::Valued(values) => values.get(step_name).cloned(),
        }
    }

    fn len(&self) -> usize {
        match self {
            StepValues::Listed(steps) => steps.len(),
            StepValues::Valued(values) => values.len(),
        }
    }

    fn to_value(&self) -> Value {
        match self {
            StepValues::Listed(steps) => {
                Value::Array(steps.iter().cloned().map(Value::String).collect())
            }
            StepValues::Valued(values) => Value::Object(
                values
                    .iter()
                    .map(|(step, v)| (step.clone(), v.to_rule_value()))
                    .collect(),
            ),
        }
    }
}

type ByAsset<T> = BTreeMap<String, StepValues<T>>;

/// Port of `NodePropertyRule`. Python's `default` dataclass field is not
/// ported: `value()` never reads it (it uses its `default` *argument*),
/// so it's dead state - see `PORTING_NOTES.md` §12.
#[derive(Debug, Clone, PartialEq)]
pub struct NodePropertyRule<T> {
    pub by_asset_type: Option<ByAsset<T>>,
    pub by_asset_name: Option<ByAsset<T>>,
}

impl<T: RuleValue> NodePropertyRule<T> {
    /// Port of `NodePropertyRule.__init__` + `__post_init__`.
    pub fn new(
        by_asset_type: Option<ByAsset<T>>,
        by_asset_name: Option<ByAsset<T>>,
    ) -> Result<Self, NodePropertyRuleError> {
        if by_asset_type.is_none() && by_asset_name.is_none() {
            return Err(NodePropertyRuleError::MissingFields);
        }
        Ok(NodePropertyRule {
            by_asset_type,
            by_asset_name,
        })
    }

    /// Port of `from_dict()` / `_validate_dict()`.
    pub fn from_value(value: &Value) -> Result<Self, NodePropertyRuleError> {
        let obj = value
            .as_object()
            .ok_or(NodePropertyRuleError::NotAMapping)?;

        const ALLOWED_FIELDS: [&str; 2] = ["by_asset_type", "by_asset_name"];
        if !ALLOWED_FIELDS.iter().any(|f| obj.contains_key(*f)) {
            return Err(NodePropertyRuleError::MissingFields);
        }
        let mut forbidden: Vec<String> = obj
            .keys()
            .filter(|k| !ALLOWED_FIELDS.contains(&k.as_str()))
            .cloned()
            .collect();
        if !forbidden.is_empty() {
            forbidden.sort();
            return Err(NodePropertyRuleError::ForbiddenFields(forbidden));
        }

        Self::new(
            parse_by_asset(obj.get("by_asset_type"))?,
            parse_by_asset(obj.get("by_asset_name"))?,
        )
    }

    /// Port of `from_optional_dict()`: an absent or `null` rule is `None`.
    pub fn from_optional_value(
        value: Option<&Value>,
    ) -> Result<Option<Self>, NodePropertyRuleError> {
        match value {
            None | Some(Value::Null) => Ok(None),
            Some(v) => Self::from_value(v).map(Some),
        }
    }

    /// Port of `to_dict()`.
    pub fn to_value(&self) -> Value {
        let by_asset_to_value = |by_asset: &Option<ByAsset<T>>| match by_asset {
            None => Value::Null,
            Some(by_asset) => Value::Object(
                by_asset
                    .iter()
                    .map(|(asset, steps)| (asset.clone(), steps.to_value()))
                    .collect::<Map<String, Value>>(),
            ),
        };
        serde_json::json!({
            "by_asset_type": by_asset_to_value(&self.by_asset_type),
            "by_asset_name": by_asset_to_value(&self.by_asset_name),
        })
    }

    /// Port of `__len__`: the number of configured asset types + names.
    /// Python callers test a rule's truthiness (`if rule:`) with this, so
    /// an empty rule behaves like no rule in those places.
    pub fn len(&self) -> usize {
        self.by_asset_type.as_ref().map_or(0, |m| m.len())
            + self.by_asset_name.as_ref().map_or(0, |m| m.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Port of `value(node, default)`. Returns `None` where Python would
    /// return `default`, so `rule.value(..).unwrap_or(default)` is the
    /// exact Python call. Precedence: asset_name > asset_type, and a falsy
    /// value falls through like Python's `or` chain.
    pub fn value(&self, node: &AttackGraphNode, model: &Model) -> Option<T> {
        let asset = model.get_asset_by_id(node.model_asset?)?;

        let lookup = |by_asset: &Option<ByAsset<T>>, key: &str| {
            by_asset
                .as_ref()
                .and_then(|m| m.get(key))
                .and_then(|steps| steps.lookup(&node.name))
                .filter(RuleValue::is_truthy)
        };

        lookup(&self.by_asset_name, &asset.name)
            .or_else(|| lookup(&self.by_asset_type, &asset.asset_type))
    }

    /// Port of `per_node()`, keyed by node id instead of full name. Like
    /// Python (`value(n, None)`, keep non-`None`), only nodes with a
    /// truthy value are included.
    pub fn per_node(&self, graph: &AttackGraph, model: &Model) -> HashMap<AttackGraphNodeId, T> {
        graph
            .nodes
            .iter()
            .filter_map(|(id, node)| self.value(node, model).map(|v| (id, v)))
            .collect()
    }

    /// Number of configured steps across all assets, used only by tests
    /// and diagnostics.
    pub fn num_steps(&self) -> usize {
        [&self.by_asset_type, &self.by_asset_name]
            .into_iter()
            .flatten()
            .flat_map(|m| m.values())
            .map(StepValues::len)
            .sum()
    }
}

fn parse_by_asset<T: RuleValue>(
    value: Option<&Value>,
) -> Result<Option<ByAsset<T>>, NodePropertyRuleError> {
    let obj = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(obj)) => obj,
        Some(other) => {
            return Err(NodePropertyRuleError::InvalidShape(format!(
                "expected a mapping of asset to steps, got {other}"
            )))
        }
    };

    obj.iter()
        .map(|(asset, steps)| Ok((asset.clone(), parse_step_values(asset, steps)?)))
        .collect::<Result<ByAsset<T>, _>>()
        .map(Some)
}

fn parse_step_values<T: RuleValue>(
    asset: &str,
    steps: &Value,
) -> Result<StepValues<T>, NodePropertyRuleError> {
    match steps {
        Value::Array(items) => {
            if T::listed().is_none() {
                return Err(NodePropertyRuleError::ListFormNotAllowed {
                    asset: asset.to_string(),
                });
            }
            items
                .iter()
                .map(|item| {
                    item.as_str().map(str::to_string).ok_or_else(|| {
                        NodePropertyRuleError::InvalidShape(format!(
                            "step list for '{asset}' must contain strings, got {item}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map(StepValues::Listed)
        }
        Value::Object(values) => values
            .iter()
            // An explicit `null` value behaves as absent in Python
            // (`_lookup` returns it, the `or` chain skips it).
            .filter(|(_, v)| !v.is_null())
            .map(|(step, v)| {
                T::from_rule_value(v)
                    .map(|parsed| (step.clone(), parsed))
                    .map_err(|reason| NodePropertyRuleError::InvalidValue {
                        asset: asset.to_string(),
                        step: step.clone(),
                        reason,
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map(StepValues::Valued),
        other => Err(NodePropertyRuleError::InvalidShape(format!(
            "steps for '{asset}' must be a list or a mapping, got {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::wiper_attack_graph;
    use serde_json::json;

    fn node<'a>(graph: &'a AttackGraph, full_name: &str) -> &'a AttackGraphNode {
        &graph.nodes[graph.get_node_by_full_name(full_name).unwrap()]
    }

    fn full_names<T>(
        graph: &AttackGraph,
        model: &Model,
        per_node: &HashMap<AttackGraphNodeId, T>,
    ) -> Vec<String> {
        let mut names: Vec<String> = per_node
            .keys()
            .map(|&id| graph.full_name_of(id, Some(model)))
            .collect();
        names.sort();
        names
    }

    #[test]
    fn by_asset_name_beats_by_asset_type() {
        let (graph, model) = wiper_attack_graph();
        let rule = NodePropertyRule::<f64>::from_value(&json!({
            "by_asset_type": {"Device": {"infect": 0.5}},
            "by_asset_name": {"InfectedDevice": {"infect": 0.9}},
        }))
        .unwrap();

        assert_eq!(
            rule.value(node(&graph, "InfectedDevice:infect"), &model),
            Some(0.9)
        );
        assert_eq!(
            rule.value(node(&graph, "VulnerableDevice:infect"), &model),
            Some(0.5)
        );
    }

    #[test]
    fn unmatched_node_falls_back_to_default() {
        let (graph, model) = wiper_attack_graph();
        let rule = NodePropertyRule::<f64>::from_value(&json!({
            "by_asset_type": {"Device": {"infect": 0.5}},
        }))
        .unwrap();

        let data_read = node(&graph, "InfectedData:read");
        assert_eq!(rule.value(data_read, &model), None);
        assert_eq!(rule.value(data_read, &model).unwrap_or(0.0), 0.0);
    }

    #[test]
    fn falsy_by_asset_name_value_falls_through_to_by_asset_type() {
        // Python: `by_asset_val or asset_type_val or default` - a 0 reward
        // by name does not override the by-type value.
        let (graph, model) = wiper_attack_graph();
        let rule = NodePropertyRule::<f64>::from_value(&json!({
            "by_asset_type": {"Device": {"infect": 0.5}},
            "by_asset_name": {"InfectedDevice": {"infect": 0}},
        }))
        .unwrap();
        assert_eq!(
            rule.value(node(&graph, "InfectedDevice:infect"), &model),
            Some(0.5)
        );

        let bool_rule = NodePropertyRule::<bool>::from_value(&json!({
            "by_asset_name": {"InfectedDevice": {"infect": false}},
        }))
        .unwrap();
        assert_eq!(
            bool_rule.value(node(&graph, "InfectedDevice:infect"), &model),
            None
        );
    }

    #[test]
    fn list_form_means_listed_value() {
        let (graph, model) = wiper_attack_graph();
        let rule = NodePropertyRule::<bool>::from_value(&json!({
            "by_asset_type": {"Device": ["infect"]},
            "by_asset_name": {"InfectedData": ["read"]},
        }))
        .unwrap();

        let per_node = rule.per_node(&graph, &model);
        assert_eq!(
            full_names(&graph, &model, &per_node),
            vec![
                "InfectedData:read",
                "InfectedDevice:infect",
                "VulnerableDevice:infect"
            ]
        );
        assert!(per_node.values().all(|v| *v));

        let reward_rule = NodePropertyRule::<f64>::from_value(&json!({
            "by_asset_type": {"Device": ["infect"]},
        }))
        .unwrap();
        assert_eq!(
            reward_rule.value(node(&graph, "VulnerableDevice:infect"), &model),
            Some(1.0)
        );
    }

    #[test]
    fn per_node_only_contains_truthy_values() {
        let (graph, model) = wiper_attack_graph();
        let rule = NodePropertyRule::<f64>::from_value(&json!({
            "by_asset_type": {"Device": {"infect": 0.0}},
            "by_asset_name": {"InfectedData": {"read": 2}},
        }))
        .unwrap();
        let per_node = rule.per_node(&graph, &model);
        assert_eq!(
            full_names(&graph, &model, &per_node),
            vec!["InfectedData:read"]
        );
        assert_eq!(per_node.values().copied().collect::<Vec<_>>(), vec![2.0]);
    }

    #[test]
    fn forbidden_or_missing_fields_are_rejected() {
        // Port of `test_apply_scenario_observability_faulty`.
        assert_eq!(
            NodePropertyRule::<bool>::from_value(&json!({
                "NotAllowedKey": {"Data": ["read", "write", "delete"]}
            })),
            Err(NodePropertyRuleError::MissingFields)
        );
        assert_eq!(
            NodePropertyRule::<bool>::from_value(&json!({
                "by_asset_type": {"Data": ["read"]},
                "NotAllowedKey": {},
            })),
            Err(NodePropertyRuleError::ForbiddenFields(vec![
                "NotAllowedKey".into()
            ]))
        );
        // Both fields present but `null` - Python's `__post_init__` check.
        assert_eq!(
            NodePropertyRule::<bool>::from_value(&json!({"by_asset_type": null})),
            Err(NodePropertyRuleError::MissingFields)
        );
    }

    #[test]
    fn old_rewards_format_is_rejected() {
        // Port of `test_apply_scenario_rewards_old_format`.
        let old_format = json!({
            "OS App:notPresent": 2,
            "Data:5:notPresent": 1,
        });
        assert!(NodePropertyRule::<f64>::from_value(&old_format).is_err());
    }

    #[test]
    fn from_optional_value_maps_absent_and_null_to_none() {
        assert_eq!(NodePropertyRule::<f64>::from_optional_value(None), Ok(None));
        assert_eq!(
            NodePropertyRule::<f64>::from_optional_value(Some(&Value::Null)),
            Ok(None)
        );
    }

    #[test]
    fn ttc_rule_parses_named_distributions_and_rejects_list_form() {
        let (graph, model) = wiper_attack_graph();
        let rule = NodePropertyRule::<TtcDist>::from_value(&json!({
            "by_asset_type": {"Device": {"infect": "HardAndUncertain"}},
        }))
        .unwrap();
        assert_eq!(
            rule.value(node(&graph, "InfectedDevice:infect"), &model),
            named_ttc_dist("HardAndUncertain")
        );

        assert_eq!(
            NodePropertyRule::<TtcDist>::from_value(&json!({
                "by_asset_type": {"Device": ["infect"]},
            })),
            Err(NodePropertyRuleError::ListFormNotAllowed {
                asset: "Device".into()
            })
        );
        assert!(matches!(
            NodePropertyRule::<TtcDist>::from_value(&json!({
                "by_asset_type": {"Device": {"infect": "NotADistribution"}},
            })),
            Err(NodePropertyRuleError::InvalidValue { .. })
        ));
    }

    #[test]
    fn to_value_round_trips() {
        let original = json!({
            "by_asset_type": {"Device": ["infect"], "Data": {"read": true}},
            "by_asset_name": null,
        });
        let rule = NodePropertyRule::<bool>::from_value(&original).unwrap();
        assert_eq!(rule.to_value(), original);
        assert_eq!(rule.len(), 2);
        assert_eq!(rule.num_steps(), 2);
    }

    #[test]
    fn node_without_model_asset_gets_default() {
        let (mut graph, model) = wiper_attack_graph();
        let step_id = graph.nodes[graph
            .get_node_by_full_name("InfectedDevice:infect")
            .unwrap()]
        .lg_attack_step;
        let orphan = graph
            .add_node(step_id, None, None, None, None, None, None)
            .unwrap();
        let rule = NodePropertyRule::<bool>::from_value(&json!({
            "by_asset_type": {"Device": ["infect"]},
        }))
        .unwrap();
        assert_eq!(rule.value(&graph.nodes[orphan], &model), None);
    }
}
