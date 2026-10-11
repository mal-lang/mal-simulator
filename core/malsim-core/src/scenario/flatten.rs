//! Resolves `AttackerSettings`/`DefenderSettings` against a concrete graph
//! into the flat, id-keyed inputs `Simulator::reset` consumes - the
//! Rust-only counterpart of `python/malsim/mal_simulator/native_settings.py`
//! (`flatten_attacker_settings`/`flatten_defender_settings`) plus
//! `attacker_state_factories.py::get_entry_points`. See `PORTING_NOTES.md`
//! §7 C3.
//!
//! Each function keeps the exact per-node semantics of the Python helper
//! it mirrors (`node_is_actionable`, `node_is_observable`,
//! `node_false_*_rate`), including the "empty rule behaves like no rule"
//! quirks those helpers get from testing `if rule:` (`NodePropertyRule.
//! __len__`).

use std::collections::{BTreeSet, HashMap, HashSet};

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use maltoolbox_model::Model;
use rand::{Rng, RngExt};

use crate::scenario::agent_settings::{AttackerSettings, DefenderSettings, EntryPoints};
use crate::scenario::node_property_rule::NodePropertyRule;
use crate::settings::{FlatAttackerSettings, FlatDefenderSettings};
use crate::ttc::TtcDist;

/// Port of `get_entry_points`: the single entry-point set, or one set
/// sampled uniformly from the alternatives. An empty list of alternatives
/// (only constructible programmatically) yields no entry points.
pub fn get_entry_points(
    attacker_settings: &AttackerSettings<AttackGraphNodeId>,
    rng: &mut impl Rng,
) -> BTreeSet<AttackGraphNodeId> {
    match &attacker_settings.entry_points {
        EntryPoints::Single(entry_points) => entry_points.clone(),
        EntryPoints::Multiple(options) if options.is_empty() => BTreeSet::new(),
        EntryPoints::Multiple(options) => options[rng.random_range(0..options.len())].clone(),
    }
}

/// Port of `node_is_actionable`: an empty rule makes every node actionable.
fn node_is_actionable(
    rule: &NodePropertyRule<bool>,
    node: &maltoolbox_attackgraph::AttackGraphNode,
    model: &Model,
) -> bool {
    if rule.is_empty() {
        return true;
    }
    rule.value(node, model).unwrap_or(false)
}

/// Port of `_flatten_actionable_steps`.
fn flatten_actionable_steps(
    rule: Option<&NodePropertyRule<bool>>,
    graph: &AttackGraph,
    model: &Model,
) -> Option<HashSet<AttackGraphNodeId>> {
    let rule = rule?;
    Some(
        graph
            .nodes
            .iter()
            .filter(|(_, node)| node_is_actionable(rule, node, model))
            .map(|(id, _)| id)
            .collect(),
    )
}

/// Port of `_flatten_observable_steps` (`node_is_observable` has no
/// empty-rule special case: an empty rule observes nothing).
fn flatten_observable_steps(
    rule: Option<&NodePropertyRule<bool>>,
    graph: &AttackGraph,
    model: &Model,
) -> Option<HashSet<AttackGraphNodeId>> {
    let rule = rule?;
    Some(
        graph
            .nodes
            .iter()
            .filter(|(_, node)| rule.value(node, model).unwrap_or(false))
            .map(|(id, _)| id)
            .collect(),
    )
}

/// Port of `_flatten_rate_map` with `node_false_positive_rate`/
/// `node_false_negative_rate` (identical bodies): every node gets an entry,
/// `0.0` where the rule gives nothing or is empty.
fn flatten_rate_map(
    rule: Option<&NodePropertyRule<f64>>,
    graph: &AttackGraph,
    model: &Model,
) -> Option<HashMap<AttackGraphNodeId, f64>> {
    let rule = rule?;
    Some(
        graph
            .nodes
            .iter()
            .map(|(id, node)| {
                let rate = if rule.is_empty() {
                    0.0
                } else {
                    rule.value(node, model).unwrap_or(0.0)
                };
                (id, rate)
            })
            .collect(),
    )
}

/// Port of `_flatten_ttc_dists`: `None` when there's no rule or it
/// matches no node.
fn flatten_ttc_dists(
    rule: Option<&NodePropertyRule<TtcDist>>,
    graph: &AttackGraph,
    model: &Model,
) -> Option<HashMap<AttackGraphNodeId, TtcDist>> {
    let per_node = rule?.per_node(graph, model);
    (!per_node.is_empty()).then_some(per_node)
}

/// Port of `flatten_attacker_settings`. `entry_points` is the
/// already-sampled single set (see `get_entry_points`).
pub fn flatten_attacker_settings(
    attacker_settings: &AttackerSettings<AttackGraphNodeId>,
    graph: &AttackGraph,
    model: &Model,
    entry_points: &BTreeSet<AttackGraphNodeId>,
) -> FlatAttackerSettings {
    FlatAttackerSettings {
        entry_points: entry_points.iter().copied().collect(),
        goals: attacker_settings.goals.iter().copied().collect(),
        actionable_steps: flatten_actionable_steps(
            attacker_settings.actionable_steps.as_ref(),
            graph,
            model,
        ),
        ttc_dists: flatten_ttc_dists(attacker_settings.ttc_dists.as_ref(), graph, model),
    }
}

/// Port of `flatten_defender_settings`.
pub fn flatten_defender_settings(
    defender_settings: &DefenderSettings,
    graph: &AttackGraph,
    model: &Model,
) -> FlatDefenderSettings {
    FlatDefenderSettings {
        actionable_steps: flatten_actionable_steps(
            defender_settings.actionable_steps.as_ref(),
            graph,
            model,
        ),
        observable_steps: flatten_observable_steps(
            defender_settings.observable_steps.as_ref(),
            graph,
            model,
        ),
        false_positive_rates: flatten_rate_map(
            defender_settings.false_positive_rates.as_ref(),
            graph,
            model,
        ),
        false_negative_rates: flatten_rate_map(
            defender_settings.false_negative_rates.as_ref(),
            graph,
            model,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::agent_settings::{agent_settings_from_dict, AgentSettings};
    use crate::test_fixtures::wiper_attack_graph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use serde_json::{json, Value};

    fn attacker(graph: &AttackGraph, d: Value) -> AttackerSettings<AttackGraphNodeId> {
        match agent_settings_from_dict("a", &d).unwrap() {
            AgentSettings::Attacker(a) => a.convert_to_attack_graph_nodes(graph).unwrap(),
            AgentSettings::Defender(_) => panic!("expected an attacker"),
        }
    }

    fn defender(d: Value) -> DefenderSettings {
        match agent_settings_from_dict("d", &d).unwrap() {
            AgentSettings::Defender(d) => d,
            AgentSettings::Attacker(_) => panic!("expected a defender"),
        }
    }

    fn ids(graph: &AttackGraph, names: &[&str]) -> HashSet<AttackGraphNodeId> {
        names
            .iter()
            .map(|n| graph.get_node_by_full_name(n).unwrap())
            .collect()
    }

    #[test]
    fn attacker_without_rules_flattens_to_none() {
        let (graph, model) = wiper_attack_graph();
        let a = attacker(
            &graph,
            json!({"type": "attacker", "entry_points": ["InfectedDevice:infect"]}),
        );
        let eps = get_entry_points(&a, &mut StdRng::seed_from_u64(0));
        let flat = flatten_attacker_settings(&a, &graph, &model, &eps);
        assert_eq!(flat.entry_points, ids(&graph, &["InfectedDevice:infect"]));
        assert!(flat.goals.is_empty());
        assert_eq!(flat.actionable_steps, None);
        assert_eq!(flat.ttc_dists, None);
    }

    #[test]
    fn attacker_rules_resolve_to_id_maps() {
        let (graph, model) = wiper_attack_graph();
        let a = attacker(
            &graph,
            json!({
                "type": "attacker",
                "goals": ["InfectedData:read"],
                "actionable_steps": {"by_asset_type": {"Device": ["infect"]}},
                "ttc_overrides": {"by_asset_name": {"VulnerableDevice": {"infect": "HardAndUncertain"}}},
            }),
        );
        let flat = flatten_attacker_settings(&a, &graph, &model, &BTreeSet::new());
        assert_eq!(flat.goals, ids(&graph, &["InfectedData:read"]));
        assert_eq!(
            flat.actionable_steps,
            Some(ids(
                &graph,
                &["InfectedDevice:infect", "VulnerableDevice:infect"]
            ))
        );
        let ttc = flat.ttc_dists.unwrap();
        assert_eq!(
            ttc.keys().copied().collect::<HashSet<_>>(),
            ids(&graph, &["VulnerableDevice:infect"])
        );
    }

    #[test]
    fn ttc_rule_matching_nothing_is_none() {
        let (graph, model) = wiper_attack_graph();
        let a = attacker(
            &graph,
            json!({"type": "attacker", "ttc_overrides": {"by_asset_type": {"Nope": {"x": "HardAndUncertain"}}}}),
        );
        assert_eq!(
            flatten_attacker_settings(&a, &graph, &model, &BTreeSet::new()).ttc_dists,
            None
        );
    }

    #[test]
    fn empty_actionability_rule_makes_everything_actionable_but_observes_nothing() {
        let (graph, model) = wiper_attack_graph();
        let d = defender(json!({
            "type": "defender",
            "actionable_steps": {"by_asset_type": {}},
            "observable_steps": {"by_asset_type": {}},
        }));
        let flat = flatten_defender_settings(&d, &graph, &model);
        assert_eq!(flat.actionable_steps.unwrap().len(), graph.nodes.len());
        assert_eq!(flat.observable_steps, Some(HashSet::new()));
    }

    #[test]
    fn rate_maps_cover_every_node() {
        let (graph, model) = wiper_attack_graph();
        let d = defender(json!({
            "type": "defender",
            "false_positive_rates": {"by_asset_type": {"Device": {"infect": 0.25}}},
            "false_negative_rates": {"by_asset_type": {}},
        }));
        let flat = flatten_defender_settings(&d, &graph, &model);
        let fpr = flat.false_positive_rates.unwrap();
        assert_eq!(fpr.len(), graph.nodes.len());
        let infect = ids(
            &graph,
            &["InfectedDevice:infect", "VulnerableDevice:infect"],
        );
        for (id, rate) in &fpr {
            assert_eq!(*rate, if infect.contains(id) { 0.25 } else { 0.0 });
        }
        let fnr = flat.false_negative_rates.unwrap();
        assert_eq!(fnr.len(), graph.nodes.len());
        assert!(fnr.values().all(|r| *r == 0.0));
        assert_eq!(flat.actionable_steps, None);
        assert_eq!(flat.observable_steps, None);
    }

    #[test]
    fn get_entry_points_samples_one_alternative() {
        let (graph, _model) = wiper_attack_graph();
        let a = attacker(
            &graph,
            json!({"type": "attacker", "entry_points": [["InfectedDevice:infect"], ["VulnerableDevice:infect"]]}),
        );
        let options = [
            ids(&graph, &["InfectedDevice:infect"]),
            ids(&graph, &["VulnerableDevice:infect"]),
        ];
        let mut rng = StdRng::seed_from_u64(7);
        let mut seen = [false, false];
        for _ in 0..64 {
            let chosen: HashSet<_> = get_entry_points(&a, &mut rng).into_iter().collect();
            let index = options
                .iter()
                .position(|o| *o == chosen)
                .expect("one of the options");
            seen[index] = true;
        }
        assert_eq!(seen, [true, true]);
    }
}
