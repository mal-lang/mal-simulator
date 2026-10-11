//! Rust port of `python/malsim/mal_simulator/attacker_step.py` - see
//! `PORTING_NOTES.md` §5 Phase A7.
//!
//! `state_query.py::node_ttc_value` is pulled forward into this module (as
//! the private `resolve_ttc_value`, below) because `attempt_attacker_step`
//! is its only caller in the hot loop - same precedent as A3 pulling
//! `TtcMode` forward from `config/sim_settings.py` into `graph_state.rs`.
//! No `AttackerState` exists on the Rust side yet (full per-agent runtime
//! state is A9's job - see `PORTING_NOTES.md` §3/§5), so the two maps
//! `node_ttc_value` reads (`attacker_state.ttc_values` - an agent-level
//! override - and `sim_state.graph_state.ttc_values` - the graph-level
//! default computed once at `reset()` by `graph_state::
//! compute_initial_graph_state`) are passed in directly, continuing A5/
//! A6's "already-flattened" pattern rather than a ported settings/state
//! struct.

use std::collections::{HashMap, HashSet};
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use rand::Rng;

use crate::attack_surface::get_effects_of_attack_step;
use crate::graph_state::{resolve_ttc_dist, GraphStateError, TtcMode};
use crate::graph_utils::{node_is_live, node_is_traversable, GraphUtilsError};
use crate::ttc::TtcDist;

#[derive(Debug, Clone, PartialEq)]
pub enum AttackerStepError {
    GraphUtils(GraphUtilsError),
    TtcDist(GraphStateError),
    /// Mirrors Python's bare `assert node in attacker_state.sim_state.
    /// graph_state.ttc_values` in `state_query.py::node_ttc_value`,
    /// reached only in `ExpectedValue`/`PreSample` mode when neither the
    /// agent-level override nor the graph-level default has a value for
    /// this node.
    MissingTtcValue(AttackGraphNodeId),
    /// Mirrors Python's `assert node == sim_state.attack_graph.nodes[node.id]`
    /// in `attacker_step` - translated to "this node id no longer exists
    /// in the graph" since the Rust side works with ids, not stale Python
    /// object references. See `PORTING_NOTES.md` §10 for why this is the
    /// chosen translation.
    NodeNotInGraph(AttackGraphNodeId),
}

impl From<GraphUtilsError> for AttackerStepError {
    fn from(e: GraphUtilsError) -> Self {
        AttackerStepError::GraphUtils(e)
    }
}

impl From<GraphStateError> for AttackerStepError {
    fn from(e: GraphStateError) -> Self {
        AttackerStepError::TtcDist(e)
    }
}

impl fmt::Display for AttackerStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttackerStepError::GraphUtils(e) => write!(f, "{e}"),
            AttackerStepError::TtcDist(e) => write!(f, "{e}"),
            AttackerStepError::MissingTtcValue(id) => {
                write!(f, "node {id:?} does not have a ttc value")
            }
            AttackerStepError::NodeNotInGraph(id) => write!(
                f,
                "tried to step a node ({id:?}) that is not part of this simulator's attack graph"
            ),
        }
    }
}

impl std::error::Error for AttackerStepError {}

/// Port of `attacker_is_terminated`. Takes the attacker's current action
/// surface/goals/performed-nodes directly rather than an `AttackerState` -
/// no such type exists on the Rust side yet (A9's job). An empty `goals`
/// set mirrors Python's falsy-empty-frozenset check (`if goals:` - the
/// `AttackerSettings.goals` field defaults to an empty `frozenset`, never
/// `None`).
pub fn attacker_is_terminated(
    action_surface: &HashSet<AttackGraphNodeId>,
    goals: &HashSet<AttackGraphNodeId>,
    performed_nodes: &HashSet<AttackGraphNodeId>,
) -> bool {
    if action_surface.is_empty() {
        return true;
    }
    if !goals.is_empty() {
        return goals.is_subset(performed_nodes);
    }
    false
}

/// Pulled-forward port of `state_query.py::node_ttc_value` - see module
/// docs. Returns `None` exactly where Python's assert would fire.
fn resolve_ttc_value(
    node_id: AttackGraphNodeId,
    ttc_value_overrides: Option<&HashMap<AttackGraphNodeId, f64>>,
    graph_ttc_values: &HashMap<AttackGraphNodeId, f64>,
) -> Option<f64> {
    ttc_value_overrides
        .and_then(|m| m.get(&node_id))
        .or_else(|| graph_ttc_values.get(&node_id))
        .copied()
}

/// Port of `attempt_attacker_step`.
///
/// `num_attempts_before` mirrors `agent.num_attempts[node]` (the attempt
/// count *before* this attempt) - defaulting a missing entry to 0 is the
/// caller's job (mirrors the `dict.fromkeys(attack_graph.attack_steps, 0)`
/// every node is seeded with at reset, per `attacker_state_factories.py`).
///
/// `ttc_dist` is resolved **unconditionally**, even in `Disabled` mode
/// where its result goes unused - ported as-is from Python's own
/// ordering (`ttc_dist = ...` happens before the `if ttc_mode ==
/// TTCMode.DISABLED` check), not simplified into a lazier short-circuit,
/// so a malformed per-node TTC dict still surfaces as an error in
/// `Disabled` mode exactly like it does in Python.
///
/// **The `ExpectedValue`/`PreSample` branch's `num_attempts + 1 >=
/// ttc_value` comparison is two attempts ahead of what the variable name
/// suggests, ported as-is - not a transcription bug.** Python's own
/// `num_attempts = agent.num_attempts[node] + 1` already adds one before
/// this branch adds a second, so the comparison actually reads
/// `agent.num_attempts[node] + 2 >= ttc_value`. See `PORTING_NOTES.md`
/// §10 for the full callout.
#[allow(clippy::too_many_arguments)]
pub fn attempt_attacker_step(
    graph: &AttackGraph,
    rng: &mut impl Rng,
    ttc_mode: TtcMode,
    node_id: AttackGraphNodeId,
    num_attempts_before: u64,
    ttc_dist_override: Option<&TtcDist>,
    ttc_value_overrides: Option<&HashMap<AttackGraphNodeId, f64>>,
    graph_ttc_values: &HashMap<AttackGraphNodeId, f64>,
) -> Result<bool, AttackerStepError> {
    let node = &graph.nodes[node_id];
    let num_attempts = num_attempts_before + 1;
    let ttc_dist = resolve_ttc_dist(node, ttc_dist_override)?;

    match ttc_mode {
        TtcMode::Disabled => Ok(true),
        TtcMode::EffortBasedPerStepSample => {
            Ok(ttc_dist.attempt_ttc_with_effort(num_attempts, rng))
        }
        TtcMode::PerStepSample => Ok(ttc_dist.sample_value(rng) <= 1.0),
        TtcMode::ExpectedValue | TtcMode::PreSample => {
            let ttc_value = resolve_ttc_value(node_id, ttc_value_overrides, graph_ttc_values)
                .ok_or(AttackerStepError::MissingTtcValue(node_id))?;
            Ok((num_attempts + 1) as f64 >= ttc_value)
        }
    }
}

/// Port of `attacker_step_effects`.
pub fn attacker_step_effects(
    graph: &AttackGraph,
    node_id: AttackGraphNodeId,
    performed_nodes: &HashSet<AttackGraphNodeId>,
    impossible_attack_steps: &HashSet<AttackGraphNodeId>,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
    necessity_per_node: &HashMap<AttackGraphNodeId, bool>,
) -> Result<Vec<AttackGraphNodeId>, GraphUtilsError> {
    let effects = get_effects_of_attack_step(
        graph,
        node_id,
        performed_nodes,
        impossible_attack_steps,
        enabled_defenses,
        necessity_per_node,
    )?;
    Ok(effects.into_iter().collect())
}

/// Port of `attacker_step`.
///
/// **Entry points bypass both the action-surface and traversability
/// checks entirely** (`can_compromise = true` unconditionally) - ported
/// as-is from Python's own `# TODO: should this actually be the case?`
/// comment on this exact branch, not tightened into a stricter check.
///
/// Like Python, a node that fails the `can_compromise` check is silently
/// skipped (Python logs a warning) rather than treated as an error - only
/// the graph-membership check below is a hard failure.
#[allow(clippy::too_many_arguments)]
pub fn attacker_step(
    graph: &AttackGraph,
    rng: &mut impl Rng,
    ttc_mode: TtcMode,
    nodes: &[AttackGraphNodeId],
    entry_points: &HashSet<AttackGraphNodeId>,
    action_surface: &HashSet<AttackGraphNodeId>,
    performed_nodes: &HashSet<AttackGraphNodeId>,
    num_attempts: &HashMap<AttackGraphNodeId, u64>,
    ttc_dist_overrides: Option<&HashMap<AttackGraphNodeId, TtcDist>>,
    ttc_value_overrides: Option<&HashMap<AttackGraphNodeId, f64>>,
    graph_ttc_values: &HashMap<AttackGraphNodeId, f64>,
    impossible_attack_steps: &HashSet<AttackGraphNodeId>,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
    necessity_per_node: &HashMap<AttackGraphNodeId, bool>,
) -> Result<(Vec<AttackGraphNodeId>, Vec<AttackGraphNodeId>), AttackerStepError> {
    let mut successful_compromises = Vec::new();
    let mut attempted_compromises = Vec::new();

    for &node_id in nodes {
        if !node_is_live(graph, node_id) {
            return Err(AttackerStepError::NodeNotInGraph(node_id));
        }

        let can_compromise = if entry_points.contains(&node_id) {
            true
        } else {
            action_surface.contains(&node_id)
                && node_is_traversable(
                    graph,
                    node_id,
                    performed_nodes,
                    impossible_attack_steps,
                    enabled_defenses,
                    necessity_per_node,
                )?
        };

        if !can_compromise {
            continue;
        }

        let num_attempts_before = num_attempts.get(&node_id).copied().unwrap_or(0);
        let ttc_override = ttc_dist_overrides.and_then(|m| m.get(&node_id));

        let succeeded = attempt_attacker_step(
            graph,
            rng,
            ttc_mode,
            node_id,
            num_attempts_before,
            ttc_override,
            ttc_value_overrides,
            graph_ttc_values,
        )?;

        if succeeded {
            successful_compromises.push(node_id);
            let effects = attacker_step_effects(
                graph,
                node_id,
                performed_nodes,
                impossible_attack_steps,
                enabled_defenses,
                necessity_per_node,
            )?;
            successful_compromises.extend(effects);
        } else {
            attempted_compromises.push(node_id);
        }
    }

    Ok((successful_compromises, attempted_compromises))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{add_dummy_node, dummy_graph};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(1)
    }

    // --- attacker_is_terminated ---

    #[test]
    fn terminated_when_action_surface_empty() {
        let empty = HashSet::new();
        assert!(attacker_is_terminated(&empty, &empty, &empty));
    }

    #[test]
    fn not_terminated_without_goals_and_nonempty_surface() {
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let surface: HashSet<_> = [node].into_iter().collect();
        let empty = HashSet::new();
        assert!(!attacker_is_terminated(&surface, &empty, &empty));
    }

    #[test]
    fn terminated_when_all_goals_performed() {
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let goal = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let surface: HashSet<_> = [node].into_iter().collect();
        let goals: HashSet<_> = [goal].into_iter().collect();
        let performed: HashSet<_> = [goal].into_iter().collect();
        assert!(attacker_is_terminated(&surface, &goals, &performed));
    }

    #[test]
    fn not_terminated_when_some_goals_unperformed() {
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let goal1 = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let goal2 = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let surface: HashSet<_> = [node].into_iter().collect();
        let goals: HashSet<_> = [goal1, goal2].into_iter().collect();
        let performed: HashSet<_> = [goal1].into_iter().collect();
        assert!(!attacker_is_terminated(&surface, &goals, &performed));
    }

    // --- attempt_attacker_step ---

    #[test]
    fn attempt_disabled_mode_always_succeeds() {
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let empty_map = HashMap::new();
        let mut r = rng();

        let result = attempt_attacker_step(
            &graph,
            &mut r,
            TtcMode::Disabled,
            node,
            0,
            None,
            None,
            &empty_map,
        )
        .unwrap();
        assert!(result);
    }

    #[test]
    fn attempt_disabled_mode_still_surfaces_malformed_ttc_dict() {
        // Ported subtlety: Python resolves `ttc_dist` before checking
        // `ttc_mode == DISABLED`, so a malformed per-node `ttc` dict still
        // errors out even though the result would never be used.
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[node].ttc = Some(serde_json::json!({}));
        let empty_map = HashMap::new();
        let mut r = rng();

        let result = attempt_attacker_step(
            &graph,
            &mut r,
            TtcMode::Disabled,
            node,
            0,
            None,
            None,
            &empty_map,
        );
        assert!(matches!(result, Err(AttackerStepError::TtcDist(_))));
    }

    #[test]
    fn attempt_effort_based_mode_uses_ttc_dist_success_probability() {
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        // Default dist for "or" steps is "Instant" (Bernoulli(1.0)):
        // success_probability(effort) is 1.0 for any effort >= 1.
        let empty_map = HashMap::new();
        let mut r = rng();

        let result = attempt_attacker_step(
            &graph,
            &mut r,
            TtcMode::EffortBasedPerStepSample,
            node,
            0,
            None,
            None,
            &empty_map,
        )
        .unwrap();
        assert!(result);
    }

    #[test]
    fn attempt_per_step_sample_mode_uses_sampled_value() {
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        // Default "Instant" dist always samples 1.0, so <= 1 is always true.
        let empty_map = HashMap::new();
        let mut r = rng();

        let result = attempt_attacker_step(
            &graph,
            &mut r,
            TtcMode::PerStepSample,
            node,
            0,
            None,
            None,
            &empty_map,
        )
        .unwrap();
        assert!(result);
    }

    #[test]
    fn attempt_expected_value_mode_requires_a_ttc_value() {
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let empty_map = HashMap::new();
        let mut r = rng();

        let result = attempt_attacker_step(
            &graph,
            &mut r,
            TtcMode::ExpectedValue,
            node,
            0,
            None,
            None,
            &empty_map,
        );
        assert!(matches!(result, Err(AttackerStepError::MissingTtcValue(_))));
    }

    #[test]
    fn attempt_expected_value_mode_agent_override_wins_over_graph_value() {
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let graph_values: HashMap<_, _> = [(node, 100.0)].into_iter().collect();
        let agent_override: HashMap<_, _> = [(node, 1.0)].into_iter().collect();
        let mut r = rng();

        // Agent override (1.0) is used instead of the graph value (100.0):
        // with num_attempts_before=0, `num_attempts + 1 == 2 >= 1.0` succeeds.
        let result = attempt_attacker_step(
            &graph,
            &mut r,
            TtcMode::ExpectedValue,
            node,
            0,
            None,
            Some(&agent_override),
            &graph_values,
        )
        .unwrap();
        assert!(result);
    }

    #[test]
    fn attempt_expected_value_mode_comparison_is_two_attempts_ahead() {
        // Documents/locks in the "two attempts ahead" oddity: with
        // num_attempts_before=0, the comparison is `0 + 2 >= ttc_value`.
        // A ttc_value of 2.0 succeeds on the very first attempt even
        // though no attempt has happened yet.
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let graph_values: HashMap<_, _> = [(node, 2.0)].into_iter().collect();
        let mut r = rng();

        let result = attempt_attacker_step(
            &graph,
            &mut r,
            TtcMode::ExpectedValue,
            node,
            0,
            None,
            None,
            &graph_values,
        )
        .unwrap();
        assert!(result);

        // A ttc_value just above that boundary (2.1) does not succeed yet.
        let graph_values: HashMap<_, _> = [(node, 2.1)].into_iter().collect();
        let mut r = rng();
        let result = attempt_attacker_step(
            &graph,
            &mut r,
            TtcMode::ExpectedValue,
            node,
            0,
            None,
            None,
            &graph_values,
        )
        .unwrap();
        assert!(!result);
    }

    // --- attacker_step ---

    fn empty_sets() -> (
        HashSet<AttackGraphNodeId>,
        HashSet<AttackGraphNodeId>,
        HashSet<AttackGraphNodeId>,
    ) {
        (HashSet::new(), HashSet::new(), HashSet::new())
    }

    #[test]
    fn step_fails_node_not_in_graph() {
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.remove_node(node).unwrap();

        let (empty_set, empty_set2, empty_set3) = empty_sets();
        let empty_map_u64: HashMap<AttackGraphNodeId, u64> = HashMap::new();
        let empty_map_f64: HashMap<AttackGraphNodeId, f64> = HashMap::new();
        let empty_map_bool: HashMap<AttackGraphNodeId, bool> = HashMap::new();
        let mut r = rng();

        let result = attacker_step(
            &graph,
            &mut r,
            TtcMode::Disabled,
            &[node],
            &empty_set,
            &empty_set2,
            &empty_set3,
            &empty_map_u64,
            None,
            None,
            &empty_map_f64,
            &empty_set,
            &empty_set,
            &empty_map_bool,
        );
        assert!(matches!(result, Err(AttackerStepError::NodeNotInGraph(_))));
    }

    #[test]
    fn step_skips_node_outside_action_surface_and_not_entry_point() {
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");

        let empty_set: HashSet<AttackGraphNodeId> = HashSet::new();
        let empty_map_u64: HashMap<AttackGraphNodeId, u64> = HashMap::new();
        let empty_map_f64: HashMap<AttackGraphNodeId, f64> = HashMap::new();
        let empty_map_bool: HashMap<AttackGraphNodeId, bool> = HashMap::new();
        let mut r = rng();

        // Not in entry_points, not in action_surface -> skipped, no error.
        let (successful, attempted) = attacker_step(
            &graph,
            &mut r,
            TtcMode::Disabled,
            &[node],
            &empty_set,
            &empty_set,
            &empty_set,
            &empty_map_u64,
            None,
            None,
            &empty_map_f64,
            &empty_set,
            &empty_set,
            &empty_map_bool,
        )
        .unwrap();
        assert!(successful.is_empty());
        assert!(attempted.is_empty());
    }

    /// Port of the first case of the old
    /// `tests/test_mal_simulator.py::test_attacker_step` ("Can not attack
    /// the notPresent step"): a defense node passed as an attacker action,
    /// neither on the action surface nor an entry point, is skipped.
    #[test]
    fn step_skips_defense_node_outside_action_surface_and_not_entry_point() {
        let mut graph = dummy_graph();
        let defense = add_dummy_node(&mut graph, "DummyDefenseAttackStep");

        let empty_set: HashSet<AttackGraphNodeId> = HashSet::new();
        let empty_map_u64: HashMap<AttackGraphNodeId, u64> = HashMap::new();
        let empty_map_f64: HashMap<AttackGraphNodeId, f64> = HashMap::new();
        let empty_map_bool: HashMap<AttackGraphNodeId, bool> = HashMap::new();
        let mut r = rng();

        let (successful, attempted) = attacker_step(
            &graph,
            &mut r,
            TtcMode::Disabled,
            &[defense],
            &empty_set,
            &empty_set,
            &empty_set,
            &empty_map_u64,
            None,
            None,
            &empty_map_f64,
            &empty_set,
            &empty_set,
            &empty_map_bool,
        )
        .unwrap();
        assert!(successful.is_empty());
        assert!(attempted.is_empty());
    }

    #[test]
    fn step_entry_point_bypasses_action_surface_and_traversability() {
        let mut graph = dummy_graph();
        // An "and" step with an unperformed necessary parent is not
        // traversable - but entry points bypass that check entirely.
        let parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let node = add_dummy_node(&mut graph, "DummyAndAttackStep");
        graph.nodes[node].parents.insert(parent);
        graph.nodes[parent].children.insert(node);

        let entry_points: HashSet<_> = [node].into_iter().collect();
        let empty_set: HashSet<AttackGraphNodeId> = HashSet::new();
        let empty_map_u64: HashMap<AttackGraphNodeId, u64> = HashMap::new();
        let empty_map_f64: HashMap<AttackGraphNodeId, f64> = HashMap::new();
        let necessity: HashMap<_, _> = [(node, true), (parent, true)].into_iter().collect();
        let mut r = rng();

        let (successful, attempted) = attacker_step(
            &graph,
            &mut r,
            TtcMode::Disabled,
            &[node],
            &entry_points,
            &empty_set,
            &empty_set,
            &empty_map_u64,
            None,
            None,
            &empty_map_f64,
            &empty_set,
            &empty_set,
            &necessity,
        )
        .unwrap();
        assert_eq!(successful, vec![node]);
        assert!(attempted.is_empty());
    }

    #[test]
    fn step_traversable_action_surface_node_succeeds_and_collects_effects() {
        let mut graph = dummy_graph();
        let parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let effect = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[parent].children.insert(node);
        graph.nodes[node].parents.insert(parent);
        graph.nodes[node].children.insert(effect);
        graph.nodes[effect].parents.insert(node);
        graph.nodes[effect].causal_mode =
            Some(maltoolbox_language::graph::attack_step::CausalMode::Effect);

        let performed: HashSet<_> = [parent].into_iter().collect();
        let action_surface: HashSet<_> = [node].into_iter().collect();
        let empty_set: HashSet<AttackGraphNodeId> = HashSet::new();
        let empty_entry_points: HashSet<AttackGraphNodeId> = HashSet::new();
        let empty_map_u64: HashMap<AttackGraphNodeId, u64> = HashMap::new();
        let empty_map_f64: HashMap<AttackGraphNodeId, f64> = HashMap::new();
        let necessity: HashMap<_, _> = [(node, true), (effect, true)].into_iter().collect();
        let mut r = rng();

        let (successful, attempted) = attacker_step(
            &graph,
            &mut r,
            TtcMode::Disabled,
            &[node],
            &empty_entry_points,
            &action_surface,
            &performed,
            &empty_map_u64,
            None,
            None,
            &empty_map_f64,
            &empty_set,
            &empty_set,
            &necessity,
        )
        .unwrap();
        assert_eq!(successful, vec![node, effect]);
        assert!(attempted.is_empty());
    }

    #[test]
    fn step_failed_attempt_is_recorded_without_compromising() {
        let mut graph = dummy_graph();
        let parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[parent].children.insert(node);
        graph.nodes[node].parents.insert(parent);

        let empty_set: HashSet<AttackGraphNodeId> = HashSet::new();
        let performed: HashSet<_> = [parent].into_iter().collect();
        let action_surface: HashSet<_> = [node].into_iter().collect();
        let empty_map_u64: HashMap<AttackGraphNodeId, u64> = HashMap::new();
        let empty_map_bool: HashMap<AttackGraphNodeId, bool> = HashMap::new();
        // ExpectedValue mode with no resolvable ttc_value is an error, not
        // a "failed attempt" - use a ttc_value that fails the comparison
        // instead (num_attempts_before=0 -> 0 + 2 >= ttc_value is false
        // for ttc_value > 2).
        let graph_values: HashMap<_, _> = [(node, 100.0)].into_iter().collect();
        let mut r = rng();

        let (successful, attempted) = attacker_step(
            &graph,
            &mut r,
            TtcMode::ExpectedValue,
            &[node],
            &empty_set,
            &action_surface,
            &performed,
            &empty_map_u64,
            None,
            None,
            &graph_values,
            &empty_set,
            &empty_set,
            &empty_map_bool,
        )
        .unwrap();
        assert!(successful.is_empty());
        assert_eq!(attempted, vec![node]);
    }
}
