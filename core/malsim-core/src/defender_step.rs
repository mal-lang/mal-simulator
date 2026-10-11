//! Rust port of `python/malsim/mal_simulator/defender_step.py` - see
//! `PORTING_NOTES.md` §5 Phase A7.

use std::collections::HashSet;
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};

use crate::attacker_step::attacker_is_terminated;
use crate::graph_utils::node_is_live;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefenderStepError {
    /// Mirrors Python's `assert node == sim_state.attack_graph.nodes[node.id]`
    /// in `defender_step` - see `attacker_step::AttackerStepError::
    /// NodeNotInGraph`'s doc comment for why this id-liveness check is the
    /// chosen translation on the Rust side.
    NodeNotInGraph(AttackGraphNodeId),
}

impl fmt::Display for DefenderStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DefenderStepError::NodeNotInGraph(id) => write!(
                f,
                "tried to step a node ({id:?}) that is not part of this simulator's attack graph"
            ),
        }
    }
}

impl std::error::Error for DefenderStepError {}

/// Port of `defender_step`. Like Python, a node outside the defender's
/// action surface is silently skipped (Python logs a warning) rather than
/// treated as an error - only the graph-membership check is a hard
/// failure.
pub fn defender_step(
    graph: &AttackGraph,
    nodes: &[AttackGraphNodeId],
    action_surface: &HashSet<AttackGraphNodeId>,
) -> Result<Vec<AttackGraphNodeId>, DefenderStepError> {
    let mut enabled_defenses = Vec::new();

    for &node_id in nodes {
        if !node_is_live(graph, node_id) {
            return Err(DefenderStepError::NodeNotInGraph(node_id));
        }
        if action_surface.contains(&node_id) {
            enabled_defenses.push(node_id);
        }
    }

    Ok(enabled_defenses)
}

/// Port of `defender_is_terminated`. Takes each attacker's
/// `(action_surface, goals, performed_nodes)` triple directly rather than
/// an `AgentStates`/`AttackerState` collection - neither exists on the
/// Rust side yet (A9's job). `Iterator::all` over an empty iterator
/// mirrors Python's `all(...)` over an empty `attacker_states(...)` dict -
/// both vacuously `true`.
pub fn defender_is_terminated<'a>(
    attackers: impl IntoIterator<
        Item = (
            &'a HashSet<AttackGraphNodeId>,
            &'a HashSet<AttackGraphNodeId>,
            &'a HashSet<AttackGraphNodeId>,
        ),
    >,
) -> bool {
    attackers
        .into_iter()
        .all(|(action_surface, goals, performed_nodes)| {
            attacker_is_terminated(action_surface, goals, performed_nodes)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{add_dummy_node, dummy_graph};

    #[test]
    fn defender_step_enables_nodes_in_action_surface() {
        let mut graph = dummy_graph();
        let defense = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let action_surface: HashSet<_> = [defense].into_iter().collect();

        let enabled = defender_step(&graph, &[defense], &action_surface).unwrap();
        assert_eq!(enabled, vec![defense]);
    }

    #[test]
    fn defender_step_skips_nodes_outside_action_surface() {
        let mut graph = dummy_graph();
        let attack_step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let empty = HashSet::new();

        let enabled = defender_step(&graph, &[attack_step], &empty).unwrap();
        assert!(enabled.is_empty());
    }

    /// Mirrors the second case of the old
    /// `tests/test_mal_simulator.py::test_defender_step` ("Can not defend
    /// attack_step"), where the defender's action surface was non-empty
    /// (it held every defense in the graph) - the requested attack step is
    /// still skipped because it isn't one of them.
    #[test]
    fn defender_step_skips_node_outside_non_empty_action_surface() {
        let mut graph = dummy_graph();
        let defense = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let attack_step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let action_surface: HashSet<_> = [defense].into_iter().collect();

        let enabled = defender_step(&graph, &[attack_step], &action_surface).unwrap();
        assert!(enabled.is_empty());
    }

    #[test]
    fn defender_step_fails_on_node_not_in_graph() {
        let mut graph = dummy_graph();
        let defense = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        graph.remove_node(defense).unwrap();
        let action_surface: HashSet<_> = [defense].into_iter().collect();

        let result = defender_step(&graph, &[defense], &action_surface);
        assert!(matches!(result, Err(DefenderStepError::NodeNotInGraph(_))));
    }

    // --- defender_is_terminated ---

    #[test]
    fn defender_terminated_vacuously_with_no_attackers() {
        assert!(defender_is_terminated(std::iter::empty()));
    }

    #[test]
    fn defender_terminated_when_all_attackers_terminated() {
        let empty = HashSet::new();
        // Empty action surface -> terminated, for every attacker.
        assert!(defender_is_terminated([
            (&empty, &empty, &empty),
            (&empty, &empty, &empty)
        ]));
    }

    #[test]
    fn defender_not_terminated_when_one_attacker_is_not() {
        let mut graph = dummy_graph();
        let node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let surface: HashSet<_> = [node].into_iter().collect();
        let empty = HashSet::new();

        assert!(!defender_is_terminated([
            (&empty, &empty, &empty),
            (&surface, &empty, &empty),
        ]));
    }
}
