//! Rust port of `python/malsim/mal_simulator/defense_surface.py` - see
//! `PORTING_NOTES.md` §5 Phase A5.
//!
//! Like `attack_surface.rs`, actionability is taken as an already-
//! flattened id-set rather than a `NodePropertyRule` - see that module's
//! docs for the exact `None`/`Some` semantics mirrored here.

use std::collections::HashSet;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};

use crate::graph_utils::{node_is_blocked, GraphUtilsError};

fn node_is_actionable_flat(
    actionable_steps: Option<&HashSet<AttackGraphNodeId>>,
    node_id: AttackGraphNodeId,
) -> bool {
    match actionable_steps {
        Some(steps) => steps.contains(&node_id),
        None => true,
    }
}

/// Port of `get_defense_surface`: all non-suppressed defense steps that are
/// actionable, not blocked, and not already enabled.
pub fn get_defense_surface(
    graph: &AttackGraph,
    actionable_steps: Option<&HashSet<AttackGraphNodeId>>,
    impossible_attack_steps: &HashSet<AttackGraphNodeId>,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
) -> Result<HashSet<AttackGraphNodeId>, GraphUtilsError> {
    let mut surface = HashSet::new();
    for &node_id in &graph.defense_steps {
        if !node_is_actionable_flat(actionable_steps, node_id) {
            continue;
        }
        if node_is_blocked(graph, node_id, impossible_attack_steps, enabled_defenses)? {
            continue;
        }
        if enabled_defenses.contains(&node_id) {
            continue;
        }
        if graph.nodes[node_id].tags.iter().any(|t| t == "suppress") {
            continue;
        }
        surface.insert(node_id);
    }
    Ok(surface)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{add_dummy_node, dummy_graph};

    #[test]
    fn defense_surface_includes_plain_unenabled_defense() {
        let mut graph = dummy_graph();
        let defense = add_dummy_node(&mut graph, "DummyDefenseAttackStep");

        let empty = HashSet::new();
        let result = get_defense_surface(&graph, None, &empty, &empty).unwrap();
        assert_eq!(result, [defense].into_iter().collect());
    }

    #[test]
    fn defense_surface_excludes_already_enabled_defense() {
        let mut graph = dummy_graph();
        let defense = add_dummy_node(&mut graph, "DummyDefenseAttackStep");

        let empty = HashSet::new();
        let enabled: HashSet<_> = [defense].into_iter().collect();
        let result = get_defense_surface(&graph, None, &empty, &enabled).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn defense_surface_excludes_suppressed_defense() {
        let mut graph = dummy_graph();
        let defense = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        graph.nodes[defense].tags.push("suppress".to_string());

        let empty = HashSet::new();
        let result = get_defense_surface(&graph, None, &empty, &empty).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn defense_surface_respects_flattened_actionable_steps() {
        let mut graph = dummy_graph();
        let actionable_defense = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let non_actionable_defense = add_dummy_node(&mut graph, "DummyDefenseAttackStep");

        let empty = HashSet::new();
        let actionable_set: HashSet<_> = [actionable_defense].into_iter().collect();
        let result = get_defense_surface(&graph, Some(&actionable_set), &empty, &empty).unwrap();
        assert_eq!(result, [actionable_defense].into_iter().collect());
        assert!(!result.contains(&non_actionable_defense));
    }
}
