//! Rust port of `python/malsim/mal_simulator/graph_utils.py`'s traversal
//! predicates - see `PORTING_NOTES.md` §5 Phase A4.
//!
//! `node_is_actionable`/`node_reward` are deliberately **not** ported here:
//! per §2.4/§4 they operate on `NodePropertyRule` directly (not the
//! flattened hot path) and stay in Python.
//!
//! Like `necessity.rs` (and unlike most of `graph_state.rs`),
//! `node_is_blocked`/`node_is_traversable` aren't split into a
//! graph-independent helper + thin wrapper: the interesting logic here
//! *is* graph traversal (resolving `node.parents` to real nodes), so
//! there's nothing graph-independent to factor out. `node_blocks_children`
//! is the exception - its own body only reads the node's `step_type`/
//! `existence_status`/id, so it's decomposed the way `graph_state.rs`
//! does, same reasoning (testable without a real `AttackGraphNode`).
//! `is_attack_step` is decomposed the same way, trivially.
//!
//! Matches on `node.step_type.as_str()` rather than the
//! `maltoolbox_language::graph::attack_step::AttackStepType` enum
//! directly, consistent with `graph_state.rs`/`necessity.rs`: that enum
//! isn't re-exported by `maltoolbox-attackgraph`, and this port keeps
//! `maltoolbox-language` a test-only (`dev-dependencies`) dependency of
//! this crate, not a runtime one - see `PORTING_NOTES.md` §10's A4 entry.

use std::collections::{HashMap, HashSet};
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNode, AttackGraphNodeId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphUtilsError {
    /// Mirrors Python's `assert isinstance(node.existence_status, bool)`
    /// in `_node_blocks_children`.
    MissingExistenceStatus(AttackGraphNodeId),
    /// Mirrors Python's bare `necessity_per_node[parent]` `KeyError` in
    /// `and_traversable` - the map must already hold every node's
    /// necessity (computed once by `necessity::calculate_necessity`).
    MissingNecessity(AttackGraphNodeId),
    /// Mirrors Python's `TypeError` in `is_and_or_traversable` for a
    /// `node.type` outside `or`/`and`. Unreachable through
    /// `node_is_traversable` itself (callers only reach this match after
    /// `is_attack_step` has already filtered to `or`/`and`), kept for the
    /// same defensive-parity reason `necessity.rs::UnknownStepType` is.
    UnknownStepType(String, AttackGraphNodeId),
}

impl fmt::Display for GraphUtilsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphUtilsError::MissingExistenceStatus(id) => {
                write!(f, "existence status not defined for node {id:?}")
            }
            GraphUtilsError::MissingNecessity(id) => {
                write!(f, "necessity not yet computed for parent node {id:?}")
            }
            GraphUtilsError::UnknownStepType(step_type, id) => {
                write!(f, "node {id:?} has an unknown type \"{step_type}\"")
            }
        }
    }
}

impl std::error::Error for GraphUtilsError {}

/// Graph-independent half of `is_attack_step`.
pub fn is_attack_step_type(step_type: &str) -> bool {
    !matches!(step_type, "defense" | "exist" | "notExist")
}

/// Port of `is_attack_step`. Only attack steps (`or`/`and`) have
/// traversability.
pub fn is_attack_step(node: &AttackGraphNode) -> bool {
    is_attack_step_type(node.step_type.as_str())
}

/// Port of `node_is_live`.
pub fn node_is_live(graph: &AttackGraph, node_id: AttackGraphNodeId) -> bool {
    graph.nodes.contains_key(node_id)
}

/// Port of `node_is_necessary` - a thin, error-checked wrapper over a
/// `necessity_per_node` lookup (`GraphState::necessity_per_node` from A3).
pub fn node_is_necessary(
    necessity_per_node: &HashMap<AttackGraphNodeId, bool>,
    node_id: AttackGraphNodeId,
) -> Result<bool, GraphUtilsError> {
    necessity_per_node
        .get(&node_id)
        .copied()
        .ok_or(GraphUtilsError::MissingNecessity(node_id))
}

/// Graph-independent half of `_node_blocks_children`.
pub fn node_blocks_children_from_parts(
    step_type: &str,
    existence_status: Option<bool>,
    node_id: AttackGraphNodeId,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
) -> Result<bool, GraphUtilsError> {
    match step_type {
        "exist" => existence_status
            .map(|exists| !exists)
            .ok_or(GraphUtilsError::MissingExistenceStatus(node_id)),
        "notExist" => existence_status.ok_or(GraphUtilsError::MissingExistenceStatus(node_id)),
        "defense" => Ok(enabled_defenses.contains(&node_id)),
        _ => Ok(false),
    }
}

/// Thin `AttackGraphNode`-reading wrapper around
/// `node_blocks_children_from_parts`.
fn node_blocks_children(
    node_id: AttackGraphNodeId,
    node: &AttackGraphNode,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
) -> Result<bool, GraphUtilsError> {
    node_blocks_children_from_parts(
        node.step_type.as_str(),
        node.existence_status,
        node_id,
        enabled_defenses,
    )
}

/// Port of `node_is_blocked`.
///
/// Note the `and`/`or` branches are not a copy-paste of each other with
/// the connective swapped by mistake: an `and` node is blocked if *any*
/// parent blocks it (one missing path is enough to cut off a step that
/// needs them all), an `or` node is blocked only if *all* parents block it
/// (any single open path keeps it reachable) - this matches Python's
/// `node_is_blocked` (`any(...)` for `and`, `all(...)` for `or`) exactly.
pub fn node_is_blocked(
    graph: &AttackGraph,
    node_id: AttackGraphNodeId,
    impossible_attack_steps: &HashSet<AttackGraphNodeId>,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
) -> Result<bool, GraphUtilsError> {
    let node = &graph.nodes[node_id];
    match node.step_type.as_str() {
        "and" => {
            if impossible_attack_steps.contains(&node_id) {
                return Ok(true);
            }
            for &parent_id in &node.parents {
                let parent = &graph.nodes[parent_id];
                if node_blocks_children(parent_id, parent, enabled_defenses)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        "or" => {
            if impossible_attack_steps.contains(&node_id) {
                return Ok(true);
            }
            for &parent_id in &node.parents {
                let parent = &graph.nodes[parent_id];
                if !node_blocks_children(parent_id, parent, enabled_defenses)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Port of `and_traversable`: all of `node`'s *necessary* parents must be
/// in `performed_nodes` (unnecessary parents are skipped, not required).
fn and_traversable(
    node: &AttackGraphNode,
    performed_nodes: &HashSet<AttackGraphNodeId>,
    necessity_per_node: &HashMap<AttackGraphNodeId, bool>,
) -> Result<bool, GraphUtilsError> {
    for &parent_id in &node.parents {
        if node_is_necessary(necessity_per_node, parent_id)?
            && !performed_nodes.contains(&parent_id)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Port of `node_is_traversable`.
///
/// Arguments mirror the Python function's `sim_state`/`performed_nodes`/
/// `node`, decomposed into the specific `GraphState`/`MalSimulatorState`
/// fields this logic actually reads (full settings/state porting is later
/// A-phase work).
pub fn node_is_traversable(
    graph: &AttackGraph,
    node_id: AttackGraphNodeId,
    performed_nodes: &HashSet<AttackGraphNodeId>,
    impossible_attack_steps: &HashSet<AttackGraphNodeId>,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
    necessity_per_node: &HashMap<AttackGraphNodeId, bool>,
) -> Result<bool, GraphUtilsError> {
    let node = &graph.nodes[node_id];

    if !is_attack_step(node) {
        return Ok(false);
    }
    if node_is_blocked(graph, node_id, impossible_attack_steps, enabled_defenses)? {
        return Ok(false);
    }
    // If no parent is reached, the node can not be traversable.
    let parents_reached = node.parents.iter().any(|p| performed_nodes.contains(p));
    if !parents_reached {
        return Ok(false);
    }

    match node.step_type.as_str() {
        "or" => Ok(true),
        "and" => and_traversable(node, performed_nodes, necessity_per_node),
        other => Err(GraphUtilsError::UnknownStepType(other.to_string(), node_id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{add_dummy_node, dummy_graph};

    // --- is_attack_step_type / is_attack_step ---

    #[test]
    fn is_attack_step_type_true_for_or_and_and() {
        assert!(is_attack_step_type("or"));
        assert!(is_attack_step_type("and"));
    }

    #[test]
    fn is_attack_step_type_false_for_defense_exist_not_exist() {
        assert!(!is_attack_step_type("defense"));
        assert!(!is_attack_step_type("exist"));
        assert!(!is_attack_step_type("notExist"));
    }

    #[test]
    fn is_attack_step_reads_node_step_type() {
        let mut graph = dummy_graph();
        let or_id = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let defense_id = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        assert!(is_attack_step(&graph.nodes[or_id]));
        assert!(!is_attack_step(&graph.nodes[defense_id]));
    }

    // --- node_blocks_children_from_parts ---

    #[test]
    fn node_blocks_children_exist_blocks_when_not_existing() {
        let defenses = HashSet::new();
        let id = AttackGraphNodeId::default();
        assert!(node_blocks_children_from_parts("exist", Some(false), id, &defenses).unwrap());
        assert!(!node_blocks_children_from_parts("exist", Some(true), id, &defenses).unwrap());
    }

    #[test]
    fn node_blocks_children_not_exist_blocks_when_existing() {
        let defenses = HashSet::new();
        let id = AttackGraphNodeId::default();
        assert!(node_blocks_children_from_parts("notExist", Some(true), id, &defenses).unwrap());
        assert!(!node_blocks_children_from_parts("notExist", Some(false), id, &defenses).unwrap());
    }

    #[test]
    fn node_blocks_children_missing_existence_status_errors() {
        let defenses = HashSet::new();
        let id = AttackGraphNodeId::default();
        assert!(matches!(
            node_blocks_children_from_parts("exist", None, id, &defenses),
            Err(GraphUtilsError::MissingExistenceStatus(_))
        ));
        assert!(matches!(
            node_blocks_children_from_parts("notExist", None, id, &defenses),
            Err(GraphUtilsError::MissingExistenceStatus(_))
        ));
    }

    #[test]
    fn node_blocks_children_defense_blocks_when_enabled() {
        let mut graph = dummy_graph();
        let defense_id = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let mut defenses = HashSet::new();
        assert!(!node_blocks_children_from_parts("defense", None, defense_id, &defenses).unwrap());
        defenses.insert(defense_id);
        assert!(node_blocks_children_from_parts("defense", None, defense_id, &defenses).unwrap());
    }

    #[test]
    fn node_blocks_children_or_and_never_block() {
        let defenses = HashSet::new();
        let id = AttackGraphNodeId::default();
        assert!(!node_blocks_children_from_parts("or", None, id, &defenses).unwrap());
        assert!(!node_blocks_children_from_parts("and", None, id, &defenses).unwrap());
    }

    // --- node_is_live ---

    #[test]
    fn node_is_live_true_for_existing_node_false_after_removal() {
        let mut graph = dummy_graph();
        let node_id = add_dummy_node(&mut graph, "DummyOrAttackStep");
        assert!(node_is_live(&graph, node_id));
        graph.remove_node(node_id).unwrap();
        assert!(!node_is_live(&graph, node_id));
    }

    // --- node_is_blocked (ported from `tests/test_graph_processing.py::test_node_is_blocked`) ---

    #[test]
    fn node_is_blocked_matches_python_test_node_is_blocked() {
        let mut graph = dummy_graph();

        let exist_node = add_dummy_node(&mut graph, "DummyExistAttackStep");
        graph.nodes[exist_node].existence_status = Some(false);

        let not_exist_node = add_dummy_node(&mut graph, "DummyNotExistAttackStep");
        graph.nodes[not_exist_node].existence_status = Some(true);

        let defense_node = add_dummy_node(&mut graph, "DummyDefenseAttackStep");

        let and_blocked_by_defense = add_dummy_node(&mut graph, "DummyAndAttackStep");
        graph.nodes[and_blocked_by_defense]
            .parents
            .insert(defense_node);

        let and_blocked_by_exist = add_dummy_node(&mut graph, "DummyAndAttackStep");
        graph.nodes[and_blocked_by_exist].parents.insert(exist_node);

        let and_blocked_by_not_exist = add_dummy_node(&mut graph, "DummyAndAttackStep");
        graph.nodes[and_blocked_by_not_exist]
            .parents
            .insert(not_exist_node);

        let or_blocked_by_defense = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[or_blocked_by_defense]
            .parents
            .insert(defense_node);

        let or_blocked_by_exist = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[or_blocked_by_exist].parents.insert(exist_node);

        let or_blocked_by_not_exist = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[or_blocked_by_not_exist]
            .parents
            .insert(not_exist_node);

        // Or node with any parent that is not blocking will not be blocked.
        let or_not_blocked = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[or_not_blocked]
            .parents
            .insert(and_blocked_by_defense);

        let impossible_and_node = add_dummy_node(&mut graph, "DummyAndAttackStep");

        let enabled_defenses: HashSet<_> = [defense_node].into_iter().collect();
        let impossible_attack_steps: HashSet<_> = [impossible_and_node].into_iter().collect();

        let blocked =
            |id| node_is_blocked(&graph, id, &impossible_attack_steps, &enabled_defenses).unwrap();

        assert!(blocked(and_blocked_by_defense));
        assert!(blocked(and_blocked_by_exist));
        assert!(blocked(and_blocked_by_not_exist));
        assert!(blocked(or_blocked_by_defense));
        assert!(blocked(or_blocked_by_exist));
        assert!(blocked(or_blocked_by_not_exist));
        assert!(!blocked(or_not_blocked));
        assert!(blocked(impossible_and_node));
    }

    // --- node_is_traversable ---

    #[test]
    fn node_is_traversable_false_for_non_attack_step() {
        let mut graph = dummy_graph();
        let defense_id = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let empty = HashSet::new();
        let empty_map = HashMap::new();
        assert!(
            !node_is_traversable(&graph, defense_id, &empty, &empty, &empty, &empty_map).unwrap()
        );
    }

    #[test]
    fn node_is_traversable_false_when_no_parent_performed() {
        let mut graph = dummy_graph();
        let parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let or_node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[or_node].parents.insert(parent);

        let empty = HashSet::new();
        let empty_map = HashMap::new();
        assert!(!node_is_traversable(&graph, or_node, &empty, &empty, &empty, &empty_map).unwrap());
    }

    #[test]
    fn node_is_traversable_or_true_once_any_parent_performed() {
        let mut graph = dummy_graph();
        let parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let or_node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[or_node].parents.insert(parent);

        let performed: HashSet<_> = [parent].into_iter().collect();
        let empty = HashSet::new();
        let empty_map = HashMap::new();
        assert!(
            node_is_traversable(&graph, or_node, &performed, &empty, &empty, &empty_map).unwrap()
        );
    }

    #[test]
    fn node_is_traversable_and_requires_all_necessary_parents_performed() {
        let mut graph = dummy_graph();
        let necessary_parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let unnecessary_parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let and_node = add_dummy_node(&mut graph, "DummyAndAttackStep");
        graph.nodes[and_node].parents.insert(necessary_parent);
        graph.nodes[and_node].parents.insert(unnecessary_parent);

        let necessity: HashMap<_, _> = [(necessary_parent, true), (unnecessary_parent, false)]
            .into_iter()
            .collect();
        let empty = HashSet::new();

        // Only the unnecessary parent performed: parents_reached is true,
        // but the necessary parent is still missing -> not traversable.
        let only_unnecessary: HashSet<_> = [unnecessary_parent].into_iter().collect();
        assert!(!node_is_traversable(
            &graph,
            and_node,
            &only_unnecessary,
            &empty,
            &empty,
            &necessity
        )
        .unwrap());

        // Necessary parent performed (unnecessary one isn't) -> traversable.
        let only_necessary: HashSet<_> = [necessary_parent].into_iter().collect();
        assert!(node_is_traversable(
            &graph,
            and_node,
            &only_necessary,
            &empty,
            &empty,
            &necessity
        )
        .unwrap());
    }

    #[test]
    fn node_is_traversable_or_true_when_only_some_parents_block() {
        let mut graph = dummy_graph();
        let defense_node = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let or_node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[or_node].parents.insert(parent);
        graph.nodes[or_node].parents.insert(defense_node);

        let performed: HashSet<_> = [parent].into_iter().collect();
        let enabled_defenses: HashSet<_> = [defense_node].into_iter().collect();
        let empty = HashSet::new();
        let empty_map = HashMap::new();

        // `or` node blocked only once *all* parents block - here only the
        // defense parent blocks, the other parent doesn't, so it's not
        // blocked (and is traversable, since that parent is performed).
        assert!(node_is_traversable(
            &graph,
            or_node,
            &performed,
            &empty,
            &enabled_defenses,
            &empty_map
        )
        .unwrap());
    }

    #[test]
    fn node_is_traversable_and_false_when_blocked_by_enabled_defense() {
        let mut graph = dummy_graph();
        let defense_node = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let and_node = add_dummy_node(&mut graph, "DummyAndAttackStep");
        graph.nodes[and_node].parents.insert(parent);
        graph.nodes[and_node].parents.insert(defense_node);

        let performed: HashSet<_> = [parent].into_iter().collect();
        let enabled_defenses: HashSet<_> = [defense_node].into_iter().collect();
        // The defense parent is marked unnecessary in both calls below, so
        // `and_traversable` alone would pass (only `parent` is required and
        // it's performed) - the only difference between the two calls is
        // whether the defense is enabled, isolating the `node_is_blocked`
        // check.
        let necessity: HashMap<_, _> = [(parent, true), (defense_node, false)]
            .into_iter()
            .collect();
        let empty = HashSet::new();

        // Defense not enabled -> traversable.
        assert!(
            node_is_traversable(&graph, and_node, &performed, &empty, &empty, &necessity).unwrap()
        );

        // `and` node blocked as soon as *any* parent blocks - the enabled
        // defense parent alone is enough, even with the other parent
        // performed.
        assert!(!node_is_traversable(
            &graph,
            and_node,
            &performed,
            &empty,
            &enabled_defenses,
            &necessity
        )
        .unwrap());
    }
}
