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
//! Matches on the `maltoolbox_language::graph::attack_step::AttackStepType`
//! enum (`node.step_type`) directly, consistent with
//! `graph_state.rs`/`necessity.rs`/`viability.rs`; `maltoolbox-language`
//! is a regular (runtime) dependency of this crate.

use crate::types::{NodeDenseMap, NodeSet};
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNode, AttackGraphNodeId};
use maltoolbox_language::graph::attack_step::AttackStepType;

/// Sorts `nodes` by the stable `AttackGraphNode.id`, the deterministic
/// order everything that draws from the RNG or produces an ordered result
/// iterates node sets in. (Slotmap keys are not used for this: the keys of
/// nodes created at runtime depend on slot reuse.) Nodes no longer in the
/// graph sort first, by key.
pub fn sort_by_node_id(graph: &AttackGraph, nodes: &mut [AttackGraphNodeId]) {
    nodes.sort_unstable_by_key(|&key| node_order_key(graph, key));
}

/// The key `sort_by_node_id` orders by, for `min_by_key` and friends.
pub fn node_order_key(
    graph: &AttackGraph,
    key: AttackGraphNodeId,
) -> (Option<i64>, AttackGraphNodeId) {
    (graph.nodes.get(key).map(|node| node.id), key)
}

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
    /// defensive parity with the Python `TypeError`.
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
pub fn is_attack_step_type(step_type: AttackStepType) -> bool {
    !matches!(
        step_type,
        AttackStepType::Defense | AttackStepType::Exist | AttackStepType::NotExist
    )
}

/// Port of `is_attack_step`. Only attack steps (`or`/`and`) have
/// traversability.
pub fn is_attack_step(node: &AttackGraphNode) -> bool {
    !matches!(
        node.step_type,
        AttackStepType::Defense | AttackStepType::Exist | AttackStepType::NotExist
    )
}

/// Port of `node_is_live`.
pub fn node_is_live(graph: &AttackGraph, node_id: AttackGraphNodeId) -> bool {
    graph.nodes.contains_key(node_id)
}

/// Port of `node_is_necessary` - a thin, error-checked wrapper over a
/// `necessity_per_node` lookup (`GraphState::necessity_per_node` from A3).
pub fn node_is_necessary(
    necessity_per_node: &NodeDenseMap<bool>,
    node_id: AttackGraphNodeId,
) -> Result<bool, GraphUtilsError> {
    necessity_per_node
        .get(node_id)
        .copied()
        .ok_or(GraphUtilsError::MissingNecessity(node_id))
}

/// Graph-independent half of `_node_blocks_children`.
pub fn node_blocks_children_from_parts(
    step_type: AttackStepType,
    existence_status: Option<bool>,
    node_id: AttackGraphNodeId,
    enabled_defenses: &NodeSet,
) -> Result<bool, GraphUtilsError> {
    match step_type {
        AttackStepType::Exist => existence_status
            .map(|exists| !exists)
            .ok_or(GraphUtilsError::MissingExistenceStatus(node_id)),
        AttackStepType::NotExist => {
            existence_status.ok_or(GraphUtilsError::MissingExistenceStatus(node_id))
        }
        AttackStepType::Defense => Ok(enabled_defenses.contains(&node_id)),
        AttackStepType::Or | AttackStepType::And => Ok(false),
    }
}

/// Thin `AttackGraphNode`-reading wrapper around
/// `node_blocks_children_from_parts`.
fn node_blocks_children(
    node_id: AttackGraphNodeId,
    node: &AttackGraphNode,
    enabled_defenses: &NodeSet,
) -> Result<bool, GraphUtilsError> {
    node_blocks_children_from_parts(
        node.step_type,
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
    impossible_attack_steps: &NodeSet,
    enabled_defenses: &NodeSet,
) -> Result<bool, GraphUtilsError> {
    let node = &graph.nodes[node_id];
    match node.step_type {
        AttackStepType::And => {
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
        AttackStepType::Or => {
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
    performed_nodes: &NodeSet,
    necessity_per_node: &NodeDenseMap<bool>,
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
    performed_nodes: &NodeSet,
    impossible_attack_steps: &NodeSet,
    enabled_defenses: &NodeSet,
    necessity_per_node: &NodeDenseMap<bool>,
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

    match node.step_type {
        AttackStepType::Or => Ok(true),
        AttackStepType::And => and_traversable(node, performed_nodes, necessity_per_node),
        other => Err(GraphUtilsError::UnknownStepType(
            other.as_str().to_string(),
            node_id,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{add_dummy_node, dummy_graph};

    // --- is_attack_step_type / is_attack_step ---

    #[test]
    fn is_attack_step_type_true_for_or_and_and() {
        assert!(is_attack_step_type(AttackStepType::Or));
        assert!(is_attack_step_type(AttackStepType::And));
    }

    #[test]
    fn is_attack_step_type_false_for_defense_exist_not_exist() {
        assert!(!is_attack_step_type(AttackStepType::Defense));
        assert!(!is_attack_step_type(AttackStepType::Exist));
        assert!(!is_attack_step_type(AttackStepType::NotExist));
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
        let defenses = NodeSet::default();
        let id = AttackGraphNodeId::default();
        assert!(
            node_blocks_children_from_parts(AttackStepType::Exist, Some(false), id, &defenses)
                .unwrap()
        );
        assert!(
            !node_blocks_children_from_parts(AttackStepType::Exist, Some(true), id, &defenses)
                .unwrap()
        );
    }

    #[test]
    fn node_blocks_children_not_exist_blocks_when_existing() {
        let defenses = NodeSet::default();
        let id = AttackGraphNodeId::default();
        assert!(node_blocks_children_from_parts(
            AttackStepType::NotExist,
            Some(true),
            id,
            &defenses
        )
        .unwrap());
        assert!(!node_blocks_children_from_parts(
            AttackStepType::NotExist,
            Some(false),
            id,
            &defenses
        )
        .unwrap());
    }

    #[test]
    fn node_blocks_children_missing_existence_status_errors() {
        let defenses = NodeSet::default();
        let id = AttackGraphNodeId::default();
        assert!(matches!(
            node_blocks_children_from_parts(AttackStepType::Exist, None, id, &defenses),
            Err(GraphUtilsError::MissingExistenceStatus(_))
        ));
        assert!(matches!(
            node_blocks_children_from_parts(AttackStepType::NotExist, None, id, &defenses),
            Err(GraphUtilsError::MissingExistenceStatus(_))
        ));
    }

    #[test]
    fn node_blocks_children_defense_blocks_when_enabled() {
        let mut graph = dummy_graph();
        let defense_id = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let mut defenses = NodeSet::default();
        assert!(!node_blocks_children_from_parts(
            AttackStepType::Defense,
            None,
            defense_id,
            &defenses
        )
        .unwrap());
        defenses.insert(defense_id);
        assert!(node_blocks_children_from_parts(
            AttackStepType::Defense,
            None,
            defense_id,
            &defenses
        )
        .unwrap());
    }

    #[test]
    fn node_blocks_children_or_and_never_block() {
        let defenses = NodeSet::default();
        let id = AttackGraphNodeId::default();
        assert!(!node_blocks_children_from_parts(AttackStepType::Or, None, id, &defenses).unwrap());
        assert!(
            !node_blocks_children_from_parts(AttackStepType::And, None, id, &defenses).unwrap()
        );
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

        let enabled_defenses: NodeSet = [defense_node].into_iter().collect();
        let impossible_attack_steps: NodeSet = [impossible_and_node].into_iter().collect();

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
        let empty = NodeSet::default();
        let empty_map = NodeDenseMap::default();
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

        let empty = NodeSet::default();
        let empty_map = NodeDenseMap::default();
        assert!(!node_is_traversable(&graph, or_node, &empty, &empty, &empty, &empty_map).unwrap());
    }

    #[test]
    fn node_is_traversable_or_true_once_any_parent_performed() {
        let mut graph = dummy_graph();
        let parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let or_node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[or_node].parents.insert(parent);

        let performed: NodeSet = [parent].into_iter().collect();
        let empty = NodeSet::default();
        let empty_map = NodeDenseMap::default();
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

        let necessity: NodeDenseMap<_> = [(necessary_parent, true), (unnecessary_parent, false)]
            .into_iter()
            .collect();
        let empty = NodeSet::default();

        // Only the unnecessary parent performed: parents_reached is true,
        // but the necessary parent is still missing -> not traversable.
        let only_unnecessary: NodeSet = [unnecessary_parent].into_iter().collect();
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
        let only_necessary: NodeSet = [necessary_parent].into_iter().collect();
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

        let performed: NodeSet = [parent].into_iter().collect();
        let enabled_defenses: NodeSet = [defense_node].into_iter().collect();
        let empty = NodeSet::default();
        let empty_map = NodeDenseMap::default();

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

        let performed: NodeSet = [parent].into_iter().collect();
        let enabled_defenses: NodeSet = [defense_node].into_iter().collect();
        // The defense parent is marked unnecessary in both calls below, so
        // `and_traversable` alone would pass (only `parent` is required and
        // it's performed) - the only difference between the two calls is
        // whether the defense is enabled, isolating the `node_is_blocked`
        // check.
        let necessity: NodeDenseMap<_> = [(parent, true), (defense_node, false)]
            .into_iter()
            .collect();
        let empty = NodeSet::default();

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
