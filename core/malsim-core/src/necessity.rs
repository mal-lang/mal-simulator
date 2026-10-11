//! Rust port of `python/malsim/mal_simulator/graph_processing.py`'s
//! necessity propagation - see `PORTING_NOTES.md` §5 Phase A3.
//!
//! Only necessity lives here; the viability/pruning half of the same
//! Python file (`calculate_viability`/`evaluate_viability`/
//! `prune_unviable_and_unnecessary_nodes`) is in `crate::viability` -
//! originally left unported as dead code at A3, ported at B7 so the Python
//! file could be deleted without losing its test coverage (see
//! `PORTING_NOTES.md` §10).
//!
//! Rust-native tests (backfilled at Phase A4, per `PORTING_NOTES.md` §10's
//! A3 entry): every case needs a real `AttackGraphNode` with a specific
//! `step_type`, which requires a real
//! `maltoolbox_language::graph::LanguageGraph` to mint (`AttackStepId` is
//! a slotmap key, not fakeable) - `maltoolbox-language` was added as a
//! test-only (`dev-dependencies`) crate dependency when A4 hit the same
//! wall for its own traversal-predicate tests. See `crate::test_fixtures`
//! and `PORTING_NOTES.md` §10 for the discussion.

use std::collections::{HashMap, HashSet};
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNode, AttackGraphNodeId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NecessityError {
    /// Mirrors Python's `assert isinstance(node.existence_status, bool)`
    /// for `exist`/`notExist` nodes.
    MissingExistenceStatus(AttackGraphNodeId),
    /// A parent's necessity was read before it was computed - mirrors
    /// Python's `necessity_per_node[parent]` `KeyError` (the dict must be
    /// pre-seeded for every node before `evaluate_necessity` runs; see
    /// `calculate_necessity`).
    MissingNecessity(AttackGraphNodeId),
    /// Mirrors Python's `ValueError` for a `node.type` outside
    /// `exist`/`notExist`/`defense`/`or`/`and`.
    UnknownStepType(String, AttackGraphNodeId),
}

impl fmt::Display for NecessityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NecessityError::MissingExistenceStatus(id) => {
                write!(f, "existence status not defined for node {id:?}")
            }
            NecessityError::MissingNecessity(id) => {
                write!(f, "necessity not yet computed for parent node {id:?}")
            }
            NecessityError::UnknownStepType(step_type, id) => write!(
                f,
                "evaluate_necessity was provided node {id:?} which is of unknown type \"{step_type}\""
            ),
        }
    }
}

impl std::error::Error for NecessityError {}

/// Port of `evaluate_necessity`. Takes `node_id` alongside `node` since
/// `AttackGraphNode` doesn't carry its own slotmap key (see
/// `maltoolbox_attackgraph::ids::AttackGraphNodeId`'s doc comment) -
/// needed both to check self-membership in `enabled_defenses` and for
/// error context.
pub fn evaluate_necessity(
    node_id: AttackGraphNodeId,
    node: &AttackGraphNode,
    necessity_per_node: &HashMap<AttackGraphNodeId, bool>,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
) -> Result<bool, NecessityError> {
    let parent_necessity = |id: &AttackGraphNodeId| -> Result<bool, NecessityError> {
        necessity_per_node
            .get(id)
            .copied()
            .ok_or(NecessityError::MissingNecessity(*id))
    };

    match node.step_type.as_str() {
        "exist" => node
            .existence_status
            .map(|exists| !exists)
            .ok_or(NecessityError::MissingExistenceStatus(node_id)),
        "notExist" => node
            .existence_status
            .ok_or(NecessityError::MissingExistenceStatus(node_id)),
        "defense" => Ok(enabled_defenses.contains(&node_id)),
        // Python: `all(necessity_per_node[p] for p in node.parents) or not
        // node.parents` - `all(())` is already `True`, so the `or not
        // node.parents` is redundant for "or" (unlike "and" below); this
        // loop mirrors that without the redundant check.
        "or" => {
            for parent in &node.parents {
                if !parent_necessity(parent)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        // Python: `any(necessity_per_node[p] for p in node.parents) or not
        // node.parents` - `any(())` is `False`, so the empty-parents case
        // genuinely needs the explicit `True` here (not redundant, unlike
        // "or").
        "and" => {
            if node.parents.is_empty() {
                return Ok(true);
            }
            for parent in &node.parents {
                if parent_necessity(parent)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        other => Err(NecessityError::UnknownStepType(other.to_string(), node_id)),
    }
}

/// Port of `_propagate_necessity_from_node`.
pub fn propagate_necessity_from_node(
    node_id: AttackGraphNodeId,
    graph: &AttackGraph,
    necessity_per_node: &mut HashMap<AttackGraphNodeId, bool>,
) -> Result<HashSet<AttackGraphNodeId>, NecessityError> {
    let mut changed_nodes = HashSet::new();
    for &child_id in &graph.nodes[node_id].children {
        let child = &graph.nodes[child_id];
        // Python passes `frozenset()` (not the real `enabled_defenses`)
        // into this inner `evaluate_necessity` call - mirrored exactly,
        // not a bug: a propagated child is never itself re-evaluated as a
        // `defense` node through this path (only `or`/`and` nodes do,
        // since propagation starts from `exist`/`notExist`/`defense`
        // nodes and walks downstream `or`/`and` children, whose necessity
        // doesn't consult `enabled_defenses` at all).
        let is_necessary =
            evaluate_necessity(child_id, child, necessity_per_node, &HashSet::new())?;
        if necessity_per_node.get(&child_id).copied() != Some(is_necessary) {
            necessity_per_node.insert(child_id, is_necessary);
            changed_nodes.insert(child_id);
            changed_nodes.extend(propagate_necessity_from_node(
                child_id,
                graph,
                necessity_per_node,
            )?);
        }
    }
    Ok(changed_nodes)
}

/// Port of `calculate_necessity`.
pub fn calculate_necessity(
    graph: &AttackGraph,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
) -> Result<HashMap<AttackGraphNodeId, bool>, NecessityError> {
    let mut necessity_per_node: HashMap<AttackGraphNodeId, bool> =
        graph.nodes.keys().map(|id| (id, true)).collect();

    let node_ids: Vec<AttackGraphNodeId> = graph.nodes.keys().collect();
    for node_id in node_ids {
        let node = &graph.nodes[node_id];
        if matches!(node.step_type.as_str(), "exist" | "notExist" | "defense") {
            let is_necessary =
                evaluate_necessity(node_id, node, &necessity_per_node, enabled_defenses)?;
            necessity_per_node.insert(node_id, is_necessary);
            if !is_necessary {
                propagate_necessity_from_node(node_id, graph, &mut necessity_per_node)?;
            }
        }
    }
    Ok(necessity_per_node)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{add_dummy_node, dummy_graph};

    // Port of `tests/test_graph_processing.py::test_necessity_necessary`.
    #[test]
    fn calculate_necessity_necessary_nodes() {
        let mut graph = dummy_graph();

        // exists, existence_status = False -> necessary
        let exist_node = add_dummy_node(&mut graph, "DummyExistAttackStep");
        graph.nodes[exist_node].existence_status = Some(false);

        // notExists, existence_status = True -> necessary
        let not_exist_node = add_dummy_node(&mut graph, "DummyNotExistAttackStep");
        graph.nodes[not_exist_node].existence_status = Some(true);

        // Defense status on -> necessary
        let enabled_defense_step = add_dummy_node(&mut graph, "DummyDefenseAttackStep");

        // or-node with necessary parents -> necessary
        let or_node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let or_node_parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[or_node].parents.insert(or_node_parent);
        graph.nodes[or_node_parent].children.insert(or_node);

        // and-node with at least one necessary parent -> necessary
        let and_node = add_dummy_node(&mut graph, "DummyAndAttackStep");
        let and_node_parent1 = add_dummy_node(&mut graph, "DummyAndAttackStep");
        let and_node_parent2 = add_dummy_node(&mut graph, "DummyAndAttackStep");
        graph.nodes[and_node].parents = [and_node_parent1, and_node_parent2].into_iter().collect();
        graph.nodes[and_node_parent1].children.insert(and_node);
        graph.nodes[and_node_parent2].children.insert(and_node);

        let enabled_defenses: HashSet<_> = [enabled_defense_step].into_iter().collect();
        let necessity_per_node = calculate_necessity(&graph, &enabled_defenses).unwrap();

        assert!(necessity_per_node[&exist_node]);
        assert!(necessity_per_node[&not_exist_node]);
        assert!(necessity_per_node[&enabled_defense_step]);
        assert!(necessity_per_node[&or_node]);
        assert!(necessity_per_node[&and_node]);
    }

    // Port of `tests/test_graph_processing.py::test_necessity_unnecessary`.
    #[test]
    fn calculate_necessity_unnecessary_nodes() {
        let mut graph = dummy_graph();

        // exists, existence_status = True -> unnecessary
        let exist_node = add_dummy_node(&mut graph, "DummyExistAttackStep");
        graph.nodes[exist_node].existence_status = Some(true);

        // notExists, existence_status = False -> unnecessary
        let not_exist_node = add_dummy_node(&mut graph, "DummyNotExistAttackStep");
        graph.nodes[not_exist_node].existence_status = Some(false);

        // Defense status off -> unnecessary
        let disabled_defense_step = add_dummy_node(&mut graph, "DummyDefenseAttackStep");

        // or-node with unnecessary parent -> unnecessary
        let or_node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[or_node].parents.insert(disabled_defense_step);
        graph.nodes[disabled_defense_step].children.insert(or_node);

        // and-node with only unnecessary parents -> unnecessary
        let and_node = add_dummy_node(&mut graph, "DummyAndAttackStep");
        graph.nodes[and_node].parents.insert(disabled_defense_step);
        graph.nodes[disabled_defense_step].children.insert(and_node);

        let enabled_defenses = HashSet::new();
        let necessity_per_node = calculate_necessity(&graph, &enabled_defenses).unwrap();

        assert!(!necessity_per_node[&exist_node]);
        assert!(!necessity_per_node[&not_exist_node]);
        assert!(!necessity_per_node[&disabled_defense_step]);
        assert!(!necessity_per_node[&or_node]);
        assert!(!necessity_per_node[&and_node]);
    }

    // Port of
    // `tests/test_graph_processing.py::test_analyzers_apriori_propagate_necessity`.
    #[test]
    fn propagate_necessity_from_node_updates_downstream_or_and_and_nodes() {
        let mut graph = dummy_graph();

        let np1 = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let np2 = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let unp1 = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let unp2 = add_dummy_node(&mut graph, "DummyOrAttackStep");

        let or_1unp = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let or_2np = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let and_1np = add_dummy_node(&mut graph, "DummyAndAttackStep");
        let and_2unp = add_dummy_node(&mut graph, "DummyAndAttackStep");

        graph.nodes[or_1unp].parents = [np1, unp1].into_iter().collect();
        graph.nodes[or_2np].parents = [np1, np2].into_iter().collect();
        graph.nodes[and_1np].parents = [np1, unp1].into_iter().collect();
        graph.nodes[and_2unp].parents = [unp1, unp2].into_iter().collect();

        graph.nodes[np1].children = [or_1unp, or_2np, and_1np].into_iter().collect();
        graph.nodes[np2].children = [or_2np].into_iter().collect();
        graph.nodes[unp1].children = [or_1unp, and_1np, and_2unp].into_iter().collect();
        graph.nodes[unp2].children = [and_2unp].into_iter().collect();

        let enabled_defenses = HashSet::new();
        let mut necessity_per_node = calculate_necessity(&graph, &enabled_defenses).unwrap();
        // Force unp1/unp2 unnecessary, then re-propagate from every
        // top-level parent - mirrors the Python test exercising
        // `_propagate_necessity_from_node` directly, independent of
        // `calculate_necessity`'s own top-level loop (none of these nodes
        // are `exist`/`notExist`/`defense`, so that loop never visits
        // them).
        necessity_per_node.insert(unp1, false);
        necessity_per_node.insert(unp2, false);

        let mut changed_nodes = HashSet::new();
        for &parent in &[np1, np2, unp1, unp2] {
            changed_nodes.extend(
                propagate_necessity_from_node(parent, &graph, &mut necessity_per_node).unwrap(),
            );
        }

        assert_eq!(changed_nodes, [or_1unp, and_2unp].into_iter().collect());

        for node in [np1, np2, or_2np, and_1np] {
            assert!(necessity_per_node[&node]);
        }
        for node in [unp1, unp2, or_1unp, and_2unp] {
            assert!(!necessity_per_node[&node]);
        }
    }
}
