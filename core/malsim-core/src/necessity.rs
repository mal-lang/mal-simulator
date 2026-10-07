//! Rust port of `python/malsim/mal_simulator/graph_processing.py`'s
//! necessity propagation - see `PORTING_NOTES.md` §5 Phase A3.
//!
//! Only necessity is ported here, not viability
//! (`calculate_viability`/`evaluate_viability`/
//! `prune_unviable_and_unnecessary_nodes`): confirmed via
//! `grep -rn` that nothing outside `graph_processing.py` itself uses
//! viability (its own module docstring already calls it
//! "(deprecated)"), and `PORTING_NOTES.md` §5's A3 description only
//! scopes in necessity. Not an oversight - see `PORTING_NOTES.md` §10 for
//! the explicit note.
//!
//! Rust-native tests are deferred for this whole module - every case
//! needs a real `AttackGraphNode` with a specific `step_type`, which
//! requires a real `maltoolbox_language::graph::LanguageGraph` to mint
//! (`AttackStepId` is a slotmap key, not fakeable), and that's a new
//! dev-dependency the port deliberately didn't add in this step. See
//! `PORTING_NOTES.md` §10 for the discussion and what unblocks it.

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
