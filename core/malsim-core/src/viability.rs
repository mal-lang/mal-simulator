//! Rust port of `python/malsim/mal_simulator/graph_processing.py`'s
//! viability propagation and pruning half - the counterpart of
//! `crate::necessity`, which ports the necessity half of the same file.
//!
//! Ports `evaluate_viability`, `_propagate_viability_from_node` (as
//! `propagate_viability_from_node`), `calculate_viability`,
//! `make_node_unviable` and `prune_unviable_and_unnecessary_nodes`.
//! Viability = whether a node can be traversed under any circumstances, or
//! whether the model structure (existence status), enabled defenses or
//! impossible attack steps make it unviable. Pruning removes `or`/`and`
//! nodes that are unviable or unnecessary (see `crate::necessity`).
//!
//! Signature shape mirrors `crate::necessity`: `node_id` is passed
//! alongside `&AttackGraphNode` (the node doesn't carry its own slotmap
//! key), per-node results are `HashMap<AttackGraphNodeId, bool>`, and node
//! sets are `HashSet<AttackGraphNodeId>`. Python's `logger.debug`/
//! `logger.error` calls are dropped - the crate has no logging dependency.
//!
//! Tests use `crate::test_fixtures`' dummy-language graphs, same as
//! `crate::necessity`'s tests.

use std::collections::{HashMap, HashSet};
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNode, AttackGraphNodeId, GraphError};

/// Not `Clone`/`PartialEq`/`Eq` (unlike `crate::necessity::NecessityError`)
/// because the `Graph` variant wraps `maltoolbox_attackgraph::GraphError`,
/// which only derives `Debug` - same trade-off as
/// `crate::model_effects::ModelEffectsError`/`crate::model_state::ModelStateError`.
#[derive(Debug)]
pub enum ViabilityError {
    /// Mirrors Python's `assert isinstance(node.existence_status, bool)`
    /// for `exist`/`notExist` nodes.
    MissingExistenceStatus(AttackGraphNodeId),
    /// A node's viability was read before it was computed - mirrors
    /// Python's `viability_per_node[node]` `KeyError` (the dict must be
    /// pre-seeded for every node before `evaluate_viability` runs; see
    /// `calculate_viability`). Also raised by
    /// `prune_unviable_and_unnecessary_nodes` for an `or`/`and` node
    /// missing from `viability_per_node`.
    MissingViability(AttackGraphNodeId),
    /// Mirrors Python's `necessity_per_node[node]` `KeyError` in
    /// `prune_unviable_and_unnecessary_nodes`.
    MissingNecessity(AttackGraphNodeId),
    /// Mirrors Python's `ValueError` for a `node.type` outside
    /// `exist`/`notExist`/`defense`/`or`/`and`.
    UnknownStepType(String, AttackGraphNodeId),
    /// Wraps an error from `AttackGraph::remove_node` during pruning -
    /// same convention as `ModelEffectsError::Graph`/`ModelStateError::Graph`.
    Graph(GraphError),
}

impl From<GraphError> for ViabilityError {
    fn from(e: GraphError) -> Self {
        ViabilityError::Graph(e)
    }
}

impl fmt::Display for ViabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViabilityError::MissingExistenceStatus(id) => {
                write!(f, "existence status not defined for node {id:?}")
            }
            ViabilityError::MissingViability(id) => {
                write!(f, "viability not yet computed for node {id:?}")
            }
            ViabilityError::MissingNecessity(id) => {
                write!(f, "necessity not computed for node {id:?}")
            }
            ViabilityError::UnknownStepType(step_type, id) => write!(
                f,
                "evaluate_viability was provided node {id:?} which is of unknown type \"{step_type}\""
            ),
            ViabilityError::Graph(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ViabilityError {}

/// Port of `evaluate_viability`. Takes `node_id` alongside `node` since
/// `AttackGraphNode` doesn't carry its own slotmap key - needed to check
/// membership in `impossible_attack_steps`/`enabled_defenses` and for
/// error context.
pub fn evaluate_viability(
    node_id: AttackGraphNodeId,
    node: &AttackGraphNode,
    viability_per_node: &HashMap<AttackGraphNodeId, bool>,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
    impossible_attack_steps: &HashSet<AttackGraphNodeId>,
) -> Result<bool, ViabilityError> {
    if impossible_attack_steps.contains(&node_id) {
        // Impossible step is not viable
        return Ok(false);
    }

    let parent_viability = |id: &AttackGraphNodeId| -> Result<bool, ViabilityError> {
        viability_per_node
            .get(id)
            .copied()
            .ok_or(ViabilityError::MissingViability(*id))
    };

    match node.step_type.as_str() {
        "exist" => node
            .existence_status
            .ok_or(ViabilityError::MissingExistenceStatus(node_id)),
        "notExist" => node
            .existence_status
            .map(|exists| !exists)
            .ok_or(ViabilityError::MissingExistenceStatus(node_id)),
        "defense" => Ok(!enabled_defenses.contains(&node_id)),
        // Python: `any(viability_per_node[p] for p in node.parents) or not
        // node.parents` - `any(())` is `False`, so the empty-parents case
        // genuinely needs the explicit `True` here (the mirror image of
        // necessity's "and" branch).
        "or" => {
            if node.parents.is_empty() {
                return Ok(true);
            }
            for parent in &node.parents {
                if parent_viability(parent)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        // Python: `all(viability_per_node[p] for p in node.parents) or not
        // node.parents` - `all(())` is already `True`, so the `or not
        // node.parents` is redundant for "and"; this loop mirrors that
        // without the redundant check.
        "and" => {
            for parent in &node.parents {
                if !parent_viability(parent)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        other => Err(ViabilityError::UnknownStepType(other.to_string(), node_id)),
    }
}

/// Port of `_propagate_viability_from_node`.
pub fn propagate_viability_from_node(
    node_id: AttackGraphNodeId,
    graph: &AttackGraph,
    viability_per_node: &mut HashMap<AttackGraphNodeId, bool>,
    impossible_attack_steps: &HashSet<AttackGraphNodeId>,
) -> Result<HashSet<AttackGraphNodeId>, ViabilityError> {
    let mut changed_nodes = HashSet::new();
    for &child_id in &graph.nodes[node_id].children {
        let child = &graph.nodes[child_id];
        // Python passes `frozenset()` (not the real `enabled_defenses`)
        // into this inner `evaluate_viability` call - mirrored exactly.
        // Note that unlike necessity, a `defense` child reached here
        // *would* be re-evaluated as viable (it's not in the empty set);
        // that is Python's behaviour too, and defense nodes normally have
        // no parents in a generated graph, so it isn't reached in practice.
        let is_viable = evaluate_viability(
            child_id,
            child,
            viability_per_node,
            &HashSet::new(),
            impossible_attack_steps,
        )?;
        // Python indexes `viability_per_node[child]` (`KeyError` if
        // missing) - surfaced as `MissingViability` rather than treating
        // a missing entry as "changed".
        let current = viability_per_node
            .get(&child_id)
            .copied()
            .ok_or(ViabilityError::MissingViability(child_id))?;
        if is_viable != current {
            viability_per_node.insert(child_id, is_viable);
            changed_nodes.insert(child_id);
            changed_nodes.extend(propagate_viability_from_node(
                child_id,
                graph,
                viability_per_node,
                impossible_attack_steps,
            )?);
        }
    }
    Ok(changed_nodes)
}

/// Port of `calculate_viability`.
pub fn calculate_viability(
    graph: &AttackGraph,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
    impossible_attack_steps: &HashSet<AttackGraphNodeId>,
) -> Result<HashMap<AttackGraphNodeId, bool>, ViabilityError> {
    let mut viability_per_node: HashMap<AttackGraphNodeId, bool> =
        graph.nodes.keys().map(|id| (id, true)).collect();

    // Unlike `calculate_necessity`, Python evaluates *every* node here,
    // not just `exist`/`notExist`/`defense` ones.
    let node_ids: Vec<AttackGraphNodeId> = graph.nodes.keys().collect();
    for node_id in node_ids {
        let node = &graph.nodes[node_id];
        let is_viable = evaluate_viability(
            node_id,
            node,
            &viability_per_node,
            enabled_defenses,
            impossible_attack_steps,
        )?;
        viability_per_node.insert(node_id, is_viable);
        if !is_viable {
            propagate_viability_from_node(
                node_id,
                graph,
                &mut viability_per_node,
                impossible_attack_steps,
            )?;
        }
    }
    Ok(viability_per_node)
}

/// Port of `make_node_unviable`. Python mutates `viability_per_node` in
/// place *and* returns it alongside the changed set; here the map is
/// mutated through `&mut` and only the set of nodes made unviable is
/// returned (returning the map too would be redundant with `&mut`).
pub fn make_node_unviable(
    node_id: AttackGraphNodeId,
    graph: &AttackGraph,
    viability_per_node: &mut HashMap<AttackGraphNodeId, bool>,
    impossible_attack_steps: &HashSet<AttackGraphNodeId>,
) -> Result<HashSet<AttackGraphNodeId>, ViabilityError> {
    viability_per_node.insert(node_id, false);
    propagate_viability_from_node(node_id, graph, viability_per_node, impossible_attack_steps)
}

/// Port of `prune_unviable_and_unnecessary_nodes`. Removes every `or`/`and`
/// node that is unviable or unnecessary from `graph`.
pub fn prune_unviable_and_unnecessary_nodes(
    graph: &mut AttackGraph,
    viability_per_node: &HashMap<AttackGraphNodeId, bool>,
    necessity_per_node: &HashMap<AttackGraphNodeId, bool>,
) -> Result<(), ViabilityError> {
    let viability = |id: AttackGraphNodeId| -> Result<bool, ViabilityError> {
        viability_per_node
            .get(&id)
            .copied()
            .ok_or(ViabilityError::MissingViability(id))
    };
    let necessity = |id: AttackGraphNodeId| -> Result<bool, ViabilityError> {
        necessity_per_node
            .get(&id)
            .copied()
            .ok_or(ViabilityError::MissingNecessity(id))
    };

    // Python collects into a `set`; a `Vec` in graph order is used here so
    // removal order is deterministic (each node is pushed at most once).
    let mut nodes_to_remove = Vec::new();
    for (node_id, node) in &graph.nodes {
        if matches!(node.step_type.as_str(), "or" | "and") {
            // Python: `not viability_per_node[node] or not
            // necessity_per_node[node]` - short-circuits, so necessity is
            // only read here for viable nodes...
            let remove = !viability(node_id)? || !necessity(node_id)?;
            if remove {
                // ...but Python's removal loop then reads
                // `necessity_per_node[node]` unconditionally for its debug
                // log ('unviable' vs 'unnecessary'), so a missing necessity
                // entry for any node to remove is a `KeyError` there.
                // Checked up front so nothing is removed before erroring.
                necessity(node_id)?;
                nodes_to_remove.push(node_id);
            }
        }
    }

    // Do the removal separately so we don't remove nodes from the
    // collection we are looping over.
    for node_id in nodes_to_remove {
        graph.remove_node(node_id)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::necessity::calculate_necessity;
    use crate::test_fixtures::{add_dummy_node, dummy_graph};

    // Port of `tests/test_graph_processing.py::test_viability_viable_nodes`.
    #[test]
    fn calculate_viability_viable_nodes() {
        let mut graph = dummy_graph();

        // Exists + existence -> viable
        let exist_node = add_dummy_node(&mut graph, "DummyExistAttackStep");
        graph.nodes[exist_node].existence_status = Some(true);

        // NotExists + nonexistence -> viable
        let not_exist_node = add_dummy_node(&mut graph, "DummyNotExistAttackStep");
        graph.nodes[not_exist_node].existence_status = Some(false);

        // defense not enabled -> viable
        let defense_step_node = add_dummy_node(&mut graph, "DummyDefenseAttackStep");

        // or-node with viable parent -> viable (Python only sets
        // `or_node.parents`, not the parent's `children` - mirrored)
        let or_node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let or_node_parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[or_node].parents.insert(or_node_parent);

        // and-node with no parents -> viable (Python's unused `ttc_values`
        // bookkeeping for this node is dropped)
        let and_node = add_dummy_node(&mut graph, "DummyAndAttackStep");

        // and-node with viable parents -> viable
        let and_node2 = add_dummy_node(&mut graph, "DummyAndAttackStep");
        let and_node_parent1 = add_dummy_node(&mut graph, "DummyAndAttackStep");
        let and_node_parent2 = add_dummy_node(&mut graph, "DummyAndAttackStep");
        graph.nodes[and_node2].parents = [and_node_parent1, and_node_parent2].into_iter().collect();

        let enabled_defenses = HashSet::new();
        let viable_nodes = calculate_viability(&graph, &enabled_defenses, &HashSet::new()).unwrap();

        // Python asserts `node in viable_nodes` (dict key membership, which
        // holds for every node); the value check below is the stronger
        // intended property.
        for node in [
            exist_node,
            not_exist_node,
            defense_step_node,
            or_node,
            and_node,
            and_node2,
        ] {
            assert!(viable_nodes.contains_key(&node));
            assert!(viable_nodes[&node]);
        }
    }

    // Port of `tests/test_graph_processing.py::test_viability_unviable_nodes`.
    #[test]
    fn calculate_viability_unviable_nodes() {
        let mut impossible_attack_steps = HashSet::new();
        let mut graph = dummy_graph();

        // exists, existence_status = False -> not viable
        let exist_node = add_dummy_node(&mut graph, "DummyExistAttackStep");
        graph.nodes[exist_node].existence_status = Some(false);

        // notExists, existence_status = True -> not viable
        let not_exist_node = add_dummy_node(&mut graph, "DummyNotExistAttackStep");
        graph.nodes[not_exist_node].existence_status = Some(true);

        // Defense status on -> not viable
        let defense_step_node = add_dummy_node(&mut graph, "DummyDefenseAttackStep");

        // or-node with no viable parent -> non viable
        let or_node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let unviable_or_node_parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[or_node].parents.insert(unviable_or_node_parent);
        graph.nodes[unviable_or_node_parent]
            .children
            .insert(or_node);
        impossible_attack_steps.insert(unviable_or_node_parent);

        // and-node with two non-viable parents -> non viable
        let and_node = add_dummy_node(&mut graph, "DummyAndAttackStep");
        let unviable_and_node_parent1 = add_dummy_node(&mut graph, "DummyAndAttackStep");
        let unviable_and_node_parent2 = add_dummy_node(&mut graph, "DummyAndAttackStep");
        graph.nodes[and_node].parents = [unviable_and_node_parent1, unviable_and_node_parent2]
            .into_iter()
            .collect();
        // Python adds `and_node` to parent1's children twice and never to
        // parent2's - mirrored (one insert into parent1 only).
        graph.nodes[unviable_and_node_parent1]
            .children
            .insert(and_node);
        impossible_attack_steps.insert(unviable_and_node_parent1);
        impossible_attack_steps.insert(unviable_and_node_parent2);

        let enabled_defenses: HashSet<_> = [defense_step_node].into_iter().collect();
        let viability_per_node =
            calculate_viability(&graph, &enabled_defenses, &impossible_attack_steps).unwrap();

        assert!(!viability_per_node[&unviable_or_node_parent]);
        assert!(!viability_per_node[&unviable_and_node_parent1]);
        assert!(!viability_per_node[&unviable_and_node_parent2]);

        assert!(!viability_per_node[&exist_node]);
        assert!(!viability_per_node[&not_exist_node]);
        assert!(!viability_per_node[&defense_step_node]);
        assert!(!viability_per_node[&or_node]);
        assert!(!viability_per_node[&and_node]);
    }

    // Port of
    // `tests/test_graph_processing.py::test_analyzers_apriori_propagate_viability`.
    #[test]
    fn propagate_viability_from_node_updates_downstream_or_and_and_nodes() {
        let mut graph = dummy_graph();

        let vp1 = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let vp2 = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let uvp1 = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let uvp2 = add_dummy_node(&mut graph, "DummyDefenseAttackStep");

        let or_1vp = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let or_2uvp = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let and_1uvp = add_dummy_node(&mut graph, "DummyAndAttackStep");
        let and_2vp = add_dummy_node(&mut graph, "DummyAndAttackStep");

        graph.nodes[or_1vp].parents = [vp1, uvp1].into_iter().collect();
        graph.nodes[or_2uvp].parents = [uvp1, uvp2].into_iter().collect();
        graph.nodes[and_1uvp].parents = [vp1, uvp1].into_iter().collect();
        graph.nodes[and_2vp].parents = [vp1, vp2].into_iter().collect();

        graph.nodes[vp1].children = [or_1vp, and_1uvp, and_2vp].into_iter().collect();
        graph.nodes[vp2].children = [and_2vp].into_iter().collect();
        graph.nodes[uvp1].children = [or_1vp, or_2uvp, and_1uvp].into_iter().collect();
        graph.nodes[uvp2].children = [or_2uvp].into_iter().collect();

        let mut viability_per_node =
            calculate_viability(&graph, &HashSet::new(), &HashSet::new()).unwrap();

        // Make unviable
        viability_per_node.insert(uvp1, false);
        viability_per_node.insert(uvp2, false);

        let mut changed_nodes = HashSet::new();
        for &parent in &[vp1, vp2, uvp1, uvp2] {
            changed_nodes.extend(
                propagate_viability_from_node(
                    parent,
                    &graph,
                    &mut viability_per_node,
                    &HashSet::new(),
                )
                .unwrap(),
            );
        }

        assert_eq!(changed_nodes, [or_2uvp, and_1uvp].into_iter().collect());

        for node in [vp1, vp2, or_1vp, and_2vp] {
            assert!(viability_per_node[&node]);
        }
        for node in [uvp1, uvp2, or_2uvp, and_1uvp] {
            assert!(!viability_per_node[&node]);
        }
    }

    // Port of
    // `tests/test_graph_processing.py::test_analyzers_apriori_prune_unviable_and_unnecessary_nodes`.
    //
    // The Python test runs against the `model` fixture (coreLang +
    // `tests/testdata/models/simple_test_model.yml`), which has no Rust
    // fixture equivalent in `crate::test_fixtures`. Ported against a
    // hand-built dummy-language graph instead, exercising the same
    // property: pick the first `or` node and make it unnecessary, the
    // first `and` node and make it unviable, prune, and check both are
    // gone.
    #[test]
    fn prune_unviable_and_unnecessary_nodes_removes_marked_nodes() {
        let mut graph = dummy_graph();

        // A plain or -> and -> or -> and chain with a parentless root:
        // everything is viable and necessary until marked otherwise, so
        // only the two manual marks below can cause pruning.
        let or_a = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let and_a = add_dummy_node(&mut graph, "DummyAndAttackStep");
        let or_b = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let and_b = add_dummy_node(&mut graph, "DummyAndAttackStep");

        graph.nodes[and_a].parents.insert(or_a);
        graph.nodes[or_b].parents.insert(and_a);
        graph.nodes[and_b].parents.insert(or_b);
        graph.nodes[or_a].children.insert(and_a);
        graph.nodes[and_a].children.insert(or_b);
        graph.nodes[or_b].children.insert(and_b);

        // Pick out an or node and make it non-necessary, and an and node
        // to make unviable - mirrors Python's `next(node for node in
        // graph.nodes.values() if node.type == ...)`.
        let node_to_make_unnecessary = graph
            .nodes
            .iter()
            .find(|(_, n)| n.step_type.as_str() == "or")
            .map(|(id, _)| id)
            .unwrap();
        let node_to_make_unviable = graph
            .nodes
            .iter()
            .find(|(_, n)| n.step_type.as_str() == "and")
            .map(|(id, _)| id)
            .unwrap();

        let mut viability_per_node =
            calculate_viability(&graph, &HashSet::new(), &HashSet::new()).unwrap();
        let mut necessity_per_node = calculate_necessity(&graph, &HashSet::new()).unwrap();
        necessity_per_node.insert(node_to_make_unnecessary, false);
        viability_per_node.insert(node_to_make_unviable, false);

        prune_unviable_and_unnecessary_nodes(&mut graph, &viability_per_node, &necessity_per_node)
            .unwrap();

        // Make sure the node was pruned
        assert!(!graph.nodes.contains_key(node_to_make_unviable));
        assert!(!graph.nodes.contains_key(node_to_make_unnecessary));
    }

    // Second port of the same Python test, this time against a real
    // generated attack graph (wiperLang + `wiper_model.yml`) rather than a
    // hand-built chain - closer to the Python test's `model` fixture, which
    // is also a language-compiled, model-generated graph.
    #[test]
    fn prune_unviable_and_unnecessary_nodes_on_generated_graph() {
        let (mut graph, _model) = crate::test_fixtures::wiper_attack_graph();

        let node_to_make_unnecessary = graph
            .nodes
            .iter()
            .find(|(_, n)| n.step_type.as_str() == "or")
            .map(|(id, _)| id)
            .unwrap();
        let node_to_make_unviable = graph
            .nodes
            .iter()
            .find(|(_, n)| n.step_type.as_str() == "and")
            .map(|(id, _)| id)
            .unwrap();

        let mut viability_per_node =
            calculate_viability(&graph, &HashSet::new(), &HashSet::new()).unwrap();
        let mut necessity_per_node = calculate_necessity(&graph, &HashSet::new()).unwrap();
        necessity_per_node.insert(node_to_make_unnecessary, false);
        viability_per_node.insert(node_to_make_unviable, false);

        prune_unviable_and_unnecessary_nodes(&mut graph, &viability_per_node, &necessity_per_node)
            .unwrap();

        assert!(!graph.nodes.contains_key(node_to_make_unviable));
        assert!(!graph.nodes.contains_key(node_to_make_unnecessary));
    }

    // Extra (no Python counterpart): `make_node_unviable` marks the node
    // unviable and returns exactly the downstream nodes it made unviable.
    #[test]
    fn make_node_unviable_propagates_and_returns_changed_nodes() {
        let mut graph = dummy_graph();

        let root = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let other = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let and_child = add_dummy_node(&mut graph, "DummyAndAttackStep");
        let or_child = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let grandchild = add_dummy_node(&mut graph, "DummyOrAttackStep");

        // and_child needs both root and other -> unviable once root is.
        // or_child has `other` as an alternative parent -> stays viable.
        graph.nodes[and_child].parents = [root, other].into_iter().collect();
        graph.nodes[or_child].parents = [root, other].into_iter().collect();
        graph.nodes[grandchild].parents.insert(and_child);
        graph.nodes[root].children = [and_child, or_child].into_iter().collect();
        graph.nodes[other].children = [and_child, or_child].into_iter().collect();
        graph.nodes[and_child].children.insert(grandchild);

        let mut viability_per_node =
            calculate_viability(&graph, &HashSet::new(), &HashSet::new()).unwrap();
        assert!(viability_per_node.values().all(|&v| v));

        let made_unviable =
            make_node_unviable(root, &graph, &mut viability_per_node, &HashSet::new()).unwrap();

        assert_eq!(made_unviable, [and_child, grandchild].into_iter().collect());
        assert!(!viability_per_node[&root]);
        assert!(!viability_per_node[&and_child]);
        assert!(!viability_per_node[&grandchild]);
        assert!(viability_per_node[&other]);
        assert!(viability_per_node[&or_child]);
    }

    // Extra (no Python counterpart): mirrors Python's
    // `assert isinstance(node.existence_status, bool)`.
    #[test]
    fn evaluate_viability_missing_existence_status_errors() {
        let mut graph = dummy_graph();
        let exist_node = add_dummy_node(&mut graph, "DummyExistAttackStep");
        let not_exist_node = add_dummy_node(&mut graph, "DummyNotExistAttackStep");

        let err = calculate_viability(&graph, &HashSet::new(), &HashSet::new()).unwrap_err();
        assert!(matches!(
            err,
            ViabilityError::MissingExistenceStatus(id) if id == exist_node
        ));

        let err = evaluate_viability(
            not_exist_node,
            &graph.nodes[not_exist_node],
            &HashMap::new(),
            &HashSet::new(),
            &HashSet::new(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ViabilityError::MissingExistenceStatus(id) if id == not_exist_node
        ));
    }

    // Extra (no Python counterpart): an impossible attack step is unviable
    // regardless of type - checked before the type match, so it wins even
    // over an otherwise-erroring node.
    #[test]
    fn evaluate_viability_impossible_step_short_circuits() {
        let mut graph = dummy_graph();
        let exist_node = add_dummy_node(&mut graph, "DummyExistAttackStep");
        let impossible: HashSet<_> = [exist_node].into_iter().collect();

        let is_viable = evaluate_viability(
            exist_node,
            &graph.nodes[exist_node],
            &HashMap::new(),
            &HashSet::new(),
            &impossible,
        )
        .unwrap();
        assert!(!is_viable);
    }

    // Extra (no Python counterpart): mirrors Python's
    // `viability_per_node[parent]` `KeyError`.
    #[test]
    fn evaluate_viability_missing_parent_viability_errors() {
        let mut graph = dummy_graph();
        let parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let child = add_dummy_node(&mut graph, "DummyAndAttackStep");
        graph.nodes[child].parents.insert(parent);

        let err = evaluate_viability(
            child,
            &graph.nodes[child],
            &HashMap::new(),
            &HashSet::new(),
            &HashSet::new(),
        )
        .unwrap_err();
        assert!(matches!(err, ViabilityError::MissingViability(id) if id == parent));
    }

    // Extra (no Python counterpart): mirrors Python's
    // `necessity_per_node[node]` `KeyError` in pruning, and checks nothing
    // is removed when it errors.
    #[test]
    fn prune_missing_necessity_errors_without_removing() {
        let mut graph = dummy_graph();
        let or_node = add_dummy_node(&mut graph, "DummyOrAttackStep");

        let viability_per_node: HashMap<_, _> = [(or_node, false)].into_iter().collect();
        let err =
            prune_unviable_and_unnecessary_nodes(&mut graph, &viability_per_node, &HashMap::new())
                .unwrap_err();
        assert!(matches!(err, ViabilityError::MissingNecessity(id) if id == or_node));
        assert!(graph.nodes.contains_key(or_node));

        let err =
            prune_unviable_and_unnecessary_nodes(&mut graph, &HashMap::new(), &HashMap::new())
                .unwrap_err();
        assert!(matches!(err, ViabilityError::MissingViability(id) if id == or_node));
    }
}
