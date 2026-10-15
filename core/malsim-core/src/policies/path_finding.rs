//! Port of `utils/path_finding.py`.
//!
//! Note: Experimental, not proven correct
//!
//! Python converts the attack graph to a networkx graph and runs one
//! shortest-path search per compromised node. Here a single Dijkstra
//! search runs backwards from the goal, which yields the same cheapest
//! path for every start at once. See `PORTING_NOTES.md` §12 for this and
//! for the unreachable AND-step validation that is not ported.

use crate::types::{NodeMap, NodeSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use maltoolbox_language::graph::attack_step::AttackStepType;

use crate::agent_state::{sorted_by_id, AgentStateError, AttackerState};
use crate::policies::PolicyError;

/// A node on the search frontier, ordered so that `BinaryHeap` pops the
/// lowest cost first and breaks ties on the lowest node id.
struct Frontier {
    cost: f64,
    id: i64,
    node: AttackGraphNodeId,
}

impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Frontier {}

impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.id.cmp(&self.id))
    }
}

/// `ttc_values[node]`, with Python's `KeyError` as an error.
fn ttc_value(
    attack_graph: &AttackGraph,
    ttc_values: &NodeMap<f64>,
    node: AttackGraphNodeId,
) -> Result<f64, PolicyError> {
    ttc_values.get(&node).copied().ok_or_else(|| {
        AgentStateError::MissingTtcValue(attack_graph.full_name_of(node, None)).into()
    })
}

/// For every node that can reach `goal_node`: the cost of its cheapest
/// path there (the TTCs of every node on the path except the node itself)
/// and the next node on that path. An edge parent -> child weighs the
/// child's TTC, as in `attack_graph_to_nx_graph`.
fn paths_to_goal(
    attack_graph: &AttackGraph,
    goal_node: AttackGraphNodeId,
    ttc_values: &NodeMap<f64>,
) -> Result<NodeMap<(f64, Option<AttackGraphNodeId>)>, PolicyError> {
    let mut best: NodeMap<(f64, Option<AttackGraphNodeId>)> = NodeMap::default();
    let mut done: NodeSet = NodeSet::default();
    let mut frontier = BinaryHeap::new();

    let Some(goal) = attack_graph.nodes.get(goal_node) else {
        // `nx.NodeNotFound`
        return Ok(best);
    };
    best.insert(goal_node, (0.0, None));
    frontier.push(Frontier {
        cost: 0.0,
        id: goal.id,
        node: goal_node,
    });

    while let Some(Frontier { cost, node, .. }) = frontier.pop() {
        if !done.insert(node) {
            continue;
        }
        let parents = sorted_by_id(
            attack_graph,
            attack_graph.nodes[node].parents.iter().copied(),
        );
        if parents.is_empty() {
            continue;
        }
        let parent_cost = cost + ttc_value(attack_graph, ttc_values, node)?;
        for parent in parents {
            let improves = best
                .get(&parent)
                .is_none_or(|&(known, _)| parent_cost < known);
            if improves && !done.contains(&parent) {
                best.insert(parent, (parent_cost, Some(node)));
                frontier.push(Frontier {
                    cost: parent_cost,
                    id: attack_graph.nodes[parent].id,
                    node: parent,
                });
            }
        }
    }
    Ok(best)
}

/// Find shortest valid attack path from compromised nodes to goal: the
/// steps still to perform, in order, ending with `goal_node`. Empty when
/// the goal can't be reached (or is already compromised).
pub fn get_shortest_path_to(
    attack_graph: &AttackGraph,
    compromised_nodes: &[AttackGraphNodeId],
    goal_node: AttackGraphNodeId,
    ttc_values: &NodeMap<f64>,
) -> Result<Vec<AttackGraphNodeId>, PolicyError> {
    let best = paths_to_goal(attack_graph, goal_node, ttc_values)?;

    // The cheapest path over all starts; the cost of a path includes its
    // start node (`sum(ttc_values[n] for n in path_of_nodes)`).
    let mut cheapest: Option<(f64, AttackGraphNodeId)> = None;
    for start in sorted_by_id(attack_graph, compromised_nodes.iter().copied()) {
        let Some(&(cost_after_start, _)) = best.get(&start) else {
            continue;
        };
        let path_cost = cost_after_start + ttc_value(attack_graph, ttc_values, start)?;
        if cheapest.is_none_or(|(cost, _)| path_cost < cost) {
            cheapest = Some((path_cost, start));
        }
    }
    let Some((_, start)) = cheapest else {
        return Ok(Vec::new());
    };

    let compromised_nodes: NodeSet = compromised_nodes.iter().copied().collect();
    let mut planned_path = Vec::new();
    let mut step = Some(start);
    while let Some(node) = step {
        if !compromised_nodes.contains(&node) {
            planned_path.push(node);
        }
        step = best[&node].1;
    }
    Ok(planned_path)
}

/// Return shortest path for each of the attackers goals, in node-id order
/// of the goals.
pub fn get_shortest_paths_for_attacker(
    attacker_state: &AttackerState<'_>,
) -> Result<Vec<(AttackGraphNodeId, Vec<AttackGraphNodeId>)>, PolicyError> {
    let attack_graph = attacker_state.attack_graph;
    let ttc_values = attack_graph
        .nodes
        .iter()
        .filter(|(_, n)| matches!(n.step_type, AttackStepType::Or | AttackStepType::And))
        .map(|(key, _)| Ok((key, attacker_state.node_ttc_value(key)?)))
        .collect::<Result<NodeMap<_>, PolicyError>>()?;

    let goals = &attacker_state.runtime.goals;
    if goals.is_empty() {
        return Err(PolicyError::NoGoals);
    }
    let performed_nodes = sorted_by_id(
        attack_graph,
        attacker_state.runtime.performed_nodes.iter().copied(),
    );
    sorted_by_id(attack_graph, goals.iter().copied())
        .into_iter()
        .map(|goal| {
            let path = get_shortest_path_to(attack_graph, &performed_nodes, goal, &ttc_values)?;
            Ok((goal, path))
        })
        .collect()
}
