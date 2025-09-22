"""
Note: Experimental, not proven correct
"""

from __future__ import annotations
from itertools import tee
from operator import itemgetter
from typing import TYPE_CHECKING

import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

if TYPE_CHECKING:
    from malsim.mal_simulator import MalSimAttackerState

def attack_graph_to_nx_graph(
    attack_graph: AttackGraph,
    ttc_values: dict[AttackGraphNode, float]
) -> nx.DiGraph:
    """
    Convert an AttackGraph object into a NetworkX directed graph.
    Add TTC value if given.

    Nodes keep attributes: id, name, type.
    Edges are directed from parent -> child.
    """
    g = nx.DiGraph()

    # Add nodes with attributes
    for node in attack_graph.nodes.values():
        g.add_node(
            node.id,
            id=node.id,
            name=node.name,
            type=node.type,
            ttc=ttc_values[node] if node.type in ('or', 'and') else 0
        )
    # Add edges based on parent/child relations
    for node in attack_graph.nodes.values():
        for child in node.children:
            g.add_edge(node.id, child.id, weight=ttc_values[child])
        for parent in node.parents:
            g.add_edge(parent.id, node.id)

    return g


def _nx_shortest_path(
    attack_graph: AttackGraph,
    nx_graph: nx.DiGraph,
    source: AttackGraphNode,
    target: AttackGraphNode,
    ttc_values: dict[AttackGraphNode, float]
) -> tuple[list[AttackGraphNode], float]:
    """Find the shortest path to the target"""
    try:
        path_of_ids: list[int] = shortest_path(
            nx_graph, source=source.id, target=target.id, weight="weight"
        )
        path_of_nodes = [attack_graph.nodes[i] for i in path_of_ids]
        path_cost = sum(ttc_values[n] for n in path_of_nodes)

    except nx.NetworkXNoPath:
        return [], 0.0

    return path_of_nodes, path_cost


def _find_path_to(
    attack_graph: AttackGraph,
    nx_graph: nx.DiGraph,
    total_path: list[AttackGraphNode],
    compromised_steps: list[AttackGraphNode],
    goal_node: AttackGraphNode,
    ttc_values: dict[AttackGraphNode, float]
) -> tuple[list[AttackGraphNode], float]:
    """Look for paths from step to goal"""

    paths = (
        _nx_shortest_path(
            attack_graph, nx_graph, start, goal_node, ttc_values
        ) for start in compromised_steps
    )

    # Go through the available paths in order of decreasing cost
    # and select the first viable one.
    for path_to_target, _ in sorted(paths, key=itemgetter(1)):
        if not path_to_target:
            continue

        try:
            return _validate_path(
                attack_graph,
                nx_graph,
                path_to_target,
                total_path,
                compromised_steps,
                ttc_values
            )
        except nx.NetworkXNoPath:
            continue

    # None of the paths worked
    raise nx.NetworkXNoPath


def get_shortest_path_to(
    attack_graph: AttackGraph,
    compromised_nodes: list[AttackGraphNode],
    goal_node: AttackGraphNode,
    ttc_values: dict[AttackGraphNode, float]
) -> list[AttackGraphNode]:
    """Find shortest valid attack path from compromised nodes to goal"""

    total_path: list[AttackGraphNode] = []
    nx_graph = attack_graph_to_nx_graph(attack_graph, ttc_values)

    try:
        path, _ = _find_path_to(
            attack_graph,
            nx_graph,
            total_path,
            compromised_nodes,
            goal_node,
            ttc_values
        )
    except nx.NodeNotFound:
        return []
    except nx.NetworkXNoPath:
        return []

    for step in path:
        _add_unique(total_path, step)

    planned_path = list(
        filter(lambda step: step not in compromised_nodes, total_path)
    )

    return planned_path


def get_shortest_paths_for_attacker(
    attacker_state: MalSimAttackerState
) -> dict[AttackGraphNode, list[AttackGraphNode]]:
    """Return shortest path for each of the attackers goals"""
    ttc_values = {
        n: attacker_state.sim.node_ttc_value(n)
        for n in attacker_state.sim.attack_graph.nodes.values()
        if n.type in ('or', 'and')
    }

    shortest_paths = {}
    assert attacker_state.goals, (
        "Attacker needs goal set for shortest path calculation"
    )
    for goal in attacker_state.goals:
        shortest_paths[goal] = get_shortest_path_to(
            attacker_state.sim.attack_graph,
            list(attacker_state.performed_nodes),
            goal,
            ttc_values
        )
    return shortest_paths


def _validate_path(
    attack_graph: AttackGraph,
    nx_graph: nx.DiGraph,
    path: list[AttackGraphNode],
    total_path: list[AttackGraphNode],
    compromised_steps: list[AttackGraphNode],
    ttc_values: dict[AttackGraphNode, float]
) -> tuple[list[AttackGraphNode], float]:

    ttc_cost = 0.0
    # Check that each step in the path is reacable with respect to AND-steps.
    for node in path:
        # If node is AND step, go to parents first.
        if node.type == 'and' and node not in path:
            paths_to_parents = [
                _find_path_to(
                    attack_graph,
                    nx_graph,
                    total_path,
                    compromised_steps,
                    parent,
                    ttc_values
                ) for parent in node.parents
            ]
            ttc_cost += _add_paths_to_total_path(
                paths_to_parents, total_path, ttc_values
            )

        ttc_cost += ttc_values[node]
        _add_unique(path, node)

    return path, ttc_cost


def _add_unique(path: list[AttackGraphNode], item: AttackGraphNode) -> int:
    if item not in path:
        path.append(item)
        return True

    return False


def _add_paths_to_total_path(
    paths: list[tuple[list[AttackGraphNode], float]],
    total_path: list[AttackGraphNode],
    ttc_values: dict[AttackGraphNode, float]
) -> float:
    """Add needed paths and return cost of adding it"""
    added_ttc_cost = 0.0
    for path, _ in sorted(paths, key=itemgetter(1)):
        steps_not_already_in_path = tee(
            filter(lambda x: x not in total_path, path)
        )
        added_ttc_cost += sum(
            ttc_values[step] for step in steps_not_already_in_path[0]
        )
        total_path.extend(steps_not_already_in_path[1])

    return added_ttc_cost
