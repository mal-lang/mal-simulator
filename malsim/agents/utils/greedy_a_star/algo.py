import heapq
import logging
from copy import copy
from itertools import count
from typing import Callable, Iterable, List, Tuple

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from .utils import ttc_map, filter_defense, NoPath, merge_paths


def naive_a_star(
    attack_graph: AttackGraph,
    source: AttackGraphNode,
    target: AttackGraphNode,
    heuristic: Callable[[AttackGraphNode, AttackGraphNode], float] = lambda u, v: 0,
    unusable_nodes: List[AttackGraphNode] = [],
) -> Tuple[List[AttackGraphNode], float]:
    """Returns a list of nodes in a heuristically shortest path between source and target using the A* ("A-star") algorithm.

    Parameters
    ----------
    attack_graph : AttackGraph

    source : node
       Starting node for path.

    target : int
       Target node for path.

    heuristic : function
       A function to evaluate the estimate of the distance from the a node to the target. The function takes two nodes arguments and must return a number.
       The default heuristic is h=0 which is same as Dijkstra's algorithm.

    Raises
    ------
    ValueError
        If no path exists between source and target.

    Adapted from NetworkX and Sandor Berglund's thesis.
    """
    if source.id not in attack_graph.nodes or target.id not in attack_graph.nodes:
        raise ValueError(f"Either source {source} or target {target} is not in attack graph")

    def cost(u, v):
        return ttc_map(v.ttc) + 1

    unusable_ids = {node.id for node in unusable_nodes}

    # g_score[node.id] = the best-known cost from source to node
    g_score = {}
    # f_score[node.id] = g_score[node.id] + heuristic(node, target)
    f_score = {}
    # came_from[node.id] = the node we came from on the best path from source
    came_from = {}

    for node_id in attack_graph.nodes:
        g_score[node_id] = float("inf")
        f_score[node_id] = float("inf")
    g_score[source.id] = 0
    f_score[source.id] = heuristic(source, target)

    # Priority queue of (f_score, tie-break counter, node)
    # Tie-break counter ensures no direct comparison of nodes
    queue = []
    c = count()
    heapq.heappush(queue, (f_score[source.id], next(c), source))

    closed_set = set()

    while queue:
        _, __, current = heapq.heappop(queue)

        # If we reached the target, reconstruct path
        if current == target:
            return _reconstruct_path(came_from, current), g_score[current.id]

        if current.id in closed_set or current.id in unusable_ids:
            continue
        closed_set.add(current.id)

        for neighbor in current.children:
            if neighbor.id in closed_set or neighbor.id in unusable_ids:
                continue
            tentative_g = g_score[current.id] + cost(current, neighbor)
            if tentative_g < g_score[neighbor.id]:
                came_from[neighbor.id] = current
                g_score[neighbor.id] = tentative_g
                f_score[neighbor.id] = tentative_g + heuristic(neighbor, target)
                heapq.heappush(queue, (f_score[neighbor.id], next(c), neighbor))

    raise NoPath(target, f"Node {target.full_name} not reachable from {source.full_name}")

def _reconstruct_path(came_from, current):
    """Reconstructs path using came_from after we pop target from the queue."""
    path = [current]
    while current.id in came_from:
        current = came_from[current.id]
        path.append(current)
    path.reverse()
    return path



def correct_and_steps(
    attack_graph: AttackGraph,
    source: AttackGraphNode,
    naive_path: List[AttackGraphNode],
    unused_entry_points: List[AttackGraphNode],
    unusable_nodes: List[AttackGraphNode],
):
    new_path = naive_path

    def node_collection_difference(
        a: Iterable[AttackGraphNode], b: Iterable[AttackGraphNode]
    ) -> List[AttackGraphNode]:
        ids = set(node.id for node in a).difference(node.id for node in b)
        return [node for node in a if node.id in ids]

    for i, node in enumerate(new_path):
        if node.type == "and" and node.id != source.id:
            # For each AND-step parent, not in the path
            for and_parent in node_collection_difference(
                filter_defense(node.parents), new_path[:i]
            ):
                min_path, min_ttc = None, float("inf")
                for sub_source in new_path[:i] + unused_entry_points:
                    try:
                        p, c = naive_a_star(attack_graph, sub_source, and_parent, unusable_nodes=unusable_nodes)
                        if c < min_ttc:
                            min_path, min_ttc = p, c
                    except NoPath:
                        pass
                if not min_path:
                    raise NoPath(
                        node, f"No path to AND-step parent: {and_parent.full_name} for {node.full_name}"
                    )
                new_path = merge_paths(new_path, min_path, i)

    return new_path


def single_source_a_star(
    attack_graph: AttackGraph,
    source: AttackGraphNode,
    target: AttackGraphNode,
    other_sources: List[AttackGraphNode],
) -> List[AttackGraphNode]:
    path, _ = naive_a_star(attack_graph, source, target)
    unusable_nodes = []

    def check_and_steps(path: List[AttackGraphNode]) -> bool:
        and_steps = [
            node for node in path
            if node.type == "and"
            and node.id != source.id
            and node.id not in {
                node.id for node in other_sources
            }
        ]
        return all(
            parent in path
            for node in and_steps
            for parent in filter_defense(node.parents)
        )

    while not check_and_steps(other_sources + path):
        try:
            path = correct_and_steps(
                attack_graph, source, path, other_sources, unusable_nodes
            )
        except NoPath as unreachable:
            logging.debug(
                unreachable.args[0]
            )
            unusable_nodes.append(unreachable.node)
            old_path = copy(path)
            path, _ = naive_a_star(
                attack_graph, source, target, unusable_nodes=unusable_nodes
            )
            if path == old_path:
                raise unreachable
    return path


def single_target_a_star(
    attack_graph: AttackGraph,
    sources: AttackGraphNode | List[AttackGraphNode],
    target: AttackGraphNode,
) -> List[AttackGraphNode]:
    if isinstance(sources, AttackGraphNode):
        sources = [sources]

    ret_path, min_ttc = [], float("inf")

    for i, source in enumerate(sources):
        try:
            path = single_source_a_star(
                attack_graph, source, target, sources[:i] + sources[i + 1 :]
            )
            path_ttc = sum(ttc_map(node.ttc) for node in path)
            if path_ttc < min_ttc:
                ret_path, min_ttc = path, path_ttc
        except NoPath:
            pass

    if len(ret_path) == 0:
        raise NoPath(target, f"No path to target: {target}")

    return ret_path


def greedy_a_star_attack(
    attack_graph: AttackGraph,
    sources: AttackGraphNode | List[AttackGraphNode],
    targets: AttackGraphNode | List[AttackGraphNode],
) -> List[AttackGraphNode]:
    if isinstance(sources, AttackGraphNode):
        sources = [sources]
    if isinstance(targets, AttackGraphNode):
        targets = [targets]

    ret_path = []

    for target in targets:
        new_path = single_target_a_star(attack_graph, sources, target)
        ret_path = merge_paths(ret_path, new_path, len(ret_path))

    return ret_path
