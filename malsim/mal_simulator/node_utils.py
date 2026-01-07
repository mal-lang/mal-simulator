"""Utilities to convert full names to nodes"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable
from maltoolbox.attackgraph import AttackGraphNode, AttackGraph


def full_name_or_node_to_node(
    attack_graph: AttackGraph, node_or_full_name: str | AttackGraphNode
) -> AttackGraphNode:
    """If `node_or_full_name` is a full_mame, return corresponding AttackGraphNode"""
    if isinstance(node_or_full_name, str):
        return get_node(attack_graph, node_or_full_name)
    return node_or_full_name


def full_names_or_nodes_to_nodes(
    attack_graph: AttackGraph,
    nodes_or_full_names: Iterable[str] | Iterable[AttackGraphNode],
) -> Iterable[AttackGraphNode]:
    """Generator converting nodes full_name to AttackGraphNode objects"""
    for n in nodes_or_full_names:
        yield get_node(attack_graph, n) if isinstance(n, str) else n


def full_name_dict_to_node_dict(
    attack_graph: AttackGraph, mapping: dict[str, Any] | dict[AttackGraphNode, Any]
) -> dict[AttackGraphNode, Any]:
    """Convert dict so it maps from AttackGraphNode instead of from full_name"""
    return {full_name_or_node_to_node(attack_graph, k): v for k, v in mapping.items()}


def get_node(
    attack_graph: AttackGraph,
    full_name: Optional[str] = None,
    node_id: Optional[int] = None,
) -> AttackGraphNode:
    """Get node from attack graph by either full name or id"""

    if full_name and not node_id:
        node = attack_graph.get_node_by_full_name(full_name)
    elif node_id and not full_name:
        node = attack_graph.nodes[node_id]
    else:
        raise ValueError("Provide either full_name or node_id to 'get_node'")

    if node is None:
        raise LookupError(f'Could not find node {full_name or node_id}')
    return node
