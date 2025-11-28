"""Utilities to convert full names to nodes"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable
from maltoolbox.attackgraph import AttackGraphNode

if TYPE_CHECKING:
    from malsim import MalSimulator


def full_name_or_node_to_node(
    sim: MalSimulator, node_or_full_name: str | AttackGraphNode
) -> AttackGraphNode:
    """If `node_or_full_name` is a full_mame, return corresponding AttackGraphNode"""
    if isinstance(node_or_full_name, str):
        return sim.get_node(node_or_full_name)
    return node_or_full_name


def full_names_or_nodes_to_nodes(
    sim: MalSimulator, nodes_or_full_names: Iterable[str] | Iterable[AttackGraphNode]
) -> Iterable[AttackGraphNode]:
    """Generator converting nodes full_name to AttackGraphNode objects"""
    for n in nodes_or_full_names:
        yield sim.get_node(n) if isinstance(n, str) else n


def full_name_dict_to_node_dict(
    sim: MalSimulator, mapping: dict[str, Any] | dict[AttackGraphNode, Any]
) -> dict[AttackGraphNode, Any]:
    """Convert dict so it maps from AttackGraphNode instead of from full_name"""
    return {full_name_or_node_to_node(sim, k): v for k, v in mapping.items()}
