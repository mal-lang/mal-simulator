"""
MAL Simulator – Attack Graph Preprocessing

This module prepares the attack graph before simulation by computing two
properties for each node:

- Viability:
  Whether the node can ever be traversed, given model structure, enabled
  defenses, or explicitly impossible attack steps.

- Necessity:
  Whether the node is required to compromise its successors, or whether
  model structure makes it effectively already satisfied.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

logger = logging.getLogger(__name__)


# ============================================================================
# Necessity
# ============================================================================

def _propagate_necessity_from_node(
    node: AttackGraphNode,
    necessity_per_node: dict[AttackGraphNode, bool],
) -> set[AttackGraphNode]:
    """
    Recompute and propagate necessity to all descendants of `node`.

    Propagation continues recursively as long as necessity values change.

    Returns:
        Set of nodes whose necessity changed.
    """
    changed: set[AttackGraphNode] = set()

    for child in node.children:
        previous = necessity_per_node[child]
        necessity_per_node[child] = evaluate_necessity(
            child,
            necessity_per_node,
            enabled_defenses=set(),
        )

        if necessity_per_node[child] != previous:
            changed.add(child)
            changed |= _propagate_necessity_from_node(
                child, necessity_per_node
            )

    return changed


def evaluate_necessity(
    node: AttackGraphNode,
    necessity_per_node: dict[AttackGraphNode, bool],
    enabled_defenses: set[AttackGraphNode],
) -> bool:
    """
    Evaluate whether a single node is necessary.
    """
    match node.type:
        case 'exist':
            assert isinstance(node.existence_status, bool), (
                f'Existence status not defined for {node.full_name}.'
            )
            return not node.existence_status

        case 'notExist':
            assert isinstance(node.existence_status, bool), (
                f'Existence status not defined for {node.full_name}.'
            )
            return node.existence_status

        case 'defense':
            return node in enabled_defenses

        case 'or':
            # Necessary only if all parents are necessary
            return (
                all(necessity_per_node[parent] for parent in node.parents)
                or not node.parents
            )

        case 'and':
            # Necessary if any parent is necessary
            return (
                any(necessity_per_node[parent] for parent in node.parents)
                or not node.parents
            )

        case _:
            msg = (
                'Evaluate necessity received node "%s"(%d) '
                'with unknown type "%s".'
            )
            logger.error(msg, node.full_name, node.id, node.type)
            raise ValueError(msg % (node.full_name, node.id, node.type))


def calculate_necessity(
    graph: AttackGraph,
    enabled_defenses: set[AttackGraphNode],
) -> dict[AttackGraphNode, bool]:
    """
    Compute necessity for all nodes in the attack graph.
    """
    necessity_per_node = dict.fromkeys(graph.nodes.values(), True)

    for node in graph.nodes.values():
        if node.type in {'exist', 'notExist', 'defense'}:
            necessity_per_node[node] = evaluate_necessity(
                node, necessity_per_node, enabled_defenses
            )
            if not necessity_per_node[node]:
                _propagate_necessity_from_node(node, necessity_per_node)

    return necessity_per_node


# ============================================================================
# Viability
# ============================================================================

def _propagate_viability_from_node(
    node: AttackGraphNode,
    viability_per_node: dict[AttackGraphNode, bool],
    impossible_attack_steps: set[AttackGraphNode],
) -> set[AttackGraphNode]:
    """
    Recompute and propagate viability to all descendants of `node`.

    Returns:
        Set of nodes whose viability changed.
    """
    changed: set[AttackGraphNode] = set()

    for child in node.children:
        new_value = evaluate_viability(
            child,
            viability_per_node,
            enabled_defenses=set(),
            impossible_attack_steps=impossible_attack_steps,
        )

        if new_value != viability_per_node[child]:
            viability_per_node[child] = new_value
            changed.add(child)
            changed |= _propagate_viability_from_node(
                child, viability_per_node, impossible_attack_steps
            )

    return changed


def evaluate_viability(
    node: AttackGraphNode,
    viability_per_node: dict[AttackGraphNode, bool],
    enabled_defenses: set[AttackGraphNode],
    impossible_attack_steps: set[AttackGraphNode],
) -> bool:
    """
    Evaluate whether a single node is viable.
    """
    if node in impossible_attack_steps:
        return False

    match node.type:
        case 'exist':
            assert isinstance(node.existence_status, bool), (
                f'Existence status not defined for {node.full_name}.'
            )
            return node.existence_status

        case 'notExist':
            assert isinstance(node.existence_status, bool), (
                f'Existence status not defined for {node.full_name}.'
            )
            return not node.existence_status

        case 'defense':
            return node not in enabled_defenses

        case 'or':
            # Viable if any parent is viable
            return (
                any(viability_per_node[parent] for parent in node.parents)
                or not node.parents
            )

        case 'and':
            # Viable only if all parents are viable
            return (
                all(viability_per_node[parent] for parent in node.parents)
                or not node.parents
            )

        case _:
            msg = (
                'Evaluate viability received node "%s"(%d) '
                'with unknown type "%s".'
            )
            logger.error(msg, node.full_name, node.id, node.type)
            raise ValueError(msg % (node.full_name, node.id, node.type))


def calculate_viability(
    graph: AttackGraph,
    enabled_defenses: set[AttackGraphNode],
    impossible_attack_steps: set[AttackGraphNode],
) -> dict[AttackGraphNode, bool]:
    """
    Compute viability for all nodes in the attack graph.
    """
    viability_per_node = dict.fromkeys(graph.nodes.values(), True)

    for node in graph.nodes.values():
        viability_per_node[node] = evaluate_viability(
            node,
            viability_per_node,
            enabled_defenses,
            impossible_attack_steps,
        )
        if not viability_per_node[node]:
            _propagate_viability_from_node(
                node, viability_per_node, impossible_attack_steps
            )

    return viability_per_node


def make_node_unviable(
    node: AttackGraphNode,
    viability_per_node: dict[AttackGraphNode, bool],
    impossible_attack_steps: set[AttackGraphNode],
) -> tuple[dict[AttackGraphNode, bool], set[AttackGraphNode]]:
    """
    Mark `node` as unviable and propagate the effect.

    Returns:
        Updated viability mapping and the set of nodes that became unviable.
    """
    viability_per_node[node] = False
    affected = _propagate_viability_from_node(
        node, viability_per_node, impossible_attack_steps
    )
    return viability_per_node, affected


# ============================================================================
# Pruning
# ============================================================================

def prune_unviable_and_unnecessary_nodes(
    graph: AttackGraph,
    viability_per_node: dict[AttackGraphNode, bool],
    necessity_per_node: dict[AttackGraphNode, bool],
) -> None:
    """
    Remove AND/OR nodes that are either unviable or unnecessary.
    """
    logger.debug('Pruning unviable and unnecessary nodes from attack graph.')

    to_remove: set[AttackGraphNode] = set()

    for node in graph.nodes.values():
        if node.type in {'or', 'and'} and (
            not viability_per_node[node] or not necessity_per_node[node]
        ):
            to_remove.add(node)

    # Remove after collection to avoid mutating during iteration
    for node in to_remove:
        logger.debug(
            'Removing %s node "%s"(%d) from attack graph.',
            'unviable' if necessity_per_node[node] else 'unnecessary',
            node.full_name,
            node.id,
        )
        graph.remove_node(node)
