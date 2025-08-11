"""
MAL Simulator Attack Graph Processing Submodule

This submodule is meant to process the attack graph before running the
simulation.
Currently it adds the following information to nodes:
- Viability = Determine if a node can be traversed under any circumstances or
  if the model structure makes it unviable.
- Necessity = Determine if a node is necessary for the attacker or if the
  model structure means it is not needed(it behaves as if it were already
  compromised) to compromise children attack steps.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

logger = logging.getLogger(__name__)

def propagate_viability_from_node(
        node: AttackGraphNode, is_viable: dict[AttackGraphNode, bool]
    ) -> set[AttackGraphNode]:
    """
    Update viability of children of node given as parameter. Propagate
    recursively via children as long as changes occur. Return all nodes which
    have changed viability.

    Arguments:
    node            - the attack graph node from which to propagate the
                      viability status

    Returns:
    changed_nodes   - set of nodes that have changed viability
    """
    logger.debug(
        'Propagate viability from "%s"(%d) with viability status %s.',
        node.full_name, node.id, is_viable[node]
    )
    changed_nodes = set()
    for child in node.children:
        original_value = is_viable[child]
        if child.type == 'or':
            is_viable[child] = any(
                is_viable[parent] for parent in child.parents)
        elif child.type == 'and':
            is_viable[child] = all(
                is_viable[parent] for parent in child.parents)

        if is_viable[child] != original_value:
            changed_nodes |= (
                {child} | propagate_viability_from_node(child, is_viable)
            )

    return changed_nodes

def propagate_necessity_from_node(
        node: AttackGraphNode, is_necessary: dict[AttackGraphNode, bool]
    ) -> set[AttackGraphNode]:
    """
    Update necessity of children of node givein as parameter. Propagate
    recursively via children as long as changes occur. Return all nodes which
    have changed necessity.

    Arguments:
    node            - the attack graph node from which to propagate the
                      necessity status

    Returns:
    changed_nodes   - set of nodes that have changed necessity
    """
    logger.debug(
        'Propagate necessity from "%s"(%d) with necessity status %s.',
        node.full_name, node.id, is_necessary[node]
    )
    changed_nodes = set()
    for child in node.children:
        original_value = is_necessary[child]
        if child.type == 'or':
            is_necessary[child] = all(
                is_necessary[parent] for parent in child.parents)

        elif child.type == 'and':
            is_necessary[child] = any(
                is_necessary[parent] for parent in child.parents)

        if is_necessary[child] != original_value:
            changed_nodes |= (
                {child} | propagate_necessity_from_node(child, is_necessary)
            )

    return changed_nodes


def evaluate_viability(
        node: AttackGraphNode,
        is_viable: dict[AttackGraphNode, bool],
        enabled_defenses: set[AttackGraphNode]
    ) -> bool:
    """
    Arguments:
    node                - the node to evaluate viability for
    is_viable           - dict mapping nodes to their viability
    enabled_defenses    - set of all enabled defenses
    """
    node_is_viable = is_viable[node]
    match (node.type):
        case 'exist':
            assert isinstance(node.existence_status, bool), \
                f'Existence status not defined for {node.full_name}.'
            node_is_viable = node.existence_status
        case 'notExist':
            assert isinstance(node.existence_status, bool), \
                f'Existence status not defined for {node.full_name}.'
            node_is_viable = not node.existence_status
        case 'defense':
            node_is_viable = node not in enabled_defenses
        case 'or':
            node_is_viable = any(is_viable[parent] for parent in node.parents)
        case 'and':
            node_is_viable = all(is_viable[parent] for parent in node.parents)
        case _:
            msg = ('Evaluate viability was provided node "%s"(%d) which '
                   'is of unknown type "%s"')
            logger.error(msg, node.full_name, node.id, node.type)
            raise ValueError(msg % (node.full_name, node.id, node.type))

    return node_is_viable

def evaluate_necessity(
        node: AttackGraphNode,
        is_necessary: dict[AttackGraphNode, bool],
        enabled_defenses: set[AttackGraphNode]
    ) -> None:
    """
    Arguments:
    node             - the node to evaluate necessity for.
    is_necessary     - dict mapping nodes to their necessity
    enabled_defenses - set of all enabled defenses
    """
    match (node.type):
        case 'exist':
            assert isinstance(node.existence_status, bool), \
                f'Existence status not defined for {node.full_name}.'
            is_necessary[node] = not node.existence_status
        case 'notExist':
            assert isinstance(node.existence_status, bool), \
                f'Existence status not defined for {node.full_name}.'
            is_necessary[node] = bool(node.existence_status)
        case 'defense':
            is_necessary[node] = node in enabled_defenses
        case 'or':
            is_necessary[node] = all(
                is_necessary[parent] for parent in node.parents
            )
        case 'and':
            is_necessary[node] = any(
                is_necessary[parent] for parent in node.parents
            )
        case _:
            msg = ('Evaluate necessity was provided node "%s"(%d) which '
                   'is of unknown type "%s"')
            logger.error(msg, node.full_name, node.id, node.type)
            raise ValueError(msg % (node.full_name, node.id, node.type))


def calculate_viability_and_necessity(
        graph: AttackGraph, enabled_defenses: set[AttackGraphNode]
    ) -> tuple[
            dict[AttackGraphNode, bool],
            dict[AttackGraphNode, bool]
        ]:
    """
    Arguments:
    graph             - the attack graph for which we wish to determine the
                        viability and necessity statuses for the nodes.
    enabled_defenses  - set of all enabled defenses

    Return:
    tuple with
        - dict mapping from node to viability
        - dict mapping from node to necessity
    """
    is_viable = {n: True for n in graph.nodes.values()}
    is_necessary = {n: True for n in graph.nodes.values()}

    for node in graph.nodes.values():
        if node.type in ['exist', 'notExist', 'defense']:
            evaluate_viability(node, is_viable, enabled_defenses)
            propagate_viability_from_node(node, is_viable)
            evaluate_necessity(node, is_necessary, enabled_defenses)
            propagate_necessity_from_node(node, is_necessary)

    return is_viable, is_necessary


def prune_unviable_and_unnecessary_nodes(
        graph: AttackGraph,
        is_viable: dict[AttackGraphNode, bool],
        is_necessary: dict[AttackGraphNode, bool]
    ) -> None:
    """
    Arguments:
    graph         - the attack graph for which we wish to remove the
                    the nodes which are not viable or necessary.
    is_viable     - dict mapping nodes to their viability
    is_necessary  - dict mapping nodes to their necessity

    """
    logger.debug(
        'Prune unviable and unnecessary nodes from the attack graph.')

    nodes_to_remove = set()
    for node in graph.nodes.values():
        if node.type in ('or', 'and') and \
            (not is_viable[node] or not is_necessary[node]):
            nodes_to_remove.add(node)

    # Do the removal separatly so we don't remove
    # nodes from a set we are looping over
    for node in nodes_to_remove:
        logger.debug(
            'Remove %s node "%s"(%d) from attack graph.',
            'unviable' if is_necessary[node] else 'unnecessary',
            node.full_name,
            node.id
        )
        graph.remove_node(node)
