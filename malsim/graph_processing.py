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
        node: AttackGraphNode, viable_nodes: set[AttackGraphNode]
    ) -> set[AttackGraphNode]:
    """
    Update viability of children of node given as parameter. Propagate
    recursively via children as long as changes occur. Return all nodes which
    have changed viability.

    Arguments:
    node            - the attack graph node from which to propagate the
                      viability status
    viable_nodes    - set of all viable nodes

    Returns:
    changed_nodes   - set of nodes that have changed viability
    """
    logger.debug(
        'Propagate viability from "%s"(%d) with viability status %s.',
        node.full_name, node.id, node in viable_nodes
    )
    changed_nodes = set()
    for child in node.children:
        was_viable = child in viable_nodes
        is_viable = None

        if child.type == 'or':
            is_viable = any(
                parent in viable_nodes for parent in child.parents)
        elif child.type == 'and':
            is_viable = all(
                parent in viable_nodes for parent in child.parents)
        else:
            raise TypeError('Child of node must be of type "or"/"and"')

        if is_viable != was_viable:

            if is_viable:
                viable_nodes.add(child)
            else:
                viable_nodes.remove(child)

            changed_nodes |= (
                {child} | propagate_viability_from_node(child, viable_nodes)
            )

    return changed_nodes

def propagate_necessity_from_node(
        node: AttackGraphNode, necessary_nodes: set[AttackGraphNode]
    ) -> set[AttackGraphNode]:
    """
    Update necessity of children of node givein as parameter. Propagate
    recursively via children as long as changes occur. Return all nodes which
    have changed necessity.

    Arguments:
    node            - the attack graph node from which to propagate the
                      necessity status
    necessary_nodes - set of all necessary nodes

    Returns:
    changed_nodes   - set of nodes that have changed necessity
    """
    logger.debug(
        'Propagate necessity from "%s"(%d) with necessity status %s.',
        node.full_name, node.id, node in necessary_nodes
    )
    changed_nodes = set()
    for child in node.children:
        was_necessary = child in necessary_nodes
        is_necessary = None
        if child.type == 'or':
            is_necessary = all(
                parent in necessary_nodes for parent in child.parents)

        elif child.type == 'and':
            is_necessary = any(
                parent in necessary_nodes for parent in child.parents)

        if is_necessary != was_necessary:

            if is_necessary:
                necessary_nodes.add(child)
            else:
                necessary_nodes.remove(child)

            changed_nodes |= (
                {child}
                | propagate_necessity_from_node(child, necessary_nodes)
            )

    return changed_nodes


def evaluate_viability(
        node: AttackGraphNode,
        viable_nodes: set[AttackGraphNode],
        enabled_defenses: set[AttackGraphNode]
    ) -> None:
    """
    Arguments:
    node                - the node to evaluate viability for
    viable_nodes        - set of all viable nodes
    enabled_defenses    - set of all enabled defenses
    """
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
            node_is_viable = any(
                parent in viable_nodes for parent in node.parents
            )
        case 'and':
            node_is_viable = all(
                parent in viable_nodes for parent in node.parents
            )
        case _:
            msg = ('Evaluate viability was provided node "%s"(%d) which '
                   'is of unknown type "%s"')
            logger.error(msg, node.full_name, node.id, node.type)
            raise ValueError(msg % (node.full_name, node.id, node.type))

    if node_is_viable:
        viable_nodes.add(node)
    else:
        viable_nodes.remove(node)


def evaluate_necessity(
        node: AttackGraphNode,
        necessary_nodes: set[AttackGraphNode],
        enabled_defenses: set[AttackGraphNode]
    ) -> None:
    """
    Arguments:
    node             - the node to evaluate necessity for.
    necessary_nodes  - set of all necessary nodes
    enabled_defenses - set of all enabled defenses
    """

    match (node.type):
        case 'exist':
            assert isinstance(node.existence_status, bool), \
                f'Existence status not defined for {node.full_name}.'
            node_is_necessary = not node.existence_status
        case 'notExist':
            assert isinstance(node.existence_status, bool), \
                f'Existence status not defined for {node.full_name}.'
            node_is_necessary = bool(node.existence_status)
        case 'defense':
            node_is_necessary = node in enabled_defenses
        case 'or':
            node_is_necessary = all(
                parent in necessary_nodes for parent in node.parents
            )
        case 'and':
            node_is_necessary = any(
                parent in necessary_nodes for parent in node.parents
            )
        case _:
            msg = ('Evaluate necessity was provided node "%s"(%d) which '
                   'is of unknown type "%s"')
            logger.error(msg, node.full_name, node.id, node.type)
            raise ValueError(msg % (node.full_name, node.id, node.type))

    if node_is_necessary:
        necessary_nodes.add(node)
    else:
        necessary_nodes.remove(node)


def calculate_viability_and_necessity(
        graph: AttackGraph, enabled_defenses: set[AttackGraphNode]
    ) -> tuple[
            set[AttackGraphNode],
            set[AttackGraphNode]
        ]:
    """
    Arguments:
    graph             - the attack graph for which we wish to determine the
                        viability and necessity statuses for the nodes.
    enabled_defenses  - set of all enabled defenses

    Return:
    tuple with
        - set of viable nodes
        - set of necessary nodes
    """
    viable_nodes = set(n for n in graph.nodes.values())
    necessary_nodes = set(n for n in graph.nodes.values())

    for node in graph.nodes.values():
        if node.type in ['exist', 'notExist', 'defense']:
            evaluate_viability(node, viable_nodes, enabled_defenses)
            propagate_viability_from_node(node, viable_nodes)
            evaluate_necessity(node, necessary_nodes, enabled_defenses)
            propagate_necessity_from_node(node, necessary_nodes)

    return viable_nodes, necessary_nodes


def prune_unviable_and_unnecessary_nodes(
        graph: AttackGraph,
        viable_nodes: set[AttackGraphNode],
        necessary_nodes: set[AttackGraphNode]
    ) -> None:
    """
    Arguments:
    graph            - the attack graph for which we wish to remove the
                       the nodes which are not viable or necessary.
    viable_nodes     - set of all viable nodes
    necessary_nodes  - set of all necessary nodes

    """
    logger.debug(
        'Prune unviable and unnecessary nodes from the attack graph.')

    nodes_to_remove = set()
    for node in graph.nodes.values():
        if node.type in ('or', 'and') and \
            (node not in viable_nodes or node not in necessary_nodes):
            nodes_to_remove.add(node)

    # Do the removal separatly so we don't remove
    # nodes from a set we are looping over
    for node in nodes_to_remove:
        logger.debug(
            'Remove %s node "%s"(%d) from attack graph.',
            'unviable' if node in necessary_nodes else 'unnecessary',
            node.full_name,
            node.id
        )
        graph.remove_node(node)
