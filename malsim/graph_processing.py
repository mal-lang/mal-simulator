"""
MAL Simulator Attack Graph Processing Submodule

This submodule is meant to process the attack graph before running the
simulation.
Currently it adds the following information to nodes:
- Viability = Determine if a node can be traversed under any circumstances or
  if the model structure, defenses or ttcs makes it unviable.
- Necessity = Determine if a node is necessary for the attacker or if the
  model structure means it is not needed (it behaves as if it were already
  compromised) to compromise children attack steps.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

logger = logging.getLogger(__name__)


def _propagate_necessity_from_node(
        node: AttackGraphNode, necessity_per_node: dict[AttackGraphNode, bool]
    ) -> set[AttackGraphNode]:
    """
    Update necessity of children of node given as parameter. Propagate
    recursively via children as long as changes occur. Return all nodes which
    have changed necessity.

    Arguments:
    node                - the attack graph node from which to propagate
                          the necessity status
    necessity_per_node  - each nodes necessity status

    Returns:
    changed_nodes   - set of nodes that have changed necessity
    """
    changed_nodes = set()
    for child in node.children:
        is_necessary = evaluate_necessity(child, necessity_per_node, set())
        if is_necessary != necessity_per_node[child]:
            necessity_per_node[child] = is_necessary
            changed_nodes |= (
                {child}
                | _propagate_necessity_from_node(child, necessity_per_node)
            )
    return changed_nodes


def evaluate_necessity(
        node: AttackGraphNode,
        necessity_per_node: dict[AttackGraphNode, bool],
        enabled_defenses: set[AttackGraphNode]
    ) -> bool:
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
            return not node.existence_status
        case 'notExist':
            assert isinstance(node.existence_status, bool), \
                f'Existence status not defined for {node.full_name}.'
            return bool(node.existence_status)
        case 'defense':
            return node in enabled_defenses
        case 'or':
            return all(
                necessity_per_node[parent] for parent in node.parents
            ) or not node.parents
        case 'and':
            return any(
                necessity_per_node[parent] for parent in node.parents
            ) or not node.parents
        case _:
            msg = ('Evaluate necessity was provided node "%s"(%d) which '
                   'is of unknown type "%s"')
            logger.error(msg, node.full_name, node.id, node.type)
            raise ValueError(msg % (node.full_name, node.id, node.type))


def calculate_necessity(
    graph: AttackGraph,
    enabled_defenses: set[AttackGraphNode],
) -> dict[AttackGraphNode, bool]:
    """Calculate necessity for an attack graph

    arguments:
    graph            - graph with nodes to calculate necessity for
    enabled_defenses - defenses that are enabled, affects necessity

    Returns necessity per node in a dict
    """

    necessity_per_node = {n: True for n in graph.nodes.values()}
    for node in graph.nodes.values():
        if node.type in ['exist', 'notExist', 'defense']:
            necessity_per_node[node] = evaluate_necessity(
                node, necessity_per_node, enabled_defenses
            )
            if not necessity_per_node[node]:
                _propagate_necessity_from_node(node, necessity_per_node)
    return necessity_per_node


def _propagate_viability_from_node(
        node: AttackGraphNode,
        viability_per_node: dict[AttackGraphNode, bool],
        impossible_attack_steps: set[AttackGraphNode]
    ) -> set[AttackGraphNode]:
    """
    Update viability of children of node given as parameter. Propagate
    recursively via children as long as changes occur. Return all nodes which
    have changed viability.

    Arguments:
    node                    - the node from which to propagate viability
    viability_per_node      - dict of node viabilities
    ttc_values              - dict with ttc values of nodes

    Returns:
    changed_nodes   - set of nodes that have changed viability as a result
    """
    changed_nodes = set()
    for child in node.children:
        is_viable = evaluate_viability(
            child, viability_per_node, set(), impossible_attack_steps
        )
        if is_viable != viability_per_node[child]:
            viability_per_node[child] = is_viable
            changed_nodes |= (
                {child} | _propagate_viability_from_node(
                    child, viability_per_node, impossible_attack_steps
                )
            )
    return changed_nodes


def evaluate_viability(
        node: AttackGraphNode,
        viability_per_node: dict[AttackGraphNode, bool],
        enabled_defenses: set[AttackGraphNode],
        impossible_attack_steps: set[AttackGraphNode]
    ) -> bool:
    """
    Arguments:
    node                - the node to evaluate viability for
    viable_nodes        - set of all viable nodes
    enabled_defenses    - set of all enabled defenses
    ttc_values          - a dictionary containing the ttc values of each node
    """
    if node in impossible_attack_steps:
        # Impossible step is not viable
        return False

    match (node.type):
        case 'exist':
            assert isinstance(node.existence_status, bool), \
                f'Existence status not defined for {node.full_name}.'
            return node.existence_status
        case 'notExist':
            assert isinstance(node.existence_status, bool), \
                f'Existence status not defined for {node.full_name}.'
            return not node.existence_status
        case 'defense':
            return node not in enabled_defenses
        case 'or':
            return any(
                viability_per_node[parent] for parent in node.parents
            ) or not node.parents
        case 'and':
            return all(
                viability_per_node[parent] for parent in node.parents
            ) or not node.parents
        case _:
            msg = ('Evaluate viability was provided node "%s"(%d) which '
                   'is of unknown type "%s"')
            logger.error(msg, node.full_name, node.id, node.type)
            raise ValueError(msg % (node.full_name, node.id, node.type))


def calculate_viability(
    graph: AttackGraph,
    enabled_defenses: set[AttackGraphNode],
    impossible_attack_steps: set[AttackGraphNode]
) -> dict[AttackGraphNode, bool]:
    """Calculate viability for an attack graph
    
    graph            - graph with nodes to calculate viability for
    enabled_defenses    - defenses that are enabled, affects viability
    ttc_values          - ttc values per node, affects viability

    Returns viability per node in a dict
    """
    viability_per_node = {n: True for n in graph.nodes.values()}
    for node in graph.nodes.values():
        viability_per_node[node] = evaluate_viability(
            node, viability_per_node,
            enabled_defenses,
            impossible_attack_steps
        )
        if not viability_per_node[node]:
            _propagate_viability_from_node(
                node, viability_per_node, impossible_attack_steps
            )
    return viability_per_node


def make_node_unviable(
        node: AttackGraphNode,
        viability_per_node: dict[AttackGraphNode, bool],
        impossible_attacksteps: set[AttackGraphNode]
    ) -> tuple[dict[AttackGraphNode, bool], set[AttackGraphNode]]:
    """Make a node unviable

    Return the new viability dict and nodes made unviable as
    a result of making `node` unviable
    """
    viability_per_node[node] = False
    nodes_made_unviable = _propagate_viability_from_node(
        node, viability_per_node, impossible_attacksteps
    )
    return viability_per_node, nodes_made_unviable


def prune_unviable_and_unnecessary_nodes(
        graph: AttackGraph,
        viability_per_node: dict[AttackGraphNode, bool],
        necessity_per_node: dict[AttackGraphNode, bool]
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
        if node.type in ('or', 'and') and (
            not viability_per_node[node] or not necessity_per_node[node]
        ):
            nodes_to_remove.add(node)

    # Do the removal separatly so we don't remove
    # nodes from a set we are looping over
    for node in nodes_to_remove:
        logger.debug(
            'Remove %s node "%s"(%d) from attack graph.',
            'unviable' if necessity_per_node[node] else 'unnecessary',
            node.full_name, node.id
        )
        graph.remove_node(node)
