"""Utilities to query nodes in an attack graph"""

from __future__ import annotations
from collections.abc import Set

from maltoolbox.attackgraph import AttackGraphNode
from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.node_getters import full_name_or_node_to_node
from malsim.mal_simulator.simulator_state import MalSimulatorState


def node_is_viable(sim_state: MalSimulatorState, node: AttackGraphNode | str) -> bool:
    """Get viability of a node"""
    node = full_name_or_node_to_node(sim_state.attack_graph, node)
    return sim_state.graph_state.viability_per_node[node]


def node_is_necessary(
    sim_state: MalSimulatorState, node: AttackGraphNode | str
) -> bool:
    """Get necessity of a node"""
    node = full_name_or_node_to_node(sim_state.attack_graph, node)
    return sim_state.graph_state.necessity_per_node[node]

def is_attack_step(node: AttackGraphNode) -> bool:
    # Only attack steps have traversability
    return node.type not in ('defense', 'exist', 'notExist')

def node_is_traversable(
    sim_state: MalSimulatorState,
    performed_nodes: Set[AttackGraphNode],
    node: AttackGraphNode,
) -> bool:
    """
    Return True or False depending if the node specified is traversable
    for given the current attacker agent state.

    A node is traversable if it is viable and:
    - if it is of type 'or' and any of its parents have been compromised
    - if it is of type 'and' and all of its necessary parents have been
        compromised

    Arguments:
    performed_nodes - the nodes we assume are compromised in this evaluation
    node            - the node we wish to evalute traversability for
    """

    def parents_reached(node: AttackGraphNode) -> bool:
        # If no parent is reached, the node can not be traversable
        return (node.parents & performed_nodes) != set()

    def or_traversable(node: AttackGraphNode) -> bool:
        return any(parent in performed_nodes for parent in node.parents)

    def and_traversable(node: AttackGraphNode) -> bool:
        return all(
            parent in performed_nodes
            for parent in node.parents
            if node_is_necessary(sim_state, parent)
        )

    def is_and_or_traversable(node: AttackGraphNode) -> bool:
        if node.type == 'or':
            return or_traversable(node)
        elif node.type == 'and':
            return and_traversable(node)
        else:
            raise TypeError(
                f'Node "{node.full_name}"({node.id})has an unknown type "{node.type}".'
            )

    return (
        is_attack_step(node)
        and node_is_viable(sim_state, node)
        and parents_reached(node)
        and is_and_or_traversable(node)
    )


def node_is_actionable(
    agent_actionability: NodePropertyRule[bool] | None,
    node: AttackGraphNode,
) -> bool:
    if agent_actionability:
        return agent_actionability.value(node, False)
    return True


def node_reward(
    node: AttackGraphNode,
    reward_rule: NodePropertyRule[float] | None = None,
) -> float:
    if reward_rule:
        # Node reward from agent settings
        return float(reward_rule.value(node, 0.0))
    return 0.0
