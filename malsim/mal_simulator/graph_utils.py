"""Utilities to query nodes in an attack graph"""

from __future__ import annotations
from collections.abc import Set
from typing import Optional, TYPE_CHECKING

from maltoolbox.attackgraph import AttackGraphNode
from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.node_getters import full_name_or_node_to_node
from malsim.mal_simulator.simulator_state import MalSimulatorState
from malsim.mal_simulator.agent_state import MalSimAttackerState, MalSimDefenderState

if TYPE_CHECKING:
    from malsim.config.agent_settings import AgentSettings


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

    if not node_is_viable(sim_state, node):
        return False

    if node.type in ('defense', 'exist', 'notExist'):
        # Only attack steps have traversability
        return False

    if not (node.parents & performed_nodes):
        # If no parent is reached, the node can not be traversable
        return False

    if node.type == 'or':
        traversable = any(parent in performed_nodes for parent in node.parents)
    elif node.type == 'and':
        traversable = all(
            parent in performed_nodes or not node_is_necessary(sim_state, parent)
            for parent in node.parents
        )
    else:
        raise TypeError(
            f'Node "{node.full_name}"({node.id})has an unknown type "{node.type}".'
        )
    return traversable


def node_is_actionable(
    agent_actionability_rule: Optional[NodePropertyRule],
    global_actionability: dict[AttackGraphNode, bool],
    node: AttackGraphNode,
) -> bool:
    if agent_actionability_rule:
        # Actionability from agent settings
        return bool(agent_actionability_rule.value(node, False))
    if global_actionability:
        # Actionability from global settings
        return global_actionability.get(node, False)
    return True


def node_reward(
    agent: MalSimDefenderState | MalSimAttackerState,
    node: AttackGraphNode,
) -> float:
    if agent.reward_rule:
        # Node reward from agent settings
        return float(agent.reward_rule.value(node, 0.0))
    if agent.sim_state.global_rewards:
        # Node reward from global settings
        return agent.sim_state.global_rewards.get(node, 0.0)
    return 0.0
