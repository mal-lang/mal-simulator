"""Query node properties affected by agent states"""

from __future__ import annotations
from collections.abc import Set

from maltoolbox.attackgraph import AttackGraphNode
from maltoolbox.attackgraph import AttackGraph

from malsim.mal_simulator.attacker_state import MalSimAttackerState
from malsim.mal_simulator.agent_states import defender_states
from malsim.mal_simulator.node_getters import full_name_or_node_to_node
from malsim.config.sim_settings import TTCMode
from malsim.mal_simulator.agent_states import AgentStates, attacker_states


def node_is_enabled_defense(
    attack_graph: AttackGraph,
    agent_states: AgentStates,
    node: AttackGraphNode | str,
) -> bool:
    """Get a nodes defense status"""
    node = full_name_or_node_to_node(attack_graph, node)
    return any(
        node in attacker_agent.performed_nodes
        for attacker_agent in defender_states(agent_states).values()
    )


def node_is_compromised(
    attack_graph: AttackGraph,
    agent_states: AgentStates,
    node: AttackGraphNode | str,
) -> bool:
    """Return True if node is compromised by any attacker agent"""
    node = full_name_or_node_to_node(attack_graph, node)
    return any(
        node in attacker_agent.performed_nodes
        for attacker_agent in attacker_states(agent_states).values()
    )


def compromised_nodes(
    agent_states: AgentStates,
) -> Set[AttackGraphNode]:
    compromised: Set[AttackGraphNode] = set()
    for attacker in attacker_states(agent_states).values():
        compromised |= attacker.performed_nodes
    return compromised


def node_ttc_value(
    attacker_state: MalSimAttackerState,
    node: AttackGraphNode | str,
) -> float:
    """Return ttc value of node if it has been sampled"""
    node = full_name_or_node_to_node(attacker_state.sim_state.attack_graph, node)
    assert attacker_state.sim_state.settings.ttc_mode in (
        TTCMode.PRE_SAMPLE,
        TTCMode.EXPECTED_VALUE,
    ), 'TTC value only when TTCMode is PRE_SAMPLE or EXPECTED_VALUE'

    # If agent overrides the global TTC values
    # return that value instead of the global
    if attacker_state.ttc_values and node in attacker_state.ttc_values:
        return attacker_state.ttc_values[node]

    assert node in attacker_state.sim_state.graph_state.ttc_values, (
        f'Node {node.full_name} does not have a ttc value'
    )
    return attacker_state.sim_state.graph_state.ttc_values[node]
