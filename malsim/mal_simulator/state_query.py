"""Query node properties affected by agent states"""

from __future__ import annotations

from maltoolbox.attackgraph import AttackGraphNode
from maltoolbox.attackgraph import AttackGraph

from malsim.mal_simulator.attacker_state import MalSimAttackerState, get_attacker_agents
from malsim.mal_simulator.defender_state import get_defender_agents
from malsim.mal_simulator.node_getters import full_name_or_node_to_node
from malsim.config.sim_settings import TTCMode
from malsim.mal_simulator.types import AgentStates


def node_is_enabled_defense(
    attack_graph: AttackGraph,
    agent_states: AgentStates,
    live_agents: set[str],
    node: AttackGraphNode | str,
) -> bool:
    """Get a nodes defense status"""
    node = full_name_or_node_to_node(attack_graph, node)
    return any(
        node in attacker_agent.performed_nodes
        for attacker_agent in get_defender_agents(agent_states, live_agents)
    )


def node_is_compromised(
    attack_graph: AttackGraph,
    agent_states: AgentStates,
    live_agents: set[str],
    node: AttackGraphNode | str,
) -> bool:
    """Return True if node is compromised by any attacker agent"""
    node = full_name_or_node_to_node(attack_graph, node)
    return any(
        node in attacker_agent.performed_nodes
        for attacker_agent in get_attacker_agents(agent_states, live_agents)
    )


def compromised_nodes(
    agent_states: AgentStates,
    live_agents: set[str],
) -> set[AttackGraphNode]:
    compromised: set[AttackGraphNode] = set()
    for attacker in get_attacker_agents(agent_states, live_agents):
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
    if node in attacker_state.ttc_value_overrides:
        return attacker_state.ttc_value_overrides[node]

    assert node in attacker_state.sim_state.graph_state.ttc_values, (
        f'Node {node.full_name} does not have a ttc value'
    )
    return attacker_state.sim_state.graph_state.ttc_values[node]
