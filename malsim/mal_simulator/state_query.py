"""Utilities to convert full names to nodes"""

from __future__ import annotations
from typing import TYPE_CHECKING

from maltoolbox.attackgraph import AttackGraphNode
from maltoolbox.attackgraph import AttackGraph

from malsim.mal_simulator.agent_state import get_attacker_agents, get_defender_agents
from malsim.mal_simulator.node_getters import full_name_or_node_to_node
from malsim.mal_simulator.settings import TTCMode
from malsim.mal_simulator.agent_state import MalSimAttackerState

if TYPE_CHECKING:
    from malsim.mal_simulator.agent_state import MalSimAgentState
    from malsim.mal_simulator.agent_state import AgentStates


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
    agent_state: MalSimAgentState,
    node: AttackGraphNode | str,
) -> float:
    """Return ttc value of node if it has been sampled"""
    node = full_name_or_node_to_node(agent_state.sim_state.attack_graph, node)
    assert agent_state.sim_state.settings.ttc_mode in (
        TTCMode.PRE_SAMPLE,
        TTCMode.EXPECTED_VALUE,
    ), 'TTC value only when TTCMode is PRE_SAMPLE or EXPECTED_VALUE'

    if agent_state:
        # If agent name is given and it overrides the global TTC values
        # Return that value instead of the global ttc value
        if not isinstance(agent_state, MalSimAttackerState):
            raise ValueError(
                f'Agent {agent_state.name} is not an attacker and has no TTC values'
            )
        if node in agent_state.ttc_value_overrides:
            return agent_state.ttc_value_overrides[node]

    assert node in agent_state.sim_state.graph_state.ttc_values, (
        f'Node {node.full_name} does not have a ttc value'
    )
    return agent_state.sim_state.graph_state.ttc_values[node]
