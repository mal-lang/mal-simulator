"""Utilities to convert full names to nodes"""

from __future__ import annotations
from collections.abc import Set
from typing import Any, Iterable, Optional
from maltoolbox.attackgraph import AttackGraphNode

from maltoolbox.attackgraph import AttackGraph

from malsim.mal_simulator.agent_state import AgentSettings
from malsim.mal_simulator.agent_state import MalSimAttackerState
from malsim.mal_simulator.agent_state import get_attacker_agents
from malsim.mal_simulator.agent_state import get_defender_agents
from malsim.mal_simulator.graph_state import GraphState
from malsim.mal_simulator.settings import TTCMode
from malsim.mal_simulator.agent_state import AgentStates
from malsim.mal_simulator.simulator_state import MalSimulatorState
from malsim.scenario import DefenderSettings

def get_node(
    attack_graph: AttackGraph,
    full_name: Optional[str] = None,
    node_id: Optional[int] = None,
) -> AttackGraphNode:
    """Get node from attack graph by either full name or id"""

    if full_name and not node_id:
        node = attack_graph.get_node_by_full_name(full_name)
    elif node_id and not full_name:
        node = attack_graph.nodes[node_id]
    else:
        raise ValueError("Provide either full_name or node_id to 'get_node'")

    if node is None:
        raise LookupError(f'Could not find node {full_name or node_id}')
    return node


def full_name_or_node_to_node(
    attack_graph: AttackGraph, node_or_full_name: str | AttackGraphNode
) -> AttackGraphNode:
    """If `node_or_full_name` is a full_mame, return corresponding AttackGraphNode"""
    if isinstance(node_or_full_name, str):
        return get_node(attack_graph, node_or_full_name)
    return node_or_full_name


def full_names_or_nodes_to_nodes(
    attack_graph: AttackGraph,
    nodes_or_full_names: Iterable[str] | Iterable[AttackGraphNode],
) -> Iterable[AttackGraphNode]:
    """Generator converting nodes full_name to AttackGraphNode objects"""
    for n in nodes_or_full_names:
        yield get_node(attack_graph, n) if isinstance(n, str) else n


def full_name_dict_to_node_dict(
    attack_graph: AttackGraph, mapping: dict[str, Any] | dict[AttackGraphNode, Any]
) -> dict[AttackGraphNode, Any]:
    """Convert dict so it maps from AttackGraphNode instead of from full_name"""
    return {full_name_or_node_to_node(attack_graph, k): v for k, v in mapping.items()}


def node_is_viable(
    sim_state: MalSimulatorState, node: AttackGraphNode | str
) -> bool:
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
            parent in performed_nodes
            or not node_is_necessary(sim_state, parent)
            for parent in node.parents
        )
    else:
        raise TypeError(
            f'Node "{node.full_name}"({node.id})has an unknown type "{node.type}".'
        )
    return traversable


def node_is_actionable(
    agent_settings: AgentSettings,
    node_actionabilities: dict[AttackGraphNode, bool],
    node: AttackGraphNode,
    agent_name: Optional[str] = None,
) -> bool:
    _agent_settings = agent_settings[agent_name] if agent_name else None
    if _agent_settings and _agent_settings.actionable_steps:
        # Actionability from agent settings
        return bool(_agent_settings.actionable_steps.value(node, False))
    if node_actionabilities:
        # Actionability from global settings
        return node_actionabilities.get(node, False)
    return True


def node_reward(
    agent_settings: AgentSettings,
    rewards: dict[AttackGraphNode, float],
    node: AttackGraphNode,
    agent_name: Optional[str] = None,
) -> float:
    _agent_settings = agent_settings[agent_name] if agent_name else None
    if _agent_settings and _agent_settings.rewards:
        # Node reward from agent settings
        return float(_agent_settings.rewards.value(node, 0.0))
    if rewards:
        # Node reward from global settings
        return rewards.get(node, 0.0)
    return 0.0


def node_is_observable(
    agent_settings: AgentSettings,
    node_observabilities: dict[AttackGraphNode, bool],
    node: AttackGraphNode,
    agent_name: Optional[str] = None,
) -> bool:
    _agent_settings = agent_settings[agent_name] if agent_name else None
    if (
        isinstance(_agent_settings, DefenderSettings)
        and _agent_settings.observable_steps
    ):
        # Observability from agent settings
        return bool(_agent_settings.observable_steps.value(node, False))
    if node_observabilities:
        # Observability from global settings
        return node_observabilities.get(node, False)
    return True


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
    agent_states: AgentStates,
    sim_state: MalSimulatorState,
    ttc_mode: TTCMode,
    node: AttackGraphNode | str,
    agent_name: Optional[str] = None,
) -> float:
    """Return ttc value of node if it has been sampled"""
    node = full_name_or_node_to_node(sim_state.attack_graph, node)
    assert ttc_mode in (
        TTCMode.PRE_SAMPLE,
        TTCMode.EXPECTED_VALUE,
    ), 'TTC value only when TTCMode is PRE_SAMPLE or EXPECTED_VALUE'

    if agent_name:
        # If agent name is given and it overrides the global TTC values
        # Return that value instead of the global ttc value
        agent_state = agent_states[agent_name]
        if not isinstance(agent_state, MalSimAttackerState):
            raise ValueError(
                f'Agent {agent_name} is not an attacker and has no TTC values'
            )
        if node in agent_state.ttc_value_overrides:
            return agent_state.ttc_value_overrides[node]

    assert node in sim_state.graph_state.ttc_values, (
        f'Node {node.full_name} does not have a ttc value'
    )
    return sim_state.graph_state.ttc_values[node]
