from typing import Optional
from malsim.mal_simulator.sim_data import SimData
from malsim.mal_simulator.graph_state import GraphState
from maltoolbox.attackgraph import AttackGraphNode

from malsim.scenario import AttackerSettings, DefenderSettings


def node_is_actionable(
    sim_data: SimData,
    node: AttackGraphNode,
    agent_settings: Optional[AttackerSettings | DefenderSettings] = None,
) -> bool:
    if agent_settings and agent_settings.actionable_steps:
        # Actionability from agent settings
        return bool(agent_settings.actionable_steps.value(node, False))
    if sim_data.node_actionabilities:
        # Actionability from global settings
        return sim_data.node_actionabilities.get(node, False)
    return True


def node_reward(
    sim_data: SimData,
    node: AttackGraphNode,
    agent_settings: Optional[AttackerSettings | DefenderSettings] = None,
) -> float:
    if agent_settings and agent_settings.rewards:
        # Node reward from agent settings
        return float(agent_settings.rewards.value(node, 0.0))
    if sim_data.rewards:
        # Node reward from global settings
        return sim_data.rewards.get(node, 0.0)
    return 0.0


def node_is_observable(
    sim_data: SimData,
    node: AttackGraphNode,
    agent_settings: Optional[AttackerSettings | DefenderSettings] = None,
) -> bool:
    if isinstance(agent_settings, DefenderSettings) and agent_settings.observable_steps:
        # Observability from agent settings
        return bool(agent_settings.observable_steps.value(node, False))
    if sim_data.node_observabilities:
        # Observability from global settings
        return sim_data.node_observabilities.get(node, False)
    return True


def node_is_viable(graph_state: GraphState, node: AttackGraphNode) -> bool:
    """Get viability of a node"""
    return graph_state.viability_per_node[node]


def node_is_necessary(graph_state: GraphState, node: AttackGraphNode) -> bool:
    """Get necessity of a node"""
    return graph_state.necessity_per_node[node]


def node_is_traversable(
    graph_state: GraphState,
    performed_nodes: frozenset[AttackGraphNode],
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

    if not node_is_viable(graph_state, node):
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
            parent in performed_nodes or not node_is_necessary(graph_state, parent)
            for parent in node.parents
        )
    else:
        raise TypeError(
            f'Node "{node.full_name}"({node.id})has an unknown type "{node.type}".'
        )
    return traversable
