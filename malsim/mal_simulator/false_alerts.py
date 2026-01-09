from typing import Optional
from malsim.mal_simulator.agent_state import AgentSettings


import numpy as np
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode


from collections.abc import Set

from malsim.scenario import DefenderSettings


def node_false_negative_rate(
    agent_settings: AgentSettings,
    node_false_negative_rates: dict[AttackGraphNode, float],
    node: AttackGraphNode,
    agent_name: Optional[str] = None,
) -> float:
    _agent_settings = agent_settings[agent_name] if agent_name else None
    if (
        isinstance(_agent_settings, DefenderSettings)
        and _agent_settings.false_negative_rates
    ):
        # FNR from agent settings
        return float(_agent_settings.false_negative_rates.value(node, 0.0))
    if node_false_negative_rates:
        # FNR from global settings
        return node_false_negative_rates.get(node, 0.0)
    return 0.0


def generate_false_negatives(
    agent_settings: AgentSettings,
    false_negative_rates: dict[AttackGraphNode, float],
    rng: np.random.Generator,
    agent_name: str,
    observed_nodes: Set[AttackGraphNode],
) -> set[AttackGraphNode]:
    """Return a set of false negative attack steps from observed nodes"""
    if false_negative_rates:
        return set(
            node
            for node in observed_nodes
            if rng.random()
            < node_false_negative_rate(
                agent_settings, false_negative_rates, node, agent_name
            )
        )
    else:
        return set()


def node_false_positive_rate(
    agent_settings: AgentSettings,
    node_false_positive_rates: dict[AttackGraphNode, float],
    node: AttackGraphNode,
    agent_name: Optional[str] = None,
) -> float:
    _agent_settings = agent_settings[agent_name] if agent_name else None
    if (
        isinstance(_agent_settings, DefenderSettings)
        and _agent_settings.false_positive_rates
    ):
        # FPR from agent settings
        return float(_agent_settings.false_positive_rates.value(node, 0.0))
    if node_false_positive_rates:
        # FPR from global settings
        return node_false_positive_rates.get(node, 0.0)
    return 0.0


def generate_false_positives(
    false_positive_rates: dict[AttackGraphNode, float],
    agent_settings: AgentSettings,
    attack_graph: AttackGraph,
    agent_name: str,
    rng: np.random.Generator,
) -> set[AttackGraphNode]:
    """Return a set of false positive attack steps from attack graph"""
    if false_positive_rates:
        return set(
            node
            for node in attack_graph.attack_steps
            if rng.random()
            < node_false_positive_rate(
                agent_settings, false_positive_rates, node, agent_name
            )
        )
    else:
        return set()
