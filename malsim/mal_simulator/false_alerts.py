from malsim.mal_simulator.node import node_is_observable
from malsim.mal_simulator.sim_data import SimData
from maltoolbox.attackgraph import AttackGraphNode
import numpy as np
from malsim.scenario import DefenderSettings
from maltoolbox.attackgraph import AttackGraph


def node_false_positive_rate(
    sim_data: SimData, node: AttackGraphNode, agent_settings: DefenderSettings
) -> float:
    if agent_settings.false_positive_rates:
        # FPR from agent settings
        return float(agent_settings.false_positive_rates.value(node, 0.0))
    if sim_data.false_positive_rates:
        # FPR from global settings
        return sim_data.false_positive_rates.get(node, 0.0)
    return 0.0


def node_false_negative_rate(
    sim_data: SimData, node: AttackGraphNode, agent_settings: DefenderSettings
) -> float:
    if agent_settings.false_negative_rates:
        # FNR from agent settings
        return float(agent_settings.false_negative_rates.value(node, 0.0))
    if sim_data.false_negative_rates:
        # FNR from global settings
        return sim_data.false_negative_rates.get(node, 0.0)
    return 0.0


def _generate_false_negatives(
    sim_data: SimData,
    agent_settings: DefenderSettings,
    observed_nodes: set[AttackGraphNode],
    rng: np.random.Generator,
) -> set[AttackGraphNode]:
    """Return a set of false negative attack steps from observed nodes"""
    return set(
        node
        for node in observed_nodes
        if rng.random() < node_false_negative_rate(sim_data, node, agent_settings)
    )


def _generate_false_positives(
    sim_data: SimData,
    attack_graph: AttackGraph,
    agent_settings: DefenderSettings,
    rng: np.random.Generator,
) -> set[AttackGraphNode]:
    """Return a set of false positive attack steps from attack graph"""
    return set(
        node
        for node in attack_graph.attack_steps
        if rng.random() < node_false_positive_rate(sim_data, node, agent_settings)
    )


def defender_observed_nodes(
    sim_data: SimData,
    agent_settings: DefenderSettings,
    compromised_nodes: set[AttackGraphNode],
    attack_graph: AttackGraph,
    rng: np.random.Generator,
) -> set[AttackGraphNode]:
    """Generate set of observed compromised nodes
    From set of compromised nodes, generate observed nodes for a defender
    in regards to observability, false negatives and false positives.
    """
    observable_steps = set(
        n for n in compromised_nodes if node_is_observable(sim_data, n, agent_settings)
    )
    false_negatives = _generate_false_negatives(
        sim_data, agent_settings, observable_steps, rng
    )
    false_positives = _generate_false_positives(
        sim_data, attack_graph, agent_settings, rng
    )

    observed_nodes = (observable_steps - false_negatives) | false_positives
    return observed_nodes
