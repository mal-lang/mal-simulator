from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Set

import numpy as np

from maltoolbox.attackgraph import AttackGraphNode

from malsim.mal_simulator.simulator_state import MalSimulatorState
from malsim.mal_simulator.graph_utils import node_is_observable

from malsim.mal_simulator.false_alerts import (
    generate_false_negatives,
    generate_false_positives,
)

if TYPE_CHECKING:
    from malsim.mal_simulator.agent_state import AgentSettings

def defender_observed_nodes(
    agent_settings: AgentSettings,
    sim_state: MalSimulatorState,
    rng: np.random.Generator,
    defender_name: str,
    compromised_nodes: Set[AttackGraphNode],
) -> set[AttackGraphNode]:
    """Generate set of observed compromised nodes
    From set of compromised nodes, generate observed nodes for a defender
    in regards to observability, false negatives and false positives.
    """
    observable_steps = set(
        n
        for n in compromised_nodes
        if node_is_observable(agent_settings, sim_state.global_observability, n, defender_name)
    )
    false_negatives = generate_false_negatives(
        agent_settings, sim_state.global_false_negative_rates, rng, defender_name, compromised_nodes
    )
    false_positives = generate_false_positives(
        sim_state.global_false_positive_rates, agent_settings, sim_state.attack_graph, defender_name, rng
    )

    observed_nodes = (observable_steps - false_negatives) | false_positives
    return observed_nodes
