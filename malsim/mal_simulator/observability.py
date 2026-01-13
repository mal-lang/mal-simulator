from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from collections.abc import Set

import numpy as np

from maltoolbox.attackgraph import AttackGraphNode

from malsim.mal_simulator.simulator_state import MalSimulatorState

from malsim.mal_simulator.false_alerts import (
    generate_false_negatives,
    generate_false_positives,
)
from malsim.scenario.agent_settings import DefenderSettings

if TYPE_CHECKING:
    from malsim.scenario.agent_settings import AgentSettings


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
