from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from collections.abc import Set

import numpy as np

from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.agent_settings import DefenderSettings
from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.simulator_state import MalSimulatorState

from malsim.mal_simulator.false_alerts import (
    generate_false_negatives,
    generate_false_positives,
)

if TYPE_CHECKING:
    from malsim.config.agent_settings import AgentSettings


def node_is_observable(
    agent_observability_rule: Optional[NodePropertyRule],
    global_observability: dict[AttackGraphNode, bool],
    node: AttackGraphNode,
) -> bool:
    if agent_observability_rule:
        # Observability from agent settings
        return bool(agent_observability_rule.value(node, False))
    if global_observability:
        # Observability from global settings
        return global_observability.get(node, False)
    return True


def defender_observed_nodes(
    observability_rule: Optional[NodePropertyRule],
    false_positive_rates_rule: Optional[NodePropertyRule],
    false_negative_rates_rule: Optional[NodePropertyRule],
    sim_state: MalSimulatorState,
    rng: np.random.Generator,
    compromised_nodes: Set[AttackGraphNode],
) -> set[AttackGraphNode]:
    """Generate set of observed compromised nodes
    From set of compromised nodes, generate observed nodes for a defender
    in regards to observability, false negatives and false positives.
    """
    observable_steps = set(
        n
        for n in compromised_nodes
        if node_is_observable(observability_rule, sim_state.global_observability, n)
    )
    false_negatives = generate_false_negatives(
        false_negative_rates_rule,
        sim_state.global_false_negative_rates,
        compromised_nodes,
        rng,
    )
    false_positives = generate_false_positives(
        false_positive_rates_rule,
        sim_state.global_false_positive_rates,
        sim_state.attack_graph,
        rng,
    )

    observed_nodes = (observable_steps - false_negatives) | false_positives
    return observed_nodes
