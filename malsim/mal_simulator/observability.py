from __future__ import annotations
from collections.abc import Set
from typing import Optional

import numpy as np

from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.simulator_state import MalSimulatorState
from malsim.mal_simulator.false_alerts import (
    generate_false_negatives,
    generate_false_positives,
)


def node_is_observable(
    agent_observability_rule: NodePropertyRule[bool],
    node: AttackGraphNode,
) -> bool:
    return bool(agent_observability_rule.value(node, False))


def observed_nodes(
    observable_steps_rule: Optional[NodePropertyRule[bool]],
    false_positive_rates_rule: Optional[NodePropertyRule[float]],
    false_negative_rates_rule: Optional[NodePropertyRule[float]],
    sim_state: MalSimulatorState,
    rng: np.random.Generator,
    compromised_nodes: Set[AttackGraphNode],
) -> Set[AttackGraphNode]:
    """Generate set of observed compromised nodes
    From set of compromised nodes, generate observed nodes for an agent
    in regards to observability, false negatives and false positives.
    """
    observable_steps = (
        {n for n in compromised_nodes if node_is_observable(observable_steps_rule, n)}
        if observable_steps_rule
        else compromised_nodes
    )
    false_negatives = generate_false_negatives(
        false_negative_rates_rule,
        compromised_nodes,
        rng,
    )
    false_positives = generate_false_positives(
        false_positive_rates_rule,
        sim_state.attack_graph,
        rng,
    )

    return (observable_steps - false_negatives) | false_positives
