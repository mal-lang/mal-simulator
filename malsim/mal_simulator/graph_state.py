"""Dataclass and function to store and create graph state used in the simulator"""

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
from numpy.random import Generator
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from malsim.mal_simulator.ttc_utils import (
    attack_step_ttc_values,
    get_pre_enabled_defenses,
    get_impossible_attack_steps,
)
from malsim.mal_simulator.graph_processing import (
    calculate_viability,
    calculate_necessity,
)

if TYPE_CHECKING:
    from malsim.mal_simulator.settings import MalSimulatorSettings


@dataclass
class GraphState:
    """Dataclass containing simulator specific graph state"""
    ttc_values: dict[AttackGraphNode, float]
    pre_enabled_defenses: set[AttackGraphNode]
    impossible_attack_steps: set[AttackGraphNode]
    viability_per_node: dict[AttackGraphNode, bool]
    necessity_per_node: dict[AttackGraphNode, bool]


def compute_initial_graph_state(
    graph: AttackGraph, settings: MalSimulatorSettings, rng: Generator
) -> GraphState:
    """Compute attack graph initial state based on probabilities and settings

    Compute ttc values, enabled defenses, impossible attack steps,
    viability and necessity for an attack graph based on given settings.
    """

    # TTC (Time to compromise) for each attack step
    # will only be set if TTCMode PRE_SAMLE/EXPECTED_VALUE is used
    ttc_values = attack_step_ttc_values(graph.attack_steps, settings.ttc_mode, rng)
    # These steps will be enabled from the start of the simulation
    # depending on if bernoullis are sampled or not
    enabled_defenses = get_pre_enabled_defenses(
        graph.defense_steps, settings.run_defense_step_bernoullis, rng
    )
    impossible_attack_steps = set()
    if settings.run_attack_step_bernoullis:
        # These steps will not be traversable
        impossible_attack_steps = get_impossible_attack_steps(graph.attack_steps, rng)

    viability_per_node = calculate_viability(
        graph, enabled_defenses, impossible_attack_steps
    )
    necessity_per_node = calculate_necessity(graph, enabled_defenses)

    return GraphState(
        ttc_values=ttc_values,
        pre_enabled_defenses=enabled_defenses,
        impossible_attack_steps=impossible_attack_steps,
        viability_per_node=viability_per_node,
        necessity_per_node=necessity_per_node,
    )
