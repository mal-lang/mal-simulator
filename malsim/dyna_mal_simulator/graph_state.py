"""Dataclass and function to store and create graph state used in the simulator"""

from __future__ import annotations
from collections.abc import Set

from numpy.random import Generator
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from malsim.mal_simulator.ttc_utils import (
    attack_step_ttc_values,
    get_pre_enabled_defenses,
    get_impossible_attack_steps,
)
from malsim.mal_simulator.graph_processing import calculate_necessity

from malsim.mal_simulator.graph_state import GraphState

from malsim.config.sim_settings import MalSimulatorSettings


def update_graph_state(
    graph_state: GraphState,
    sim_settings: MalSimulatorSettings,
    attack_graph: AttackGraph,
    new_nodes: set[AttackGraphNode],
    rng: Generator,
) -> tuple[GraphState, Set[AttackGraphNode]]:
    """Update graph state based on new nodes added to the attack graph."""

    new_attack_steps = {node for node in new_nodes if node.type in ('and', 'or')}
    new_defense_steps = [node for node in new_nodes if node.type == 'defense']

    # TTC (Time to compromise) for each attack step
    # will only be set if TTCMode PRE_SAMLE/EXPECTED_VALUE is used
    new_attack_step_ttc_values = attack_step_ttc_values(
        new_attack_steps, rng=rng, ttc_mode=sim_settings.ttc_mode
    )
    # These steps will be enabled from the start of the simulation
    # depending on if bernoullis are sampled or not
    new_steps_enabled_defenses = get_pre_enabled_defenses(
        new_defense_steps, sim_settings.run_defense_step_bernoullis, rng
    )
    enabled_defenses = graph_state.pre_enabled_defenses | new_steps_enabled_defenses
    if sim_settings.run_attack_step_bernoullis:
        # These steps will not be traversable
        new_impossible_attack_steps = get_impossible_attack_steps(new_attack_steps, rng)
    else:
        new_impossible_attack_steps = set()

    necessity_per_node = calculate_necessity(attack_graph, enabled_defenses)

    return GraphState(
        ttc_values={**graph_state.ttc_values, **new_attack_step_ttc_values},
        pre_enabled_defenses=graph_state.pre_enabled_defenses
        | new_steps_enabled_defenses,
        impossible_attack_steps=graph_state.impossible_attack_steps
        | new_impossible_attack_steps,
        necessity_per_node=necessity_per_node,
    ), new_steps_enabled_defenses
