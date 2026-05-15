"""Creation/manipulation of defender state"""

from collections.abc import Set

import numpy as np
from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.agent_settings import DefenderSettings
from malsim.mal_simulator.defender_state import DefenderState
from malsim.mal_simulator.defense_surface import get_defense_surface
from malsim.mal_simulator.observability import observed_nodes
from malsim.mal_simulator.simulator_state import MalSimulatorState


def create_defender_state(
    sim_state: MalSimulatorState,
    name: str,
    rng: np.random.Generator,
    defender_settings: DefenderSettings,
    new_compromised_nodes: Set[AttackGraphNode] = frozenset(),
    new_enabled_defenses: Set[AttackGraphNode] = frozenset(),
    previous_state: DefenderState | None = None,
) -> DefenderState:
    """
    Update a previous defender state based on what steps
    were enabled/compromised during last step
    """
    previous_enabled_defenses = (
        previous_state.performed_nodes if previous_state else set()
    )
    previous_compromised_nodes = (
        previous_state.compromised_nodes if previous_state else set()
    )
    previous_performed_nodes = (
        previous_state.performed_nodes if previous_state else set()
    )
    previous_observed_nodes = previous_state.observed_nodes if previous_state else set()
    performed_nodes_order = (
        dict(previous_state.performed_nodes_order) if previous_state else {}
    )

    action_surface = (
        get_defense_surface(sim_state, defender_settings.actionable_steps)
        - previous_performed_nodes
    )

    iteration = previous_state.iteration if previous_state else 0
    if new_enabled_defenses:
        performed_nodes_order[iteration] = frozenset(new_enabled_defenses)

    new_observed_nodes = observed_nodes(
        defender_settings.observable_steps,
        defender_settings.false_positive_rates,
        defender_settings.false_negative_rates,
        sim_state,
        rng,
        new_compromised_nodes,
    )

    return DefenderState(
        name,
        sim_state=sim_state,
        settings=defender_settings,
        performed_nodes=frozenset(previous_enabled_defenses | new_enabled_defenses),
        compromised_nodes=frozenset(previous_compromised_nodes | new_compromised_nodes),
        observed_nodes=frozenset(previous_observed_nodes | new_observed_nodes),
        action_surface=frozenset(action_surface),
        iteration=iteration + 1,
        performed_nodes_order=performed_nodes_order,
        previous_state=previous_state,
    )


def initial_defender_state(
    sim_state: MalSimulatorState,
    defender_settings: DefenderSettings,
    pre_compromised_nodes: Set[AttackGraphNode],
    pre_enabled_defenses: Set[AttackGraphNode],
    rng: np.random.Generator,
) -> DefenderState:
    """Create a defender state from defender settings"""
    return create_defender_state(
        sim_state=sim_state,
        name=defender_settings.name,
        new_compromised_nodes=pre_compromised_nodes,
        new_enabled_defenses=pre_enabled_defenses,
        rng=rng,
        defender_settings=defender_settings,
    )
