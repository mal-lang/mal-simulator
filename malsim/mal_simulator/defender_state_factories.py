"""Creation/manipulation of defender state"""

from collections.abc import Set

import numpy as np
from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.agent_settings import DefenderSettings
from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.defender_state import MalSimDefenderState
from malsim.mal_simulator.defense_surface import get_defense_surface
from malsim.mal_simulator.observability import observed_nodes
from malsim.mal_simulator.simulator_state import MalSimulatorState


def create_defender_state(
    sim_state: MalSimulatorState,
    sim_settings: MalSimulatorSettings,
    name: str,
    rng: np.random.Generator,
    defender_settings: DefenderSettings,
    step_compromised_nodes: Set[AttackGraphNode] = frozenset(),
    step_enabled_defenses: Set[AttackGraphNode] = frozenset(),
    step_nodes_made_unviable: Set[AttackGraphNode] = frozenset(),
    previous_state: Optional[MalSimDefenderState] = None,
) -> MalSimDefenderState:
    """
    Update a previous defender state based on what steps
    were enabled/compromised during last step
    """

    if previous_state is None:
        # Initialize
        action_surface = get_defense_surface(
            sim_state, defender_settings.actionable_steps
        )
        previous_enabled_defenses: Set[AttackGraphNode] = frozenset()
        previous_compromised_nodes: Set[AttackGraphNode] = frozenset()
        previous_observed_nodes: Set[AttackGraphNode] = frozenset()
        action_surface_additions: Set[AttackGraphNode] = action_surface
        action_surface_removals: Set[AttackGraphNode] = frozenset()
        performed_nodes_order: dict[int, Set[AttackGraphNode]] = {}

        if step_enabled_defenses:
            # Pre enabled defenses go into iteration 0
            performed_nodes_order[0] = frozenset(step_enabled_defenses)
    else:
        # Initialize
        action_surface = (
            get_defense_surface(sim_state, defender_settings.actionable_steps)
            - previous_state.performed_nodes
        )

        previous_enabled_defenses = previous_state.performed_nodes
        previous_compromised_nodes = previous_state.compromised_nodes
        previous_observed_nodes = previous_state.observed_nodes
        action_surface_additions = frozenset()
        action_surface_removals = step_enabled_defenses
        performed_nodes_order = dict(previous_state.performed_nodes_order)
        if step_enabled_defenses:
            performed_nodes_order[previous_state.iteration] = frozenset(
                step_enabled_defenses
            )

    step_observed_nodes = observed_nodes(
        sim_state=sim_state,
        observable_steps_rule=defender_settings.observable_steps,
        false_positive_rates_rule=defender_settings.false_positive_rates
        or sim_settings.false_positive_rates,
        rng=rng,
        false_negative_rates_rule=defender_settings.false_negative_rates
        or sim_settings.false_negative_rates,
        compromised_nodes=step_compromised_nodes,
    )
    return MalSimDefenderState(
        name,
        sim_state=sim_state,
        performed_nodes=frozenset(previous_enabled_defenses | step_enabled_defenses),
        compromised_nodes=frozenset(
            previous_compromised_nodes | step_compromised_nodes
        ),
        step_compromised_nodes=frozenset(step_compromised_nodes),
        observed_nodes=frozenset(previous_observed_nodes | step_observed_nodes),
        step_observed_nodes=frozenset(step_observed_nodes),
        step_action_surface_additions=frozenset(action_surface_additions),
        step_action_surface_removals=frozenset(action_surface_removals),
        action_surface=frozenset(action_surface),
        step_performed_nodes=frozenset(step_enabled_defenses),
        step_unviable_nodes=frozenset(step_nodes_made_unviable),
        iteration=(previous_state.iteration + 1) if previous_state else 1,
        performed_nodes_order=performed_nodes_order,
        settings=defender_settings,
    )


def initial_defender_state(
    sim_state: MalSimulatorState,
    sim_settings: MalSimulatorSettings,
    defender_settings: DefenderSettings,
    pre_compromised_nodes: Set[AttackGraphNode],
    pre_enabled_defenses: Set[AttackGraphNode],
    rng: np.random.Generator,
) -> MalSimDefenderState:
    """Create a defender state from defender settings"""
    return create_defender_state(
        sim_state=sim_state,
        sim_settings=sim_settings,
        name=defender_settings.name,
        step_compromised_nodes=pre_compromised_nodes,
        step_enabled_defenses=pre_enabled_defenses,
        rng=rng,
        defender_settings=defender_settings,
    )
