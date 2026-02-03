"""Creation/manipulation of defender state"""

from collections.abc import Set
from types import MappingProxyType
from typing import Optional

import numpy as np
from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.agent_settings import DefenderSettings
from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.defender_state import MalSimDefenderState
from malsim.mal_simulator.defense_surface import get_defense_surface
from malsim.mal_simulator.event_logger import collect_logs
from malsim.mal_simulator.observability import defender_observed_nodes
from malsim.mal_simulator.simulator_state import MalSimulatorState


def create_defender_state(
    sim_state: MalSimulatorState,
    name: str,
    rng: np.random.Generator,
    step_compromised_nodes: Set[AttackGraphNode] = frozenset(),
    step_enabled_defenses: Set[AttackGraphNode] = frozenset(),
    step_nodes_made_unviable: Set[AttackGraphNode] = frozenset(),
    reward_rule: Optional[NodePropertyRule] = None,
    actionability_rule: Optional[NodePropertyRule] = None,
    observability_rule: Optional[NodePropertyRule] = None,
    false_positive_rates_rule: Optional[NodePropertyRule] = None,
    false_negative_rates_rule: Optional[NodePropertyRule] = None,
    previous_state: Optional[MalSimDefenderState] = None,
) -> MalSimDefenderState:
    """
    Update a previous defender state based on what steps
    were enabled/compromised during last step
    """

    if previous_state is None:
        # Initialize
        action_surface = get_defense_surface(
            sim_state, actionability_rule, sim_state.global_actionability
        )
        previous_enabled_defenses: Set[AttackGraphNode] = frozenset()
        previous_compromised_nodes: Set[AttackGraphNode] = frozenset()
        previous_observed_nodes: Set[AttackGraphNode] = frozenset()
        action_surface_additions: Set[AttackGraphNode] = action_surface
        action_surface_removals: Set[AttackGraphNode] = frozenset()
        performed_nodes_order: dict[int, frozenset[AttackGraphNode]] = {}
        logs = collect_logs(0, step_compromised_nodes, previous_compromised_nodes)

        if step_enabled_defenses:
            # Pre enabled defenses go into iteration 0
            performed_nodes_order[0] = frozenset(step_enabled_defenses)
    else:
        # Previous rules used if previous state given
        reward_rule = previous_state.reward_rule
        actionability_rule = previous_state.actionability_rule
        observability_rule = previous_state.observability_rule
        false_positive_rates_rule = previous_state.false_positive_rates_rule
        false_negative_rates_rule = previous_state.false_negative_rates_rule

        # Initialize
        action_surface = (
            get_defense_surface(
                sim_state, actionability_rule, sim_state.global_actionability
            )
            - previous_state.performed_nodes
        )

        previous_enabled_defenses = previous_state.performed_nodes
        previous_compromised_nodes = previous_state.compromised_nodes
        previous_observed_nodes = previous_state.observed_nodes
        action_surface_additions = frozenset()
        action_surface_removals = step_enabled_defenses
        performed_nodes_order = dict(previous_state.performed_nodes_order)

        logs = collect_logs(
            previous_state.iteration, step_compromised_nodes, previous_compromised_nodes
        )

        if step_enabled_defenses:
            performed_nodes_order[previous_state.iteration] = frozenset(
                step_enabled_defenses
            )

    step_observed_nodes = defender_observed_nodes(
        observability_rule,
        false_positive_rates_rule,
        false_negative_rates_rule,
        sim_state,
        rng,
        step_compromised_nodes,
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
        reward_rule=reward_rule,
        observability_rule=observability_rule,
        actionability_rule=actionability_rule,
        false_negative_rates_rule=false_negative_rates_rule,
        false_positive_rates_rule=false_positive_rates_rule,
        performed_nodes_order=MappingProxyType(performed_nodes_order),
        logs=tuple(previous_state.logs + logs) if previous_state else tuple(logs),
    )


def initial_defender_state(
    sim_state: MalSimulatorState,
    defender_settings: DefenderSettings,
    pre_compromised_nodes: set[AttackGraphNode],
    pre_enabled_defenses: set[AttackGraphNode],
    rng: np.random.Generator,
) -> MalSimDefenderState:
    """Create a defender state from defender settings"""
    return create_defender_state(
        sim_state=sim_state,
        name=defender_settings.name,
        step_compromised_nodes=pre_compromised_nodes,
        step_enabled_defenses=pre_enabled_defenses,
        rng=rng,
        reward_rule=defender_settings.rewards,
        actionability_rule=defender_settings.actionable_steps,
        observability_rule=defender_settings.observable_steps,
        false_negative_rates_rule=defender_settings.false_negative_rates,
        false_positive_rates_rule=defender_settings.false_positive_rates,
    )
