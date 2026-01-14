"""Creation/manipulation of agent states used in the simulator"""

from __future__ import annotations
from collections.abc import Set, Mapping
from types import MappingProxyType
from typing import Optional, TYPE_CHECKING

from malsim.mal_simulator.attack_surface import get_attack_surface
from malsim.mal_simulator.defense_surface import get_defense_surface

from malsim.mal_simulator.agent_state import (
    MalSimAttackerState,
    MalSimDefenderState,
)

from malsim.mal_simulator.node_getters import (
    full_name_dict_to_node_dict,
    full_name_or_node_to_node,
    full_names_or_nodes_to_nodes
)
from malsim.mal_simulator.observability import defender_observed_nodes
from malsim.mal_simulator.ttc_utils import (
    TTCDist,
    attack_step_ttc_values,
    get_impossible_attack_steps
)
from malsim.config.agent_settings import AttackerSettings, DefenderSettings
from malsim.mal_simulator.settings import TTCMode

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
    from malsim.config.agent_settings import AgentSettings
    from malsim.mal_simulator.simulator_state import MalSimulatorState
    import numpy as np

def create_attacker_state(
    sim_state: MalSimulatorState,
    attacker_settings: AttackerSettings,
    name: str,
    entry_points: Set[AttackGraphNode],
    goals: Set[AttackGraphNode] = frozenset(),
    step_compromised_nodes: Set[AttackGraphNode] = frozenset(),
    step_attempted_nodes: Set[AttackGraphNode] = frozenset(),
    step_nodes_made_unviable: Set[AttackGraphNode] = frozenset(),
    ttc_overrides: Mapping[AttackGraphNode, TTCDist] = MappingProxyType({}),
    ttc_value_overrides: Mapping[AttackGraphNode, float] = MappingProxyType({}),
    impossible_step_overrides: Set[AttackGraphNode] = frozenset(),
    previous_state: Optional[MalSimAttackerState] = None,
) -> MalSimAttackerState:
    """
    Update a previous attacker state based on what the agent compromised
    and what nodes became unviable.
    """

    if previous_state is None:
        # Initial compromised nodes
        if sim_state.settings.compromise_entrypoints_at_start:
            step_compromised_nodes |= entry_points
        compromised_nodes = step_compromised_nodes

        # Create an initial attack surface
        new_action_surface = get_attack_surface(
            sim_state.settings,
            sim_state,
            attacker_settings.actionable_steps,
            sim_state.global_actionability,
            compromised_nodes
        )
        action_surface_removals: set[AttackGraphNode] = set()
        action_surface_additions = new_action_surface

        if not sim_state.settings.compromise_entrypoints_at_start:
            # If entrypoints not compromised at start,
            # we need to put them in action surface
            new_action_surface |= entry_points
            action_surface_additions |= entry_points

        previous_num_attempts: Mapping[AttackGraphNode, int] = {
            n: 0 for n in sim_state.attack_graph.attack_steps
        }

    else:
        ttc_value_overrides = previous_state.ttc_value_overrides
        impossible_step_overrides = previous_state.impossible_step_overrides
        compromised_nodes = previous_state.performed_nodes | step_compromised_nodes

        # Build on previous attack surface (for performance)
        action_surface_additions = (
            get_attack_surface(
                sim_state.settings,
                sim_state,
                attacker_settings.actionable_steps,
                sim_state.global_actionability,
                compromised_nodes | step_compromised_nodes,
                from_nodes=step_compromised_nodes,
            )
            - previous_state.action_surface
        )
        action_surface_removals = set(
            (step_nodes_made_unviable & previous_state.action_surface)
            | step_compromised_nodes
        )
        new_action_surface = frozenset(
            (previous_state.action_surface - action_surface_removals)
            | action_surface_additions
        )
        previous_num_attempts = previous_state.num_attempts

    new_num_attempts = dict(previous_num_attempts)
    for node in step_attempted_nodes:
        new_num_attempts[node] += 1

    return MalSimAttackerState(
        name,
        entry_points=frozenset(entry_points),
        goals=frozenset(goals),
        sim_state=sim_state,
        performed_nodes=frozenset(compromised_nodes | step_compromised_nodes),
        action_surface=new_action_surface,
        step_action_surface_additions=action_surface_additions,
        step_action_surface_removals=frozenset(action_surface_removals),
        step_performed_nodes=frozenset(step_compromised_nodes),
        step_unviable_nodes=frozenset(step_nodes_made_unviable),
        step_attempted_nodes=frozenset(step_attempted_nodes),
        num_attempts=MappingProxyType(new_num_attempts),
        ttc_overrides=MappingProxyType(ttc_overrides),
        ttc_value_overrides=MappingProxyType(ttc_value_overrides),
        impossible_step_overrides=frozenset(impossible_step_overrides),
        iteration=(previous_state.iteration + 1) if previous_state else 1,
        reward_rule=attacker_settings.rewards,
        actionability_rule=attacker_settings.rewards
    )


def initial_attacker_state(
    sim_state: MalSimulatorState,
    attacker_settings: AttackerSettings,
    ttc_mode: TTCMode,
    rng: np.random.Generator,
) -> MalSimAttackerState:
    """Create an attacker state from attacker settings"""
    ttc_overrides, ttc_value_overrides, impossible_steps = (
        attacker_overriding_ttc_settings(sim_state.attack_graph, attacker_settings, ttc_mode, rng)
    )
    return create_attacker_state(
        attacker_settings=attacker_settings,
        sim_state=sim_state,
        name=attacker_settings.name,
        entry_points=set(
            full_names_or_nodes_to_nodes(sim_state.attack_graph, attacker_settings.entry_points)
        ),
        goals=set(full_names_or_nodes_to_nodes(sim_state.attack_graph, attacker_settings.goals)),
        ttc_overrides=ttc_overrides,
        ttc_value_overrides=ttc_value_overrides,
        impossible_step_overrides=impossible_steps,
    )


def create_defender_state(
    sim_state: MalSimulatorState,
    defender_settings: DefenderSettings,
    rng: np.random.Generator,
    step_compromised_nodes: Set[AttackGraphNode] = frozenset(),
    step_enabled_defenses: Set[AttackGraphNode] = frozenset(),
    step_nodes_made_unviable: Set[AttackGraphNode] = frozenset(),
    previous_state: Optional[MalSimDefenderState] = None,
) -> MalSimDefenderState:
    """
    Update a previous defender state based on what steps
    were enabled/compromised during last step
    """

    action_surface = get_defense_surface(
        sim_state, defender_settings.actionable_steps, sim_state.global_actionability
    )

    if previous_state is None:
        # Initialize
        previous_enabled_defenses: Set[AttackGraphNode] = frozenset()
        previous_compromised_nodes: Set[AttackGraphNode] = frozenset()
        previous_observed_nodes: Set[AttackGraphNode] = frozenset()
        action_surface_additions: Set[AttackGraphNode] = action_surface
        action_surface_removals: Set[AttackGraphNode] = frozenset()
    else:
        previous_enabled_defenses = previous_state.performed_nodes
        previous_compromised_nodes = previous_state.compromised_nodes
        previous_observed_nodes = previous_state.observed_nodes
        action_surface_additions = frozenset()
        action_surface_removals = step_enabled_defenses
        action_surface -= previous_state.performed_nodes

    step_observed_nodes = defender_observed_nodes(
        defender_settings.observable_steps,
        defender_settings.false_positive_rates,
        defender_settings.false_negative_rates,
        sim_state,
        rng,
        step_compromised_nodes
    )
    return MalSimDefenderState(
        defender_settings.name,
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
        reward_rule=defender_settings.rewards,
        observability_rule=defender_settings.observable_steps,
        actionability_rule=defender_settings.actionable_steps,
        false_negatives_rule=defender_settings.false_negative_rates,
        false_positives_rule=defender_settings.false_positive_rates,
    )


def attacker_overriding_ttc_settings(
    attack_graph: AttackGraph,
    attacker_settings: AttackerSettings,
    ttc_mode: TTCMode,
    rng: np.random.Generator,
) -> tuple[
    dict[AttackGraphNode, TTCDist],
    dict[AttackGraphNode, float],
    set[AttackGraphNode],
]:
    """
    Get overriding TTC distributions, TTC values, and impossible attack steps
    from attacker settings if they exist.

    Returns three separate collections:
        - a dict of TTC distributions
        - a dict of TTC values
        - a set of impossible steps
    """

    if not attacker_settings.ttc_overrides:
        return {}, {}, set()

    ttc_overrides_names = attacker_settings.ttc_overrides.per_node(attack_graph)

    # Convert names to TTCDist objects and map from AttackGraphNode
    # objects instead of from full names
    ttc_overrides = {
        full_name_or_node_to_node(attack_graph, node): TTCDist.from_name(name)
        for node, name in ttc_overrides_names.items()
    }
    ttc_value_overrides = attack_step_ttc_values(
        ttc_overrides.keys(),
        ttc_mode,
        rng,
        ttc_dists=full_name_dict_to_node_dict(attack_graph, ttc_overrides),
    )
    impossible_step_overrides = get_impossible_attack_steps(
        ttc_overrides.keys(),
        rng,
        ttc_dists=full_name_dict_to_node_dict(attack_graph, ttc_overrides),
    )
    return ttc_overrides, ttc_value_overrides, impossible_step_overrides

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
        defender_settings=defender_settings,
        step_compromised_nodes=pre_compromised_nodes,
        step_enabled_defenses=pre_enabled_defenses,
        rng=rng,
    )
