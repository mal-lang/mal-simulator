"""Creation/manipulation of attacker state"""

from __future__ import annotations
from collections.abc import Set, Mapping
from types import MappingProxyType
from typing import Optional, TYPE_CHECKING

from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.attack_surface import get_attack_surface
from malsim.mal_simulator.attacker_state import MalSimAttackerState
from malsim.mal_simulator.node_getters import (
    full_name_dict_to_node_dict,
    full_name_or_node_to_node,
    full_names_or_nodes_to_nodes,
)
from malsim.mal_simulator.ttc_utils import (
    TTCDist,
    attack_step_ttc_values,
    get_impossible_attack_steps,
)
from malsim.config.agent_settings import AttackerSettings
from malsim.config.sim_settings import TTCMode

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
    from malsim.mal_simulator.simulator_state import MalSimulatorState
    import numpy as np


def create_attacker_state(
    sim_state: MalSimulatorState,
    name: str,
    entry_points: Set[AttackGraphNode],
    goals: Set[AttackGraphNode] = frozenset(),
    step_compromised_nodes: Set[AttackGraphNode] = frozenset(),
    step_attempted_nodes: Set[AttackGraphNode] = frozenset(),
    step_nodes_made_unviable: Set[AttackGraphNode] = frozenset(),
    ttc_overrides: Mapping[AttackGraphNode, TTCDist] = MappingProxyType({}),
    ttc_value_overrides: Mapping[AttackGraphNode, float] = MappingProxyType({}),
    impossible_step_overrides: Set[AttackGraphNode] = frozenset(),
    reward_rule: Optional[NodePropertyRule] = None,
    actionability_rule: Optional[NodePropertyRule] = None,
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
            actionability_rule,
            sim_state.global_actionability,
            compromised_nodes,
        )
        action_surface_removals: set[AttackGraphNode] = set()
        action_surface_additions = new_action_surface
        performed_nodes_order: dict[int, frozenset[AttackGraphNode]] = {}

        if sim_state.settings.compromise_entrypoints_at_start:
            performed_nodes_order[0] = frozenset(entry_points)
        else:
            # If entrypoints not compromised at start,
            # we need to put them in action surface
            new_action_surface |= entry_points
            action_surface_additions |= entry_points

        previous_num_attempts: Mapping[AttackGraphNode, int] = {
            n: 0 for n in sim_state.attack_graph.attack_steps
        }

    else:
        # Previous state rules will be used if previous state is given
        reward_rule = previous_state.reward_rule
        actionability_rule = previous_state.actionability_rule

        ttc_value_overrides = previous_state.ttc_value_overrides
        impossible_step_overrides = previous_state.impossible_step_overrides
        compromised_nodes = previous_state.performed_nodes | step_compromised_nodes
        performed_nodes_order = dict(previous_state.performed_nodes_order)

        if step_compromised_nodes:
            performed_nodes_order[previous_state.iteration] = frozenset(
                step_compromised_nodes
            )

        # Build on previous attack surface (for performance)
        action_surface_additions = (
            get_attack_surface(
                sim_state.settings,
                sim_state,
                actionability_rule,
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
        reward_rule=reward_rule,
        actionability_rule=actionability_rule,
        performed_nodes_order=MappingProxyType(performed_nodes_order),
    )


def initial_attacker_state(
    sim_state: MalSimulatorState,
    attacker_settings: AttackerSettings,
    ttc_mode: TTCMode,
    rng: np.random.Generator,
) -> MalSimAttackerState:
    """Create an attacker state from attacker settings"""
    ttc_overrides, ttc_value_overrides, impossible_steps = (
        attacker_overriding_ttc_settings(
            sim_state.attack_graph, attacker_settings, ttc_mode, rng
        )
    )
    return create_attacker_state(
        sim_state=sim_state,
        name=attacker_settings.name,
        entry_points=set(
            full_names_or_nodes_to_nodes(
                sim_state.attack_graph, attacker_settings.entry_points
            )
        ),
        goals=set(
            full_names_or_nodes_to_nodes(
                sim_state.attack_graph, attacker_settings.goals
            )
        ),
        ttc_overrides=ttc_overrides,
        ttc_value_overrides=ttc_value_overrides,
        impossible_step_overrides=impossible_steps,
        reward_rule=attacker_settings.rewards,
        actionability_rule=attacker_settings.actionable_steps,
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
