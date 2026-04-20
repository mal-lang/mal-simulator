"""Creation/manipulation of attacker state"""

from __future__ import annotations
from collections.abc import MutableSet, Set, Mapping
from typing import TYPE_CHECKING

from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.attack_surface import (
    get_attack_surface,
    get_effects_of_attack_step,
)
from malsim.mal_simulator.attacker_state import AttackerState
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
from malsim.config.sim_settings import (
    AttackSurfaceSettings,
    MalSimulatorSettings,
    TTCMode,
)

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
    from malsim.mal_simulator.simulator_state import MalSimulatorState
    import numpy as np


def create_attacker_state(
    sim_state: MalSimulatorState,
    attack_surface_settings: AttackSurfaceSettings,
    attacker_settings: AttackerSettings[AttackGraphNode],
    name: str,
    new_performed_nodes: Set[AttackGraphNode],
    ttc_values: Mapping[AttackGraphNode, float],
    impossible_steps: Set[AttackGraphNode],
    new_attempted_nodes: Set[AttackGraphNode] = frozenset(),
    previous_state: AttackerState | None = None,
) -> AttackerState:
    """
    Update a previous attacker state based on what the agent compromised
    """

    previous_performed_nodes = (
        previous_state.performed_nodes if previous_state else set()
    )
    previous_performed_nodes_order = (
        previous_state.performed_nodes_order if previous_state else {}
    )
    previous_attempted_nodes = (
        previous_state.attempted_nodes if previous_state else set()
    )
    previous_action_surface = previous_state.action_surface if previous_state else set()
    previous_num_attempts = (
        previous_state.num_attempts
        if previous_state
        else dict.fromkeys(sim_state.attack_graph.attack_steps, 0)
    )

    performed_nodes = previous_performed_nodes | new_performed_nodes
    performed_nodes_order = dict(previous_performed_nodes_order)
    attempted_nodes = previous_attempted_nodes | new_attempted_nodes
    num_attempts = dict(previous_num_attempts)
    for node in new_attempted_nodes:
        num_attempts[node] += 1

    action_surface = get_attack_surface(
        attack_surface_settings,
        sim_state,
        attacker_settings.actionable_steps,
        previous_performed_nodes | new_performed_nodes,
    )

    if not previous_state and not sim_state.settings.compromise_entrypoints_at_start:
        action_surface |= attacker_settings.entry_points

    iteration = previous_state.iteration if previous_state else 0
    if new_performed_nodes:
        performed_nodes_order[iteration] = frozenset(new_performed_nodes)

    return AttackerState(
        name,
        sim_state=sim_state,
        iteration=iteration + 1,
        performed_nodes_order=performed_nodes_order,
        settings=attacker_settings,
        ttc_values=ttc_values,
        impossible_steps=impossible_steps,
        performed_nodes=frozenset(performed_nodes),
        attempted_nodes=frozenset(attempted_nodes),
        action_surface=action_surface,
        num_attempts=num_attempts,
        previous_state=previous_state,
    )


def get_entrypoint_compromises(
    sim_state: MalSimulatorState,
    entry_points: Set[AttackGraphNode],
) -> Set[AttackGraphNode]:
    """Compromise entry points and return compromised nodes including effects"""
    step_compromised_nodes: MutableSet[AttackGraphNode] = set()
    for entry_point in entry_points:
        step_compromised_nodes.add(entry_point)
        # Perform effects of entry point compromises
        step_compromised_nodes |= get_effects_of_attack_step(
            sim_state, entry_point, step_compromised_nodes
        )
    return step_compromised_nodes


def initial_attacker_state(
    sim_state: MalSimulatorState,
    sim_settings: MalSimulatorSettings,
    attacker_settings: AttackerSettings[AttackGraphNode],
    rng: np.random.Generator,
) -> AttackerState:
    """Create an attacker state from attacker settings"""

    ttc_values, impossible_steps = (
        attacker_overriding_ttc_settings(
            sim_state.attack_graph,
            attacker_settings.ttc_dists,
            sim_settings.ttc_mode,
            rng,
        )
        if attacker_settings.ttc_dists
        else (
            sim_state.graph_state.ttc_values,
            sim_state.graph_state.impossible_attack_steps,
        )
    )
    entry_points = set(
        full_names_or_nodes_to_nodes(
            sim_state.attack_graph, attacker_settings.entry_points
        )
    )
    new_compromised_nodes: Set[AttackGraphNode] = set()

    if sim_state.settings.compromise_entrypoints_at_start:
        new_compromised_nodes = get_entrypoint_compromises(sim_state, entry_points)

    return create_attacker_state(
        sim_state=sim_state,
        attack_surface_settings=sim_settings.attack_surface,
        attacker_settings=attacker_settings,
        name=attacker_settings.name,
        ttc_values=ttc_values,
        impossible_steps=impossible_steps,
        new_performed_nodes=new_compromised_nodes,
    )


def attacker_overriding_ttc_settings(
    attack_graph: AttackGraph,
    ttc_overrides_rule: NodePropertyRule[TTCDist],
    ttc_mode: TTCMode,
    rng: np.random.Generator,
) -> tuple[
    Mapping[AttackGraphNode, float],
    Set[AttackGraphNode],
]:
    """
    Get overriding TTC distributions, TTC values, and impossible attack steps
    from attacker settings if they exist.

    Returns three separate collections:
        - a dict of TTC distributions
        - a dict of TTC values
        - a set of impossible steps
    """

    ttc_overrides_names = ttc_overrides_rule.per_node(attack_graph)

    # Convert names to TTCDist objects and map from AttackGraphNode
    # objects instead of from full names
    ttc_overrides = {
        full_name_or_node_to_node(attack_graph, node): TTCDist.from_name(name)
        for node, name in ttc_overrides_names.items()
    }
    ttc_value_overrides = attack_step_ttc_values(
        ttc_overrides.keys(),
        rng,
        ttc_mode,
        ttc_dists=full_name_dict_to_node_dict(attack_graph, ttc_overrides),
    )
    impossible_step_overrides = get_impossible_attack_steps(
        ttc_overrides.keys(),
        rng,
        ttc_dists=full_name_dict_to_node_dict(attack_graph, ttc_overrides),
    )
    return ttc_value_overrides, impossible_step_overrides
