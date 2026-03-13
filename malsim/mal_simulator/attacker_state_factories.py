"""Creation/manipulation of attacker state"""

from __future__ import annotations
from collections.abc import MutableSet, Set, Mapping
from typing import Optional, TYPE_CHECKING

from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.attack_surface import (
    get_attack_surface,
    get_effects_of_attack_step,
)
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
    step_compromised_nodes: Set[AttackGraphNode],
    ttc_values: Mapping[AttackGraphNode, float],
    impossible_steps: Set[AttackGraphNode],
    step_attempted_nodes: Set[AttackGraphNode] = frozenset(),
    step_nodes_made_unviable: Set[AttackGraphNode] = frozenset(),
    previous_state: Optional[MalSimAttackerState] = None,
) -> MalSimAttackerState:
    """
    Update a previous attacker state based on what the agent compromised
    and what nodes became unviable.
    """
    entry_points = attacker_settings.entry_points
    if previous_state is None:
        # Initial compromised nodes are just the step compromised nodes
        compromised_nodes = step_compromised_nodes

        # Create an initial attack surface
        new_action_surface = get_attack_surface(
            attack_surface_settings,
            sim_state,
            attacker_settings.actionable_steps,
            compromised_nodes,
        )
        action_surface_removals: Set[AttackGraphNode] = set()
        action_surface_additions = new_action_surface
        performed_nodes_order: dict[int, Set[AttackGraphNode]] = {}

        if sim_state.settings.compromise_entrypoints_at_start:
            performed_nodes_order[0] = frozenset(entry_points)
        else:
            # If entrypoints not compromised at start,
            # we need to put them in action surface
            new_action_surface |= entry_points
            action_surface_additions |= entry_points

        previous_num_attempts: Mapping[AttackGraphNode, int] = dict.fromkeys(
            sim_state.attack_graph.attack_steps, 0
        )

    else:
        compromised_nodes = previous_state.performed_nodes | step_compromised_nodes
        performed_nodes_order = dict(previous_state.performed_nodes_order)

        if step_compromised_nodes:
            performed_nodes_order[previous_state.iteration] = frozenset(
                step_compromised_nodes
            )

        # Build on previous attack surface (for performance)
        action_surface_additions = (
            get_attack_surface(
                attack_surface_settings,
                sim_state,
                attacker_settings.actionable_steps,
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
        sim_state=sim_state,
        performed_nodes=frozenset(compromised_nodes | step_compromised_nodes),
        action_surface=new_action_surface,
        step_action_surface_additions=action_surface_additions,
        step_action_surface_removals=frozenset(action_surface_removals),
        step_performed_nodes=frozenset(step_compromised_nodes),
        step_unviable_nodes=frozenset(step_nodes_made_unviable),
        step_attempted_nodes=frozenset(step_attempted_nodes),
        num_attempts=new_num_attempts,
        iteration=(previous_state.iteration + 1) if previous_state else 1,
        performed_nodes_order=performed_nodes_order,
        settings=attacker_settings,
        ttc_values=ttc_values,
        impossible_steps=impossible_steps,
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
) -> MalSimAttackerState:
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

    step_compromised_nodes: Set[AttackGraphNode] = set()
    if sim_state.settings.compromise_entrypoints_at_start:
        step_compromised_nodes = get_entrypoint_compromises(sim_state, entry_points)

    return create_attacker_state(
        sim_state=sim_state,
        attack_surface_settings=sim_settings.attack_surface,
        name=attacker_settings.name,
        attacker_settings=attacker_settings,
        step_compromised_nodes=step_compromised_nodes,
        ttc_values=ttc_values,
        impossible_steps=impossible_steps,
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
