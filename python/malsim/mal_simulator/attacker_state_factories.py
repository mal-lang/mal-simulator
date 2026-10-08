"""Creation/manipulation of attacker state"""

from __future__ import annotations
from collections.abc import MutableSet, Set, Mapping
from typing import Any, TYPE_CHECKING

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
    entry_points: Set[AttackGraphNode],
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
        settings=attack_surface_settings,
        sim_state=sim_state,
        actionability=attacker_settings.actionable_steps,
        performed_nodes=performed_nodes,
    )

    if not previous_state and not sim_state.settings.compromise_entrypoints_at_start:
        action_surface |= entry_points

    iteration = previous_state.iteration if previous_state else 0
    if new_performed_nodes:
        performed_nodes_order[iteration] = frozenset(new_performed_nodes)

    return AttackerState(
        name,
        entry_points=entry_points,
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
        goals=attacker_settings.goals,
    )


def create_attacker_state_from_native(
    sim_state: MalSimulatorState,
    name: str,
    attacker_settings: AttackerSettings[AttackGraphNode],
    entry_points: Set[AttackGraphNode],
    native_agent_out: Mapping[str, Any],
    previous_state: AttackerState | None,
) -> AttackerState:
    """Build an `AttackerState` from one attacker's `reset_native`/
    `step_native` output dict (PORTING_NOTES.md §5 Phase A9, plus the
    post-A10 delta-wire-format perf fix - see §10) - the
    native-output-driven counterpart of `create_attacker_state` above.

    `create_attacker_state`/`initial_attacker_state` are deliberately left
    untouched by this function: `DynaMalSimulator` still calls them
    directly with its own pure-Python recompute (§2.3 - its own port is a
    later phase), so changing their signature/behavior would break it. See
    PORTING_NOTES.md §10 for why this is a new, separate function rather
    than an in-place rewrite of `create_attacker_state`.

    Unlike `create_attacker_state`, most fields here are resolved directly
    from native's output rather than recomputed in Python - but since the
    delta-wire-format fix (§10), `step_native`'s output is a *delta*
    (`step_performed_nodes`/`step_attempted_nodes`), not the full
    episode-accumulated value `reset_native` still returns under the old
    full-field names. On the
    first call after reset (`previous_state is None`) the full fields are
    read as before; on every subsequent call the delta is merged against
    `previous_state`. `ttc_values`/`impossible_steps` are resolved once
    from `reset_native`'s output and carried forward unchanged - native
    never re-sends them (they're static after reset even when
    `ttc_dists` overrides are configured - see `simulator.rs`'s
    `attacker_ttc_overrides` doc comment). `performed_nodes_order` is
    Python-only bookkeeping native has no concept of, built incrementally
    via a diff against `previous_state`, same as before.
    """
    attack_graph = sim_state.attack_graph

    action_surface = frozenset(
        attack_graph.nodes[node_id] for node_id in native_agent_out['action_surface']
    )

    performed_nodes: Set[AttackGraphNode]
    attempted_nodes: Set[AttackGraphNode]
    num_attempts: dict[AttackGraphNode, int]
    ttc_values: Mapping[AttackGraphNode, float]
    impossible_steps: Set[AttackGraphNode]

    if previous_state is None:
        new_performed_nodes = frozenset(
            attack_graph.nodes[node_id]
            for node_id in native_agent_out['performed_nodes']
        )
        new_attempted_nodes = frozenset(
            attack_graph.nodes[node_id]
            for node_id in native_agent_out['attempted_nodes']
        )
        # Dense by default (every attack step present, defaulting to 0) -
        # `attempt_attacker_step` (pure Python, still used directly by
        # `DynaMalSimulator` and by some existing tests - §2.3/§10)
        # indexes `agent.num_attempts[node]` unconditionally for any node
        # about to be attempted, matching `create_attacker_state`'s own
        # `dict.fromkeys(sim_state.attack_graph.attack_steps, 0)` -
        # native's own map is sparse (only nodes actually attempted), so
        # it's overlaid on top of the dense default rather than used
        # as-is. Only built once here, at reset - every subsequent call
        # starts from `previous_state.num_attempts` instead.
        num_attempts = dict.fromkeys(attack_graph.attack_steps, 0)
        num_attempts.update(
            {
                attack_graph.nodes[node_id]: count
                for node_id, count in native_agent_out['num_attempts'].items()
            }
        )
        ttc_values = {
            attack_graph.nodes[node_id]: value
            for node_id, value in native_agent_out['ttc_values'].items()
        }
        impossible_steps = frozenset(
            attack_graph.nodes[node_id]
            for node_id in native_agent_out['impossible_steps']
        )
        performed_nodes = new_performed_nodes
        attempted_nodes = new_attempted_nodes
    else:
        new_performed_nodes = frozenset(
            attack_graph.nodes[node_id]
            for node_id in native_agent_out['step_performed_nodes']
        )
        new_attempted_nodes = frozenset(
            attack_graph.nodes[node_id]
            for node_id in native_agent_out['step_attempted_nodes']
        )
        performed_nodes = previous_state.performed_nodes | new_performed_nodes
        attempted_nodes = previous_state.attempted_nodes | new_attempted_nodes
        num_attempts = dict(previous_state.num_attempts)
        for node in new_attempted_nodes:
            num_attempts[node] += 1
        ttc_values = previous_state.ttc_values
        impossible_steps = previous_state.impossible_steps

    performed_nodes_order = dict(
        previous_state.performed_nodes_order if previous_state else {}
    )
    # native's returned `iteration` is 0-indexed from reset and is exactly
    # the key `performed_nodes_order` should use for this call's new
    # nodes; `AttackerState.iteration` itself is 1-indexed (§5/A9, §10).
    native_iteration = native_agent_out['iteration']
    if new_performed_nodes:
        performed_nodes_order[native_iteration] = frozenset(new_performed_nodes)

    return AttackerState(
        name,
        entry_points=entry_points,
        sim_state=sim_state,
        iteration=native_iteration + 1,
        performed_nodes_order=performed_nodes_order,
        settings=attacker_settings,
        ttc_values=ttc_values,
        impossible_steps=impossible_steps,
        performed_nodes=performed_nodes,
        attempted_nodes=attempted_nodes,
        action_surface=action_surface,
        num_attempts=num_attempts,
        previous_state=previous_state,
        goals=attacker_settings.goals,
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


def get_entry_points(
    attack_graph: AttackGraph,
    attacker_settings: AttackerSettings[AttackGraphNode],
    rng: np.random.Generator,
) -> Set[AttackGraphNode]:
    """
    Get entry points as set of AttackGraphNodes from attacker settings.
    If multiple sets of entry points are given, sample one set from the options.

    Takes `attack_graph` directly (not a `MalSimulatorState`) since this is
    the only thing its body reads - this lets it be called before a
    `MalSimulatorState` exists yet (PORTING_NOTES.md §5/§10 Phase A9: the
    native-backed `reset()` must resolve entry points before it has built
    one, to pass the single resolved set into `reset_native`).
    """

    if isinstance(attacker_settings.entry_points, Set):
        return frozenset(
            full_names_or_nodes_to_nodes(attack_graph, attacker_settings.entry_points)
        )
    else:
        # Multiple potential entry point sets given
        # - sample one set of entry points from the options
        chosen_entry_points = rng.choice(list(attacker_settings.entry_points))
        return set(
            full_names_or_nodes_to_nodes(attack_graph, chosen_entry_points)  # type: ignore
        )


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
    entry_points = get_entry_points(sim_state.attack_graph, attacker_settings, rng)
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
        entry_points=entry_points,
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
