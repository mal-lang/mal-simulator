"""Creation/manipulation of attacker state"""

from __future__ import annotations
from collections.abc import Set, Mapping
from typing import Any, TYPE_CHECKING

from malsim.mal_simulator.attacker_state import AttackerState
from malsim.mal_simulator.node_getters import (
    full_names_or_nodes_to_nodes,
)
from malsim.config.agent_settings import AttackerSettings

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
    from malsim.mal_simulator.simulator_state import MalSimulatorState
    import numpy as np


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
