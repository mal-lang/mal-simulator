"""Creation/manipulation of defender state"""

from __future__ import annotations
from collections.abc import Mapping, Set
from typing import Any, TYPE_CHECKING

import numpy as np
from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.agent_settings import DefenderSettings
from malsim.mal_simulator.defender_state import DefenderState
from malsim.mal_simulator.defense_surface import get_defense_surface
from malsim.mal_simulator.event_logger import (
    LogEntry,
    collect_logs,
    collect_false_positives,
)
from malsim.mal_simulator.observability import observed_nodes
from malsim.mal_simulator.simulator_state import MalSimulatorState

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraph


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

    logs = collect_logs(
        previous_state.iteration if previous_state else 0,
        new_compromised_nodes,
        previous_compromised_nodes,
        rng,
    ) + collect_false_positives(
        previous_state.iteration if previous_state else 0,
        sim_state.attack_graph.detectors,
        rng,
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
        logs=tuple(previous_state.logs + logs) if previous_state else tuple(logs),
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


def _log_entry_from_native(
    attack_graph: AttackGraph, native_log: Mapping[str, Any]
) -> LogEntry:
    """Resolve one native `LogEntry` dict (detector/trigger/context as ids)
    back into the real Python `LogEntry` dataclass (§3: "only plain data
    crosses the FFI boundary" - `LogEntry` itself is rebuilt on the Python
    side, same pattern as `AttackerState`/`DefenderState`)."""
    detector_node = attack_graph.nodes[native_log['detector_node_id']]
    detector = detector_node.detectors[native_log['detector_label']]
    return LogEntry(
        timestep=native_log['timestep'],
        detector_name=str(detector.name),
        detector=detector,
        trigger=attack_graph.nodes[native_log['trigger']],
        context={
            label: attack_graph.nodes[node_id]
            for label, node_id in native_log['context'].items()
        },
        false_positive=native_log['false_positive'],
    )


def create_defender_state_from_native(
    sim_state: MalSimulatorState,
    name: str,
    defender_settings: DefenderSettings,
    native_agent_out: Mapping[str, Any],
    previous_state: DefenderState | None,
) -> DefenderState:
    """Build a `DefenderState` from one defender's `reset_native`/
    `step_native` output dict (PORTING_NOTES.md §5 Phase A9) - the
    native-output-driven counterpart of `create_defender_state` above.

    `create_defender_state`/`initial_defender_state` are deliberately left
    untouched (§2.3/§10 - `DynaMalSimulator` still calls them directly),
    same reasoning as `create_attacker_state_from_native`.

    Unlike `create_defender_state`, `performed_nodes`/`compromised_nodes`/
    `observed_nodes`/`logs` are resolved directly from native's output
    with no merging against `previous_state`: native already returns the
    full episode-accumulated history for all four, not just this step's
    delta. Only `performed_nodes_order` is still built incrementally via a
    diff against `previous_state`, same as the attacker counterpart.
    """
    attack_graph = sim_state.attack_graph

    performed_nodes = frozenset(
        attack_graph.nodes[node_id] for node_id in native_agent_out['performed_nodes']
    )
    compromised_nodes = frozenset(
        attack_graph.nodes[node_id] for node_id in native_agent_out['compromised_nodes']
    )
    observed_nodes_ = frozenset(
        attack_graph.nodes[node_id] for node_id in native_agent_out['observed_nodes']
    )
    action_surface = frozenset(
        attack_graph.nodes[node_id] for node_id in native_agent_out['action_surface']
    )
    logs = tuple(
        _log_entry_from_native(attack_graph, log) for log in native_agent_out['logs']
    )

    previous_performed_nodes = (
        previous_state.performed_nodes if previous_state else frozenset()
    )
    new_performed_nodes = performed_nodes - previous_performed_nodes
    performed_nodes_order = dict(
        previous_state.performed_nodes_order if previous_state else {}
    )
    native_iteration = native_agent_out['iteration']
    if new_performed_nodes:
        performed_nodes_order[native_iteration] = frozenset(new_performed_nodes)

    return DefenderState(
        name,
        sim_state=sim_state,
        settings=defender_settings,
        performed_nodes=performed_nodes,
        compromised_nodes=compromised_nodes,
        observed_nodes=observed_nodes_,
        action_surface=action_surface,
        iteration=native_iteration + 1,
        performed_nodes_order=performed_nodes_order,
        previous_state=previous_state,
        logs=logs,
    )
