"""Creation/manipulation of defender state"""

from __future__ import annotations
from collections.abc import Mapping, Set
from typing import Any, TYPE_CHECKING

from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.agent_settings import DefenderSettings
from malsim.mal_simulator.defender_state import DefenderState
from malsim.mal_simulator.event_logger import LogEntry
from malsim.mal_simulator.simulator_state import MalSimulatorState

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraph


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
    `step_native` output dict (PORTING_NOTES.md §5 Phase A9, plus the
    post-A10 delta-wire-format perf fix - see §10) - the
    native-output-driven counterpart of `create_defender_state` above.

    `create_defender_state`/`initial_defender_state` are deliberately left
    untouched (§2.3/§10 - `DynaMalSimulator` still calls them directly),
    same reasoning as `create_attacker_state_from_native`.

    Unlike `create_defender_state`, `performed_nodes`/`compromised_nodes`/
    `observed_nodes`/`logs` are resolved directly from native's output on
    the first call after reset (`previous_state is None`), where
    `reset_native`'s output still carries the full fields under their old
    names. On every subsequent call, `step_native`'s output is a *delta*
    (`step_performed_nodes`/`step_compromised_nodes`/
    `step_observed_nodes`/`step_logs` - §10) merged against
    `previous_state` instead - this is also what fixes the O(episode^2)
    bug `logs` used to have, where every log ever fired was re-parsed into
    a `LogEntry` on every single step. `performed_nodes_order` is still
    built incrementally via a diff against `previous_state`, same as the
    attacker counterpart.
    """
    attack_graph = sim_state.attack_graph

    action_surface = frozenset(
        attack_graph.nodes[node_id] for node_id in native_agent_out['action_surface']
    )

    performed_nodes: Set[AttackGraphNode]
    compromised_nodes: Set[AttackGraphNode]
    observed_nodes_: Set[AttackGraphNode]

    if previous_state is None:
        new_performed_nodes = frozenset(
            attack_graph.nodes[node_id]
            for node_id in native_agent_out['performed_nodes']
        )
        new_compromised_nodes = frozenset(
            attack_graph.nodes[node_id]
            for node_id in native_agent_out['compromised_nodes']
        )
        new_observed_nodes = frozenset(
            attack_graph.nodes[node_id]
            for node_id in native_agent_out['observed_nodes']
        )
        new_logs = tuple(
            _log_entry_from_native(attack_graph, log)
            for log in native_agent_out['logs']
        )
        performed_nodes = new_performed_nodes
        compromised_nodes = new_compromised_nodes
        observed_nodes_ = new_observed_nodes
        logs = new_logs
    else:
        new_performed_nodes = frozenset(
            attack_graph.nodes[node_id]
            for node_id in native_agent_out['step_performed_nodes']
        )
        new_compromised_nodes = frozenset(
            attack_graph.nodes[node_id]
            for node_id in native_agent_out['step_compromised_nodes']
        )
        new_observed_nodes = frozenset(
            attack_graph.nodes[node_id]
            for node_id in native_agent_out['step_observed_nodes']
        )
        new_logs = tuple(
            _log_entry_from_native(attack_graph, log)
            for log in native_agent_out['step_logs']
        )
        performed_nodes = previous_state.performed_nodes | new_performed_nodes
        compromised_nodes = previous_state.compromised_nodes | new_compromised_nodes
        observed_nodes_ = previous_state.observed_nodes | new_observed_nodes
        logs = previous_state.logs + new_logs

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
