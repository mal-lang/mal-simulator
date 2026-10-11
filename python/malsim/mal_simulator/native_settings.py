"""Flatten Python-side settings objects into the plain-data shapes
`malsim._native.Simulator.reset_native` expects (see PORTING_NOTES.md
§2.4/§5 Phase A9).

`NodePropertyRule` itself is never handed across the FFI boundary (§2.4) -
every function here resolves a rule against a concrete `AttackGraph`
*once*, reusing the existing per-node helper functions
(`node_is_actionable`/`node_is_observable`/`node_false_positive_rate`/
`node_false_negative_rate`) for exact semantic parity with the pure-Python
path `DynaMalSimulator` still uses, rather than re-implementing
`NodePropertyRule.value()`'s precedence/truthiness rules a second time.
"""

from __future__ import annotations

from collections.abc import Set
from typing import Any

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim.config.agent_settings import AttackerSettings, DefenderSettings
from malsim.config.node_property_rule import NodePropertyRule
from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.graph_utils import node_is_actionable
from malsim.mal_simulator.observability import node_is_observable
from malsim.mal_simulator.false_alerts import (
    node_false_negative_rate,
    node_false_positive_rate,
)
from malsim.mal_simulator.ttc_utils import TTCDist


def flatten_sim_settings(settings: MalSimulatorSettings) -> dict[str, Any]:
    """Flatten the `MalSimulatorSettings` fields `reset_native` reads.

    `uncompromise_untraversable_steps` is deliberately not included - it's
    unread by the current implementation on the Python side too (§2.5).
    """
    return {
        'ttc_mode': settings.ttc_mode.name,
        'run_defense_step_bernoullis': settings.run_defense_step_bernoullis,
        'run_attack_step_bernoullis': settings.run_attack_step_bernoullis,
        'skip_compromised': settings.attack_surface.skip_compromised,
        'skip_unnecessary': settings.attack_surface.skip_unnecessary,
        'compromise_entrypoints_at_start': settings.compromise_entrypoints_at_start,
    }


def _node_ids(nodes: Set[AttackGraphNode]) -> list[int]:
    return [n.id for n in nodes]


def _flatten_actionable_steps(
    rule: NodePropertyRule[bool] | None, attack_graph: AttackGraph
) -> list[int] | None:
    if rule is None:
        return None
    return [n.id for n in attack_graph.nodes.values() if node_is_actionable(rule, n)]


def _flatten_observable_steps(
    rule: NodePropertyRule[bool] | None, attack_graph: AttackGraph
) -> list[int] | None:
    if rule is None:
        return None
    return [n.id for n in attack_graph.nodes.values() if node_is_observable(rule, n)]


def _flatten_rate_map(
    rule: NodePropertyRule[float] | None,
    attack_graph: AttackGraph,
    rate_fn: Any,
) -> dict[int, float] | None:
    if rule is None:
        return None
    return {n.id: rate_fn(n, rule) for n in attack_graph.nodes.values()}


def _flatten_ttc_dists(
    rule: NodePropertyRule[Any] | None, attack_graph: AttackGraph
) -> dict[int, dict[str, Any]] | None:
    """Flatten `AttackerSettings.ttc_dists` the same way
    `attacker_state_factories.py::attacker_overriding_ttc_settings` reads
    it: `.per_node()`'s resolved values are predefined-distribution *name*
    strings (`TTCDist.from_name`), not arbitrary `TTCDist` objects -
    mirrored here rather than widened, to stay bug-for-bug compatible with
    the pure-Python path `DynaMalSimulator` still uses.
    """
    if rule is None:
        return None
    per_node = rule.per_node(attack_graph)
    if not per_node:
        return None
    return {
        attack_graph.get_node_by_full_name(full_name).id: TTCDist.from_name(
            name
        ).to_dict()
        for full_name, name in per_node.items()
    }


def flatten_attacker_settings(
    attacker_settings: AttackerSettings[AttackGraphNode],
    attack_graph: AttackGraph,
    entry_points: Set[AttackGraphNode],
) -> dict[str, Any]:
    """Flatten one attacker's settings into `reset_native`'s per-agent dict
    shape. `entry_points` is the already-sampled single set (multiple
    entry-point-set sampling happens in Python before this call - see
    PORTING_NOTES.md §10, A9).
    """
    cfg: dict[str, Any] = {
        'type': 'attacker',
        'entry_points': _node_ids(entry_points),
    }
    if attacker_settings.goals:
        cfg['goals'] = _node_ids(attacker_settings.goals)
    actionable = _flatten_actionable_steps(
        attacker_settings.actionable_steps, attack_graph
    )
    if actionable is not None:
        cfg['actionable_steps'] = actionable
    ttc_dists = _flatten_ttc_dists(attacker_settings.ttc_dists, attack_graph)
    if ttc_dists is not None:
        cfg['ttc_dists'] = ttc_dists
    return cfg


def flatten_defender_settings(
    defender_settings: DefenderSettings, attack_graph: AttackGraph
) -> dict[str, Any]:
    """Flatten one defender's settings into `reset_native`'s per-agent dict
    shape."""
    cfg: dict[str, Any] = {'type': 'defender'}
    actionable = _flatten_actionable_steps(
        defender_settings.actionable_steps, attack_graph
    )
    if actionable is not None:
        cfg['actionable_steps'] = actionable
    observable = _flatten_observable_steps(
        defender_settings.observable_steps, attack_graph
    )
    if observable is not None:
        cfg['observable_steps'] = observable
    fpr = _flatten_rate_map(
        defender_settings.false_positive_rates, attack_graph, node_false_positive_rate
    )
    if fpr is not None:
        cfg['false_positive_rates'] = fpr
    fnr = _flatten_rate_map(
        defender_settings.false_negative_rates, attack_graph, node_false_negative_rate
    )
    if fnr is not None:
        cfg['false_negative_rates'] = fnr
    return cfg
