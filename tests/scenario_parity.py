"""Golden "resolved scenario shape" for the Rust/Python scenario-loader
parity check (PORTING_NOTES.md §7 C6, §11).

For every fixture in `tests/testdata/scenarios` this computes the resolved,
non-random shape the Python `Scenario` produces: language/model paths,
graph size, simulator settings, and per agent the policy name, reward mode,
entry points/goals (by full name), every `NodePropertyRule` resolved per
node (`per_node()`, by full name) and the flattened `reset_native` inputs
(`native_settings.py`). A fixture Python refuses to load records only the
exception type.

The result is committed as `tests/testdata/scenario_parity.json`.
`tests/test_scenario_parity.py` asserts the file is current, and
`core/malsim-core/tests/scenario_parity.rs` asserts the Rust loader
produces the same shape. Regenerate after changing a fixture or the
Python loader with:

    python -m tests.scenario_parity
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable, Iterable, Mapping, Set
from typing import Any

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim.config.agent_settings import AttackerSettings
from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.native_settings import (
    flatten_attacker_settings,
    flatten_defender_settings,
    flatten_sim_settings,
)
from malsim.mal_simulator.ttc_utils import TTCDist
from malsim.scenario.scenario import Scenario

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCENARIOS_DIR = os.path.join(REPO_ROOT, 'tests', 'testdata', 'scenarios')
GOLDEN_FILE = os.path.join(REPO_ROOT, 'tests', 'testdata', 'scenario_parity.json')

# Needs network access (git clone of the language) - `integration`-marked
# in test_scenario.py, and not supported by the Rust loader (§12, C4).
SKIPPED_FIXTURES = {'simple_scenario_git_url.yml'}


def _repo_relative(path: str) -> str:
    return os.path.relpath(os.path.realpath(path), REPO_ROOT)


def _sorted_names(nodes: Iterable[AttackGraphNode]) -> list[str]:
    return sorted(n.full_name for n in nodes)


def _rule_per_node(
    rule: NodePropertyRule[Any] | None,
    graph: AttackGraph,
    convert: Callable[[Any], Any] = lambda v: v,
) -> dict[str, Any] | None:
    if rule is None:
        return None
    return {k: convert(v) for k, v in sorted(rule.per_node(graph).items())}


def _id_names(graph: AttackGraph, ids: Iterable[int] | None) -> list[str] | None:
    if ids is None:
        return None
    return sorted(graph.nodes[i].full_name for i in ids)


def _rate_map(
    graph: AttackGraph, rates: Mapping[int, float] | None
) -> dict[str, Any] | None:
    """Rate maps cover every node, so only the length and the non-zero
    entries are recorded."""
    if rates is None:
        return None
    return {
        'len': len(rates),
        'nonzero': dict(
            sorted((graph.nodes[i].full_name, r) for i, r in rates.items() if r)
        ),
    }


def _attacker_shape(
    settings: AttackerSettings[AttackGraphNode], graph: AttackGraph
) -> dict[str, Any]:
    entry_points = settings.entry_points
    shape: dict[str, Any] = {}
    if isinstance(entry_points, Set):
        shape['entry_points'] = _sorted_names(entry_points)
        single_set: Set[AttackGraphNode] = entry_points
    else:
        shape['entry_points'] = [_sorted_names(eps) for eps in entry_points]
        # Sampled at reset; the flattened shape below doesn't include it.
        single_set = frozenset()
    shape['goals'] = _sorted_names(settings.goals)
    shape['rewards'] = _rule_per_node(settings.rewards, graph)
    shape['actionable_steps'] = _rule_per_node(settings.actionable_steps, graph)
    shape['ttc_dists'] = _rule_per_node(
        settings.ttc_dists, graph, lambda name: TTCDist.from_name(name).to_dict()
    )
    flat = flatten_attacker_settings(settings, graph, single_set)
    ttc_dists = flat.get('ttc_dists')
    shape['flat'] = {
        'actionable_steps': _id_names(graph, flat.get('actionable_steps')),
        'ttc_dists': None
        if ttc_dists is None
        else dict(sorted((graph.nodes[i].full_name, d) for i, d in ttc_dists.items())),
    }
    return shape


def scenario_shape(scenario_file: str) -> dict[str, Any]:
    """The resolved, non-random shape of one scenario file."""
    scenario = Scenario.load_from_file(scenario_file)
    graph = scenario.attack_graph
    agents: list[dict[str, Any]] = []
    for settings in scenario.agent_settings:
        agent: dict[str, Any] = {
            'name': settings.name,
            'type': settings.type.value,
            'policy': settings.policy.__name__ if settings.policy else None,
            'reward_mode': settings.reward_mode.name,
        }
        if isinstance(settings, AttackerSettings):
            agent.update(_attacker_shape(settings, graph))
        else:
            agent['rewards'] = _rule_per_node(settings.rewards, graph)
            agent['actionable_steps'] = _rule_per_node(settings.actionable_steps, graph)
            agent['observable_steps'] = _rule_per_node(settings.observable_steps, graph)
            agent['false_positive_rates'] = _rule_per_node(
                settings.false_positive_rates, graph
            )
            agent['false_negative_rates'] = _rule_per_node(
                settings.false_negative_rates, graph
            )
            flat = flatten_defender_settings(settings, graph)
            agent['flat'] = {
                'actionable_steps': _id_names(graph, flat.get('actionable_steps')),
                'observable_steps': _id_names(graph, flat.get('observable_steps')),
                'false_positive_rates': _rate_map(
                    graph, flat.get('false_positive_rates')
                ),
                'false_negative_rates': _rate_map(
                    graph, flat.get('false_negative_rates')
                ),
            }
        agents.append(agent)

    sim_settings = scenario.sim_settings
    return {
        'lang_file': _repo_relative(scenario._lang_file),
        'model_file': _repo_relative(scenario._model_file)
        if scenario._model_file
        else None,
        'num_nodes': len(graph.nodes),
        'sim_settings': {
            **flatten_sim_settings(sim_settings),
            'seed': sim_settings.seed,
            'uncompromise_untraversable_steps': (
                sim_settings.uncompromise_untraversable_steps
            ),
        },
        'agents': agents,
    }


def generate() -> dict[str, Any]:
    """Shape (or error type) of every fixture, keyed by path relative to
    `tests/testdata/scenarios`."""
    shapes: dict[str, Any] = {}
    pattern = os.path.join(SCENARIOS_DIR, '**', '*.yml')
    for path in sorted(glob.glob(pattern, recursive=True)):
        if os.path.basename(path) in SKIPPED_FIXTURES:
            continue
        fixture = os.path.relpath(path, SCENARIOS_DIR)
        try:
            shapes[fixture] = scenario_shape(path)
        except Exception as e:
            shapes[fixture] = {'error': type(e).__name__}
    return shapes


def dumps(shapes: Mapping[str, Any]) -> str:
    return json.dumps(shapes, indent=1) + '\n'


if __name__ == '__main__':
    with open(GOLDEN_FILE, 'w', encoding='utf-8') as f:
        f.write(dumps(generate()))
    print(f'Wrote {GOLDEN_FILE}')
