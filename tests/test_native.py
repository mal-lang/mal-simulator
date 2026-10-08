"""Smoke tests for `malsim._native` (Phase A1/A8 - see PORTING_NOTES.md
§5/A1, §5/A8).

Proves end to end that a `maltoolbox.AttackGraph` built on the Python side
can be read from inside `malsim._native`, a separately-compiled PyO3
extension module, via the shared `Rc<RefCell<AttackGraph>>` handle that
mal-toolbox's `PyAttackGraph.__inner_capsule__()` hands out as a
`PyCapsule` (see PORTING_NOTES.md §10 for why a direct pyclass downcast
across extension modules doesn't work and a capsule is needed instead).

The `test_native_simulator_*` tests below exercise `_native.Simulator`
(Phase A8) - *not* wired into `MalSimulator` yet (that's A9's job), so
these build the `reset_native`/`step_native` input dicts by hand from a
`Scenario`'s already-resolved `attacker_settings`/`defender_settings`
rather than going through `MalSimulator`.
"""

from collections.abc import Set

import pytest
from maltoolbox.attackgraph import AttackGraphNode

from .test_scenario import path_relative_to_tests

from malsim.config.agent_settings import AttackerSettings
from malsim.scenario.scenario import Scenario
from malsim import _native


def test_native_node_count_matches_python() -> None:
    scenario = Scenario.load_from_file(
        path_relative_to_tests('./testdata/scenarios/simple_scenario.yml')
    )
    attack_graph = scenario.attack_graph

    assert _native.node_count(attack_graph) == len(attack_graph.nodes)


def test_native_model_asset_count_matches_python_and_sees_both_sides_mutations() -> (
    None
):
    """Phase B3 (PORTING_NOTES.md §6/B3) - A1-equivalent proof for
    `maltoolbox.Model`/`PyModel.__inner_capsule__()`. Unlike A1's original
    smoke test, this also proves *double visibility*: a mutation made
    through either side (Python `model.add_asset(...)` or the native
    handle, read back via `model.assets`) is seen by the other - the one
    thing A1's own smoke test didn't need to prove, since Phase A never
    mutates the shared graph from both sides at once (see §6's "New
    architectural wrinkle" note).
    """
    scenario = Scenario.load_from_file(
        path_relative_to_tests('./testdata/scenarios/simple_scenario.yml')
    )
    model = scenario.model

    assert _native.model_asset_count(model) == len(model.assets)

    # Mutate from the Python side; the native handle must see it immediately
    # (same shared `Rc<RefCell<Model>>`, not a copy).
    model.add_asset('Application', name='NativeB3TestAsset')
    assert _native.model_asset_count(model) == len(model.assets)

    # Mutate from the native side; the Python object must see it immediately
    # too - the direction A1's own smoke test never needed to prove.
    new_id = _native.model_add_asset_native(model, 'Application')
    assert new_id in model.assets
    assert _native.model_asset_count(model) == len(model.assets)


def _load_simple_scenario() -> Scenario:
    return Scenario.load_from_file(
        path_relative_to_tests('./testdata/scenarios/simple_scenario.yml')
    )


def _single_entry_point(
    attacker: AttackerSettings[AttackGraphNode],
) -> AttackGraphNode:
    """`simple_scenario.yml`'s `Attacker1` has one flat entry point set
    (not the `tuple[Set, ...]` "multiple entry point sets" shape) - A8's
    native `Simulator` doesn't support that shape yet (see
    `simulator.rs`'s module docs), so these tests only ever deal with the
    single-`Set` case.
    """
    assert isinstance(attacker.entry_points, Set)
    (entry_point,) = attacker.entry_points
    return entry_point


def test_native_simulator_reset_compromises_entry_points_by_default() -> None:
    scenario = _load_simple_scenario()
    attack_graph = scenario.attack_graph
    attacker = scenario.attacker_settings['Attacker1']
    entry_point = _single_entry_point(attacker)

    sim = _native.Simulator(attack_graph)
    out = sim.reset_native(
        {},
        {
            'Attacker1': {
                'type': 'attacker',
                'entry_points': [entry_point.id],
            },
            'Defender1': {'type': 'defender'},
        },
        42,
    )

    attacker_out = out['agents']['Attacker1']
    defender_out = out['agents']['Defender1']

    # compromise_entrypoints_at_start defaults to True.
    assert entry_point.id in attacker_out['performed_nodes']
    # skip_compromised defaults to True - the entry point itself should
    # not be back in its own action surface.
    assert entry_point.id not in attacker_out['action_surface']
    assert len(attacker_out['action_surface']) > 0
    assert attacker_out['terminated'] is False
    assert attacker_out['iteration'] == 0

    assert isinstance(defender_out['action_surface'], list)
    assert isinstance(defender_out['logs'], list)


def test_native_simulator_reset_without_compromise_keeps_entry_actionable() -> None:
    scenario = _load_simple_scenario()
    attack_graph = scenario.attack_graph
    attacker = scenario.attacker_settings['Attacker1']
    entry_point = _single_entry_point(attacker)

    sim = _native.Simulator(attack_graph)
    out = sim.reset_native(
        {'compromise_entrypoints_at_start': False},
        {
            'Attacker1': {
                'type': 'attacker',
                'entry_points': [entry_point.id],
            },
        },
        42,
    )

    attacker_out = out['agents']['Attacker1']
    assert entry_point.id not in attacker_out['performed_nodes']
    assert entry_point.id in attacker_out['action_surface']


def test_native_simulator_step_advances_attacker_state() -> None:
    scenario = _load_simple_scenario()
    attack_graph = scenario.attack_graph
    attacker = scenario.attacker_settings['Attacker1']
    entry_point = _single_entry_point(attacker)

    sim = _native.Simulator(attack_graph)
    reset_out = sim.reset_native(
        {},
        {
            'Attacker1': {
                'type': 'attacker',
                'entry_points': [entry_point.id],
            },
            'Defender1': {'type': 'defender'},
        },
        42,
    )
    action_surface = reset_out['agents']['Attacker1']['action_surface']
    assert action_surface, 'expected a non-empty action surface to step into'
    next_node_id = action_surface[0]

    step_out = sim.step_native({'Attacker1': [next_node_id], 'Defender1': []})
    attacker_out = step_out['agents']['Attacker1']

    assert attacker_out['iteration'] == 1
    assert next_node_id in attacker_out['step_performed_nodes']
    assert next_node_id not in attacker_out['action_surface']


def test_native_simulator_step_output_is_delta_only() -> None:
    """`step_native`'s return shape is deltas-only (`step_*` keys) for the
    fields that are episode-accumulated internally - see
    `simulator.rs`'s `build_step_output` module docs and
    `PORTING_NOTES.md` §10's post-A10/pre-A11 differences-log entry for
    the wire-format tables. `reset_native`'s shape is untouched (full
    fields) and isn't asserted here.
    """
    scenario = _load_simple_scenario()
    attack_graph = scenario.attack_graph
    attacker = scenario.attacker_settings['Attacker1']
    entry_point = _single_entry_point(attacker)

    sim = _native.Simulator(attack_graph)
    reset_out = sim.reset_native(
        {},
        {
            'Attacker1': {
                'type': 'attacker',
                'entry_points': [entry_point.id],
            },
            'Defender1': {'type': 'defender'},
        },
        42,
    )
    action_surface = reset_out['agents']['Attacker1']['action_surface']
    assert action_surface

    step_out = sim.step_native({'Attacker1': [action_surface[0]], 'Defender1': []})

    assert set(step_out['sim_state'].keys()) == {'step_enabled_defenses'}

    attacker_out = step_out['agents']['Attacker1']
    assert set(attacker_out.keys()) == {
        'type',
        'step_performed_nodes',
        'step_attempted_nodes',
        'action_surface',
        'iteration',
        'terminated',
    }

    defender_out = step_out['agents']['Defender1']
    assert set(defender_out.keys()) == {
        'type',
        'step_performed_nodes',
        'step_compromised_nodes',
        'step_observed_nodes',
        'action_surface',
        'iteration',
        'terminated',
        'step_logs',
    }


def test_native_simulator_step_output_is_delta_only_attacker_only() -> None:
    """Same as `test_native_simulator_step_output_is_delta_only` but with
    no defender agent registered at all, per the plan's requirement to
    cover both an attacker-only and an attacker+defender fixture.
    """
    scenario = _load_simple_scenario()
    attack_graph = scenario.attack_graph
    attacker = scenario.attacker_settings['Attacker1']
    entry_point = _single_entry_point(attacker)

    sim = _native.Simulator(attack_graph)
    reset_out = sim.reset_native(
        {},
        {
            'Attacker1': {
                'type': 'attacker',
                'entry_points': [entry_point.id],
            },
        },
        42,
    )
    action_surface = reset_out['agents']['Attacker1']['action_surface']
    assert action_surface

    step_out = sim.step_native({'Attacker1': [action_surface[0]]})

    assert set(step_out['sim_state'].keys()) == {'step_enabled_defenses'}
    assert set(step_out['agents'].keys()) == {'Attacker1'}

    attacker_out = step_out['agents']['Attacker1']
    assert set(attacker_out.keys()) == {
        'type',
        'step_performed_nodes',
        'step_attempted_nodes',
        'action_surface',
        'iteration',
        'terminated',
    }


def test_native_simulator_step_before_reset_raises() -> None:
    scenario = _load_simple_scenario()
    sim = _native.Simulator(scenario.attack_graph)

    with pytest.raises(ValueError):
        sim.step_native({})


def test_native_simulator_step_unknown_agent_raises() -> None:
    scenario = _load_simple_scenario()
    attack_graph = scenario.attack_graph
    attacker = scenario.attacker_settings['Attacker1']
    entry_point = _single_entry_point(attacker)

    sim = _native.Simulator(attack_graph)
    sim.reset_native(
        {},
        {'Attacker1': {'type': 'attacker', 'entry_points': [entry_point.id]}},
        42,
    )

    with pytest.raises(ValueError):
        sim.step_native({'NoSuchAgent': []})
