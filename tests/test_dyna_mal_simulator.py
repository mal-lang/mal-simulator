"""Test DynaMalSimulator class"""

from __future__ import annotations
from copy import copy
import gc
import logging
import random
import weakref
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from maltoolbox.attackgraph import AttackGraph
from maltoolbox.model import Model, ModelAsset
from maltoolbox.language.language_graph_model_effect import ModelEffectType
from malsim.config.sim_settings import TTCMode
from malsim.dyna_mal_simulator.simulator_state import AssetOp, AssocOp
from malsim.mal_simulator import (
    MalSimulatorSettings,
    run_simulation,
)
from malsim import Scenario

from malsim.config.agent_settings import AttackerSettings

import numpy as np
import pytest
from scipy.stats import chisquare

from malsim.mal_simulator.graph_utils import node_is_blocked
from malsim.policies.attackers.searchers import BreadthFirstAttacker, DepthFirstAttacker
from malsim.policies.attackers.ttc_soft_min import TTCSoftMinAttacker
from malsim.policies.random_agent import RandomAgent
from malsim.dyna_mal_simulator import DynaMalSimulator
from malsim.mal_simulator.attacker_state import AttackerState
from malsim.mal_simulator.defender_state import DefenderState

if TYPE_CHECKING:
    from maltoolbox.model import Model
    from typing import Any


def assert_no_dangling_associations(model: Model) -> None:
    """Assert no association points at a stale, orphaned ModelAsset object."""
    live_assets = set(model.assets.values())
    for asset in model.assets.values():
        for field_name, others in asset.associated_assets.items():
            for other in others:
                assert other in live_assets, (
                    f'{asset.name}.{field_name} points to a stale {other.name} object'
                )


def check_graph_equivalence(true: AttackGraph, other: AttackGraph) -> None:
    """Helper function to check that two graphs are equivalent in terms of
    nodes and their relationships."""
    assert len(true.nodes) == len(other.nodes), (
        f'Number of nodes differ: True AttackGraph {len(true.nodes)} != Other: '
        f'{len(other.nodes)}'
    )

    true_full_names = set(true.full_name_to_node.keys())
    other_full_names = set(other.full_name_to_node.keys())
    assert true_full_names == other_full_names, (
        f'Node full_names differ: '
        f'only in true: {true_full_names - other_full_names}; '
        f'only in other: {other_full_names - true_full_names}'
    )

    true_node_id_to_full_name = {
        node.id: node.full_name for node in true.nodes.values()
    }
    other_node_id_to_full_name = {
        node.id: node.full_name for node in other.nodes.values()
    }

    for true_node_full_name, true_node in true.full_name_to_node.items():
        try:
            part_node = other.full_name_to_node[true_node_full_name]
        except LookupError:
            pytest.fail(
                reason=f'{true_node.full_name} not in partially regenerated graph.'
            )
        true_node_child_names = {
            true_node_id_to_full_name[node.id] for node in true_node.children
        }
        part_node_child_names = {
            other_node_id_to_full_name[node.id] for node in part_node.children
        }
        assert true_node_child_names == part_node_child_names, (
            f'Different children between true and partially regenerated graphs for '
            f'{true_node.full_name}'
        )
        true_node_parent_names = {
            true_node_id_to_full_name[node.id] for node in true_node.parents
        }
        part_node_parent_names = {
            other_node_id_to_full_name[node.id] for node in part_node.parents
        }
        assert true_node_parent_names == part_node_parent_names, (
            f'Different parents between true and partially regenerated graphs for '
            f'{true_node.full_name}'
        )


def test_error_without_model(wiperLang_attack_graph: AttackGraph) -> None:
    """Make sure error is raised if the instance model is not available"""
    wiperLang_attack_graph.model = None
    with pytest.raises(ValueError):
        DynaMalSimulator(wiperLang_attack_graph, agents=())


def test_init_with_agent_settings(
    wiperLang_attack_graph: AttackGraph, wiperLang_model: Model
) -> None:
    """Make sure the simulator can be initialized with agent settings"""
    entry_points = frozenset(
        {wiperLang_attack_graph.get_node_by_full_name('InfectedDevice:infect')}
    )
    goals = frozenset(
        {wiperLang_attack_graph.get_node_by_full_name('InfectedData:read')}
    )

    agent_settings = (
        AttackerSettings(
            name='WiperAttacker',
            entry_points=entry_points,
            goals=goals,
            policy=RandomAgent,
        ),
    )
    sim = DynaMalSimulator(
        wiperLang_attack_graph,
        agents=agent_settings,
        sim_settings=MalSimulatorSettings(compromise_entrypoints_at_start=False),
    )

    # Make sure the agents were registered
    assert sim.agent_states.keys() == {'WiperAttacker'}
    assert sim.agent_reward_by_name('WiperAttacker') == 0.0
    assert sim.alive_agents == {'WiperAttacker'}


def test_init_from_scenario(wiperLang_scenario: Scenario) -> None:
    """Make sure the simulator can be initialized from a scenario"""
    sim = DynaMalSimulator.from_scenario(wiperLang_scenario)
    assert not sim.sim_settings.compromise_entrypoints_at_start

    # Make sure the agents were registered
    assert sim.agent_states.keys() == {'WiperController'}
    assert sim.agent_reward_by_name('WiperController') == 0.0
    assert sim.alive_agents == {'WiperController'}


def test_reset(wiperLang_scenario: Scenario) -> None:
    """Make sure attack graph is reset"""
    sim_settings = MalSimulatorSettings(seed=10, compromise_entrypoints_at_start=False)
    sim = DynaMalSimulator.from_scenario(wiperLang_scenario, sim_settings=sim_settings)
    attacker_name = wiperLang_scenario.agent_settings[0].name

    starting_nodes = set(sim._attack_graph.nodes.values())

    blocked_before = {
        n.full_name: node_is_blocked(sim.sim_state, n)
        for n in sim.sim_state.attack_graph.nodes.values()
    }
    necessity_before = {
        n.full_name: v for n, v in sim.sim_state.graph_state.necessity_per_node.items()
    }
    enabled_defenses = {
        n.full_name for n in sim.sim_state.graph_state.pre_enabled_defenses
    }
    assert attacker_name in sim.agent_states
    assert len(sim.agent_states) == 1
    attacker_state = sim.agent_states[attacker_name]
    action_surface_before = {n.full_name for n in attacker_state.action_surface}

    sim.reset()

    attacker_state = sim.agent_states[attacker_name]
    assert action_surface_before == {n.full_name for n in attacker_state.action_surface}
    assert enabled_defenses == {
        n.full_name for n in sim.sim_state.graph_state.pre_enabled_defenses
    }

    sim.reset()
    attacker_state = sim.agent_states[attacker_name]
    assert action_surface_before == {n.full_name for n in attacker_state.action_surface}

    # Step with action surface
    sim.step({attacker_name: list(attacker_state.action_surface)})

    # Make sure action surface back to normal
    sim.reset()
    attacker_state = sim.agent_states[attacker_name]
    assert action_surface_before == {n.full_name for n in attacker_state.action_surface}

    # Re-creating the simulator object with the same seed
    # should result in getting the same viability and necessity values
    sim = DynaMalSimulator.from_scenario(wiperLang_scenario, sim_settings=sim_settings)
    # Dynamic addition to test:
    # The attack graph may now contain new nodes from partial regeneration
    # Use intersection to only check the nodes
    # that were present in the original attack graph
    for node in starting_nodes.intersection(sim.sim_state.attack_graph.nodes.values()):
        # blocked is the same after reset
        assert blocked_before[node.full_name] == node_is_blocked(sim.sim_state, node)

    for node, necessary in sim.sim_state.graph_state.necessity_per_node.items():
        # necessity is the same after reset
        if node in starting_nodes:
            assert necessity_before[node.full_name] == necessary


def test_apply_model_effect_modification_record_partially_regenerates_graph(
    dynamic_remove_many_assoc_scenario: Scenario,
) -> None:
    attack_graph = dynamic_remove_many_assoc_scenario.attack_graph
    model = attack_graph.model
    assert model

    remove_steps = [
        node
        for node in attack_graph.nodes.values()
        if node.name == 'remove'
        and node.model_asset
        and node.model_asset.type != 'GodAsset'
    ]
    assert remove_steps

    fuzz_step = random.choice(remove_steps)
    assert fuzz_step.subtractive_model_effects

    # Drive the removal through the simulator (native `execute_model_effects`
    # + `partially_regenerate_graph`), with the step as an entry point so it
    # can be performed directly regardless of the scenario's action surface.
    sim = DynaMalSimulator(
        attack_graph,
        agents=[AttackerSettings(name='Fuzzer', entry_points={fuzz_step})],
        sim_settings=MalSimulatorSettings(
            ttc_mode=TTCMode.DISABLED, compromise_entrypoints_at_start=False
        ),
    )
    sim.reset()
    sim.step({'Fuzzer': [fuzz_step]})

    subtractive_ops = [
        op
        for op in sim.sim_state.modification_record
        if op.type == ModelEffectType.SUBTRACTIVE
    ]
    assert subtractive_ops, f'{fuzz_step.full_name} removed nothing'
    for op in subtractive_ops:
        if isinstance(op, AssetOp):
            assert op.asset.id not in model.assets, (
                f'{op.asset.name} is recorded as removed but is still in the model'
            )
        else:
            left, field_name, right = op.assoc
            if isinstance(left, ModelAsset) and left.id in model.assets:
                assert right.id not in {
                    a.id for a in left.associated_assets.get(field_name, set())
                }, f'{left.name}.{field_name} still points to {right.name}'
    assert_no_dangling_associations(model)

    fresh_attack_graph = AttackGraph(
        dynamic_remove_many_assoc_scenario.lang_graph, model
    )
    check_graph_equivalence(fresh_attack_graph, attack_graph)


def test_attacker_step(wiperLang_scenario: Scenario) -> None:
    sim = DynaMalSimulator.from_scenario(wiperLang_scenario)
    wiperLang_attack_graph = wiperLang_scenario.attack_graph
    model = wiperLang_attack_graph.model
    assert model
    attacker_name = next(iter(wiperLang_scenario.attacker_settings.keys()))
    state = sim.reset()[attacker_name]

    infected_device = model.get_asset_by_name('InfectedDevice')
    assert infected_device
    assert model.get_asset_by_name('Wiper') is None, (
        'Wiper asset should not exist before applying model effect of infect step'
    )
    assert 'malware' not in infected_device.associated_assets, (
        'InfectedDevice should not have malware associated before applying model effect'
    )
    infect_step = wiperLang_attack_graph.get_node_by_full_name('InfectedDevice:infect')
    assert infect_step in state.action_surface, (
        'InfectedDevice:infect step should be in action surface according to '
        'scenario settings'
    )
    state = sim.step({attacker_name: [infect_step]})[attacker_name]
    assert 'malware' in infected_device.associated_assets, (
        'InfectedDevice should have malware associated after applying model '
        'effect of infect step'
    )
    wiper = model.get_asset_by_name('Wiper-7')
    assert wiper
    assert wiper is not None, (
        'Wiper asset should exist after applying model effect of infect step'
    )
    assert wiper in infected_device.associated_assets['malware'], (
        'Wiper should be associated with InfectedDevice after applying model '
        'effect of infect step'
    )

    activate_step = wiperLang_attack_graph.get_node_by_full_name('Wiper-7:activate')
    assert activate_step in state.action_surface, (
        'Wiper:activate step should be in action surface after performing '
        'InfectedDevice:infect'
    )
    state = sim.step({attacker_name: [activate_step]})[attacker_name]
    assert activate_step in state.performed_nodes, (
        'Wiper:activate step should be in performed nodes after performing it'
    )

    infected_data = model.get_asset_by_name('InfectedData')
    assert infected_data
    assert infected_data.associated_assets.get('node') == {infected_device}, (
        'InfectedData should be associated to InfectedDevice before applying '
        'model effect'
    )
    exfiltrate_step = wiperLang_attack_graph.get_node_by_full_name('Wiper-7:exfiltrate')
    assert exfiltrate_step in state.action_surface, (
        'Wiper:exfiltrate step should be in action surface after performing '
        'Wiper:activate'
    )
    state = sim.step({attacker_name: [exfiltrate_step]})[attacker_name]
    c2_server = model.get_asset_by_name('C2Server')
    assert c2_server
    assert infected_data.associated_assets.get('node') == {
        infected_device,
        c2_server,
    }, (
        'InfectedData should be associated to InfectedDevice and C2Server after '
        'applying model effect'
    )

    access_data_step = wiperLang_attack_graph.get_node_by_full_name(
        'C2Server:accessData'
    )
    state = sim.step({attacker_name: [access_data_step]})[attacker_name]
    assert access_data_step in state.performed_nodes, (
        'C2Server:accessData step should be in performed nodes after performing it'
    )


@pytest.mark.parametrize(
    'attacker_class, config',
    [
        (RandomAgent, {}),
        (RandomAgent, {'wait_prob': 0.01}),
        (TTCSoftMinAttacker, {'beta': 1.0}),
        (TTCSoftMinAttacker, {'beta': 0.1}),
        (BreadthFirstAttacker, {'action_ordering': 'random'}),
        (BreadthFirstAttacker, {'action_ordering': 'sorted'}),
        (BreadthFirstAttacker, {'action_ordering': 'nothing'}),
        (BreadthFirstAttacker, {'action_ordering': 'list_random'}),
        (DepthFirstAttacker, {'action_ordering': 'random'}),
        (DepthFirstAttacker, {'action_ordering': 'sorted'}),
        (DepthFirstAttacker, {'action_ordering': 'nothing'}),
        (DepthFirstAttacker, {'action_ordering': 'list_random'}),
    ],
)
def test_different_attackers(attacker_class: type, config: dict[str, Any]) -> None:
    """Test a bunch of attacker agents with different configurations
    to make sure they don't crash the simulator
    """

    dynamal_example_scenarios = Path(
        'tests/testdata/scenarios/dynamal_example_scenarios'
    )
    for scenario_file in dynamal_example_scenarios.rglob('*.yml'):
        scenario = Scenario.load_from_file(str(scenario_file))
        attacker_name = next(iter(scenario.attacker_settings.keys()))

        sim_settings = MalSimulatorSettings(
            ttc_mode=TTCMode.PRE_SAMPLE, compromise_entrypoints_at_start=False
        )
        scenario.attacker_settings[attacker_name].policy = attacker_class
        scenario.attacker_settings[attacker_name].config = config
        sim = DynaMalSimulator.from_scenario(scenario, sim_settings=sim_settings)
        run_simulation(sim)


def test_remove_before_add(dynamic_add_remove_scenario: Scenario) -> None:
    attack_graph = dynamic_add_remove_scenario.attack_graph
    sim = DynaMalSimulator.from_scenario(dynamic_add_remove_scenario)
    state = sim.reset()['Attacker']

    # Non-existent assets cannot be removed
    # The error should be logged, but the simulation should not crash
    action = attack_graph.get_node_by_full_name('Start:remove')
    assert action in state.action_surface, (
        'Start:remove step should be in action surface according to scenario settings'
    )
    sim.step({'Attacker': [action]})

    # Non-existent associations cannot be removed
    # The error should be logged, but the simulation should not crash
    action = attack_graph.get_node_by_full_name('Start:remove_association')
    sim.step({'Attacker': [action]})

    # Only one start asset per object in the example language (association
    # condition) so the asset cannot be added twice
    # The error should be logged, but the simulation should not crash
    action = attack_graph.get_node_by_full_name('Start:add')
    sim.step({'Attacker': [action]})
    action = attack_graph.get_node_by_full_name('Object-1:addStart')
    sim.step({'Attacker': [action]})

    action = attack_graph.get_node_by_full_name('Object-1:addStartAssoc')
    sim.step({'Attacker': [action]})


def test_int_dynamic_test_lang6_transfer_apple_multi_step_and_reset(
    int_dynamic_test_lang6_scenario: Scenario,
) -> None:
    """Walk the full 3-step chain of the intDynamicTestLang6 example, asserting
    the model after every step, then verify sim.reset() fully undoes it.
    """
    sim = DynaMalSimulator.from_scenario(
        int_dynamic_test_lang6_scenario,
        sim_settings=MalSimulatorSettings(compromise_entrypoints_at_start=False),
    )
    attack_graph = int_dynamic_test_lang6_scenario.attack_graph
    model = attack_graph.model
    assert model
    attacker_name = next(iter(int_dynamic_test_lang6_scenario.attacker_settings))
    state = sim.reset()[attacker_name]

    table1 = model.get_asset_by_name('Table:1')
    bowl1 = model.get_asset_by_name('Bowl:1')
    apple1 = model.get_asset_by_name('Apple:1')
    assert table1 and bowl1 and apple1
    assert bowl1.associated_assets == {'table': {table1}, 'apples': {apple1}}
    assert apple1.associated_assets == {'container': {bowl1}}
    assert len(model.assets) == 3

    # Step 1: access
    access = attack_graph.get_node_by_full_name('Table:1:access')
    assert access in state.action_surface
    state = sim.step({attacker_name: [access]})[attacker_name]
    assert {n.full_name for n in state.action_surface} == {
        'Bowl:1:addApple',
        'Bowl:1:transferApple',
    }
    assert bowl1.associated_assets == {'table': {table1}, 'apples': {apple1}}

    # Step 2: addApple creates a new Apple asset associated to Bowl:1
    add_apple = attack_graph.get_node_by_full_name('Bowl:1:addApple')
    assert add_apple in state.action_surface
    state = sim.step({attacker_name: [add_apple]})[attacker_name]
    assert len(model.assets) == 4
    new_apple = next(
        a for a in model.assets.values() if a.type == 'Apple' and a is not apple1
    )
    assert bowl1.associated_assets == {'table': {table1}, 'apples': {apple1, new_apple}}
    assert new_apple.associated_assets == {'container': {bowl1}}
    assert {n.full_name for n in state.action_surface} == {'Bowl:1:transferApple'}

    # Step 3: transferApple orphans both apples (unlinked, not deleted)
    transfer = attack_graph.get_node_by_full_name('Bowl:1:transferApple')
    assert transfer in state.action_surface
    state = sim.step({attacker_name: [transfer]})[attacker_name]
    assert 'apples' not in bowl1.associated_assets
    assert bowl1.associated_assets == {'table': {table1}}
    assert apple1.associated_assets == {}
    assert new_apple.associated_assets == {}
    assert len(model.assets) == 4
    assert apple1 in model.assets.values()
    assert new_apple in model.assets.values()
    assert state.action_surface == frozenset()

    # Order between apple1/new_apple isn't meaningful, compare as a multiset
    assert Counter(sim.sim_state.modification_record) == Counter(
        [
            AssetOp(type=ModelEffectType.ADDITIVE, asset=new_apple),
            AssocOp(type=ModelEffectType.ADDITIVE, assoc=(bowl1, 'apples', new_apple)),
            AssocOp(
                type=ModelEffectType.SUBTRACTIVE, assoc=(bowl1, 'apples', new_apple)
            ),
            AssocOp(type=ModelEffectType.SUBTRACTIVE, assoc=(bowl1, 'apples', apple1)),
        ]
    )

    state = sim.reset()[attacker_name]
    assert len(model.assets) == 3
    assert model.get_asset_by_name('Apple') is None
    table1 = model.get_asset_by_name('Table:1')
    bowl1 = model.get_asset_by_name('Bowl:1')
    apple1 = model.get_asset_by_name('Apple:1')
    assert table1 and bowl1 and apple1
    assert bowl1.associated_assets == {'table': {table1}, 'apples': {apple1}}
    assert apple1.associated_assets == {'container': {bowl1}}
    assert {n.full_name for n in state.action_surface} == {'Table:1:access'}
    assert sim.sim_state.modification_record == []
    assert_no_dangling_associations(model)


def test_int_dynamic_test_lang3_contaminate_bananas_multi_step_and_reset(
    int_dynamic_test_lang3_scenario: Scenario,
) -> None:
    """Walk the full 3-step chain of the intDynamicTestLang3 example, asserting
    the model after every step, then verify sim.reset() fully undoes it.
    """
    sim = DynaMalSimulator.from_scenario(
        int_dynamic_test_lang3_scenario,
        sim_settings=MalSimulatorSettings(compromise_entrypoints_at_start=False),
    )
    attack_graph = int_dynamic_test_lang3_scenario.attack_graph
    model = attack_graph.model
    assert model
    attacker_name = next(iter(int_dynamic_test_lang3_scenario.attacker_settings))
    state = sim.reset()[attacker_name]

    alarm1 = model.get_asset_by_name('Alarm:1')
    tree1 = model.get_asset_by_name('BananaTree:1')
    banana1 = model.get_asset_by_name('Banana:1')
    poison1 = model.get_asset_by_name('Poison:1')
    assert alarm1 and tree1 and banana1 and poison1
    assert tree1.associated_assets == {'alarm': {alarm1}, 'bananas': {banana1}}
    assert alarm1.associated_assets == {'trees': {tree1}}
    assert banana1.associated_assets == {'poison': {poison1}, 'tree': {tree1}}
    assert len(model.assets) == 4

    # Step 1: access
    access = attack_graph.get_node_by_full_name('BananaTree:1:access')
    assert access in state.action_surface
    state = sim.step({attacker_name: [access]})[attacker_name]
    assert {n.full_name for n in state.action_surface} == {
        'BananaTree:1:contaminateBananas'
    }
    assert tree1.associated_assets == {'alarm': {alarm1}, 'bananas': {banana1}}

    # Step 2: contaminateBananas
    contaminate = attack_graph.get_node_by_full_name('BananaTree:1:contaminateBananas')
    assert contaminate in state.action_surface
    state = sim.step({attacker_name: [contaminate]})[attacker_name]
    assert 'alarm' not in tree1.associated_assets
    assert tree1.associated_assets == {'bananas': {banana1}}
    assert alarm1.associated_assets == {}
    assert alarm1 in model.assets.values()
    assert len(model.assets) == 4
    assert banana1.associated_assets == {'poison': {poison1}, 'tree': {tree1}}
    assert {n.full_name for n in state.action_surface} == {'Banana:1:poisoned'}
    assert sim.sim_state.modification_record == [
        AssocOp(type=ModelEffectType.SUBTRACTIVE, assoc=(tree1, 'alarm', alarm1)),
    ]

    # Step 3: poisoned
    poisoned = attack_graph.get_node_by_full_name('Banana:1:poisoned')
    assert poisoned in state.action_surface
    state = sim.step({attacker_name: [poisoned]})[attacker_name]
    assert poisoned in state.performed_nodes
    assert state.action_surface == frozenset()
    assert sim.sim_state.modification_record == [
        AssocOp(type=ModelEffectType.SUBTRACTIVE, assoc=(tree1, 'alarm', alarm1)),
    ]

    state = sim.reset()[attacker_name]
    assert len(model.assets) == 4
    alarm1 = model.get_asset_by_name('Alarm:1')
    tree1 = model.get_asset_by_name('BananaTree:1')
    banana1 = model.get_asset_by_name('Banana:1')
    poison1 = model.get_asset_by_name('Poison:1')
    assert alarm1 and tree1 and banana1 and poison1
    assert tree1.associated_assets == {'alarm': {alarm1}, 'bananas': {banana1}}
    assert alarm1.associated_assets == {'trees': {tree1}}
    assert {n.full_name for n in state.action_surface} == {'BananaTree:1:access'}
    assert sim.sim_state.modification_record == []
    assert_no_dangling_associations(model)


def test_int_dynamic_test_lang12_create_and_destroy_asset_within_one_record(
    int_dynamic_test_lang12_scenario: Scenario,
) -> None:
    """An asset created and later deleted in the same run must round-trip
    cleanly through reset(), including assets that existed beforehand and
    are deleted alongside the newly-created ones.
    """
    sim = DynaMalSimulator.from_scenario(
        int_dynamic_test_lang12_scenario,
        sim_settings=MalSimulatorSettings(compromise_entrypoints_at_start=False),
    )
    attack_graph = int_dynamic_test_lang12_scenario.attack_graph
    model = attack_graph.model
    assert model
    attacker_name = next(iter(int_dynamic_test_lang12_scenario.attacker_settings))
    state = sim.reset()[attacker_name]

    table1 = model.get_asset_by_name('Table:1')
    bowl1 = model.get_asset_by_name('Bowl:1')
    basket1 = model.get_asset_by_name('Basket:1')
    apple1 = model.get_asset_by_name('Apple:1')
    assert table1 and bowl1 and basket1 and apple1
    assert bowl1.associated_assets['apples'] == {apple1}
    assert len(model.assets) == 9

    def step(full_name: str) -> None:
        nonlocal state
        node = attack_graph.get_node_by_full_name(full_name)
        assert node in state.action_surface, (
            f'{full_name} not in action surface: '
            f'{sorted(n.full_name for n in state.action_surface)}'
        )
        state = sim.step({attacker_name: [node]})[attacker_name]

    step('Table:1:access')
    step('Table:1:testNodeAdditions')
    step('Table:1:testAddLeftUnionTwo')
    assert bowl1.associated_assets['apples'] == {
        apple1,
        next(a for a in bowl1.associated_assets['apples'] if a is not apple1),
    }
    assert len(model.assets) == 11

    step('Table:1:testNodeRemovals')
    step('Table:1:testLeftUnionRemoveTwo')
    assert 'apples' not in bowl1.associated_assets
    assert 'apples' not in basket1.associated_assets
    assert len(model.assets) == 8

    state = sim.reset()[attacker_name]
    assert len(model.assets) == 9
    table1 = model.get_asset_by_name('Table:1')
    bowl1 = model.get_asset_by_name('Bowl:1')
    apple1 = model.get_asset_by_name('Apple:1')
    assert table1 and bowl1 and apple1
    assert bowl1.associated_assets['apples'] == {apple1}
    assert apple1.associated_assets == {'container': {bowl1}}
    assert_no_dangling_associations(model)


def test_easy_ransomware_lang_attack_and_reset(
    easy_ransomware_lang_scenario: Scenario,
) -> None:
    """A two-step attack that deletes two pre-existing assets must still
    allow a clean, fully-restoring reset() afterwards.
    """
    sim = DynaMalSimulator.from_scenario(
        easy_ransomware_lang_scenario,
        sim_settings=MalSimulatorSettings(compromise_entrypoints_at_start=False),
    )
    attack_graph = easy_ransomware_lang_scenario.attack_graph
    model = attack_graph.model
    assert model
    attacker_name = next(iter(easy_ransomware_lang_scenario.attacker_settings))
    state = sim.reset()[attacker_name]

    data1 = model.get_asset_by_name('Data:1')
    locked_data1 = model.get_asset_by_name('LockedData:1')
    host1 = model.get_asset_by_name('Host:1')
    ransomware1 = model.get_asset_by_name('Ransomware:1')
    assert data1 and locked_data1 and host1 and ransomware1
    assert data1.associated_assets == {'host': {host1}, 'locked': {locked_data1}}
    assert locked_data1.associated_assets == {
        'plain': {data1},
        'host': {host1},
        'victim': {host1},
    }
    assert host1.associated_assets == {
        'data': {data1, locked_data1},
        'infected': {locked_data1},
        'ransomware': {ransomware1},
    }
    assert len(model.assets) == 4

    def step(full_name: str) -> None:
        nonlocal state
        node = attack_graph.get_node_by_full_name(full_name)
        assert node in state.action_surface
        state = sim.step({attacker_name: [node]})[attacker_name]

    step('Host:1:connect')
    step('Ransomware:1:attack')

    assert model.get_asset_by_name('Data:1') is None
    assert model.get_asset_by_name('LockedData:1') is None
    assert 'data' not in host1.associated_assets
    assert_no_dangling_associations(model)

    state = sim.reset()[attacker_name]
    assert len(model.assets) == 4
    data1 = model.get_asset_by_name('Data:1')
    locked_data1 = model.get_asset_by_name('LockedData:1')
    host1 = model.get_asset_by_name('Host:1')
    ransomware1 = model.get_asset_by_name('Ransomware:1')
    assert data1 and locked_data1 and host1 and ransomware1
    assert data1.associated_assets == {'host': {host1}, 'locked': {locked_data1}}
    assert locked_data1.associated_assets == {
        'plain': {data1},
        'host': {host1},
        'victim': {host1},
    }
    assert host1.associated_assets == {
        'data': {data1, locked_data1},
        'infected': {locked_data1},
        'ransomware': {ransomware1},
    }
    assert {n.full_name for n in state.action_surface} == {'Host:1:connect'}
    assert_no_dangling_associations(model)


def assert_uniform_over_interval(
    counts: np.ndarray, low: int, high: int, alpha: float = 0.05
) -> None:
    """Assert `counts` (integer samples) look uniformly distributed over [low, high]."""
    assert int(counts.min()) >= low and int(counts.max()) <= high, (
        f'counts must be in [{low}, {high}], '
        f'but got min={int(counts.min())}, max={int(counts.max())}'
    )
    observed = np.bincount(counts - low, minlength=high - low + 1)
    expected = np.full(high - low + 1, len(counts) / (high - low + 1))
    _, p_value = chisquare(observed, expected)
    assert p_value > alpha, (
        f'counts do not look uniform over [{low}, {high}]: '
        f'observed={observed.tolist()}, p={p_value}'
    )


def test_rand_multiplicity_scenario(rand_multiplicity_scenario: Scenario) -> None:
    """Test the multiplicity scenario"""
    # Seeded for reproducibility: without a fixed seed this test draws from
    # OS entropy each run, making failures non-reproducible.
    sim = DynaMalSimulator.from_scenario(
        rand_multiplicity_scenario,
        sim_settings=MalSimulatorSettings(seed=1337),
    )
    attack_graph = rand_multiplicity_scenario.attack_graph
    model = attack_graph.model
    assert model is not None

    A = model.get_asset_by_name('A')
    OtherA = model.get_asset_by_name('OtherA')
    assert A and OtherA, 'A and OtherA assets should exist in the model for scenario'

    original_A2B = copy(A.associated_assets.get('children', set()))
    original_OtherA2B = copy(OtherA.associated_assets.get('children', set()))

    num_runs = 1000
    addRandB = np.zeros(num_runs, dtype=int)
    addRandB2OtherA = np.zeros(num_runs, dtype=int)
    linkBfromOtherA = np.zeros(num_runs, dtype=int)
    removeRandB = np.zeros(num_runs, dtype=int)
    unlinkRandB = np.zeros(num_runs, dtype=int)

    def addB(model: Model, A: ModelAsset) -> None:
        """Add a new B asset and associate it to A"""
        new_B = model.add_asset(name=f'B:{model.next_id}', asset_type='B')
        A.add_associated_assets('children', {new_B})

    for i in range(num_runs):
        sim.reset()['TestAttacker']

        sim.step({'TestAttacker': [attack_graph.get_node_by_full_name('A:addRandB')]})[
            'TestAttacker'
        ]
        own_new_Bs = set(A.associated_assets['children'] - original_A2B)
        addRandB[i] = len(own_new_Bs)

        sim.step(
            {'TestAttacker': [attack_graph.get_node_by_full_name('A:addRandB2OtherA')]}
        )['TestAttacker']
        addRandB2OtherA[i] = len(
            OtherA.associated_assets['children'] - original_OtherA2B
        )

        sim.reset()['TestAttacker']
        for _ in range(20):
            addB(model, A)
            addB(model, OtherA)
        pre_linked_Bs = copy(A.associated_assets.get('children', set()))
        sim.step(
            {'TestAttacker': [attack_graph.get_node_by_full_name('A:linkBfromOtherA')]}
        )['TestAttacker']
        shared_new_Bs = copy(A.associated_assets.get('children', set())) - pre_linked_Bs
        assert all(
            b in OtherA.associated_assets.get('children', set()) for b in shared_new_Bs
        )
        linkBfromOtherA[i] = len(shared_new_Bs)

        sim.reset()['TestAttacker']
        for _ in range(20):
            addB(model, A)
            addB(model, OtherA)
        pre_removed_Bs = copy(A.associated_assets.get('children', set()))
        sim.step(
            {'TestAttacker': [attack_graph.get_node_by_full_name('A:removeRandB')]}
        )['TestAttacker']
        removed_Bs = pre_removed_Bs - A.associated_assets.get('children', set())
        assert all(b not in set(model.assets.values()) for b in removed_Bs)
        removeRandB[i] = len(removed_Bs)

        sim.reset()['TestAttacker']
        for _ in range(20):
            addB(model, A)
            addB(model, OtherA)
        pre_unlinked_Bs = copy(OtherA.associated_assets.get('children', set()))
        sim.step(
            {'TestAttacker': [attack_graph.get_node_by_full_name('A:unlinkRandB')]}
        )['TestAttacker']
        unlinked_Bs = pre_unlinked_Bs - OtherA.associated_assets.get('children', set())
        assert all(b in set(model.assets.values()) for b in unlinked_Bs)
        assert all(
            b not in OtherA.associated_assets.get('children', set())
            for b in unlinked_Bs
        )
        unlinkRandB[i] = len(unlinked_Bs)

    alpha = 0.05 / 5
    assert_uniform_over_interval(addRandB, 4, 10, alpha=alpha)
    assert_uniform_over_interval(addRandB2OtherA, 8, 12, alpha=alpha)
    assert_uniform_over_interval(linkBfromOtherA, 2, 5, alpha=alpha)
    assert_uniform_over_interval(removeRandB, 2, 10, alpha=alpha)
    assert_uniform_over_interval(unlinkRandB, 1, 3, alpha=alpha)


def test_no_memory_leak_on_teardown() -> None:
    """Make sure a DynaMalSimulator, its attack graph and its agent states
    can all be fully garbage collected once the last external reference to
    the simulator (and the scenario it was built from) is dropped.

    If any of these objects survive a gc.collect(), something (a reference
    cycle involving a __del__, a callback registered on a global/module
    level object, a thread, etc.) is keeping the simulator alive - i.e. a
    memory leak.
    """
    scenario = Scenario.load_from_file('tests/testdata/scenarios/wiper_scenario.yml')
    sim = DynaMalSimulator.from_scenario(scenario)
    states = sim.reset()
    attacker_name = next(iter(states.keys()))
    attacker_state = states[attacker_name]
    action = next(iter(attacker_state.action_surface))
    states = sim.step({attacker_name: [action]})

    sim_ref = weakref.ref(sim)
    graph_ref = weakref.ref(sim.sim_state.attack_graph)
    attacker_state_ref = weakref.ref(attacker_state)

    del sim, scenario, states, attacker_state, action
    gc.collect()

    assert sim_ref() is None, (
        'DynaMalSimulator instance was not garbage collected after all '
        'external references were dropped - likely a memory leak'
    )
    assert graph_ref() is None, (
        'AttackGraph held by the simulator was not garbage collected - '
        'likely a memory leak'
    )
    assert attacker_state_ref() is None, (
        'Attacker state held by the simulator was not garbage collected - '
        'likely a memory leak'
    )


def test_no_memory_growth_over_repeated_simulations() -> None:
    """Run many independent DynaMalSimulator instances to completion and
    make sure the number of live objects tracked by the garbage collector
    does not keep growing.

    Unbounded growth here would indicate objects leaking across simulator
    instances, e.g. via a module-level cache/registry that keeps
    accumulating entries instead of being cleaned up.
    """
    scenario_file = next(
        Path('tests/testdata/scenarios/dynamal_example_scenarios').rglob('*.yml')
    )

    def run_once() -> None:
        scenario = Scenario.load_from_file(str(scenario_file))
        attacker_name = next(iter(scenario.attacker_settings.keys()))
        scenario.attacker_settings[attacker_name].policy = RandomAgent
        sim_settings = MalSimulatorSettings(
            ttc_mode=TTCMode.PRE_SAMPLE,
            compromise_entrypoints_at_start=False,
        )
        sim = DynaMalSimulator.from_scenario(scenario, sim_settings=sim_settings)
        run_simulation(sim)

    # Logging increases memory usage, so disable it
    logging.disable(logging.CRITICAL)
    try:
        # Warm-up runs so one-off allocations (imports, lru_caches, interned
        # objects, ...) don't get mistaken for a leak.
        for _ in range(3):
            run_once()
        gc.collect()
        baseline = len(gc.get_objects())

        for _ in range(50):
            run_once()
        gc.collect()
        after_growth = len(gc.get_objects())
    finally:
        logging.disable(logging.NOTSET)

    growth = after_growth - baseline
    assert growth < max(baseline * 0.05, 500), (
        f'Number of live objects grew by {growth} ({baseline} -> {after_growth}) '
        'after repeatedly running independent simulations - this suggests a '
        'memory leak in DynaMalSimulator'
    )


def test_inherited_query_methods_follow_graph_mutated_by_model_effects() -> None:
    """PORTING_NOTES.md §6 Phase B6: `DynaMalSimulator` inherits all of
    `MalSimulator`'s public query methods unmodified, but unlike the base
    class its graph (and therefore the native-computed `GraphState` those
    methods read) grows and shrinks mid-episode. Make sure the inherited
    methods give correct answers for a node that only exists after a step.
    """
    sim = DynaMalSimulator.from_scenario(
        'tests/testdata/scenarios/wiper_scenario.yml',
        sim_settings=MalSimulatorSettings(ttc_mode=TTCMode.PRE_SAMPLE, seed=1),
    )
    attacker = 'WiperController'
    infect = sim.get_node(full_name='InfectedDevice:infect')
    with pytest.raises(LookupError):
        sim.get_node(full_name='Wiper-7:activate')

    sim.step({attacker: [infect]})

    new = sim.get_node(full_name='Wiper-7:activate')
    assert sim.get_node(node_id=new.id) is new
    assert sim.node_ttc_value(new) == sim.node_ttc_value(new, attacker) == 1.0
    assert sim.node_is_necessary(new)
    assert not sim.node_is_blocked(new)
    assert not sim.node_is_blocked(new.full_name)
    assert not sim.node_is_compromised(new)
    assert not sim.node_is_enabled_defense(new)
    assert sim.node_is_actionable(new, attacker)
    assert sim.node_reward(new, attacker) == 0.0
    assert sim.node_is_traversable(set(sim.compromised_nodes), new)
    assert infect in sim.compromised_nodes

    sim.step({attacker: [new]})
    assert sim.node_is_compromised(new)
    assert new in sim.compromised_nodes
    assert not sim.done()

    # Reset restores the pristine graph: the added node is gone again.
    sim.reset()
    with pytest.raises(LookupError):
        sim.get_node(full_name='Wiper-7:activate')
    assert not sim.compromised_nodes - set(sim.agent_states[attacker].performed_nodes)


def _restorable_object_scenario(
    attacker: dict[str, Any], defender: dict[str, Any] | None = None
) -> Scenario:
    """`dynamic_remove_add.mal` scenario whose pristine model has `Object:1`
    associated to `Start:0`: stepping `Start:0:remove` deletes `Object:1`,
    and `reset()` restores it with freshly generated nodes.
    `attacker`/`defender` are merged into the agents' settings dicts.
    """
    lang_file = str(
        Path(__file__).parent / 'testdata' / 'langs' / 'dynamic_remove_add.mal'
    )
    agents: dict[str, Any] = {
        'Attacker': {'type': 'attacker', 'policy': None, **attacker},
    }
    if defender is not None:
        agents['Defender'] = {'type': 'defender', 'policy': None, **defender}
    return Scenario.from_dict(
        {
            'lang_file': lang_file,
            'model': {
                'metadata': {
                    'name': 'restore_model',
                    'langVersion': '1.0.0',
                    'langID': 'org.mal-lang.dynamicRemoveAdd',
                    'malVersion': '0.1.0-SNAPSHOT',
                    'MAL-Toolbox Version': '2.10.0',
                    'info': '',
                },
                'assets': {
                    0: {
                        'name': 'Start:0',
                        'type': 'Start',
                        'associated_assets': {'objects': {1: 'Object:1'}},
                    },
                    1: {
                        'name': 'Object:1',
                        'type': 'Object',
                        'associated_assets': {'start': {0: 'Start:0'}},
                    },
                },
            },
            'agents': agents,
        }
    )


def test_reset_restored_nodes_keep_rule_settings() -> None:
    """Nodes removed by one episode's model effects and restored by
    `reset()` must get their rule-derived agent settings in the next
    episode (PORTING_NOTES.md §12, C5/dyna-reset entry).

    `Start:0:remove` deletes `Object:1` (and its attack steps). The next
    `reset()` restores it with freshly generated nodes, so per-node settings
    must be resolved against the restored graph, not the mutated one.
    """
    scenario = _restorable_object_scenario(
        attacker={'entry_points': ['Start:0:access']},
        defender={'observable_steps': {'by_asset_type': {'Object': ['addStart']}}},
    )
    sim = DynaMalSimulator.from_scenario(scenario)
    attack_graph = scenario.attack_graph
    model = attack_graph.model
    assert model

    def defender_observes_add_start() -> bool:
        """One episode: reach and compromise `Object:1:addStart`, return
        whether the defender observed it."""
        sim.reset()
        sim.step({'Attacker': [attack_graph.get_node_by_full_name('Start:0:add')]})
        add_start = attack_graph.get_node_by_full_name('Object:1:addStart')
        states = sim.step({'Attacker': [add_start]})
        assert add_start in states['Attacker'].performed_nodes
        defender_state = states['Defender']
        assert isinstance(defender_state, DefenderState)
        return add_start in defender_state.observed_nodes

    assert defender_observes_add_start()

    # An episode whose model effect removes `Object:1`.
    sim.reset()
    sim.step({'Attacker': [attack_graph.get_node_by_full_name('Start:0:remove')]})
    assert model.get_asset_by_name('Object:1') is None

    # After reset `Object:1` is back and must still be observable.
    assert defender_observes_add_start()


@pytest.mark.parametrize(
    'entry_points',
    [
        ['Start:0:access', 'Object:1:addStartAssoc'],
        # Multiple alternative entry-point sets, sampled at reset.
        [['Start:0:access', 'Object:1:addStartAssoc']],
    ],
)
def test_reset_re_resolves_entry_points_and_goals_on_restored_nodes(
    entry_points: list[Any],
) -> None:
    """Entry points and goals on an asset that one episode's model effects
    removed must point at the restored (regenerated) nodes after `reset()`,
    instead of the removed ones (PORTING_NOTES.md §12, C5/dyna-reset entry).
    """
    scenario = _restorable_object_scenario(
        attacker={'entry_points': entry_points, 'goals': ['Object:1:addStart']}
    )
    sim = DynaMalSimulator.from_scenario(scenario)
    attack_graph = scenario.attack_graph
    model = attack_graph.model
    assert model

    sim.reset()
    sim.step({'Attacker': [attack_graph.get_node_by_full_name('Start:0:remove')]})
    assert model.get_asset_by_name('Object:1') is None

    state = sim.reset()['Attacker']
    assert isinstance(state, AttackerState)
    entry_point = attack_graph.get_node_by_full_name('Object:1:addStartAssoc')
    goal = attack_graph.get_node_by_full_name('Object:1:addStart')
    assert entry_point in state.entry_points
    assert entry_point in state.performed_nodes
    assert state.goals == frozenset({goal})
    settings = sim.agent_settings['Attacker']
    assert isinstance(settings, AttackerSettings)
    assert settings.goals == frozenset({goal})

    # The episode runs to the restored goal.
    sim.step({'Attacker': [attack_graph.get_node_by_full_name('Start:0:add')]})
    state = sim.step({'Attacker': [goal]})['Attacker']
    assert isinstance(state, AttackerState)
    assert goal in state.performed_nodes
    assert sim.done()
