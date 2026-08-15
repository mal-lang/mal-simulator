"""Test DynaMalSimulator class"""

from __future__ import annotations
import gc
import logging
import random
import weakref
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from maltoolbox.attackgraph import AttackGraph
from maltoolbox.language.language_graph_model_effect import ModelEffectType
from malsim.config.sim_settings import TTCMode
from malsim.dyna_mal_simulator.model_effects import (
    _apply_model_effect,
)
from malsim.dyna_mal_simulator.process_assoc_traversal import traverse_association_chain
from malsim.dyna_mal_simulator.simulator_state import AssetOp, AssocOp
from malsim.mal_simulator import (
    MalSimulatorSettings,
    run_simulation,
)
from malsim import Scenario

from malsim.config.agent_settings import AttackerSettings

import numpy as np
import pytest

from malsim.mal_simulator.graph_utils import node_is_blocked
from malsim.policies.attackers.searchers import BreadthFirstAttacker, DepthFirstAttacker
from malsim.policies.attackers.ttc_soft_min import TTCSoftMinAttacker
from malsim.policies.random_agent import RandomAgent
from malsim.dyna_mal_simulator import DynaMalSimulator

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


def test_assoc_traversal(wiperLang_attack_graph: AttackGraph) -> None:
    """Test the association traversal logic"""
    rng = np.random.default_rng(42)
    # Check Device:infect step has self as base, so the traversal should return
    # the same asset
    infect_step = wiperLang_attack_graph.get_node_by_full_name('InfectedDevice:infect')
    assert infect_step.model_asset, 'Infect step should have a model asset'
    assert infect_step.additive_model_effects, 'Infect step should have model effects'
    terminating_assets = traverse_association_chain(
        {infect_step.model_asset}, infect_step.additive_model_effects[0].base, rng
    )
    assert terminating_assets == {infect_step.model_asset}, (
        'Infect step has self as base'
    )
    assert infect_step.additive_model_effects

    # Check that target of Device:infect doesn't resolve to any assets
    resolved_assets = traverse_association_chain(
        terminating_assets,
        infect_step.additive_model_effects[0].targets[0].assoc_traversal,
        rng,
    )
    assert len(resolved_assets) == 0, 'Infect step should not have any targets yet'

    # Add Wiper asset into model
    model = wiperLang_attack_graph.model
    assert model is not None, 'Model should be available in attack graph'
    wiper = model.add_asset('Wiper', 'Wiper', max(model.assets.keys()) + 1)
    infected_device = model.get_asset_by_name('InfectedDevice')
    assert infected_device
    wiper.add_associated_assets('victim', {infected_device})
    wiperLang_attack_graph.regenerate_graph()

    # Check that Device:infect step has Wiper as target, so the traversal should
    # return the Wiper asset
    terminating_assets = traverse_association_chain(
        terminating_assets,
        infect_step.additive_model_effects[0].targets[0].assoc_traversal,
        rng,
    )
    assert terminating_assets == {wiper}, 'Infect step should have Wiper as target'

    # Check that Wiper:exfiltrate step has the InfectedData as base
    exfiltrate_step = wiperLang_attack_graph.get_node_by_full_name('Wiper:exfiltrate')
    assert exfiltrate_step.model_asset
    assert exfiltrate_step.additive_model_effects
    terminating_assets = traverse_association_chain(
        {exfiltrate_step.model_asset},
        exfiltrate_step.additive_model_effects[0].base,
        rng,
    )
    assert terminating_assets == {model.get_asset_by_name('InfectedData')}, (
        'Exfiltrate step should have InfectedData as base'
    )

    # Check that target of Wiper:exfiltrate doesn't resolve to any assets,
    # this is because the InfectedData is not yet associated to the C2Server
    resolved_assets = traverse_association_chain(
        terminating_assets,
        exfiltrate_step.additive_model_effects[0].targets[0].assoc_traversal,
        rng,
    )
    assert len(resolved_assets) == 0, (
        'Exfiltrate step should not have any targets yet, because InfectedData is '
        'not associated to C2Server'
    )

    infected_data = model.get_asset_by_name('InfectedData')
    assert infected_data
    c2_server = model.get_asset_by_name('C2Server')
    assert c2_server
    infected_data.add_associated_assets('node', {c2_server})

    # This target refers to an additive assoc op,
    # so the instigating asset is the asset where the step is defined
    terminating_assets = traverse_association_chain(
        {exfiltrate_step.model_asset},
        exfiltrate_step.additive_model_effects[0].targets[0].assoc_traversal,
        rng,
    )
    assert terminating_assets == {infected_data}, (
        'Exfiltrate step target resolve to InfectedData after association to C2Server'
    )


def test_apply_model_effect(wiperLang_attack_graph: AttackGraph) -> None:
    """Test that applying a model effect modifies the model as expected"""
    rng = np.random.default_rng(42)

    model = wiperLang_attack_graph.model
    assert model
    infected_device = model.get_asset_by_name('InfectedDevice')
    assert infected_device
    assert 'malware' not in infected_device.associated_assets, (
        'InfectedDevice should not have malware associated before applying model effect'
    )
    infect_step = wiperLang_attack_graph.get_node_by_full_name('InfectedDevice:infect')
    assert infect_step.additive_model_effects
    model_effect_record = _apply_model_effect(
        infect_step, infect_step.additive_model_effects[0], model, rng
    )
    assert 'malware' in infected_device.associated_assets, (
        'InfectedDevice should have malware associated after applying model effect'
    )
    assert any(
        assoc_op.assoc
        == (
            model.get_asset_by_name('InfectedDevice'),
            'malware',
            model.get_asset_by_name('Wiper'),
        )
        for assoc_op in model_effect_record
        if isinstance(assoc_op, AssocOp)
    )

    wiper = next(asset for asset in model.assets.values() if asset.type == 'Wiper')
    assert infected_device in wiper.associated_assets['victim'], (
        'Wiper should have InfectedDevice as victim after applying model effect'
    )
    assert any(
        asset_op.asset == wiper
        for asset_op in model_effect_record
        if isinstance(asset_op, AssetOp)
    )

    infected_data = model.get_asset_by_name('InfectedData')
    assert infected_data
    assert infected_data.associated_assets.get('node') == {infected_device}, (
        'InfectedData should be associated to InfectedDevice before applying '
        'model effect'
    )
    wiperLang_attack_graph.regenerate_graph()
    exfiltrate_step = wiperLang_attack_graph.get_node_by_full_name('Wiper:exfiltrate')
    assert exfiltrate_step.additive_model_effects
    model_effect_record = _apply_model_effect(
        exfiltrate_step, exfiltrate_step.additive_model_effects[0], model, rng
    )
    c2_server = model.get_asset_by_name('C2Server')
    assert c2_server
    assert infected_data.associated_assets.get('node') == {
        infected_device,
        c2_server,
    }, (
        'InfectedData should be associated to InfectedDevice and C2Server after '
        'applying model effect'
    )
    assert any(
        assoc_op.assoc == (c2_server, 'data', infected_data)
        for assoc_op in model_effect_record
        if isinstance(assoc_op, AssocOp)
    )


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
    modification_record = _apply_model_effect(
        fuzz_step,
        fuzz_step.subtractive_model_effects[0],
        model,
        np.random.default_rng(),
    )
    removed_assets = {
        op.asset
        for op in modification_record
        if isinstance(op, AssetOp) and op.type == ModelEffectType.SUBTRACTIVE
    }
    removed_associations = {
        op.assoc
        for op in modification_record
        if isinstance(op, AssocOp) and op.type == ModelEffectType.SUBTRACTIVE
    }
    attack_graph.partially_regenerate_graph(
        removed_assets=removed_assets, removed_associations=removed_associations
    )
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
    wiper = model.get_asset_by_name('Wiper')
    assert wiper
    assert wiper is not None, (
        'Wiper asset should exist after applying model effect of infect step'
    )
    assert wiper in infected_device.associated_assets['malware'], (
        'Wiper should be associated with InfectedDevice after applying model '
        'effect of infect step'
    )

    activate_step = wiperLang_attack_graph.get_node_by_full_name('Wiper:activate')
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
    exfiltrate_step = wiperLang_attack_graph.get_node_by_full_name('Wiper:exfiltrate')
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
    action = attack_graph.get_node_by_full_name('Object:addStart')
    sim.step({'Attacker': [action]})

    action = attack_graph.get_node_by_full_name('Object:addStartAssoc')
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
