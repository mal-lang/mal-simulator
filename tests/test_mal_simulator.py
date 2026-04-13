"""Test MalSimulator class"""

from __future__ import annotations
from collections.abc import MutableSet, Set
import random
from typing import TYPE_CHECKING, Any

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from maltoolbox.language import LanguageGraphAttackStep
from malsim.config.node_property_rule import NodePropertyRule
from malsim.config.sim_settings import AttackSurfaceSettings
from malsim.mal_simulator import (
    MalSimulator,
    MalSimulatorSettings,
    DefenderState,
    AttackerState,
    TTCMode,
    RewardMode,
)
from malsim.mal_simulator.attacker_step import attacker_is_terminated, attacker_step
from malsim.mal_simulator.agent_states import (
    attacker_states,
    defender_states,
)
from malsim.mal_simulator.defender_step import defender_is_terminated, defender_step
from malsim.mal_simulator import TTCDist
from malsim import Scenario, run_simulation

from malsim.config.agent_settings import AttackerSettings, DefenderSettings

from dataclasses import asdict
import numpy as np
import pytest

from malsim.mal_simulator.graph_utils import node_is_blocked
from malsim.policies.random_agent import RandomAgent
from .conftest import get_node

if TYPE_CHECKING:
    from maltoolbox.language import LanguageGraph
    from maltoolbox.model import Model


def test_init_with_agent_settings(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_points = frozenset(
        {attack_graph.get_node_by_full_name('OS App:localConnect')}
    )
    goals = frozenset({attack_graph.get_node_by_full_name('OS App:fullAccess')})

    agent_settings = (
        AttackerSettings(
            name='Attacker1', entry_points=entry_points, goals=goals, policy=RandomAgent
        ),
        DefenderSettings(name='Defender1', policy=RandomAgent),
    )
    sim = MalSimulator(attack_graph, agents=agent_settings)

    # Make sure the agents were registered
    assert sim.agent_states.keys() == {'Attacker1', 'Defender1'}
    assert sim.agent_reward_by_name('Attacker1') == 0.0
    assert sim.agent_reward_by_name('Defender1') == 0.0
    assert sim.alive_agents == {'Attacker1', 'Defender1'}


def test_reset(corelang_lang_graph: LanguageGraph, model: Model) -> None:
    """Make sure attack graph is reset"""
    attack_graph = AttackGraph(corelang_lang_graph, model)

    agent_entry_point = 'OS App:localConnect'
    attacker_name = 'testagent'

    sim = MalSimulator(
        attack_graph,
        sim_settings=MalSimulatorSettings(seed=10),
        agents=(
            AttackerSettings(
                name=attacker_name,
                entry_points=frozenset(
                    {attack_graph.get_node_by_full_name(agent_entry_point)}
                ),
            ),
        ),
    )

    viability_before = {
        n.full_name: v for n, v in sim.sim_state.graph_state.viability_per_node.items()
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
    sim = MalSimulator(
        attack_graph, sim_settings=MalSimulatorSettings(seed=10), agents=()
    )
    for node, viable in sim.sim_state.graph_state.viability_per_node.items():
        # viability is the same after reset
        assert viability_before[node.full_name] == viable

    for node, necessary in sim.sim_state.graph_state.necessity_per_node.items():
        # necessity is the same after reset
        assert necessity_before[node.full_name] == necessary


def test_register_agent_action_surface(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph, agents=(DefenderSettings(name='defender1'),))

    agent_name = 'defender1'

    defender_state = sim.agent_states[agent_name]
    action_surface = defender_state.action_surface
    for node in action_surface:
        assert node not in sim.sim_state.graph_state.pre_enabled_defenses


def test_simulator_actionable_action_surface(model: Model) -> None:
    scenario = Scenario(
        lang_file='tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar',
        model=model,
        agents=(
            AttackerSettings(
                name='Attacker1',
                entry_points=frozenset({'OS App:fullAccess'}),
                actionable_steps=NodePropertyRule(
                    by_asset_type={
                        'Application': {
                            'attemptRead': True,
                            'successfulRead': True,
                            'read': True,
                            'notPresent': True,
                        }
                    }
                ),
            ),
            DefenderSettings(
                name='Defender',
                actionable_steps=NodePropertyRule(
                    by_asset_type={
                        'Application': {
                            'attemptRead': True,
                            'successfulRead': True,
                            'read': True,
                            'notPresent': True,
                        }
                    }
                ),
            ),
        ),
    )
    sim = MalSimulator.from_scenario(scenario)

    # Only three nodes ever in action surface
    states = sim.step({'Attacker1': ['OS App:attemptRead']})
    states = sim.step({'Attacker1': ['OS App:successfulRead']})
    assert states['Attacker1'].action_surface == {sim.get_node('OS App:read')}

    # Only Application:notPresent defenses in defenders action surface
    assert states['Defender'].action_surface == {
        sim.get_node('OS App:notPresent'),
        sim.get_node('Program 2:notPresent'),
    }


def test_simulator_initialize_agents(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    """Test _initialize_agents"""

    scenario = Scenario.load_from_file('tests/testdata/scenarios/simple_scenario.yml')
    sim = MalSimulator.from_scenario(scenario)

    # Register the agents
    attacker_name = 'Attacker1'
    defender_name = 'Defender1'

    agent_states = sim.reset()

    assert set(agent_states.keys()) == {attacker_name, defender_name}


def test_get_agents() -> None:
    """Test _get_attacker_agents and _get_defender_agents"""

    scenario = Scenario.load_from_file('tests/testdata/scenarios/simple_scenario.yml')
    sim = MalSimulator.from_scenario(scenario)
    sim.reset()

    assert list(attacker_states(sim.agent_states)) == ['Attacker1']
    assert list(defender_states(sim.agent_states)) == ['Defender1']


def test_attacker_step(corelang_lang_graph: LanguageGraph, model: Model) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_point = get_node(attack_graph, 'OS App:fullAccess')

    sim = MalSimulator(
        attack_graph,
        agents=(
            AttackerSettings(name='attacker', entry_points=frozenset({entry_point})),
        ),
    )

    attacker_name = 'attacker'

    sim.reset()

    attacker_agent = sim._agent_states[attacker_name]
    assert isinstance(attacker_agent, AttackerState)

    # Can not attack the notPresent step
    defense_step = get_node(attack_graph, 'OS App:notPresent')
    actions, _ = attacker_step(sim.sim_state, attacker_agent, [defense_step], sim.rng)

    assert not actions

    attack_step = get_node(attack_graph, 'OS App:attemptRead')
    actions, _ = attacker_step(sim.sim_state, attacker_agent, [attack_step], sim.rng)
    assert actions == [attack_step]


def test_defender_step(corelang_lang_graph: LanguageGraph, model: Model) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph, agents=(DefenderSettings(name='defender'),))

    defender_name = 'defender'

    sim.reset()

    defender_agent = sim._agent_states[defender_name]
    assert isinstance(defender_agent, DefenderState)

    defense_step = get_node(attack_graph, 'OS App:notPresent')
    enabled, made_unviable = defender_step(
        sim.sim_state,
        defender_agent,
        [defense_step],
    )
    assert enabled == [defense_step]
    assert made_unviable

    # Can not defend attack_step
    attack_step = get_node(attack_graph, 'OS App:attemptUseVulnerability')
    assert attack_step
    enabled, made_unviable = defender_step(
        sim.sim_state,
        defender_agent,
        [attack_step],
    )
    assert enabled == []
    assert not made_unviable


def test_node_full_names_to_simulator(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)

    # Give nodes as full names - important to test
    asset_name = 'OS App'
    entry_point = 'fullAccess'
    attempt_read = 'attemptRead'
    access_network_and_conn = 'accessNetworkAndConnections'

    entry_point_full_name = 'OS App:fullAccess'
    access_network_and_conn_full_name = 'OS App:accessNetworkAndConnections'

    rewards = NodePropertyRule(
        by_asset_name={
            asset_name: {
                entry_point: 10.0,
                attempt_read: 100.0,
                access_network_and_conn: 50.4,
            }
        }
    )
    observability_per_node = NodePropertyRule(
        by_asset_name={
            asset_name: {
                entry_point: True,
                attempt_read: False,
                access_network_and_conn: True,
            }
        }
    )

    defender_name = 'Test Defender'
    attacker_name = 'Test Attacker'
    sim = MalSimulator(
        attack_graph,
        sim_settings=MalSimulatorSettings(),
        agents=(
            DefenderSettings(
                name=defender_name,
                observable_steps=observability_per_node,
                rewards=rewards,
            ),
            AttackerSettings(
                name=attacker_name,
                entry_points=frozenset(
                    {attack_graph.get_node_by_full_name(entry_point_full_name)}
                ),
                rewards=rewards,
            ),
        ),
    )

    states = sim.reset()

    states = sim.step({attacker_name: [asset_name + ':' + attempt_read]})
    defender_state = states[defender_name]
    assert isinstance(defender_state, DefenderState)
    # Make sure observability worked
    assert not defender_state.step_observed_nodes

    states = sim.step({attacker_name: [access_network_and_conn_full_name]})
    defender_state = states[defender_name]
    assert isinstance(defender_state, DefenderState)
    assert {n.full_name for n in defender_state.observed_nodes} == {
        entry_point_full_name,
        access_network_and_conn_full_name,
    }

    # Make sure rewards worked
    as_dict = rewards.to_dict()
    assert as_dict
    by_asset_name = as_dict['by_asset_name']
    assert by_asset_name
    assert sim.agent_reward(states[attacker_name]) and sim.agent_reward(
        states[attacker_name]
    ) == sum(by_asset_name[asset_name].values())
    assert sim.agent_reward(states[attacker_name]) and sim.agent_reward(
        states[defender_name]
    ) == -sum(by_asset_name[asset_name].values())


def test_attacker_step_rewards_cumulative(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)

    entry_point = get_node(attack_graph, 'OS App:fullAccess')
    attempt_read = get_node(attack_graph, 'OS App:attemptRead')
    access_network_and_conn = get_node(
        attack_graph, 'OS App:accessNetworkAndConnections'
    )

    rewards = NodePropertyRule(
        by_asset_name={
            'OS App': {
                'fullAccess': 10.0,
                'attemptRead': 100.0,
                'accessNetworkAndConnections': 50.4,
            }
        }
    )
    attacker_name = 'Test Attacker'
    sim = MalSimulator(
        attack_graph,
        sim_settings=MalSimulatorSettings(),
        agents=(
            AttackerSettings(
                name=attacker_name,
                entry_points=frozenset({entry_point}),
                rewards=rewards,
                reward_mode=RewardMode.CUMULATIVE,
            ),
        ),
    )

    sim.reset()

    states = sim.step({attacker_name: [attempt_read]})
    attacker_state = states[attacker_name]
    assert sim.agent_reward(attacker_state) == (
        rewards[entry_point] + rewards[attempt_read]
    )

    states = sim.step({attacker_name: [access_network_and_conn]})
    attacker_state = states[attacker_name]
    assert sim.agent_reward(attacker_state) == (
        rewards[entry_point] + rewards[attempt_read] + rewards[access_network_and_conn]
    )

    # Recording of the simulation
    assert sim.recording == {
        1: {attacker_name: [attempt_read]},
        2: {attacker_name: [access_network_and_conn]},
    }


def test_is_traversable(corelang_lang_graph: LanguageGraph, model: Model) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_point = attack_graph.get_node_by_full_name('OS App:fullAccess')
    attacker_name = 'Test Attacker'
    sim = MalSimulator(
        attack_graph,
        agents=(
            AttackerSettings(
                name=attacker_name,
                entry_points=frozenset({entry_point}),
            ),
        ),
    )

    attacker_state = sim.agent_states[attacker_name]

    # Compromise all or steps it can
    or_steps = list(attacker_state.action_surface)
    while or_steps:
        attacker_state = sim.step({attacker_name: or_steps})[attacker_name]
        or_steps = [
            n for n in sim.agent_states[attacker_name].action_surface if n.type == 'or'
        ]
    assert isinstance(attacker_state, AttackerState)
    children_of_reached_nodes = set()
    for n in attacker_state.performed_nodes:
        children_of_reached_nodes |= n.children

    for node in sim.sim_state.attack_graph.nodes.values():
        if node in attacker_state.settings.entry_points:
            # Unclear traversability of entry points
            continue

        if node in children_of_reached_nodes:
            if node.type == 'and' and not sim.node_is_traversable(
                attacker_state.performed_nodes, node
            ):
                assert not all(
                    p in attacker_state.performed_nodes
                    for p in node.parents
                    if p.type in ('or', 'and')
                ) or sim.node_is_blocked(node)
            if node.type == 'or' and not sim.node_is_traversable(
                attacker_state.performed_nodes, node
            ):
                assert sim.node_is_blocked(node)
        else:
            assert not sim.node_is_traversable(attacker_state.performed_nodes, node)


def test_not_initial_compromise_entrypoints(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_point = get_node(attack_graph, 'OS App:fullAccess')
    attacker_name = 'Test Attacker'
    sim = MalSimulator(
        attack_graph,
        sim_settings=MalSimulatorSettings(compromise_entrypoints_at_start=False),
        agents=(
            AttackerSettings(name=attacker_name, entry_points=frozenset({entry_point})),
        ),
    )

    attacker_state = sim.reset()[attacker_name]

    # No performed nodes, action surface is only the entrypoint
    assert attacker_state.performed_nodes == set()
    assert attacker_state.action_surface == {entry_point}

    # Step through entrypoint adds it to performed nodes and extends the action surface
    attacker_state = sim.step({attacker_name: [entry_point]})[attacker_name]
    assert attacker_state.performed_nodes == {entry_point}
    assert attacker_state.action_surface == set(entry_point.children)


def test_not_initial_compromise_entrypoints_unviable_step(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    attacker_name = 'Test Attacker'
    defender_name = 'Test Defender'
    sim = MalSimulator(
        attack_graph,
        sim_settings=MalSimulatorSettings(compromise_entrypoints_at_start=False),
        agents=(
            AttackerSettings(
                name=attacker_name,
                entry_points=frozenset(
                    {attack_graph.get_node_by_full_name('OS App:fullAccess')}
                ),
            ),
            DefenderSettings(name=defender_name),
        ),
    )

    attacker_state = sim.reset()[attacker_name]

    # Step should not succeed if defender defended the entrypoint
    attacker_state = sim.step(
        {attacker_name: ['OS App:fullAccess'], defender_name: ['OS App:notPresent']}
    )[attacker_name]

    node = attack_graph.get_node_by_full_name('OS App:fullAccess')
    assert (
        node_is_blocked(sim.sim_state, node)
        or node not in attacker_state.action_surface
    )

    assert attacker_state.performed_nodes == set()
    assert attacker_state.action_surface == set()


def test_is_compromised(corelang_lang_graph: LanguageGraph, model: Model) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_point = get_node(attack_graph, 'OS App:fullAccess')
    attacker_name = 'Test Attacker'
    sim = MalSimulator(
        attack_graph,
        agents=(
            AttackerSettings(name=attacker_name, entry_points=frozenset({entry_point})),
        ),
    )
    attacker_state = sim.agent_states[attacker_name]

    # Compromise all or steps it can
    or_steps = list(attacker_state.action_surface)
    while or_steps:
        attacker_state = sim.step({attacker_name: or_steps})[attacker_name]
        or_steps = [
            n for n in sim.agent_states[attacker_name].action_surface if n.type == 'or'
        ]
    assert isinstance(attacker_state, AttackerState)

    for node in sim.sim_state.attack_graph.nodes.values():
        if node in attacker_state.performed_nodes:
            # Unclear traversability of entry points
            assert sim.node_is_compromised(node)
        else:
            assert not sim.node_is_compromised(node)


def test_get_node(corelang_lang_graph: LanguageGraph, model: Model) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph, agents=())

    assert isinstance(sim.get_node('OS App:fullAccess'), AttackGraphNode)

    with pytest.raises(LookupError):
        sim.get_node('nonExisting:node')


def test_simulation_done(corelang_lang_graph: LanguageGraph, model: Model) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_point = get_node(attack_graph, 'OS App:fullAccess')
    defender_name = 'Test defender'
    attacker_name = 'Test Attacker'
    sim = MalSimulator(
        attack_graph,
        agents=(
            AttackerSettings(name=attacker_name, entry_points=frozenset({entry_point})),
            DefenderSettings(name=defender_name),
        ),
    )
    states = sim.reset()

    for _ in range(10):
        # Do nothing 10 steps
        states = sim.step({})

    attacker_state = states[attacker_name]
    assert isinstance(attacker_state, AttackerState)
    defender_state = states[defender_name]
    assert isinstance(defender_state, DefenderState)

    assert not sim.done()  # simulation is done because truncated
    assert not defender_is_terminated(sim._agent_states)  # not terminated
    assert not attacker_is_terminated(attacker_state)  # not terminated


def test_simulation_terminations(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    defender_name = 'Test defender'
    attacker_name = 'Test Attacker'
    entry_point = get_node(attack_graph, 'OS App:fullAccess')
    sim = MalSimulator(
        attack_graph,
        agents=(
            DefenderSettings(name=defender_name),
            AttackerSettings(name=attacker_name, entry_points=frozenset({entry_point})),
        ),
    )

    states = sim.reset()
    attacker_state = states[attacker_name]
    assert isinstance(attacker_state, AttackerState)

    while attacker_state.action_surface:
        # Perform entire action surface of attacker
        states = sim.step({attacker_name: list(attacker_state.action_surface)})
        attacker_state = states[attacker_name]

    attacker_state = states[attacker_name]
    assert isinstance(attacker_state, AttackerState)
    defender_state = states[defender_name]
    assert isinstance(defender_state, DefenderState)

    assert sim.done()  # simulation is done because all agents terminated
    assert defender_is_terminated(sim._agent_states)
    assert attacker_is_terminated(attacker_state)


def test_attacker_step_rewards_one_off(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)

    entry_point = get_node(attack_graph, 'OS App:fullAccess')
    attempt_read = get_node(attack_graph, 'OS App:attemptRead')
    access_network_and_conn = get_node(
        attack_graph, 'OS App:accessNetworkAndConnections'
    )

    assert entry_point.model_asset

    rewards = NodePropertyRule(
        by_asset_type={
            entry_point.model_asset.type: {
                entry_point.name: 10.0,
                attempt_read.name: 100.0,
                access_network_and_conn.name: 50.4,
            }
        }
    )

    attacker_name = 'Test Attacker'

    sim = MalSimulator(
        attack_graph,
        sim_settings=MalSimulatorSettings(),
        agents=(
            AttackerSettings(
                name=attacker_name,
                reward_mode=RewardMode.ONE_OFF,
                entry_points=frozenset({entry_point}),
                rewards=rewards,
            ),
        ),
    )

    agent_states = sim.reset()

    sim.step({attacker_name: [attempt_read]})
    state1 = agent_states[attacker_name]
    assert isinstance(state1, AttackerState)
    assert sim.agent_reward(state1) == rewards[attempt_read] - float(
        len(state1.step_attempted_nodes)
    )

    sim.step({attacker_name: [access_network_and_conn]})
    state2 = agent_states[attacker_name]
    assert isinstance(state2, AttackerState)
    assert sim.agent_reward(state2) == rewards[access_network_and_conn] - float(
        len(state2.step_attempted_nodes)
    )


def test_attacker_step_rewards_expected_ttc(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_point = get_node(attack_graph, 'OS App:fullAccess')

    # Set some random rewards for each node
    rng = np.random.default_rng(22)
    rewards = NodePropertyRule(
        by_asset_name={
            n.type: {
                x.name: rng.random() * 100
                for x in filter(lambda x: x.type == n.type, attack_graph.nodes.values())
            }
            for n in attack_graph.attack_steps
        }
    )
    attacker_name = 'Test Attacker'
    sim = MalSimulator(
        attack_graph,
        sim_settings=MalSimulatorSettings(),
        agents=(
            AttackerSettings(
                name=attacker_name,
                entry_points=frozenset({entry_point}),
                reward_mode=RewardMode.EXPECTED_TTC,
                rewards=rewards,
            ),
        ),
    )

    state = sim.reset()[attacker_name]

    while not sim.done():
        # Run a simulation and make sure rewards are as they should be
        state = sim.step({attacker_name: list(state.action_surface)})[attacker_name]
        assert isinstance(state, AttackerState)

        # Penalized with expected ttc value (since ttc mode is disabled)
        ttc_penalty = sum(
            TTCDist.from_node(node).expected_value if node.ttc else 0.0
            for node in state.step_performed_nodes
        )
        # Rewarded by node rewards
        reward = sum(rewards.value(node, 0) for node in state.step_performed_nodes)
        assert sim.agent_reward(state) == reward - ttc_penalty


def test_defender_step_rewards_cumulative(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)

    entry_point = get_node(attack_graph, 'OS App:fullAccess')
    attempt_read = get_node(attack_graph, 'OS App:attemptRead')
    not_present = get_node(attack_graph, 'OS App:notPresent')
    access_network_and_conn = get_node(
        attack_graph, 'OS App:accessNetworkAndConnections'
    )
    assert entry_point.model_asset

    attacker_name = 'Test Attacker'  # To be able to step
    defender_name = 'defender'
    rewards = NodePropertyRule(
        by_asset_name={
            entry_point.model_asset.name: {
                not_present.name: 100,
                entry_point.name: 10.0,
                attempt_read.name: 105.0,
                access_network_and_conn.name: 35.04,
            }
        }
    )

    sim = MalSimulator(
        attack_graph,
        sim_settings=MalSimulatorSettings(),
        agents=(
            DefenderSettings(name=defender_name, rewards=rewards),
            AttackerSettings(name=attacker_name, entry_points=frozenset({entry_point})),
        ),
    )

    agent_states = sim.reset()

    sim.step({attacker_name: [attempt_read]})
    assert sim.agent_reward(agent_states[defender_name]) == -(
        rewards[entry_point] + rewards[attempt_read]
    )

    sim.step({attacker_name: [access_network_and_conn]})
    assert sim.agent_reward(agent_states[defender_name]) == -(
        rewards[entry_point] + rewards[attempt_read] + rewards[access_network_and_conn]
    )


def test_defender_step_rewards_one_off(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)

    entry_point = get_node(attack_graph, 'OS App:fullAccess')
    attempt_read = get_node(attack_graph, 'OS App:attemptRead')
    not_present = get_node(attack_graph, 'OS App:notPresent')
    access_network_and_conn = get_node(
        attack_graph, 'OS App:accessNetworkAndConnections'
    )

    assert entry_point.model_asset

    rewards = NodePropertyRule(
        by_asset_type={
            entry_point.model_asset.type: {
                not_present.name: 100,
                entry_point.name: 10.0,
                attempt_read.name: 105.0,
                access_network_and_conn.name: 35.04,
            }
        }
    )
    defender_name = 'defender'
    attacker_name = 'Test Attacker'  # To be able to step

    sim = MalSimulator(
        attack_graph,
        sim_settings=MalSimulatorSettings(),
        agents=(
            DefenderSettings(
                name=defender_name, reward_mode=RewardMode.ONE_OFF, rewards=rewards
            ),
            AttackerSettings(name=attacker_name, entry_points=frozenset({entry_point})),
        ),
    )

    sim.reset()

    states = sim.step({attacker_name: [attempt_read]})
    defender_state = states[defender_name]
    assert sim.agent_reward(defender_state) == -rewards[attempt_read]

    states = sim.step({attacker_name: [access_network_and_conn]})
    defender_state = states[defender_name]
    assert sim.agent_reward(defender_state) == -rewards[access_network_and_conn]


# TODO: Some of the assert values in this test have changed when updating the
# attacker logic. We should check to see if the behaviour is the new behaviour
# is correct.
def test_agent_state_views_simple(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_point = get_node(attack_graph, 'OS App:fullAccess')

    mss = MalSimulatorSettings(seed=13, ttc_mode=TTCMode.PER_STEP_SAMPLE)
    attacker_name = 'attacker'
    defender_name = 'defender'
    # Create simulator and register agents
    sim = MalSimulator(
        attack_graph,
        sim_settings=mss,
        agents=(
            AttackerSettings(name=attacker_name, entry_points=frozenset({entry_point})),
            DefenderSettings(name=defender_name),
        ),
    )

    # Evaluate the agent state views after reset
    state_views = sim.agent_states
    entry_point = get_node(attack_graph, 'OS App:fullAccess')

    pre_enabled_defenses = set(sim.sim_state.graph_state.pre_enabled_defenses)

    asv = state_views['attacker']
    dsv = state_views['defender']

    assert asv.step_performed_nodes == {entry_point}
    assert dsv.step_performed_nodes == pre_enabled_defenses

    assert asv.performed_nodes == {entry_point}
    assert dsv.performed_nodes == pre_enabled_defenses

    assert len(asv.action_surface) == 6
    assert {n.full_name for n in dsv.action_surface} == {
        'Credentials:10:notPhishable',  # Disabled in lang
        'Data:5:notPresent',  # Disabled in lang
        'Credentials:9:unique',  # Enabled in lang, Disabled in model
        'User:12:noPasswordReuse',  # Enabled in lang, Disabled in model
        'Group:13:notPresent',  # Disabled in lang
        'IDPS 1:notPresent',  # Disabled in lang
        'OS App:supplyChainAuditing',  # Not set in lang, Disabled by default
        'OS App:notPresent',  # Disabled in lang
        'Credentials:6:unique',  # Enabled in lang, Disabled in model
        'Program 2:notPresent',  # Disabled in lang
        'Credentials:9:notPhishable',  # Disabled in lang
        'Program 2:supplyChainAuditing',  # Not set in lang, Disabled by default
        'User:12:securityAwareness',  # Not set in lang, Disabled by default
        'Identity:11:notPresent',  # Disabled in lang
        'Credentials:7:notPhishable',  # Disabled in lang
        'Identity:8:notPresent',  # Disabled in lang
        'Credentials:7:unique',  # Enabled in lang, Disabled in model
        'Credentials:6:notPhishable',  # Disabled in lang
        'IDPS 1:supplyChainAuditing',  # Not set in lang, Disabled by default
        'SoftwareVulnerability:4:notPresent',  # Disabled in lang
        'Program 1:supplyChainAuditing',  # Not set in lang, Disabled by default
    }

    assert len(dsv.step_action_surface_additions) == len(dsv.action_surface)
    assert len(asv.step_action_surface_additions) == len(asv.action_surface)

    assert asv.step_action_surface_removals == set()
    assert dsv.step_action_surface_removals == set()

    # Save all relvant nodes in variables
    program2_not_present = get_node(attack_graph, 'Program 2:notPresent')
    os_app_attempt_deny = get_node(attack_graph, 'OS App:attemptDeny')
    os_app_success_deny = get_node(attack_graph, 'OS App:successfulDeny')
    os_app_not_present = get_node(attack_graph, 'OS App:notPresent')
    os_app_access_netcon = get_node(attack_graph, 'OS App:accessNetworkAndConnections')
    os_app_spec_access = get_node(attack_graph, 'OS App:specificAccess')

    # Evaluate the agent state views after stepping through an attack step and
    # a defense that will not impact it in any way

    state_views = sim.step(
        {'defender': [program2_not_present], 'attacker': [os_app_attempt_deny]}
    )
    asv = state_views['attacker']
    dsv = state_views['defender']
    assert isinstance(asv, AttackerState)
    assert isinstance(dsv, DefenderState)

    assert asv.step_performed_nodes == {os_app_attempt_deny}
    assert dsv.step_performed_nodes == {program2_not_present}

    assert asv.performed_nodes == {os_app_attempt_deny, entry_point}
    assert dsv.performed_nodes == pre_enabled_defenses | {program2_not_present}

    assert asv.step_action_surface_additions == {os_app_success_deny}
    assert dsv.step_action_surface_additions == set()
    assert asv.step_action_surface_removals == {os_app_attempt_deny}
    assert os_app_attempt_deny not in asv.action_surface
    assert dsv.step_action_surface_removals == {program2_not_present}
    assert dsv.step_compromised_nodes == {os_app_attempt_deny}
    assert len(dsv.step_unviable_nodes) == 48

    # Go through an attack step that already has some children in the attack
    # surface(OS App:accessNetworkAndConnections in this case)
    assert os_app_access_netcon in asv.action_surface
    state_views = sim.step({'defender': [], 'attacker': [os_app_spec_access]})
    asv = state_views['attacker']
    dsv = state_views['defender']
    assert isinstance(asv, AttackerState)
    assert isinstance(dsv, DefenderState)

    assert asv.step_performed_nodes == {os_app_spec_access}
    assert dsv.step_performed_nodes == set()
    assert os_app_access_netcon in asv.action_surface
    assert os_app_access_netcon not in asv.step_action_surface_additions
    assert dsv.step_action_surface_additions == set()
    assert asv.step_action_surface_removals == {os_app_spec_access}
    assert os_app_spec_access not in asv.action_surface
    assert dsv.step_action_surface_removals == set()
    assert dsv.step_compromised_nodes == {os_app_spec_access}
    assert len(dsv.step_unviable_nodes) == 0

    # Evaluate the agent state views after stepping through an attack step and
    # a defense that would prevent it from occurring
    state_views = sim.step(
        {'defender': [os_app_not_present], 'attacker': [os_app_success_deny]}
    )
    asv = state_views['attacker']
    dsv = state_views['defender']
    assert isinstance(asv, AttackerState)
    assert isinstance(dsv, DefenderState)

    assert asv.step_performed_nodes == set()
    assert dsv.step_performed_nodes == {os_app_not_present}
    assert asv.step_action_surface_additions == set()
    assert dsv.step_action_surface_additions == set()
    assert {a.full_name for a in asv.step_action_surface_removals} == {
        'OS App:accessNetworkAndConnections',
        'OS App:attemptApplicationRespondConnectThroughData',
        'OS App:attemptAuthorizedApplicationRespondConnectThroughData',
        'OS App:attemptModify',
        'OS App:attemptRead',
        'OS App:specificAccessDelete',
        'OS App:specificAccessModify',
        # 'OS App:bypassContainerization',
        'OS App:specificAccessRead',
        'OS App:successfulDeny',
        'Program 1:localConnect',
        'Program 2:localConnect',
        # 'IDPS 1:localConnect',
    }
    assert dsv.step_action_surface_removals == {os_app_not_present}
    assert dsv.step_compromised_nodes == set()
    assert len(dsv.step_unviable_nodes) == 53

    # Recording of the simulation
    assert sim.recording == {
        1: {'defender': [program2_not_present], 'attacker': [os_app_attempt_deny]},
        2: {'defender': [], 'attacker': [os_app_spec_access]},
        3: {'defender': [os_app_not_present], 'attacker': []},
    }


def test_step_attacker_defender_action_surface_updates() -> None:
    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_scenario.yml'
    )

    sim = MalSimulator.from_scenario(scenario)
    attacker_agent_id = 'Attacker1'
    defender_agent_id = 'Defender1'

    states = sim.agent_states

    attacker_agent = states[attacker_agent_id]
    defender_agent = states[defender_agent_id]

    # Run step() with action crafted in test
    attacker_step = sim.get_node('User:3:compromise')
    assert attacker_step in attacker_agent.action_surface

    defender_step = sim.get_node('User:3:notPresent')
    assert defender_step in defender_agent.action_surface

    actions = {
        attacker_agent.name: [attacker_step],
        defender_agent.name: [defender_step],
    }

    states = sim.step(actions)
    attacker_agent = states[attacker_agent_id]
    defender_agent = states[defender_agent_id]

    # Make sure no nodes added to action surface
    assert not attacker_agent.step_action_surface_additions
    assert not defender_agent.step_action_surface_additions

    # Make sure the steps are removed from the action surfaces
    assert attacker_step in attacker_agent.step_action_surface_removals
    assert defender_step in defender_agent.step_action_surface_removals

    assert attacker_step not in attacker_agent.action_surface
    assert defender_step not in defender_agent.action_surface


def test_default_simulator_default_settings_eviction() -> None:
    """Test attacker node eviction using MalSimulatorSettings default"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_scenario.yml'
    )
    sim = MalSimulator.from_scenario(scenario)

    attacker_agent_id = 'Attacker1'
    defender_agent_id = 'Defender1'

    attacker_agent = sim.agent_states[attacker_agent_id]
    defender_agent = sim.agent_states[defender_agent_id]

    # Get a step to compromise and its defense parent
    user_3_compromise = get_node(sim.sim_state.attack_graph, 'User:3:compromise')
    user_3_compromise_defense = next(
        n for n in user_3_compromise.parents if n.type == 'defense'
    )
    assert user_3_compromise not in attacker_agent.performed_nodes
    assert user_3_compromise_defense not in defender_agent.performed_nodes

    # First let the attacker compromise User:3:compromise
    actions = {attacker_agent_id: [user_3_compromise], defender_agent_id: []}
    states = sim.step(actions)

    # Check that the compromise happened and that the defense did not
    assert user_3_compromise in states[attacker_agent_id].performed_nodes
    assert user_3_compromise_defense not in states[defender_agent_id].performed_nodes

    # Now let the defender defend, and the attacker waits
    actions = {attacker_agent_id: [], defender_agent_id: [user_3_compromise_defense]}
    states = sim.step(actions)

    # Verify defense was performed and attacker NOT kicked out
    assert user_3_compromise in states[attacker_agent_id].performed_nodes
    assert user_3_compromise_defense in states[defender_agent_id].performed_nodes


def test_simulator_false_positives() -> None:
    """Create a simulator with false positives"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_fp_fn_scenario.yml',
        sim_settings=MalSimulatorSettings(seed=30),
    )
    scenario.defender_settings['defender'].false_negative_rates = None

    sim = MalSimulator.from_scenario(scenario)
    run_simulation(sim)

    defender_state = sim.agent_states['defender']
    assert isinstance(defender_state, DefenderState)
    attacker_state = sim.agent_states['Attacker1']
    assert isinstance(attacker_state, AttackerState)

    # Should be false positive in defender state
    assert len(defender_state.observed_nodes) > len(defender_state.compromised_nodes)


def test_simulator_false_positives_after_done() -> None:
    """Create a simulator with false positives"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_fp_fn_scenario.yml',
        sim_settings=MalSimulatorSettings(seed=30),
    )
    scenario.defender_settings['defender'].false_negative_rates = None

    sim = MalSimulator.from_scenario(scenario)
    run_simulation(sim)
    assert sim.done()

    # Simulation is done, but we can still observe false positives
    false_positives: Set[AttackGraphNode] = set()
    for _ in range(100):
        states = sim.step({})
        defender_state = states['defender']
        assert isinstance(defender_state, DefenderState)
        false_positives |= defender_state.observed_nodes

    assert false_positives


def test_simulator_false_positives_reset() -> None:
    """Create a simulator with false positives"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_fp_fn_scenario.yml',
        sim_settings=MalSimulatorSettings(seed=9),
    )
    scenario.defender_settings['defender'].false_negative_rates = None

    sim = MalSimulator.from_scenario(scenario)
    defender_state = sim.reset()['defender']
    assert isinstance(defender_state, DefenderState)
    # Should be false positive in defender state even on reset
    assert len(defender_state.observed_nodes) > len(defender_state.compromised_nodes)


def test_simulator_false_negatives() -> None:
    """Create a simulator with false negatives"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_fp_fn_scenario.yml',
        sim_settings=MalSimulatorSettings(seed=100),
    )
    scenario.defender_settings['defender'].false_positive_rates = None
    sim = MalSimulator.from_scenario(
        scenario,
    )
    run_simulation(sim)

    defender_state = sim.agent_states['defender']
    assert isinstance(defender_state, DefenderState)
    attacker_state = sim.agent_states['Attacker1']
    assert isinstance(attacker_state, AttackerState)

    # Should be false negatives in defender state
    assert len(defender_state.observed_nodes) < len(defender_state.compromised_nodes)


def test_simulator_no_fpr_fnr() -> None:
    """Create a simulator with no fnr fpr"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_fp_fn_scenario.yml',
        sim_settings=MalSimulatorSettings(seed=100),
    )

    sim = MalSimulator.from_scenario(scenario)

    sim = MalSimulator.from_scenario(scenario)
    run_simulation(sim)

    defender_state = sim.agent_states['defender']
    assert isinstance(defender_state, DefenderState)
    attacker_state = sim.agent_states['Attacker1']
    assert isinstance(attacker_state, AttackerState)

    # No false positives or negatives
    assert defender_state.compromised_nodes == attacker_state.performed_nodes


def test_simulator_ttcs() -> None:
    """Create a simulator and check TTCs, then reset and check TTCs again"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_scenario.yml',
        sim_settings=MalSimulatorSettings(ttc_mode=TTCMode.PER_STEP_SAMPLE),
    )
    sim = MalSimulator.from_scenario(scenario)

    # host_0_notPresent = sim.get_node("Host:0:notPresent")
    # host_0_auth = sim.get_node("Host:0:authenticate")
    # host_0_connect = sim.get_node("Host:0:connect")
    # host_0_access = sim.get_node("Host:0:access")
    # host_1_notPresent = sim.get_node("Host:1:notPresent")
    # host_1_auth = sim.get_node("Host:1:authenticate")
    # host_1_connect = sim.get_node("Host:1:connect")
    # host_1_access = sim.get_node("Host:1:access")
    # data_2_notPresent = sim.get_node("Data:2:notPresent")
    # data_2_read = sim.get_node("Data:2:read")
    # data_2_modify = sim.get_node("Data:2:modify")
    # user_3_notPresent = sim.get_node("User:3:notPresent")
    # user_3_compromise = sim.get_node("User:3:compromise")
    # user_3_phishing = sim.get_node("User:3:phishing")
    # network_3_access = sim.get_node("Network:3:access")

    # expected_bernoullis = {
    #     host_0_notPresent: math.inf,
    #     host_0_auth: 1.0,
    #     host_0_connect: 1.0,
    #     host_0_access: 1.0,
    #     host_1_notPresent: math.inf,
    #     host_1_auth: 1.0,
    #     host_1_connect: 1.0,
    #     host_1_access: 1.0,
    #     data_2_notPresent: math.inf,
    #     data_2_read: 1.0,
    #     data_2_modify: 1.0,
    #     user_3_notPresent: math.inf,
    #     user_3_compromise: 1.0,
    #     user_3_phishing: 1.0,
    #     network_3_access: 1.0
    # }

    assert not sim.sim_state.graph_state.impossible_attack_steps
    assert not sim.sim_state.graph_state.pre_enabled_defenses

    sim.reset()

    assert not sim.sim_state.graph_state.impossible_attack_steps
    assert not sim.sim_state.graph_state.pre_enabled_defenses


def test_simulator_multiple_attackers() -> None:
    """
    Have two attackers from different entrypoints perform their
    full action surface every step. Defender is passive.
    """

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_scenario.yml',
        sim_settings=MalSimulatorSettings(seed=100),
    )

    scenario.agent_settings = [
        *scenario.agent_settings,
        AttackerSettings(
            name='Attacker2',
            entry_points=frozenset(
                {scenario.attack_graph.get_node_by_full_name('Network:3:access')}
            ),
        ),
    ]

    sim = MalSimulator.from_scenario(
        scenario,
    )

    states = sim.reset()

    while not sim.done():
        states = sim.step(
            {
                'Attacker1': sorted(
                    states['Attacker1'].action_surface, key=lambda n: n.id
                ),
                'Attacker2': sorted(
                    states['Attacker2'].action_surface, key=lambda n: n.id
                ),
                'Defender1': [],
            }
        )

    # Verify that it is possible to select more than one action
    # for more than one agent
    assert sim.recording == {
        1: {
            'Defender1': [],
            'Attacker1': [sim.get_node('User:3:compromise')],
            'Attacker2': [
                sim.get_node('Host:0:connect'),
                sim.get_node('Host:1:connect'),
            ],
        },
        2: {
            'Defender1': [],
            'Attacker1': [sim.get_node('Host:0:authenticate')],
            'Attacker2': [],
        },
        3: {
            'Defender1': [],
            'Attacker1': [sim.get_node('Host:0:access')],
            'Attacker2': [],
        },
        4: {
            'Defender1': [],
            'Attacker1': [
                sim.get_node('Data:2:read'),
                sim.get_node('Data:2:modify'),
                sim.get_node('Network:3:access'),
            ],
            'Attacker2': [],
        },
        5: {
            'Defender1': [],
            'Attacker1': [sim.get_node('Host:1:connect')],
            'Attacker2': [],
        },
    }


def test_simulator_multiple_defenders() -> None:
    """
    Should only be possible to have more than one defender
    if use forces it. It makes no sense.
    """

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_scenario.yml',
        sim_settings=MalSimulatorSettings(seed=100),
    )

    scenario.agent_settings = [
        *scenario.agent_settings,
        DefenderSettings(name='Defender2'),
    ]

    sim = MalSimulator.from_scenario(
        scenario,
    )

    states = sim.reset()

    while not sim.done():
        states = sim.step(
            {
                'Defender1': sorted(
                    states['Defender1'].action_surface, key=lambda n: n.id
                ),
                'Defender2': sorted(
                    states['Defender2'].action_surface, key=lambda n: n.id
                ),
            }
        )

    assert sim.recording == {
        1: {
            'Defender1': [
                sim.get_node('Host:0:notPresent'),
                sim.get_node('Host:1:notPresent'),
                sim.get_node('Data:2:notPresent'),
                sim.get_node('User:3:notPresent'),
            ],
            'Defender2': [
                sim.get_node('Host:0:notPresent'),
                sim.get_node('Host:1:notPresent'),
                sim.get_node('Data:2:notPresent'),
                sim.get_node('User:3:notPresent'),
            ],
            'Attacker1': [],
        }
    }


def test_simulator_attacker_override_ttcs_state() -> None:
    """
    Have an attacker that overrides ttcs
    """

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/ttc_lang_scenario_override_ttcs.yml',
        sim_settings=MalSimulatorSettings(seed=100, ttc_mode=TTCMode.PRE_SAMPLE),
    )

    sim = MalSimulator.from_scenario(
        scenario,
    )
    states = sim.reset()

    bad_attacker_settings = sim.agent_settings['BadAttacker']
    assert isinstance(bad_attacker_settings, AttackerSettings)
    assert bad_attacker_settings.ttc_dists is not None
    bad_attacker_state = states['BadAttacker']
    assert isinstance(bad_attacker_state, AttackerState)

    assert bad_attacker_settings.ttc_dists
    assert set(bad_attacker_settings.ttc_dists.per_node(scenario.attack_graph)) == {
        'ComputerC:easyConnect',
        'ComputerA:easyConnect',
        'ComputerD:easyConnect',
        'ComputerB:easyConnect',
    }

    assert bad_attacker_state.ttc_values
    assert {n.full_name: v for n, v in bad_attacker_state.ttc_values.items()} == {
        'ComputerA:easyConnect': 7.4543483865750755,
        'ComputerB:easyConnect': 15.661809565462281,
        'ComputerC:easyConnect': 5.434482312470439,
        'ComputerD:easyConnect': 35.14904078865208,
    }
    assert {n.full_name for n in bad_attacker_state.impossible_steps} == {
        'ComputerB:easyConnect'
    }

    good_attacker_state = states['GoodAttacker']
    good_attacker_settings = sim.agent_settings['GoodAttacker']
    assert isinstance(good_attacker_state, AttackerState)
    assert isinstance(good_attacker_settings, AttackerSettings)
    assert not good_attacker_settings.ttc_dists
    assert not good_attacker_state.impossible_steps


def test_simulator_attacker_override_ttcs_step() -> None:
    """
    Have an attacker that overrides ttcs step
    """

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/ttc_lang_scenario_override_ttcs.yml',
        sim_settings=MalSimulatorSettings(
            seed=100,
            ttc_mode=TTCMode.PRE_SAMPLE,
            attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
        ),
    )
    sim = MalSimulator.from_scenario(scenario)
    max_iter = 1000

    states = sim.reset()
    attacker_name = 'GoodAttacker'
    attacker_state = None
    while not sim.agent_is_terminated(attacker_name):
        # Good attacker should be fast
        attacker_state = states[attacker_name]
        agent_conf = scenario.attacker_settings[attacker_name]
        assert agent_conf.agent is not None
        next_action = agent_conf.agent.get_next_action(attacker_state)
        states = sim.step({attacker_name: [next_action]})
        if attacker_state.iteration > max_iter:
            break
    assert attacker_state is not None
    assert attacker_state.iteration == 8

    states = sim.reset()
    attacker_name = 'BadAttacker'
    attacker_state = None
    while not sim.agent_is_terminated(attacker_name):
        # Bad attacker should be slow
        attacker_state = states[attacker_name]
        agent_conf = scenario.attacker_settings[attacker_name]
        assert agent_conf.agent is not None
        next_action = agent_conf.agent.get_next_action(attacker_state)
        states = sim.step({attacker_name: [next_action]})
        if attacker_state.iteration > max_iter:
            break
    assert attacker_state is not None
    assert attacker_state.iteration == 15


def test_simulator_seed_setting() -> None:
    """Test that the seed setting works"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/socialEngineering_scenario.yml',
        sim_settings=MalSimulatorSettings(
            uncompromise_untraversable_steps=False,
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            seed=100,
            attack_surface=AttackSurfaceSettings(
                skip_compromised=True,
                skip_unviable=True,
                skip_unnecessary=False,
            ),
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
        ),
    )
    sim = MalSimulator.from_scenario(scenario)

    ttcs = []
    for _ in range(100):
        state = sim.reset()['Attacker1']
        node = sim.get_node('Human:successfulSocialEngineering')
        assert node in state.action_surface
        state = sim.step({'Attacker1': [node]})['Attacker1']
        node = sim.get_node('Human:socialEngineering')
        assert node in state.action_surface
        state = sim.step({'Attacker1': [node]})['Attacker1']
        node = sim.get_node('Human:unsafeUserActivity')
        assert node in state.action_surface
        i = 0
        while node in state.action_surface:
            state = sim.step({'Attacker1': [node]})['Attacker1']
            i += 1
        ttcs.append(i)

    ttc_array = np.array(ttcs)
    variance = ttc_array.var()
    assert variance > 0, 'Variance is 0, which means the TTCs are not random'


def test_settings_serialization() -> None:
    """Test that the settings serialization works"""
    settings = MalSimulatorSettings(
        ttc_mode=TTCMode.PER_STEP_SAMPLE,
    )
    deserialized_settings = MalSimulatorSettings(**asdict(settings))
    assert deserialized_settings == settings


def test_simulator_picklable(tmp_path: Any) -> None:
    import pickle

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_observability_scenario.yml'
    )
    sim = MalSimulator.from_scenario(scenario)

    pickle_path = tmp_path / 'sim.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(sim, f)

    with open(pickle_path, 'rb') as f:
        restored: MalSimulator = pickle.load(f)

    assert type(restored) is type(sim)
    assert restored.sim_settings == sim.sim_settings

    # Compare attack graph dicts
    assert (
        restored.sim_state.attack_graph._to_dict()
        == sim.sim_state.attack_graph._to_dict()
    )


def test_scenario_advanced_agent_settings() -> None:
    """Verify:
    - scenario loads correctly using new format
    - agent settings are parsed
    - NodePropertyRule objects are created
    - Applies correctly in simulator methods
    """

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_scenario_advanced_agent_settings.yml'
    )
    sim = MalSimulator.from_scenario(scenario)

    defender_name = 'Defender1'
    attacker_name = 'Attacker1'

    # Start with defender
    assert (
        sim.node_false_negative_rate(sim.get_node('Host:0:access'), defender_name)
        == 0.5
    )
    assert (
        sim.node_false_negative_rate(sim.get_node('Data:2:read'), defender_name) == 0.0
    )

    assert (
        sim.node_false_positive_rate(sim.get_node('Host:0:connect'), defender_name)
        == 0.5
    )
    assert (
        sim.node_false_positive_rate(sim.get_node('Data:2:read'), defender_name) == 0.0
    )

    assert sim.node_is_actionable(sim.get_node('Host:0:notPresent'), defender_name)
    assert not sim.node_is_actionable(sim.get_node('Data:2:notPresent'), defender_name)

    assert sim.node_is_observable(sim.get_node('Host:0:connect'), defender_name)
    assert not sim.node_is_observable(sim.get_node('Data:2:read'), defender_name)

    assert sim.node_reward(sim.get_node('Host:0:notPresent'), defender_name) == 100.0
    assert sim.node_reward(sim.get_node('Host:0:access'), defender_name) == 0.0

    # Attacker
    assert sim.node_is_actionable(sim.get_node('Host:0:authenticate'), attacker_name)
    assert not sim.node_is_actionable(sim.get_node('Data:2:read'), attacker_name)

    assert sim.node_reward(sim.get_node('Host:0:authenticate'), attacker_name) == 1000
    assert sim.node_reward(sim.get_node('Host:0:access'), attacker_name) == 4.0


def test_active_defenses() -> None:
    """Verify that active defenses are correctly applied"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/credentials_scenario.yml',
        sim_settings=MalSimulatorSettings(
            ttc_mode=TTCMode.DISABLED,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface=AttackSurfaceSettings(
                skip_unnecessary=False,
            ),
            compromise_entrypoints_at_start=True,
        ),
    )
    sim = MalSimulator.from_scenario(scenario)

    assert len(sim.sim_state.graph_state.pre_enabled_defenses) == 2
    assert (
        sim.get_node('Creds:notGuessable')
        in sim.sim_state.graph_state.pre_enabled_defenses
    )
    assert (
        sim.get_node('Creds:notDisclosed')
        in sim.sim_state.graph_state.pre_enabled_defenses
    )


def test_compromise_order() -> None:
    """Verify that the compromise order is correctly recorded"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/socialEngineering_scenario.yml'
    )
    sim = MalSimulator.from_scenario(scenario)
    states = sim.reset()

    attacker_record = (
        {0: set(states['Attacker1'].performed_nodes_order[0])}
        if len(states['Attacker1'].performed_nodes_order) > 0
        else {}
    )
    defender_record = (
        {0: set(states['Defender1'].performed_nodes_order[0])}
        if len(states['Defender1'].performed_nodes_order) > 0
        else {}
    )

    for i in range(1, 101):
        actions: dict[str, list[AttackGraphNode]] = {}
        if len(states['Attacker1'].action_surface) == 0 or random.random() < 0.5:
            actions['Attacker1'] = []
        else:
            actions['Attacker1'] = [
                random.choice(list(states['Attacker1'].action_surface))
            ]

        if len(states['Defender1'].action_surface) == 0 or random.random() < 0.3:
            actions['Defender1'] = []
        else:
            actions['Defender1'] = [
                random.choice(list(states['Defender1'].action_surface))
            ]
        states = sim.step(actions)
        if len(sim.recording[i]['Attacker1']) > 0:
            attacker_record[i] = set(sim.recording[i]['Attacker1'])
        if len(sim.recording[i]['Defender1']) > 0:
            defender_record[i] = set(sim.recording[i]['Defender1'])
        if sim.done():
            break

    for i in range(max(max(attacker_record.keys()), max(defender_record.keys())) + 1):
        if i in attacker_record and i in states['Attacker1'].performed_nodes_order:
            assert attacker_record[i] == states['Attacker1'].performed_nodes_order[i], (
                f'Attacker record does not match simulator at time {i}'
            )
        elif i in attacker_record:
            assert False, (
                f'Attacker record has steps for time {i} but simulator does not'
            )
        elif i in states['Attacker1'].performed_nodes_order:
            assert False, (
                f'Simulator has steps for time {i} but attacker record does not'
            )

    for i in range(100):
        if i in defender_record and i in states['Defender1'].performed_nodes_order:
            assert defender_record[i] == states['Defender1'].performed_nodes_order[i], (
                f'Defender record does not match simulator at time {i}'
            )
        elif i in defender_record:
            assert False, (
                f'Defender record has steps for time {i} but simulator does not'
            )
        elif i in states['Defender1'].performed_nodes_order:
            assert False, (
                f'Simulator has steps for time {i} but defender record does not'
            )

    assert states['Attacker1'].performed_nodes_order == attacker_record
    assert states['Defender1'].performed_nodes_order == defender_record


def test_actions_effects() -> None:
    """Verify actions and effects works as intended"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/actions_effects_scenario.yml',
        sim_settings=MalSimulatorSettings(
            ttc_mode=TTCMode.DISABLED,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface=AttackSurfaceSettings(
                skip_unnecessary=False,
            ),
            compromise_entrypoints_at_start=True,
        ),
    )

    sim = MalSimulator.from_scenario(scenario)
    run_simulation(sim)
    for i in sorted(sim.recording.keys()):
        node_list = sim.recording[i]['Attacker']
        action_nodes = [node for node in node_list if node.causal_mode == 'action']
        effect_nodes = [node for node in node_list if node.causal_mode == 'effect']

        def get_all_effect_children(
            node: LanguageGraphAttackStep,
        ) -> Set[LanguageGraphAttackStep]:
            children: MutableSet[LanguageGraphAttackStep] = set()
            for child in node.children:
                if child.causal_mode == 'effect':
                    children.add(child)
                else:
                    children |= get_all_effect_children(child)
            return children

        all_effect_children: MutableSet[LanguageGraphAttackStep] = set()
        for action_node in action_nodes:
            action_lg_step = action_node.lg_attack_step
            all_effect_children |= get_all_effect_children(action_lg_step)

        for effect_node in effect_nodes:
            assert effect_node.lg_attack_step in all_effect_children
