"""Test MalSimulator class"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

from maltoolbox.attackgraph import AttackGraph
from malsim.mal_simulator import MalSimulator, MalSimAttackerState
from malsim.envs import MalSimVectorizedObsEnv
from malsim.scenario.scenario import Scenario

from ..conftest import get_node

if TYPE_CHECKING:
    from maltoolbox.language import LanguageGraph
    from maltoolbox.model import Model


def test_create_blank_observation(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    """Make sure blank observation contains correct default values"""

    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimVectorizedObsEnv(MalSimulator(attack_graph))

    num_objects = len(attack_graph.nodes)
    blank_observation = sim._create_blank_observation()

    assert len(blank_observation['is_observable']) == num_objects
    for state in blank_observation['is_observable']:
        # Default is that all nodes are observable,
        # unless anything else is given through its extras field
        assert state == 1

    assert len(blank_observation['observed_state']) == num_objects
    for state in blank_observation['observed_state']:
        assert state == -1  # This is the default (which we get in blank observation)

    assert len(blank_observation['remaining_ttc']) == num_objects
    for ttc in blank_observation['remaining_ttc']:
        assert ttc == 0  # TTC is currently always 0 no matter what

    # asset_type_index points us to an asset type in sim._index_to_asset_type
    # the index where asset_type_index lies on will point to an attack step id in
    # sim._index_to_id
    # The type we get from sim._index_to_asset_type[asset_type_index]
    # should be the same as the asset type of attack step with id index in attack graph
    assert len(blank_observation['asset_type']) == num_objects
    for index, asset_type_index in enumerate(blank_observation['asset_type']):
        # Note: offset is decremented from asset_type_index
        expected_type = sim._index_to_asset_type[asset_type_index]
        node = sim.index_to_node(index)
        assert node.lg_attack_step.asset.name == expected_type

    # asset_id on index X in blank_observation['asset_id']
    # should be the same as the id of the asset of attack step X
    assert len(blank_observation['asset_id']) == num_objects
    for index, expected_asset_id in enumerate(blank_observation['asset_id']):
        node = sim.index_to_node(index)
        assert node.model_asset and node.model_asset.id == expected_asset_id

    assert len(blank_observation['step_name']) == num_objects

    expected_num_edges = sum(
        [1 for step in attack_graph.nodes.values() for child in step.children]
        +
        # We expect all defenses again (reversed)
        [
            1
            for step in attack_graph.nodes.values()
            for child in step.children
            if step.type == 'defense'
        ]
    )
    assert len(blank_observation['attack_graph_edges']) == expected_num_edges


def test_create_blank_observation_deterministic(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    """Make sure blank observation is deterministic with seed given"""

    attack_graph = AttackGraph(corelang_lang_graph, model)
    os_app_fa = get_node(attack_graph, 'OS App:fullAccess')

    sim = MalSimVectorizedObsEnv(MalSimulator(attack_graph))
    sim.register_attacker('test_attacker', {os_app_fa})
    sim.register_defender('test_defender')

    obs1, _ = sim.reset(seed=123)
    obs2, _ = sim.reset(seed=123)

    assert list(obs1['test_attacker']['is_observable']) == list(
        obs2['test_attacker']['is_observable']
    )
    assert list(obs1['test_attacker']['is_actionable']) == list(
        obs2['test_attacker']['is_actionable']
    )
    assert list(obs1['test_attacker']['observed_state']) == list(
        obs2['test_attacker']['observed_state']
    )
    assert list(obs1['test_attacker']['remaining_ttc']) == list(
        obs2['test_attacker']['remaining_ttc']
    )
    assert list(obs1['test_attacker']['asset_type']) == list(
        obs2['test_attacker']['asset_type']
    )
    assert list(obs1['test_attacker']['asset_id']) == list(
        obs2['test_attacker']['asset_id']
    )
    assert list(obs1['test_attacker']['step_name']) == list(
        obs2['test_attacker']['step_name']
    )

    for i, elem in enumerate(obs1['test_attacker']['attack_graph_edges']):
        assert list(obs2['test_attacker']['attack_graph_edges'][i]) == list(elem)

    assert list(obs1['test_attacker']['model_asset_id']) == list(
        obs2['test_attacker']['model_asset_id']
    )
    assert list(obs1['test_attacker']['model_asset_type']) == list(
        obs2['test_attacker']['model_asset_type']
    )

    for i, elem in enumerate(obs1['test_attacker']['model_edges_ids']):
        assert list(obs2['test_attacker']['model_edges_ids'][i]) == list(elem)

    assert list(obs1['test_attacker']['model_edges_type']) == list(
        obs2['test_attacker']['model_edges_type']
    )


def test_step_deterministic(corelang_lang_graph: LanguageGraph, model: Model) -> None:
    """Make sure blank observation is deterministic with seed given"""

    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimVectorizedObsEnv(MalSimulator(attack_graph))
    os_app_fa = get_node(attack_graph, 'OS App:fullAccess')

    sim.register_attacker('test_attacker', {os_app_fa})
    sim.register_defender('test_defender')

    obs1: dict[str, Any] = {}
    obs2: dict[str, Any] = {}

    # Run 1
    sim.reset(seed=123)
    for _ in range(10):
        attacker_state = sim.get_agent_state('test_attacker')
        attacker_node = next(n for n in attacker_state.action_surface)
        attacker_action = (1, sim.node_to_index(attacker_node))
        obs1, _, _, _, _ = sim.step(
            {'test_defender': (0, None), 'test_attacker': attacker_action}
        )

    # Run 2 - identical
    sim.reset(seed=123)
    for _ in range(10):
        attacker_state = sim.get_agent_state('test_attacker')
        attacker_node = next(n for n in attacker_state.action_surface)
        attacker_action = (1, sim.node_to_index(attacker_node))
        obs2, _, _, _, _ = sim.step(
            {'test_defender': (0, None), 'test_attacker': attacker_action}
        )

    assert list(obs1['test_attacker']['observed_state']) == list(
        obs2['test_attacker']['observed_state']
    )
    assert list(obs1['test_defender']['observed_state']) == list(
        obs2['test_defender']['observed_state']
    )


def test_create_blank_observation_observability_given(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    """Make sure observability propagates correctly from extras field/scenario
    to observation in mal simulator"""

    # Load Scenario with observability rules set
    scenario_file = 'tests/testdata/scenarios/traininglang_observability_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    env = MalSimVectorizedObsEnv(MalSimulator.from_scenario(scenario))

    num_objects = len(env.sim.sim_state.attack_graph.nodes)
    blank_observation = env._create_blank_observation()

    assert len(blank_observation['is_observable']) == num_objects

    for index, observable in enumerate(blank_observation['is_observable']):
        node = env.index_to_node(index)

        # Below are the rules from the traininglang observability scenario
        # made into if statements
        if (
            (node.lg_attack_step.asset.name == 'Host' and node.name in ('access'))
            or (
                node.lg_attack_step.asset.name == 'Host'
                and node.name in ('authenticate')
            )
            or (node.lg_attack_step.asset.name == 'Data' and node.name in ('read'))
        ) or (
            node.model_asset
            and node.model_asset.name == 'User:3'
            and node.name in ('phishing')
        ):
            assert observable
        else:
            assert not observable


def test_create_blank_observation_actionability_given(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    """Make sure actionability propagates correctly from extras field/scenario
    to observation in mal simulator"""

    # Load Scenario with observability rules set
    scenario_file = 'tests/testdata/scenarios/traininglang_actionability_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    env = MalSimVectorizedObsEnv(MalSimulator.from_scenario(scenario))

    num_objects = len(env.sim.sim_state.attack_graph.nodes)
    blank_observation = env._create_blank_observation()

    assert len(blank_observation['is_actionable']) == num_objects

    for index, actionable in enumerate(blank_observation['is_actionable']):
        node = env.index_to_node(index)

        # Below are the rules from the traininglang observability scenario
        # made into if statements
        if (
            (node.lg_attack_step.asset.name == 'Host' and node.name in ('notPresent'))
            or (
                node.lg_attack_step.asset.name == 'Data' and node.name in ('notPresent')
            )
        ) or (
            node.model_asset
            and node.model_asset.name == 'User:3'
            and node.name in ('notPresent')
        ):
            assert actionable
        else:
            assert not actionable


def test_malsimulator_observe_attacker() -> None:
    scenario = Scenario.load_from_file('tests/testdata/scenarios/simple_scenario.yml')

    # Create the simulator
    env = MalSimVectorizedObsEnv(MalSimulator(scenario.attack_graph))

    # Register the agents
    defender_agent_name = 'defender'
    attacker_agent_name = 'attacker'

    os_app_fa = get_node(scenario.attack_graph, 'OS App:fullAccess')

    env.register_attacker(attacker_agent_name, {os_app_fa})
    env.register_defender(defender_agent_name)

    # Must reset after registering agents
    env.reset()

    # Make alteration to the attack graph attacker
    attacker_state = env.get_agent_state(attacker_agent_name)

    assert len(attacker_state.performed_nodes) == 1
    reached_step = next(iter(attacker_state.performed_nodes))

    # Select actions for the attacker
    # In the end the attacker will have three reached steps
    # where two are children of the first one
    actions_to_take = [
        child_node
        for child_node in reached_step.children
        if child_node.type in ('and', 'or')
    ]

    num_reached_steps_before = len(attacker_state.performed_nodes)

    for attacker_action in actions_to_take:
        obs, _, _, _, _ = env.step(
            {
                defender_agent_name: (0, None),
                attacker_agent_name: (1, env.node_to_index(attacker_action)),
            }
        )

        attacker_state = env.get_agent_state(attacker_agent_name)
        num_reached_steps_now = len(attacker_state.performed_nodes)
        assert num_reached_steps_now == num_reached_steps_before + 1
        num_reached_steps_before = num_reached_steps_now

    attacker_observation = obs[attacker_agent_name]['observed_state']

    for node in attacker_state.performed_nodes:
        node_index = env._id_to_index[node.id]
        node_obs_state = attacker_observation[node_index]
        assert node_obs_state == 1

    for index, state in enumerate(attacker_observation):
        node = env.index_to_node(index)

        if node in attacker_state.performed_nodes:
            assert state == 1
        else:
            if state == -1:
                for parent in node.parents:
                    assert parent not in attacker_state.performed_nodes
            else:
                assert state == 0


def test_malsimulator_observe_and_reward_attacker_defender() -> None:
    """Run attacker and defender actions and make sure
    rewards and observation states are updated correctly"""

    def verify_attacker_obs_state(
        observed_state: list[int],
        expected_reached: list[int],
        expected_children_of_reached: list[int],
    ) -> None:
        """Make sure obs state looks as expected"""
        for index, state in enumerate(observed_state):
            node_id = env._index_to_id[index]
            if state == 1:
                assert node_id in expected_reached
            elif state == 0:
                assert node_id in expected_children_of_reached
            else:
                assert state == -1

    def verify_defender_obs_state(
        observed_state: list[int], expected_enabled_nodes: list[int]
    ) -> None:
        """Make sure obs state looks as expected"""
        for index, state in enumerate(observed_state):
            if state == 1:
                assert index in expected_enabled_nodes
            elif state == 0:
                assert index not in expected_enabled_nodes, (
                    f'{index} not correct state {state}'
                )
            else:
                assert state == -1

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/traininglang_scenario.yml'
    )
    # Create the simulator
    env = MalSimVectorizedObsEnv(
        MalSimulator.from_scenario(scenario, register_agents=False)
    )

    user3_phish = get_node(scenario.attack_graph, 'User:3:phishing')
    host0_connect = get_node(scenario.attack_graph, 'Host:0:connect')

    # Register an attacker
    attacker_name = 'attacker'
    entry_points = {user3_phish, host0_connect}
    env.register_attacker(attacker_name, entry_points)

    defender_agent_name = 'Defender1'
    env.register_defender(defender_agent_name)
    env.reset()

    defender_state = env.get_agent_state(defender_agent_name)

    defender_enabled_steps = [
        n.id
        for n in env.attack_graph.nodes.values()
        if n in defender_state.performed_nodes
    ]

    attacker_reached_steps = [n.id for n in entry_points]
    attacker_reached_step_children = []
    for reached in entry_points:
        attacker_reached_step_children.extend([n.id for n in reached.children])

    # Prepare nodes that will be stepped through in order
    user_3_compromise = get_node(scenario.attack_graph, 'User:3:compromise')
    host_0_authenticate = get_node(scenario.attack_graph, 'Host:0:authenticate')
    host_0_access = get_node(scenario.attack_graph, 'Host:0:access')
    host_0_notPresent = get_node(scenario.attack_graph, 'Host:0:notPresent')
    data_2_read = get_node(scenario.attack_graph, 'Data:2:read')

    # Step with attacker action
    obs, rew, _, _, _ = env.step(
        {
            defender_agent_name: (0, None),
            attacker_name: (1, env.node_to_index(user_3_compromise)),
        }
    )

    # Verify obs state
    attacker_reached_steps.append(user_3_compromise.id)
    attacker_reached_step_children.extend([n.id for n in user_3_compromise.children])

    verify_attacker_obs_state(
        obs[attacker_name]['observed_state'],
        attacker_reached_steps,
        attacker_reached_step_children,
    )
    verify_defender_obs_state(
        obs[defender_agent_name]['observed_state'],
        attacker_reached_steps + defender_enabled_steps,
    )

    # Verify rewards
    assert rew[defender_agent_name] == 0
    assert rew[attacker_name] == 0

    # Step with attacker again
    obs, rew, _, _, _ = env.step(
        {
            defender_agent_name: (0, None),
            attacker_name: (1, env.node_to_index(host_0_authenticate)),
        }
    )

    # Verify obs state
    attacker_reached_steps.append(host_0_authenticate.id)
    attacker_reached_step_children.extend([n.id for n in host_0_authenticate.children])
    verify_attacker_obs_state(
        obs[attacker_name]['observed_state'],
        attacker_reached_steps,
        attacker_reached_step_children,
    )
    verify_defender_obs_state(
        obs[defender_agent_name]['observed_state'],
        attacker_reached_steps + defender_enabled_steps,
    )

    # Verify rewards
    assert rew[defender_agent_name] == 0
    assert rew[attacker_name] == 0

    # Step attacker again
    obs, rew, _, _, _ = env.step(
        {
            defender_agent_name: (0, None),
            attacker_name: (1, env.node_to_index(host_0_access)),
        }
    )

    # Verify obs state
    attacker_reached_steps.append(host_0_access.id)
    attacker_reached_step_children.extend([n.id for n in host_0_access.children])
    verify_attacker_obs_state(
        obs[attacker_name]['observed_state'],
        attacker_reached_steps,
        attacker_reached_step_children,
    )
    verify_defender_obs_state(
        obs[defender_agent_name]['observed_state'],
        attacker_reached_steps + defender_enabled_steps,
    )

    reward_host_0_access = 4
    # Verify rewards
    assert rew[attacker_name] == reward_host_0_access
    assert rew[defender_agent_name] == -rew[attacker_name]

    # Step defender and attacker
    # Attacker wont be able to traverse Data:2:read since
    # Host:0:notPresent is activated before
    obs, rew, _, _, _ = env.step(
        {
            defender_agent_name: (1, env.node_to_index(host_0_notPresent)),
            attacker_name: (1, env.node_to_index(data_2_read)),
        }
    )
    defender_enabled_steps.append(host_0_notPresent.id)

    # Attacker obs state should look the same as before
    verify_attacker_obs_state(
        obs[attacker_name]['observed_state'],
        attacker_reached_steps,
        attacker_reached_step_children,
    )
    verify_defender_obs_state(
        obs[defender_agent_name]['observed_state'],
        attacker_reached_steps + defender_enabled_steps,
    )

    # Verify rewards
    reward_host_0_not_present = 2
    assert rew[attacker_name] == reward_host_0_access  # no additional reward
    assert rew[defender_agent_name] == -rew[attacker_name] - reward_host_0_not_present


def test_malsimulator_initial_observation_defender(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    """Make sure ._observe_defender observes nodes and set observed state"""

    attack_graph = AttackGraph(corelang_lang_graph, model)
    env = MalSimVectorizedObsEnv(MalSimulator(attack_graph))

    defender_agent_name = 'defender'
    env.register_defender(defender_agent_name)
    obs, _ = env.reset()

    defender_obs_state = obs[defender_agent_name]['observed_state']
    defender_agent_state = env.get_agent_state(defender_agent_name)

    nodes_to_observe = [
        node
        for node in env.sim.sim_state.attack_graph.nodes.values()
        if node in defender_agent_state.performed_nodes
    ]

    assert nodes_to_observe

    # Assert that observed state is 1 after observe_defender
    for node in nodes_to_observe:
        index = env._id_to_index[node.id]
        # Make sure observed after
        assert defender_obs_state[index] == 1


def test_malsimulator_observe_and_reward_attacker_no_entrypoints(
    corelang_lang_graph: LanguageGraph, model: Model
) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimVectorizedObsEnv(MalSimulator(attack_graph))

    # Register an attacker
    sim.register_attacker('attacker', set())
    sim.reset()

    obs, rew, _, _, _ = sim.step({})

    # Observe and reward with no new actions
    # Since attacker has no entry points and no steps have been performed
    # the observed state should be empty
    for state in obs['attacker']['observed_state']:
        assert state == -1
    assert rew['attacker'] == 0


def test_malsimulator_observe_and_reward_attacker_entrypoints(
    traininglang_lang_graph: LanguageGraph, traininglang_model: Model
) -> None:
    attack_graph = AttackGraph(traininglang_lang_graph, traininglang_model)
    env = MalSimVectorizedObsEnv(MalSimulator(attack_graph))

    user3_phish = get_node(attack_graph, 'User:3:phishing')
    host0_connect = get_node(attack_graph, 'Host:0:connect')

    # Register an attacker
    attacker_name = 'attacker'
    entry_points = {user3_phish, host0_connect}
    env.register_attacker(attacker_name, entry_points)

    # We need to reinitialize to initialize agent
    obs, _ = env.reset()

    for index, state in enumerate(obs[attacker_name]['observed_state']):
        attacker_state = env.get_agent_state(attacker_name)
        assert isinstance(attacker_state, MalSimAttackerState)

        node = env.index_to_node(index)
        if state == -1:
            assert node not in attacker_state.entry_points
            assert node not in attacker_state.performed_nodes
            assert not any(p in attacker_state.performed_nodes for p in node.parents)
        elif state == 0:
            assert node not in attacker_state.entry_points
            assert node not in attacker_state.performed_nodes
            assert any(p in attacker_state.performed_nodes for p in node.parents)
        elif state == 1:
            assert node in attacker_state.entry_points
            assert node in attacker_state.performed_nodes
