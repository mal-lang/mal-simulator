"""Test MalSimulator class"""

from maltoolbox.attackgraph import AttackGraph, Attacker
from malsim.mal_simulator import MalSimulator, MalSimAttackerState
from malsim.envs import MalSimVectorizedObsEnv
from malsim.scenario import load_scenario

def test_create_blank_observation(corelang_lang_graph, model):
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
    # the index where asset_type_index lies on will point to an attack step id in sim._index_to_id
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
        assert node.model_asset.id == expected_asset_id

    assert len(blank_observation['step_name']) == num_objects

    expected_num_edges = sum([1 for step in attack_graph.nodes.values()
                                for child in step.children] +
                                # We expect all defenses again (reversed)
                             [1 for step in attack_graph.nodes.values()
                                for child in step.children
                                if step.type == "defense"])
    assert len(blank_observation['attack_graph_edges']) == expected_num_edges


def test_create_blank_observation_deterministic(
        corelang_lang_graph, model
    ):
    """Make sure blank observation is deterministic with seed given"""

    attack_graph = AttackGraph(corelang_lang_graph, model)
    attack_graph.attach_attackers()
    attacker = next(iter(attack_graph.attackers.values()))

    sim = MalSimVectorizedObsEnv(MalSimulator(attack_graph))
    sim.register_attacker("test_attacker", attacker.id)
    sim.register_defender("test_defender")

    obs1, _ = sim.reset(seed=123)
    obs2, _ = sim.reset(seed=123)

    assert list(obs1['test_attacker']['is_observable']) == list(obs2['test_attacker']['is_observable'])
    assert list(obs1['test_attacker']['is_actionable']) == list(obs2['test_attacker']['is_actionable'])
    assert list(obs1['test_attacker']['observed_state']) == list(obs2['test_attacker']['observed_state'])
    assert list(obs1['test_attacker']['remaining_ttc']) == list(obs2['test_attacker']['remaining_ttc'])
    assert list(obs1['test_attacker']['asset_type']) == list(obs2['test_attacker']['asset_type'])
    assert list(obs1['test_attacker']['asset_id']) == list(obs2['test_attacker']['asset_id'])
    assert list(obs1['test_attacker']['step_name']) == list(obs2['test_attacker']['step_name'])

    for i, elem in  enumerate(obs1['test_attacker']['attack_graph_edges']):
        assert list(obs2['test_attacker']['attack_graph_edges'][i]) == list(elem)

    assert list(obs1['test_attacker']['model_asset_id']) == list(obs2['test_attacker']['model_asset_id'])
    assert list(obs1['test_attacker']['model_asset_type']) == list(obs2['test_attacker']['model_asset_type'])

    for i, elem in  enumerate(obs1['test_attacker']['model_edges_ids']):
        assert list(obs2['test_attacker']['model_edges_ids'][i]) == list(elem)

    assert list(obs1['test_attacker']['model_edges_type']) == list(obs2['test_attacker']['model_edges_type'])


def test_step_deterministic(
        corelang_lang_graph, model
    ):
    """Make sure blank observation is deterministic with seed given"""

    attack_graph = AttackGraph(corelang_lang_graph, model)
    attack_graph.attach_attackers()
    attacker = next(iter(attack_graph.attackers.values()))

    sim = MalSimVectorizedObsEnv(MalSimulator(attack_graph))
    sim.register_attacker("test_attacker", attacker.id)
    sim.register_defender("test_defender")

    obs1 = {}
    obs2 = {}

    # Run 1
    sim.reset(seed=123)
    for _ in range(10):
        attacker_node = next(
            n for n in sim.get_agent_state('test_attacker').action_surface
            if not n.is_compromised()
        )
        attacker_action = (1, sim.node_to_index(attacker_node))
        obs1, _, _, _, _ = sim.step(
            {'test_defender': (0, None), 'test_attacker': attacker_action}
        )

    # Run 2 - identical
    sim.reset(seed=123)
    for _ in range(10):
        attacker_node = next(
            n for n in sim.get_agent_state('test_attacker').action_surface
            if not n.is_compromised()
        )
        attacker_action = (1, sim.node_to_index(attacker_node))
        obs2, _, _, _, _ = sim.step(
            {'test_defender': (0, None), 'test_attacker': attacker_action}
        )

    assert list(obs1['test_attacker']['observed_state']) == list(obs2['test_attacker']['observed_state'])
    assert list(obs1['test_defender']['observed_state']) == list(obs2['test_defender']['observed_state'])


def test_create_blank_observation_observability_given(
        corelang_lang_graph, model
    ):
    """Make sure observability propagates correctly from extras field/scenario
    to observation in mal simulator"""

    # Load Scenario with observability rules set
    scenario_file = \
        'tests/testdata/scenarios/traininglang_observability_scenario.yml'
    ag, _ = load_scenario(scenario_file)
    env = MalSimVectorizedObsEnv(MalSimulator(ag))

    num_objects = len(env.sim.attack_graph.nodes)
    blank_observation = env._create_blank_observation()

    assert len(blank_observation['is_observable']) == num_objects

    for index, observable in enumerate(blank_observation['is_observable']):
        node = env.index_to_node(index)

        # Below are the rules from the traininglang observability scenario
        # made into if statements
        if node.lg_attack_step.asset.name == 'Host' and node.name in ('access'):
            assert observable
        elif node.lg_attack_step.asset.name == 'Host' and node.name in ('authenticate'):
            assert observable
        elif node.lg_attack_step.asset.name == 'Data' and node.name in ('read'):
            assert observable
        elif node.model_asset.name == 'User:3' and node.name in ('phishing'):
            assert observable
        else:
            assert not observable

def test_create_blank_observation_actionability_given(
        corelang_lang_graph, model
    ):
    """Make sure actionability propagates correctly from extras field/scenario
    to observation in mal simulator"""

    # Load Scenario with observability rules set
    scenario_file = 'tests/testdata/scenarios/traininglang_actionability_scenario.yml'
    ag, _ = load_scenario(scenario_file)
    env = MalSimVectorizedObsEnv(MalSimulator(ag))

    num_objects = len(env.sim.attack_graph.nodes)
    blank_observation = env._create_blank_observation()

    assert len(blank_observation['is_actionable']) == num_objects

    for index, actionable in enumerate(blank_observation['is_actionable']):
        node = env.index_to_node(index)

        # Below are the rules from the traininglang observability scenario
        # made into if statements
        if node.lg_attack_step.asset.name == 'Host' and node.name in ('notPresent'):
            assert actionable
        elif node.lg_attack_step.asset.name == 'Data' and node.name in ('notPresent'):
            assert actionable
        elif node.model_asset.name == 'User:3' and node.name in ('notPresent'):
            assert actionable
        else:
            assert not actionable

def test_step(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_point = attack_graph.get_node_by_full_name('OS App:fullAccess')

    attacker = Attacker(
        'attacker1',
        reached_attack_steps = {entry_point},
        entry_points = {entry_point},
        attacker_id = 100)
    attack_graph.add_attacker(attacker, attacker.id)
    env = MalSimVectorizedObsEnv(MalSimulator(attack_graph))

    # Refresh attack graph reference to the one deepcopied during the reset
    attack_graph = env.sim.attack_graph

    agent_info = MalSimAttackerState(attacker.name, attacker.id)

    # Can not attack the notPresent step
    defense_step = attack_graph\
        .get_node_by_full_name('OS App:notPresent')
    actions, new_surface = env.sim._attacker_step(agent_info, {defense_step})
    assert not actions
    assert not new_surface

    attack_step = attack_graph.get_node_by_full_name('OS App:attemptRead')

    # Action needs to be in action surface to be an allowed action
    agent_info.action_surface = {attack_step}

    # Since action is in attack surface and since it is traversable,
    # action will be performed.
    actions, new_surface = env.sim._attacker_step(agent_info, {attack_step})
    assert actions == {attack_step}
    assert new_surface == attack_step.children


def test_malsimulator_defender_step(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    env = MalSimVectorizedObsEnv(MalSimulator(attack_graph))

    agent_name = "defender1"
    env.register_defender(agent_name)
    env.reset()

    defense_step = env.sim.attack_graph.get_node_by_full_name(
        'OS App:notPresent')
    enabled, _ = env.sim._defender_step(env.sim.agents[agent_name], {defense_step})
    assert enabled == {defense_step}

    # Can not defend attack_step
    attack_step = env.sim.attack_graph.get_node_by_full_name(
        'OS App:attemptUseVulnerability')
    actions, _ = env.sim._defender_step(env.sim.agents[agent_name], {attack_step})
    assert not actions


def test_malsimulator_observe_attacker():
    attack_graph, _ = load_scenario(
        'tests/testdata/scenarios/simple_scenario.yml')

    # Create the simulator
    env = MalSimVectorizedObsEnv(MalSimulator(attack_graph))

    # Register the agents
    defender_agent_name = 'defender'
    attacker_agent_name = 'attacker'

    attacker = next(iter(attack_graph.attackers.values()))

    env.register_attacker(attacker_agent_name, attacker.id)
    env.register_defender(defender_agent_name)

    # Must reset after registering agents
    env.reset()

    # Make alteration to the attack graph attacker
    assert len(env.sim.attack_graph.attackers) == 1
    attacker = next(iter(env.sim.attack_graph.attackers.values()))

    assert len(attacker.reached_attack_steps) == 1
    reached_step = next(iter(attacker.reached_attack_steps))

    # Select actions for the attacker
    actions_to_take = []
    for child_node in reached_step.children:
        if child_node.type in ('and', 'or'):
            # In the end the attacker will have three reached steps
            # where two are children of the first one
            actions_to_take.append(child_node)

    num_reached_steps_before = len(attacker.reached_attack_steps)

    for attacker_action in actions_to_take:
        obs, _, _, _, _ = env.step({
            defender_agent_name: (0, None),
            attacker_agent_name: (1, env.node_to_index(attacker_action))
        })

        num_reached_steps_now = len(attacker.reached_attack_steps)
        assert num_reached_steps_now == num_reached_steps_before + 1
        num_reached_steps_before = num_reached_steps_now

    attacker_observation = obs[attacker_agent_name]["observed_state"]

    for node in attacker.reached_attack_steps:
        node_index = env._id_to_index[node.id]
        node_obs_state = attacker_observation[node_index]
        assert node_obs_state == 1

    for index, state in enumerate(attacker_observation):
        node = env.index_to_node(index)

        if node.is_compromised():
            assert state == 1
        else:
            if state == -1:
                for parent in node.parents:
                    assert parent not in attacker.reached_attack_steps
            else:
                assert state == 0


def test_malsimulator_observe_and_reward_attacker_defender():
    """Run attacker and defender actions and make sure
    rewards and observation states are updated correctly"""

    def verify_attacker_obs_state(
            observed_state,
            expected_reached,
            expected_children_of_reached
        ):
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
            observed_state
        ):
        """Make sure obs state looks as expected"""
        for index, state in enumerate(observed_state):
            node = env.index_to_node(index)
            if state == 1:
                assert node.is_compromised() or node.is_enabled_defense()
            elif state == 0:
                assert not node.is_compromised() and not node.is_enabled_defense(), f"{node.full_name} not correct state {state}"
            else:
                assert state == -1

    attack_graph, _ = load_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml')
    # Create the simulator
    env = MalSimVectorizedObsEnv(MalSimulator(attack_graph))

    attacker = next(iter(attack_graph.attackers.values()))
    attacker_agent_name = "Attacker1"
    env.register_attacker(attacker_agent_name, attacker.id)

    defender_agent_name = "Defender1"
    env.register_defender(defender_agent_name)

    env.reset()

    attacker_reached_steps = [n.id for n in attacker.entry_points]
    attacker_reached_step_children = []
    for reached in attacker.entry_points:
        attacker_reached_step_children.extend(
            [n.id for n in reached.children]
        )

    # Prepare nodes that will be stepped through in order
    user_3_compromise = env.sim.attack_graph\
        .get_node_by_full_name("User:3:compromise")
    host_0_authenticate = env.sim.attack_graph\
        .get_node_by_full_name("Host:0:authenticate")
    host_0_access = env.sim.attack_graph\
        .get_node_by_full_name("Host:0:access")
    host_0_notPresent = env.sim.attack_graph\
        .get_node_by_full_name("Host:0:notPresent")
    data_2_read = env.sim.attack_graph\
        .get_node_by_full_name("Data:2:read")

    # Step with attacker action
    obs, rew, _, _, _ = env.step({
            defender_agent_name: (0, None),
            attacker_agent_name: (1, env.node_to_index(user_3_compromise))
        }
    )

    # Verify obs state
    attacker_reached_steps.append(user_3_compromise.id)
    attacker_reached_step_children.extend(
        [n.id for n in user_3_compromise.children])

    verify_attacker_obs_state(
        obs[attacker_agent_name]['observed_state'],
        attacker_reached_steps,
        attacker_reached_step_children)
    verify_defender_obs_state(
        obs[defender_agent_name]['observed_state']
    )

    # Verify rewards
    assert rew[defender_agent_name] == 0
    assert rew[attacker_agent_name] == 0

    # Step with attacker again
    obs, rew, _, _, _ = env.step({
            defender_agent_name: (0, None),
            attacker_agent_name: (1, env.node_to_index(host_0_authenticate))
        }
    )

    # Verify obs state
    attacker_reached_steps.append(host_0_authenticate.id)
    attacker_reached_step_children.extend(
        [n.id for n in host_0_authenticate.children])
    verify_attacker_obs_state(
        obs[attacker_agent_name]['observed_state'],
        attacker_reached_steps,
        attacker_reached_step_children)
    verify_defender_obs_state(
        obs[defender_agent_name]['observed_state']
    )

    # Verify rewards
    assert rew[defender_agent_name] == 0
    assert rew[attacker_agent_name] == 0

    # Step attacker again
    obs, rew, _, _, _ = env.step({
            defender_agent_name: (0, None),
            attacker_agent_name: (1, env.node_to_index(host_0_access))
        }
    )

    # Verify obs state
    attacker_reached_steps.append(host_0_access.id)
    attacker_reached_step_children.extend(
        [n.id for n in host_0_access.children])
    verify_attacker_obs_state(
        obs[attacker_agent_name]['observed_state'],
        attacker_reached_steps,
        attacker_reached_step_children)
    verify_defender_obs_state(
        obs[defender_agent_name]['observed_state']
    )

    reward_host_0_access = 4
    # Verify rewards
    assert rew[attacker_agent_name] == reward_host_0_access
    assert rew[defender_agent_name] == -rew[attacker_agent_name]

    # Step defender and attacker
    # Attacker wont be able to traverse Data:2:read since
    # Host:0:notPresent is activated before
    obs, rew, _, _, _ = env.step({
            defender_agent_name: (1, env.node_to_index(host_0_notPresent)),
            attacker_agent_name: (1, env.node_to_index(data_2_read))
        }
    )

    # Attacker obs state should look the same as before
    verify_attacker_obs_state(
        obs[attacker_agent_name]['observed_state'],
        attacker_reached_steps,
        attacker_reached_step_children)
    verify_defender_obs_state(
        obs[defender_agent_name]['observed_state']
    )

    # Verify rewards
    reward_host_0_not_present = 2
    assert rew[attacker_agent_name] == reward_host_0_access  # no additional reward
    assert rew[defender_agent_name] == - rew[attacker_agent_name] - reward_host_0_not_present


def test_malsimulator_initial_observation_defender(corelang_lang_graph, model):
    """Make sure ._observe_defender observes nodes and set observed state"""

    attack_graph = AttackGraph(corelang_lang_graph, model)
    env = MalSimVectorizedObsEnv(MalSimulator(attack_graph))

    defender_agent_name = "defender"
    env.register_defender(defender_agent_name)
    obs, _ = env.reset()

    defender_obs_state = obs[defender_agent_name]["observed_state"]

    nodes_to_observe = [
        node for node in env.sim.attack_graph.nodes.values()
        if node.is_enabled_defense() or node.is_compromised()
    ]

    # Assert that observed state is 1 after observe_defender
    for node in nodes_to_observe:
        index = env._id_to_index[node.id]
        # Make sure observed after
        assert defender_obs_state[index] == 1


def test_malsimulator_observe_and_reward_attacker_no_entrypoints(
        corelang_lang_graph, model
    ):

    attack_graph = AttackGraph(corelang_lang_graph, model)
    attacker = Attacker("TestAttacker", [], [])
    attack_graph.add_attacker(attacker)
    sim = MalSimVectorizedObsEnv(MalSimulator(attack_graph))

    # Register an attacker
    sim.register_attacker(attacker.name, attacker.id)
    sim.reset()

    obs, rew, _, _, _ = sim.step({})

    # Observe and reward with no new actions
    # Since attacker has no entry points and no steps have been performed
    # the observed state should be empty
    for state in obs[attacker.name]['observed_state']:
        assert state == -1
    assert rew[attacker.name] == 0


def test_malsimulator_observe_and_reward_attacker_entrypoints(
        traininglang_lang_graph, traininglang_model
    ):

    attack_graph = AttackGraph(
        traininglang_lang_graph, traininglang_model)
    attack_graph.attach_attackers()
    env = MalSimVectorizedObsEnv(MalSimulator(attack_graph))

    # Register an attacker
    attacker = env.sim.attack_graph.attackers[0]
    env.register_attacker(attacker.name, attacker.id)

    # We need to reinitialize to initialize agent
    obs, _ = env.reset()

    # Since reset deepcopies attack graph we
    # need to fetch attacker again
    attacker = env.sim.attack_graph.attackers[0]

    for index, state in enumerate(
            obs[attacker.name]['observed_state']):

        node = env.index_to_node(index)
        if state == -1:
            assert node not in attacker.entry_points
            assert node not in attacker.reached_attack_steps
            assert not node.is_compromised()
            assert not any([p.is_compromised() for p in node.parents])
        elif state == 0:
            assert node not in attacker.entry_points
            assert node not in attacker.reached_attack_steps
            assert not node.is_compromised()
            assert any([p.is_compromised() for p in node.parents])
        elif state == 1:
            assert node in attacker.entry_points
            assert node in attacker.reached_attack_steps
            assert node.is_compromised()
