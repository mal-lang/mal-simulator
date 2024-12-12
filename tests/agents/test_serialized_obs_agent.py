"""Test MalSimulator class"""

from maltoolbox.attackgraph import AttackGraph, Attacker
from malsim.sims.mal_simulator import MalSimulator
from malsim.scenario import load_scenario, create_simulator_from_scenario, load_scenario
from malsim.sims import MalSimulatorSettings

from malsim.agents import SerializedObsAttacker, SerializedObsDefender

def test_create_blank_observation(corelang_lang_graph, model):
    """Make sure blank observation contains correct default values"""

    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    agent_name = "attacker1"
    agent = SerializedObsAttacker(agent_name, None, sim)

    num_objects = len(attack_graph.nodes)
    blank_observation = agent._create_blank_observation()

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
        expected_type = agent._index_to_asset_type[asset_type_index]
        node = agent.get_attack_graph_node_by_index(index)
        assert node.asset.type == expected_type

    # asset_id on index X in blank_observation['asset_id']
    # should be the same as the id of the asset of attack step X
    assert len(blank_observation['asset_id']) == num_objects
    for index, expected_asset_id in enumerate(blank_observation['asset_id']):
        node = agent.get_attack_graph_node_by_index(index)
        assert node.asset.id == expected_asset_id

    assert len(blank_observation['step_name']) == num_objects

    expected_num_edges = sum([1 for step in attack_graph.nodes
                                for child in step.children] +
                                # We expect all defenses again (reversed)
                             [1 for step in attack_graph.nodes
                                for child in step.children
                                if step.type == "defense"])
    assert len(blank_observation['attack_graph_edges']) == expected_num_edges


def test_create_blank_observation_observability_given(
        corelang_lang_graph, model
    ):
    """Make sure observability propagates correctly from extras field/scenario
    to observation in mal simulator"""

    # Load Scenario with observability rules set
    scenario_file = \
        'tests/testdata/scenarios/traininglang_observability_scenario.yml'
    ag, _ = load_scenario(scenario_file)
    sim = MalSimulator(ag)

    agent_name = "attacker1"
    agent = SerializedObsAttacker(agent_name, 1, sim)

    num_objects = len(sim.attack_graph.nodes)
    blank_observation = agent._create_blank_observation()

    assert len(blank_observation['is_observable']) == num_objects

    for index, observable in enumerate(blank_observation['is_observable']):
        node = agent.get_attack_graph_node_by_index(index)

        # Below are the rules from the traininglang observability scenario
        # made into if statements
        if node.asset.type == 'Host' and node.name in ('access'):
            assert observable
        elif node.asset.type == 'Host' and node.name in ('authenticate'):
            assert observable
        elif node.asset.type == 'Data' and node.name in ('read'):
            assert observable
        elif node.asset.name == 'User:3' and node.name in ('phishing'):
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
    sim = MalSimulator(ag)

    agent_name = "attacker1"
    agent = SerializedObsAttacker(agent_name, 1, sim)

    num_objects = len(sim.attack_graph.nodes)
    blank_observation = agent._create_blank_observation()

    assert len(blank_observation['is_actionable']) == num_objects

    for index, actionable in enumerate(blank_observation['is_actionable']):
        node = agent.get_attack_graph_node_by_index(index)

        # Below are the rules from the traininglang observability scenario
        # made into if statements
        if node.asset.type == 'Host' and node.name in ('notPresent'):
            assert actionable
        elif node.asset.type == 'Data' and node.name in ('notPresent'):
            assert actionable
        elif node.asset.name == 'User:3' and node.name in ('notPresent'):
            assert actionable
        else:
            assert not actionable


def test_step(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)

    attacker = Attacker('attacker1', id=0)
    attack_graph.add_attacker(attacker, attacker.id)
    sim = MalSimulator(attack_graph)

    agent_name = "attacker1"
    agent = SerializedObsAttacker(agent_name, attacker.id, sim)
    sim.reset()

    # Can not attack the notPresent step
    defense_step = attack_graph.get_node_by_full_name('OS App:notPresent')
    actions = sim._attacker_step(agent, [defense_step])
    assert not actions

    # Can attack the attemptUseVulnerability step!
    attack_step = attack_graph.get_node_by_full_name('OS App:attemptUseVulnerability')
    actions = sim._attacker_step(agent, [attack_step])
    assert actions == [attack_step]


def test_malsimulator_defender_step(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    agent_name = "defender1"
    defender_agent = SerializedObsDefender(agent_name, sim)
    sim.register_agent(defender_agent)
    sim.reset()

    defense_step = attack_graph.get_node_by_full_name(
        'OS App:notPresent')
    actions, _ = sim._defender_step(defender_agent, [defense_step])
    assert actions == [defense_step]

    # Can not defend attack_step
    attack_step = attack_graph.get_node_by_full_name(
        'OS App:attemptUseVulnerability')
    actions, _ = sim._defender_step(defender_agent, [attack_step])
    assert not actions


def test_malsimulator_observe_attacker():
    attack_graph, _ = load_scenario(
        'tests/testdata/scenarios/simple_scenario.yml')

    # Create the simulator
    sim = MalSimulator(attack_graph)

    # Register the agents
    attacker_agent = SerializedObsAttacker(
        "attacker",
        attacker_id=attack_graph.attackers[0].id,
        simulator=sim
    )
    defender_agent = SerializedObsDefender(
        "defender",
        simulator=sim
    )

    sim.register_agent(attacker_agent)
    sim.register_agent(defender_agent)

    # Must reset after registering agents
    sim.reset()

    # Make alteration to the attack graph attacker
    assert len(sim.attack_graph.attackers) == 1
    attacker = sim.attack_graph.attackers[0]
    assert len(attacker.reached_attack_steps) == 1
    reached_step = attacker.reached_attack_steps[0]

    # Select actions for the attacker
    actions_to_take = []
    for child_node in reached_step.children:
        if child_node.type in ('and', 'or'):
            # In the end the attacker will have three reached steps
            # where two are children of the first one
            actions_to_take.append(child_node)

    attacker = sim.attack_graph.attackers[0]
    num_reached_steps_before = len(attacker.reached_attack_steps)

    for attacker_action in actions_to_take:
        sim.step({
            defender_agent.name: [],
            attacker_agent.name: [attacker_action]
        })

        num_reached_steps_now = len(attacker.reached_attack_steps)
        assert num_reached_steps_now == num_reached_steps_before + 1
        num_reached_steps_before = num_reached_steps_now

    attacker_observation = attacker_agent.observation["observed_state"]

    for node in attacker.reached_attack_steps:
        node_index = attacker_agent._id_to_index[node.id]
        node_obs_state = attacker_observation[node_index]
        assert node_obs_state == 1

    for index, state in enumerate(attacker_observation):
        node = attacker_agent.get_attack_graph_node_by_index(index)

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
            agent,
            expected_reached,
            expected_children_of_reached
        ):
        """Make sure obs state looks as expected"""
        for index, state in enumerate(agent.observation['observed_state']):
            node_id = agent._index_to_id[index]
            if state == 1:
                assert node_id in expected_reached
            elif state == 0:
                assert node_id in expected_children_of_reached
            else:
                assert state == -1

    def verify_defender_obs_state(
            agent: SerializedObsDefender
        ):
        """Make sure obs state looks as expected"""
        for index, state in enumerate(agent.observation['observed_state']):
            node = agent.get_attack_graph_node_by_index(index)
            if state == 1:
                assert node.is_compromised() or node.is_enabled_defense()
            elif state == 0:
                assert not node.is_compromised() and not node.is_enabled_defense()
            else:
                assert state == -1


    attack_graph, _ = load_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml')
    # Create the simulator
    sim = MalSimulator(attack_graph)

    attacker = sim.attack_graph.attackers[0]
    attacker_agent = SerializedObsAttacker("Attacker1", attacker.id, sim)
    sim.register_agent(attacker_agent)

    defender_agent = SerializedObsDefender("Defender1", sim)
    sim.register_agent(defender_agent)

    sim.reset()

    attacker_reached_steps = [n.id for n in attacker.entry_points]
    attacker_reached_step_children = []
    for reached in attacker.entry_points:
        attacker_reached_step_children.extend(
            [n.id for n in reached.children])

    # Prepare nodes that will be stepped through in order
    user_3_compromise = sim.attack_graph\
        .get_node_by_full_name("User:3:compromise")
    host_0_authenticate = sim.attack_graph\
        .get_node_by_full_name("Host:0:authenticate")
    host_0_access = sim.attack_graph\
        .get_node_by_full_name("Host:0:access")
    host_0_notPresent = sim.attack_graph\
        .get_node_by_full_name("Host:0:notPresent")
    data_2_read = sim.attack_graph\
        .get_node_by_full_name("Data:2:read")

    # Step with attacker action
    sim.step({
            defender_agent.name: [],
            attacker_agent.name: [user_3_compromise]
        }
    )

    # Verify obs state
    attacker_reached_steps.append(user_3_compromise.id)
    attacker_reached_step_children.extend(
        [n.id for n in user_3_compromise.children])

    verify_attacker_obs_state(
        attacker_agent,
        attacker_reached_steps,
        attacker_reached_step_children)
    verify_defender_obs_state(
        defender_agent
    )

    # Verify rewards
    assert defender_agent.reward == 0
    assert attacker_agent.reward == 0

    # Step with attacker again
    sim.step({
        defender_agent.name: [],
        attacker_agent.name: [host_0_authenticate]
    })

    # Verify obs state
    attacker_reached_steps.append(host_0_authenticate.id)
    attacker_reached_step_children.extend(
        [n.id for n in host_0_authenticate.children])
    verify_attacker_obs_state(
        attacker_agent,
        attacker_reached_steps,
        attacker_reached_step_children)
    verify_defender_obs_state(
        defender_agent
    )

    # Verify rewards
    assert defender_agent.reward == 0
    assert attacker_agent.reward == 0

    # Step attacker again
    sim.step({
        defender_agent.name: [],
        attacker_agent.name: [host_0_access]
    })

    # Verify obs state
    attacker_reached_steps.append(host_0_access.id)
    attacker_reached_step_children.extend(
        [n.id for n in host_0_access.children])
    verify_attacker_obs_state(
        attacker_agent,
        attacker_reached_steps,
        attacker_reached_step_children)
    verify_defender_obs_state(
        defender_agent
    )

    reward_host_0_access = 4
    # Verify rewards
    assert attacker_agent.reward == reward_host_0_access
    assert defender_agent.reward == -attacker_agent.reward

    # Step defender and attacker
    # Attacker wont be able to traverse Data:2:read since
    # Host:0:notPresent is activated before
    sim.step({
        defender_agent.name: [host_0_notPresent],
        attacker_agent.name: [data_2_read]
    })

    # Attacker obs state should look the same as before
    verify_attacker_obs_state(
        attacker_agent,
        attacker_reached_steps,
        attacker_reached_step_children)
    verify_defender_obs_state(
        defender_agent
    )

    # Verify rewards
    reward_host_0_not_present = 2
    assert attacker_agent.reward == reward_host_0_access  # no additional reward
    assert defender_agent.reward == - attacker_agent.reward - reward_host_0_not_present


def test_malsimulator_initial_observation_defender(corelang_lang_graph, model):
    """Make sure ._observe_defender observes nodes and set observed state"""

    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    defender_agent = SerializedObsDefender("defender", sim)
    sim.register_agent(defender_agent)
    sim.reset()

    defender_obs_state = defender_agent.observation["observed_state"]

    nodes_to_observe = [
        node for node in sim.attack_graph.nodes
        if node.is_enabled_defense() or node.is_compromised()
    ]

    # Assert that observed state is 1 after observe_defender
    for node in nodes_to_observe:
        index = defender_agent._id_to_index[node.id]
        # Make sure observed after
        assert defender_obs_state[index] == 1


def test_malsimulator_observe_and_reward_attacker_no_entrypoints(
        corelang_lang_graph, model
    ):

    attack_graph = AttackGraph(corelang_lang_graph, model)
    attacker = Attacker("TestAttacker", [], [])
    attack_graph.add_attacker(attacker)
    sim = MalSimulator(attack_graph)

    # Register an attacker
    attacker_agent = SerializedObsAttacker(attacker.name, attacker.id, sim)
    sim.register_agent(attacker_agent)
    sim.reset()

    agents = sim.step({})
    assert list(agents.keys()) == [attacker_agent.name]

    # Observe and reward with no new actions
    # Since attacker has no entry points and no steps have been performed
    # the observed state should be empty
    for state in attacker_agent.observation['observed_state']:
        assert state == -1
    assert attacker_agent.reward == 0


def test_malsimulator_observe_and_reward_attacker_entrypoints(
        traininglang_lang_graph, traininglang_model
    ):

    attack_graph = AttackGraph(
        traininglang_lang_graph, traininglang_model)
    attack_graph.attach_attackers()
    sim = MalSimulator(attack_graph)

    # Register an attacker
    attacker = sim.attack_graph.attackers[0]
    attacker_agent = SerializedObsAttacker(attacker.name, attacker.id, sim)
    sim.register_agent(attacker_agent)

    # We need to reinitialize to initialize agent
    sim.reset()

    # Since reset deepcopies attack graph we
    # need to fetch attacker again
    attacker = sim.attack_graph.attackers[0]

    # Observe and reward with no new actions
    attacker_agent.update_obs([], [])

    for index, state in enumerate(
            attacker_agent.observation['observed_state']):

        node = attacker_agent.get_attack_graph_node_by_index(index)
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

    assert attacker_agent.reward == 0


