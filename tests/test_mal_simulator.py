"""Test MalSimulator class"""
import copy

from maltoolbox.attackgraph import AttackGraph, Attacker
from malsim.sims.mal_simulator import MalSimulator
from malsim.scenario import load_scenario
from malsim.scenario import create_simulator_from_scenario
from malsim.sims import MalSimulatorSettings

def test_malsimulator(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    MalSimulator(corelang_lang_graph, model, attack_graph)


def test_malsimulator_create_blank_observation(corelang_lang_graph, model):
    """Make sure blank observation contains correct default values"""
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)

    num_objects = len(attack_graph.nodes)
    blank_observation = sim.create_blank_observation()

    assert len(blank_observation['is_observable']) == num_objects

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
        node_id = sim._index_to_id[index]
        node = attack_graph.get_node_by_id(node_id)
        assert node.asset.type == expected_type

    # asset_id on index X in blank_observation['asset_id']
    # should be the same as the id of the asset of attack step X
    assert len(blank_observation['asset_id']) == num_objects
    for index, expected_asset_id in enumerate(blank_observation['asset_id']):
        node_id = sim._index_to_id[index]
        node = attack_graph.get_node_by_id(node_id)
        assert node.asset.id == expected_asset_id

    assert len(blank_observation['step_name']) == num_objects

    expected_num_edges = sum([1 for step in attack_graph.nodes
                                for child in step.children] +
                                # We expect all defenses again (reversed)
                             [1 for step in attack_graph.nodes
                                for child in step.children
                                if step.type == "defense"])
    assert len(blank_observation['attack_graph_edges']) == expected_num_edges


def test_malsimulator_format_info(corelang_lang_graph, model):
    """Make sure format info works as expected"""
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)

    # Preparations of info to send to _format_info
    can_wait = {"attacker": 0, "defender": 1}
    infos = {}
    agent = "attacker1"
    agent_type = "attacker"
    can_act = 1
    available_actions = [0] * len(attack_graph.nodes)
    available_actions[0] = 1  # Only first action is available

    infos[agent] = {
        "action_mask": (
            [can_wait[agent_type], can_act],
            available_actions
        )
    }
    formatted = sim._format_info(infos[agent])
    assert formatted == "Can act? Yes\n0 OS App:notPresent\n"

    # Add an action and change 'can_act' to false
    available_actions[1] = 1  # Also second action is available
    can_act = False
    infos[agent] = {
        "action_mask": (
            [can_wait[agent_type], can_act],
            available_actions
        )
    }
    formatted = sim._format_info(infos[agent])
    assert formatted == "Can act? No\n0 OS App:notPresent\n1 OS App:attemptUseVulnerability\n"


def test_malsimulator_observation_space(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    observation_space = sim.observation_space()
    assert set(observation_space.keys()) == {
        'is_observable', 'observed_state', 'remaining_ttc',
        'asset_type', 'asset_id', 'step_name', 'attack_graph_edges'
    }
    # All values in the observation space dict are of type Box
    # which comes from gymnasium.spaces (Box is a Space)
    # spaces have a shape (tuple) and a datatype (from numpy)


def test_malsimulator_action_space(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    action_space = sim.action_space()
    # action_space is a 'MultiDiscrete' (from gymnasium.spaces)
    assert action_space.shape == (2,)


def test_malsimulator_reset(corelang_lang_graph, model):
    """Make sure attack graph is reset"""
    attack_graph = AttackGraph(corelang_lang_graph, model)
    agent_name = "testagent"
    agent_id = 0
    agent_entry_point = attack_graph.get_node_by_full_name(
        'OS App:networkConnectUninspected')

    attacker = Attacker(
        agent_name,
        entry_points=[agent_entry_point],
        reached_attack_steps=[agent_entry_point]
    )

    attack_graph.add_attacker(attacker, agent_id)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)

    assert sim._index_to_id
    assert sim._index_to_full_name
    assert sim._id_to_index

    attack_graph_before = sim.attack_graph
    sim.register_attacker(agent_name, agent_id)
    assert agent_name in sim.possible_agents
    assert agent_name in sim.agents_dict
    assert agent_name not in sim.agents

    sim.reset()

    attack_graph_after = sim.attack_graph

    # Make sure agent was added (and not removed)
    assert agent_name in sim.agents
    # Make sure the attack graph is not the same object but identical
    assert attack_graph_before != attack_graph_after
    assert attack_graph_before._to_dict() == attack_graph_after._to_dict()


def test_malsimulator_register_attacker(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    agent_name = "attacker1"
    attacker = 1
    sim.register_attacker(agent_name, attacker)
    assert agent_name in sim.possible_agents
    assert agent_name in sim.agents_dict


def test_malsimulator_register_defender(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    agent_name = "defender1"
    sim.register_defender(agent_name)
    assert agent_name in sim.possible_agents
    assert agent_name in sim.agents_dict


def test_malsimulator_attacker_step(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)

    attacker = Attacker('attacker1', id=0)
    attack_graph.add_attacker(attacker, attacker.id)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)

    sim.register_attacker(attacker.name, attacker.id)
    sim.reset()

    # Can not attack the notPresent step
    defense_step = attack_graph.get_node_by_full_name('OS App:notPresent')
    actions = sim._attacker_step(attacker.name, defense_step.id)
    assert not actions

    # Can attack the attemptUseVulnerability step!
    attack_step = attack_graph.get_node_by_full_name('OS App:attemptUseVulnerability')
    actions = sim._attacker_step(attacker.name, attack_step.id)
    assert actions == [attack_step.id]


def test_malsimulator_defender_step(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)

    defender_name = "defender"
    sim.register_defender(defender_name)
    sim.reset()

    defense_step = attack_graph.get_node_by_full_name(
        'OS App:notPresent')
    actions = sim._defender_step(defender_name, defense_step.id)
    assert actions == [defense_step.id]

    # Can not defend attack_step
    attack_step = attack_graph.get_node_by_full_name('OS App:attemptUseVulnerability')
    actions = sim._defender_step(defender_name, attack_step.id)
    assert not actions


def test_malsimulator_observe_attacker():
    attack_graph, conf = load_scenario(
        'tests/testdata/scenarios/simple_scenario.yml'
    )

    # Make alteration to the attack graph attacker
    assert len(attack_graph.attackers) == 1
    attacker = attack_graph.attackers[0]
    assert len(attacker.reached_attack_steps) == 1
    reached_step = attacker.reached_attack_steps[0]
    for child_node in reached_step.children:
        if child_node.type in ('and', 'or'):
            # compromise children of reached step so in the end the
            # attacker will have three reached attack steps where
            # two are children of the first one
            attacker.compromise(child_node)
    assert len(attacker.reached_attack_steps) == 3

    #Create the simulator
    sim = MalSimulator(
        attack_graph.lang_graph, attack_graph.model, attack_graph)

    # Register the agents
    attacker_agent_id = "attacker"
    defender_agent_if = "defender"
    sim.register_attacker(attacker_agent_id, 0)
    sim.register_defender(defender_agent_if)

    obs, _ = sim.reset()

    attacker_agent_id = next(iter(sim.get_attacker_agents()))
    attacker_observation = obs[attacker_agent_id]["observed_state"]
    attacker = sim.attack_graph.attackers[0]

    for node in attacker.reached_attack_steps:
        node_index = sim._id_to_index[node.id]
        node_obs_state = attacker_observation[node_index]
        assert node_obs_state == 1

def test_malsimulator_observe_defender(corelang_lang_graph, model):
    """Make sure ._observe_defender observes nodes and set observed state"""
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    defender_name = "defender"
    sim.register_defender(defender_name)
    sim.reset()

    observation = copy.deepcopy(sim._blank_observation)

    # Assert that observed state is not 1 before observe_defender
    nodes_to_observe = [
        node for node in sim.attack_graph.nodes
        if node.is_enabled_defense() or node.is_compromised()
    ]

    # Assert that observed state is not set before observe_defender
    # assert len(nodes_to_observe) == 3 # TODO: why did behavior change here?
    for node in nodes_to_observe:
        index = sim._id_to_index[node.id]
        # Make sure not observed before
        assert observation["observed_state"][index] == -1

    sim._observe_defender(defender_name, [], observation)

    # Assert that observed state is 1 after observe_defender
    for node in nodes_to_observe:
        index = sim._id_to_index[node.id]
        # Make sure observed after
        assert observation["observed_state"][index] == 1


def test_malsimulator_observe_and_reward(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    sim._observe_and_reward([])
    # TODO: test that things happen as they should


def test_malsimulator_update_viability_with_eviction(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    attempt_vuln_node = attack_graph.get_node_by_full_name('OS App:attemptUseVulnerability')
    assert attempt_vuln_node.is_viable
    success_vuln_node = attack_graph.get_node_by_full_name('OS App:successfulUseVulnerability')
    assert success_vuln_node.is_viable

    # Make attempt unviable
    attempt_vuln_node.is_viable = False
    sim.update_viability_with_eviction(attempt_vuln_node)
    # Should make success unviable
    assert not success_vuln_node.is_viable


def test_malsimulator_step(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    attack_graph.attach_attackers()
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)

    agent_name = "attacker1"
    attacker_id = attack_graph.attackers[0].id
    sim.register_attacker(agent_name, attacker_id)
    assert not sim.action_surfaces

    obs, infos = sim.reset()

    # Run step() with action crafted in test
    action = 1
    step = attack_graph.get_node_by_full_name('OS App:attemptUseVulnerability')
    step_index = sim._id_to_index[step.id]
    actions = {agent_name: (action, step_index)}
    observations, rewards, terminations, truncations, infos = sim.step(actions)
    assert len(observations[agent_name]['observed_state']) == len(attack_graph.nodes)
    assert agent_name in sim.action_surfaces

    # Make sure 'OS App:attemptUseVulnerability' is observed and set to 1 (active)
    assert observations[agent_name]['observed_state'][step_index] == 1
    for child in step.children:
        child_step_index = sim._id_to_index[child.id]
        # Make sure 'OS App:attemptUseVulnerability' children are observed and set to 0 (not active)
        assert observations[agent_name]['observed_state'][child_step_index] == 0

def test_default_simulator_settings_eviction():
    """Test using the MalSimulatorSettings default"""
    sim, _ = create_simulator_from_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml',
    )

    sim.reset()

    attacker = sim.attack_graph.attackers[0]
    attacker_agent_id = next(iter(sim.get_attacker_agents()))
    defender_agent_id = next(iter(sim.get_defender_agents()))

    # Get a step to compromise and its defense parent
    user_3_compromise = sim.attack_graph.get_node_by_full_name('User:3:compromise')
    assert attacker not in user_3_compromise.compromised_by
    user_3_compromise_defense = next(n for n in user_3_compromise.parents if n.type=='defense')
    assert not user_3_compromise_defense.is_enabled_defense()

    # First let the attacker compromise User:3:compromise
    defender_action = (0, None)
    attacker_action = (1, sim._id_to_index[user_3_compromise.id])
    actions = {
        attacker_agent_id: attacker_action,
        defender_agent_id: defender_action
    }
    sim.step(actions)

    # Check that the compromise happened and that the defense not
    assert not user_3_compromise_defense.is_enabled_defense()
    assert attacker in user_3_compromise.compromised_by

    # Now let the defender defend, and the attacker waits
    defender_action = (1, sim._id_to_index[user_3_compromise_defense.id])
    attacker_action = (0, None)
    actions = {
        attacker_agent_id: attacker_action,
        defender_agent_id: defender_action
    }
    sim.step(actions)

    # Check that the defense was performed,
    # and that the attacker was NOT kicked out
    assert user_3_compromise_defense.is_enabled_defense()
    assert attacker in user_3_compromise.compromised_by


def test_simulator_settings_evict_attacker():
    """Test using the MalSimulatorSettings when evicting attacker"""

    settings_evict_attacker = MalSimulatorSettings(
        uncompromise_untraversable_steps=True
    )

    sim, _ = create_simulator_from_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml',
        sim_settings=settings_evict_attacker
    )

    sim.reset()

    attacker = sim.attack_graph.attackers[0]
    attacker_agent_id = next(iter(sim.get_attacker_agents()))
    defender_agent_id = next(iter(sim.get_defender_agents()))

   # Get a step to compromise and its defense parent
    user_3_compromise = sim.attack_graph.get_node_by_full_name('User:3:compromise')
    assert attacker not in user_3_compromise.compromised_by
    user_3_compromise_defense = next(n for n in user_3_compromise.parents if n.type=='defense')
    assert not user_3_compromise_defense.is_enabled_defense()

    # First let the attacker compromise User:3:compromise
    defender_action = (0, None)
    attacker_action = (1, sim._id_to_index[user_3_compromise.id])
    actions = {
        attacker_agent_id: attacker_action,
        defender_agent_id: defender_action
    }
    sim.step(actions)

    # Check that the compromise happened and that the defense not
    assert not user_3_compromise_defense.is_enabled_defense()
    assert attacker in user_3_compromise.compromised_by

    # Now let the defender defend, and the attacker waits
    defender_action = (1, sim._id_to_index[user_3_compromise_defense.id])
    attacker_action = (0, None)
    actions = {
        attacker_agent_id: attacker_action,
        defender_agent_id: defender_action
    }
    sim.step(actions)

    # Check that the defense was performed,
    # and that the attacker WAS kicked out
    assert user_3_compromise_defense.is_enabled_defense()
    assert attacker not in user_3_compromise.compromised_by

def test_simulator_default_settings_defender_observation():
    """Test MalSimulatorSettings show previous steps in obs"""

    sim, _ = create_simulator_from_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml'
    )
    sim.reset()

    attacker = sim.attack_graph.attackers[0]
    attacker_agent_id = next(iter(sim.get_attacker_agents()))
    defender_agent_id = next(iter(sim.get_defender_agents()))

   # Get an uncompromised step
    user_3_compromise = sim.attack_graph.get_node_by_full_name(
        'User:3:compromise')
    assert attacker not in user_3_compromise.compromised_by

    # Get a defense for the uncompromised step
    user_3_compromise_defense = next(
        n for n in user_3_compromise.parents if n.type=='defense')
    assert not user_3_compromise_defense.is_enabled_defense()

    # First let the attacker compromise User:3:compromise
    defender_action = (0, None)
    attacker_action = (1, sim._id_to_index[user_3_compromise.id])
    actions = {
        attacker_agent_id: attacker_action,
        defender_agent_id: defender_action
    }

    obs, _, _, _, _ = sim.step(actions)
    defender_observation = obs[defender_agent_id]['observed_state']

    for index, state in enumerate(defender_observation):
        step_id = sim._index_to_id[index]
        node = sim.attack_graph.get_node_by_id(step_id)
        if state == 1:
            assert node.is_compromised()
        else:
            assert not node.is_compromised()

    # Now let the defender defend, and the attacker waits
    defender_action = (1, sim._id_to_index[user_3_compromise_defense.id])
    attacker_action = (0, None)
    actions = {
        attacker_agent_id: attacker_action,
        defender_agent_id: defender_action
    }
    obs, _, _, _, _ = sim.step(actions)
    defender_observation = obs[defender_agent_id]['observed_state']

    for index, state in enumerate(defender_observation):
        step_id = sim._index_to_id[index]
        node = sim.attack_graph.get_node_by_id(step_id)
        if state == 1:
            assert node.is_compromised() or node.is_enabled_defense()
        else:
            assert not node.is_compromised() and not node.is_enabled_defense()

def test_simulator_settings_defender_observation():
    """Test MalSimulatorSettings only show last steps in obs"""

    settings_dont_show_previous = MalSimulatorSettings(
        cumulative_actions_in_defender_obs=False
    )

    sim, _ = create_simulator_from_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml',
        sim_settings=settings_dont_show_previous
    )
    sim.reset()

    attacker = sim.attack_graph.attackers[0]
    attacker_agent_id = next(iter(sim.get_attacker_agents()))
    defender_agent_id = next(iter(sim.get_defender_agents()))

   # Get an uncompromised step
    user_3_compromise = sim.attack_graph.get_node_by_full_name(
        'User:3:compromise')
    assert attacker not in user_3_compromise.compromised_by

    # Get a defense for the uncompromised step
    user_3_compromise_defense = next(
        n for n in user_3_compromise.parents if n.type=='defense')
    assert not user_3_compromise_defense.is_enabled_defense()

    # First let the attacker compromise User:3:compromise
    actions = {
        attacker_agent_id: (0, None),
        defender_agent_id: (1, sim._id_to_index[user_3_compromise.id])
    }

    obs, _, _, _, _ = sim.step(actions)
    defender_observation = obs[defender_agent_id]['observed_state']

    for index, state in enumerate(defender_observation):
        step_id = sim._index_to_id[index]
        node = sim.attack_graph.get_node_by_id(step_id)
        if state == 1:
            assert node == user_3_compromise

    # Now let the defender defend, and the attacker waits
    actions = {
        attacker_agent_id: (0, None),
        defender_agent_id: (1, sim._id_to_index[user_3_compromise_defense.id])
    }
    obs, _, _, _, _ = sim.step(actions)
    defender_observation = obs[defender_agent_id]['observed_state']

    for index, state in enumerate(defender_observation):
        step_id = sim._index_to_id[index]
        node = sim.attack_graph.get_node_by_id(step_id)
        if state == 1:
            assert node == user_3_compromise_defense
