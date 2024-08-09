"""Test MalSimulator class"""
import copy

from maltoolbox.attackgraph import AttackGraph, Attacker
from malsim.sims.mal_simulator import MalSimulator

def test_malsimulator(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    MalSimulator(corelang_lang_graph, model, attack_graph)


def test_malsimulator_num_assets(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    assert (
        sim.num_assets == len(corelang_lang_graph.assets) + sim.offset
    )


def test_malsimulator_num_step_names(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    assert (
        sim.num_step_names == len(
            corelang_lang_graph.attack_steps
        ) + sim.offset
    )


def test_malsimulator_asset_type(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    step = attack_graph.nodes[0]
    assert (
        sim.asset_type(step) == \
            sim._asset_type_to_index[step.asset.type] + sim.offset
    )

def test_malsimulator_step_name(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    step = attack_graph.nodes[0]
    assert (
        sim.step_name(step) == \
            sim._step_name_to_index[step.asset.type + ":" + step.name] + sim.offset
    )


def test_malsimulator_asset_id(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    step = attack_graph.nodes[0]
    assert sim.asset_id(step) == int(step.asset.id)


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
    # The type we get from sim._index_to_asset_type[asset_type_index - sim.offset]
    # should be the same as the asset type of attack step with id index in attack graph
    assert len(blank_observation['asset_type']) == num_objects
    for index, asset_type_index in enumerate(blank_observation['asset_type']):
        # Note: offset is decremented from asset_type_index
        expected_type = sim._index_to_asset_type[asset_type_index - sim.offset]
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
    # TODO: Understand how step_name is created

    expected_num_edges = sum([1 for step in attack_graph.nodes
                                for child in step.children] +
                                # We expect all defenses again (reversed)
                             [1 for step in attack_graph.nodes
                                for child in step.children
                                if step.type == "defense"])
    assert len(blank_observation['edges']) == expected_num_edges


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
    assert formatted == "Can act? Yes\n0\n"

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
    assert formatted == "Can act? No\n0\n1\n"


def test_malsimulator_observation_space(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    observation_space = sim.observation_space()
    assert set(observation_space.keys()) == {
        'is_observable', 'observed_state', 'remaining_ttc',
        'asset_type', 'asset_id', 'step_name', 'edges'
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
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)

    attack_graph_before = sim.attack_graph
    sim.reset()
    attack_graph_after = sim.attack_graph

    # Make sure the attack graph is not the same object but identical
    assert attack_graph_before != attack_graph_after
    assert attack_graph_before._to_dict() == attack_graph_after._to_dict()

def test_malsimulator_init(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    obs, infos = sim.init()
    assert sim._index_to_id
    assert sim._index_to_full_name
    assert sim._id_to_index
    # TODO: test agent populating


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
    sim.init() # Have to do again after register_attacker

    # Can not attack the notPresent step
    defense_step = attack_graph.get_node_by_full_name('OS App:notPresent')
    actions = sim._attacker_step(attacker.name, defense_step.id)
    assert not actions

    # Can attack the attemptUseVulnerability step!
    attack_step = attack_graph.get_node_by_full_name('OS App:attemptUseVulnerability')
    actions = sim._attacker_step(attacker.name, attack_step.id)
    assert actions == [attack_step.id]


# def test_malsimulator_update_viability_with_eviction(corelang_lang_graph, model):
#     pass

def test_malsimulator_defender_step(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)

    defender_name = "defender"
    sim.register_defender(defender_name)
    sim.init() # Have to do again after register

    defense_step = attack_graph.get_node_by_full_name(
        'OS App:notPresent')
    actions = sim._defender_step(defender_name, defense_step.id)
    assert actions == [defense_step.id]

    # Can not defend attack_step
    attack_step = attack_graph.get_node_by_full_name('OS App:attemptUseVulnerability')
    actions = sim._defender_step(defender_name, attack_step.id)
    assert not actions

def test_malsimulator_observe_attacker(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    attacker = Attacker('attacker1', id=0)
    attack_graph.add_attacker(attacker, attacker.id)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)

    sim.register_attacker(attacker.name, attacker.id)
    sim.init() # Have to do again after register_attacker

    agent = attacker.name
    agent_observation = copy.deepcopy(sim._blank_observation)
    # .observe_attacker goes through reached_attack_steps and action surface
    # TODO: set reached attack steps and see the behaviour.
    # TODO: where are reached_attack_steps actually set??
    sim._observe_attacker(agent, agent_observation)


def test_malsimulator_observe_defender(corelang_lang_graph, model):
    """Make sure ._observe_defender observes nodes and set observed state"""
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    defender_name = "defender"
    sim.register_defender(defender_name)
    sim.init() # Have to do again after register

    observation = copy.deepcopy(sim._blank_observation)

    # Assert that observed state is not 1 before observe_defender
    nodes_to_observe = [
        node for node in sim.attack_graph.nodes
        if node.is_enabled_defense() or node.is_compromised()
    ]

    # Assert that observed state is not set before observe_defender
    assert len(nodes_to_observe) == 3
    for node in nodes_to_observe:
        index = sim._id_to_index[node.id]
        # Make sure not observed before
        assert observation["observed_state"][index] == -1

    sim._observe_defender(defender_name, observation)

    # Assert that observed state is 1 after observe_defender
    for node in nodes_to_observe:
        index = sim._id_to_index[node.id]
        # Make sure observed after
        assert observation["observed_state"][index] == 1


def test_malsimulator_observe_and_reward(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    sim._observe_and_reward()
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

    # run init to enable attackers
    obs, infos = sim.init()

    # Run step() with action crafted in test
    action = 1
    step = attack_graph.get_node_by_full_name('OS App:attemptUseVulnerability')
    step_index = sim._id_to_index[step.id]
    actions = {agent_name: (action, step_index)}
    observations, rewards, terminations, truncations, infos = sim.step(actions)
    assert len(observations[agent_name]['observed_state']) == len(attack_graph.nodes)

    # Make sure 'OS App:attemptUseVulnerability' is observed and set to 1 (active)
    assert observations[agent_name]['observed_state'][step_index] == 1
    for child in step.children:
        child_step_index = sim._id_to_index[child.id]
        # Make sure 'OS App:attemptUseVulnerability' children are observed and set to 0 (not active)
        assert observations[agent_name]['observed_state'][child_step_index] == 0

    assert agent_name in sim.action_surfaces
    assert step in sim.action_surfaces[agent_name]
