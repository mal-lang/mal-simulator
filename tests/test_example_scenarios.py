from malsim.scenario import create_simulator_from_scenario

def test_bfs_vs_bfs_state_and_reward():
    sim, sim_config = create_simulator_from_scenario(
        'tests/testdata/scenarios/bfs_vs_bfs_scenario.yml')
    obs, infos = sim.reset()

    attacker_agent_id = next(iter(sim.get_attacker_agents()))
    defender_agent_id = next(iter(sim.get_defender_agents()), None)

    # Initialize defender and attacker according to classes
    defender_class = sim_config['agents'][defender_agent_id]['agent_class']
    defender_agent = defender_class({})

    attacker_class = sim_config['agents'][attacker_agent_id]['agent_class']
    attacker_agent = attacker_class({})

    total_reward_defender = 0
    total_reward_attacker = 0

    attacker = sim.attack_graph.attackers[0]
    attacker_actions = [sim._id_to_index[n.id] for n in attacker.entry_points]
    defender_actions = [sim._id_to_index[n.id]
                        for n in sim.attack_graph.nodes
                        if n.is_enabled_defense()]

    while True:
        defender_action = defender_agent.compute_action_from_dict(
                          obs[defender_agent_id],
                          infos[defender_agent_id]["action_mask"])

        attacker_action = attacker_agent.compute_action_from_dict(
                          obs[attacker_agent_id],
                          infos[attacker_agent_id]["action_mask"])

        if attacker_action[0]:
            attacker_actions.append(int(attacker_action[1]))
        if defender_action[0]:
            defender_actions.append(int(defender_action[1]))

        actions = {
            'defender': defender_action,
            'attacker': attacker_action
        }
        obs, rewards, terminated, truncated, infos = sim.step(actions)

        total_reward_defender += rewards.get(defender_agent_id, 0)
        total_reward_attacker += rewards.get(attacker_agent_id, 0)

        if terminated[defender_agent_id] or terminated[attacker_agent_id]:
            break

    assert attacker_actions == [328, 329, 353, 330, 354, 355, 356, 331, 357, 283, 332, 375, 358, 376, 377]
    assert defender_actions == [68, 249, 324, 325, 349, 350, 396, 397, 421, 422, 423, 457, 0, 31, 88, 113, 144, 181, 212, 252, 276, 326, 327, 351, 352, 374]

    assert rewards[attacker_agent_id] == 0
    assert rewards[defender_agent_id] == -31

    assert total_reward_attacker == 0
    assert total_reward_defender == -307


def test_bfs_vs_tripwire_state_and_reward():
    sim, sim_config = create_simulator_from_scenario(
        'tests/testdata/scenarios/demo1_bfs_attacker_vs_tripwire_defender.yml'
    )
    obs, infos = sim.reset()

    attacker_agent_id = next(iter(sim.get_attacker_agents()))
    defender_agent_id = next(iter(sim.get_defender_agents()), None)

    # Initialize defender and attacker according to classes
    defender_class = sim_config['agents'][defender_agent_id]['agent_class']
    defender_agent = defender_class({}, simulator=sim)

    attacker_class = sim_config['agents'][attacker_agent_id]['agent_class']
    attacker_agent = attacker_class({})

    total_reward_defender = 0
    total_reward_attacker = 0

    attacker = sim.attack_graph.attackers[0]
    attacker_actions = [sim._id_to_index[n.id] for n in attacker.entry_points]
    defender_actions = [sim._id_to_index[n.id]
                        for n in sim.attack_graph.nodes
                        if n.is_enabled_defense()]

    while True:
        # Run through the full simulation
        defender_action = defender_agent.compute_action_from_dict(
                          obs[defender_agent_id],
                          infos[defender_agent_id]["action_mask"])

        attacker_action = attacker_agent.compute_action_from_dict(
                          obs[attacker_agent_id],
                          infos[attacker_agent_id]["action_mask"])

        if attacker_action[0]:
            attacker_actions.append(int(attacker_action[1]))
        if defender_action[0]:
            defender_actions.append(int(defender_action[1]))

        actions = {
            'defender': defender_action,
            'attacker': attacker_action
        }
        obs, rewards, terminated, truncated, infos = sim.step(actions)

        total_reward_defender += rewards.get(defender_agent_id, 0)
        total_reward_attacker += rewards.get(attacker_agent_id, 0)

        if terminated[defender_agent_id] or terminated[attacker_agent_id]:
            break

    enabled_defense_nodes = []
    for step in defender_actions:
        # Make sure all defenses chosen by TripWireAgent are notPresent
        defense_node = sim.attack_graph.get_node_by_id(
            sim._index_to_id[step])

        enabled_defense_nodes.append(defense_node)
        assert defense_node.name == "notPresent"

    for step in attacker_actions:
        # Make sure all attacks chosen by attacker are also defended against
        # if they have a 'notPresent' defense step
        attack_node = sim.attack_graph.get_node_by_id(
            sim._index_to_id[step])

        not_present_defense = sim.attack_graph.get_node_by_full_name(
            attack_node.asset.name + ":notPresent")
        if not_present_defense:
            assert not_present_defense in enabled_defense_nodes

    # Check rewards
    assert rewards[attacker_agent_id] == 0
    assert rewards[defender_agent_id] == -11

    # Check attacker observed state
    for i, state in enumerate(obs[attacker_agent_id]['observed_state']):
        node = sim.attack_graph.get_node_by_id(
            sim._index_to_id[i]
        )
        if state == -1:
            assert not node.is_compromised()
        elif state == 0:
            assert not node.is_compromised()
        elif state == 1:
            assert node.is_compromised()

    # Check defender observed state
    for i, state in enumerate(obs[defender_agent_id]['observed_state']):
        node = sim.attack_graph.get_node_by_id(
            sim._index_to_id[i]
        )
        if state == -1:
            assert False # Defender always knows..
        elif state == 0:
            assert not node.is_compromised() and not node.is_enabled_defense()
        elif state == 1:
            assert node.is_compromised() or node.is_enabled_defense()
