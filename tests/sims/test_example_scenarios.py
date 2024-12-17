from malsim.scenario import create_simulator_from_scenario

def test_bfs_vs_bfs_state_and_reward():
    sim, agents = create_simulator_from_scenario(
        'tests/testdata/scenarios/bfs_vs_bfs_scenario.yml')
    obs, infos = sim.reset()

    attacker_agent = next(iter(sim.get_attacker_agents()))
    defender_agent = next(iter(sim.get_defender_agents()), None)

    total_reward_defender = 0
    total_reward_attacker = 0
    attacker = sim.attack_graph.get_attacker_by_id(attacker_agent.attacker_id)
    attacker_actions = [n.id for n in attacker.entry_points]
    defender_actions = [n.id for n in sim.attack_graph.nodes
                        if n.is_enabled_defense()]

    while True:
        attacker_action = attacker_agent.get_next_action()
        defender_action = defender_agent.get_next_action()

        if attacker_action:
            attacker_actions.append(int(attacker_action[0].id))
        if defender_action:
            defender_actions.append(int(defender_action[0].id))

        actions = {
            'defender': defender_action,
            'attacker': attacker_action
        }
        sim.step(actions)

        total_reward_defender += defender_agent.reward
        total_reward_attacker += attacker_agent.reward

        if defender_agent.terminated or attacker_agent.terminated:
            break
    
    breakpoint()
    assert attacker_actions == [328, 329, 353, 330, 354, 355, 356, 331, 357, 283, 332, 375, 358, 376, 377]
    assert defender_actions == [68, 249, 324, 325, 349, 350, 396, 397, 421, 422, 423, 457, 0, 31, 88, 113, 144, 181, 212, 252, 276, 326, 327, 351, 352, 374]

    for step_index in attacker_actions:
        node = sim.attack_graph.get_node_by_id(sim._index_to_id[step_index])
        if node.is_compromised():
            assert defender_agent.observation['observed_state'][step_index]

    for step_index in defender_actions:
        assert defender_agent.observation['observed_state'][step_index]

    assert attacker_agent.reward == 0
    assert defender_agent.reward == -31

    assert total_reward_attacker == 0
    assert total_reward_defender == -307
