"""
Run a scenario and make sure expected actions are chosen and
expected reward is given to agents
"""

from malsim.scenario import create_simulator_from_scenario
from malsim.sims import MalSimVectorizedObsEnv

def test_bfs_vs_bfs_state_and_reward():
    sim, agents = create_simulator_from_scenario(
        'tests/testdata/scenarios/bfs_vs_bfs_scenario.yml',
    )
    env = MalSimVectorizedObsEnv(sim)
    obs, infos = env.reset()

    defender_agent_name = 'defender1'
    attacker_agent_name = 'attacker1'

    attacker_agent_info = next(
        agent for agent in agents
        if agent['name'] == attacker_agent_name
    )
    defender_agent_info = next(
        agent for agent in agents
        if agent['name'] == defender_agent_name
    )

    attacker_agent = attacker_agent_info['agent']
    defender_agent = defender_agent_info['agent']

    total_reward_defender = 0
    total_reward_attacker = 0

    attacker = env.attack_graph.get_attacker_by_id(attacker_agent_info['attacker_id'])
    attacker_actions = [env.node_to_index(n) for n in attacker.entry_points]
    defender_actions = [env.node_to_index(n) for n in env.attack_graph.nodes
                        if n.is_enabled_defense()]

    while True:
        attacker_agent_state = env.get_agent(attacker_agent_info['name'])
        attacker_node = attacker_agent.get_next_action(
            attacker_agent_state
        )
        defender_agent_state = env.get_agent(defender_agent_info['name'])
        defender_node = defender_agent.get_next_action(
            defender_agent_state
        )

        attacker_action = (0, None)
        defender_action = (0, None)

        if attacker_node:
            node_index = env.node_to_index(attacker_node)
            attacker_action = (1, node_index)
            attacker_actions.append(node_index)

        if defender_node:
            node_index = env.node_to_index(defender_node)
            defender_action = (1, node_index)
            defender_actions.append(node_index)

        actions = {
            defender_agent_name: defender_action,
            attacker_agent_name: attacker_action
        }
        obs, rew, trunc, term, infos = env.step(actions)

        total_reward_defender += rew[defender_agent_name]
        total_reward_attacker += rew[attacker_agent_name]
        if term[defender_agent_name] or term[attacker_agent_name]:
            break

        if trunc[defender_agent_name] or trunc[attacker_agent_name]:
            break

    assert attacker_actions == [328, 329, 353, 330, 354, 355, 356, 357, 331, 358, 375, 283, 332, 376, 377]
    assert defender_actions == [68, 249, 324, 325, 349, 350, 396, 397, 421, 422, 423, 457, 0, 31, 88, 113, 144, 181, 212, 252, 276, 326, 327, 351, 352, 374]

    for step_index in attacker_actions:
        node = env.index_to_node(step_index)
        if node.is_compromised():
            assert obs[defender_agent_name]['observed_state'][step_index]

    for step_index in defender_actions:
        assert obs[defender_agent_name]['observed_state'][step_index]

    assert rew[attacker_agent_name] == 0
    assert rew[defender_agent_name] == -31

    assert total_reward_attacker == 0
    assert total_reward_defender == -307
