"""
Run a scenario and make sure expected actions are chosen and
expected reward is given to agents
"""

from malsim.scenario import create_simulator_from_scenario
from malsim.sims import VectorizedObsMalSimulator

def test_bfs_vs_bfs_state_and_reward():
    sim, agents = create_simulator_from_scenario(
        'tests/testdata/scenarios/bfs_vs_bfs_scenario.yml',
        sim_class=VectorizedObsMalSimulator
    )
    obs, infos = sim.reset()

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

    attacker = sim.attack_graph.get_attacker_by_id(attacker_agent_info['attacker_id'])
    attacker_actions = [sim.node_to_index(n) for n in attacker.entry_points]
    defender_actions = [sim.node_to_index(n) for n in sim.attack_graph.nodes
                        if n.is_enabled_defense()]

    while True:
        attacker_action = attacker_agent\
            .get_next_action(sim.get_agent(attacker_agent_info['name']))
        defender_action = defender_agent\
            .get_next_action(sim.get_agent(defender_agent_info['name']))

        if attacker_action:
            attacker_actions.append(
                int(attacker_action[1])
            )
        if defender_action:
            defender_actions.append(
                int(defender_action[1])
            )
        actions = {
            defender_agent_name: defender_action,
            attacker_agent_name: attacker_action
        }
        obs, rew, trunc, term, infos = sim.step(actions)

        total_reward_defender += rew[defender_agent_name]
        total_reward_attacker += rew[attacker_agent_name]

        if term[defender_agent_name] or term[attacker_agent_name]:
            break

        if trunc[defender_agent_name] or trunc[attacker_agent_name]:
            break
        
    assert attacker_actions == [328, 329, 353, 330, 354, 355, 356, 331, 357, 283, 332, 375, 358, 376, 377]
    assert defender_actions == [68, 249, 324, 325, 349, 350, 396, 397, 421, 422, 423, 457, 0, 31, 88, 113, 144, 181, 212, 252, 276, 326, 327, 351, 352, 374]

    for step_index in attacker_actions:
        node = sim.index_to_node(step_index)
        if node.is_compromised():
            assert obs[defender_agent_name]['observed_state'][step_index]

    for step_index in defender_actions:
        assert obs[defender_agent_name]['observed_state'][step_index]

    assert rew[attacker_agent_name] == 0
    assert rew[defender_agent_name] == -31

    assert total_reward_attacker == 0
    assert total_reward_defender == -307
