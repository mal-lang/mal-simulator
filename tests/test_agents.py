"""Test MalSimulator agents"""

import pytest

from malsim.sims.mal_simulator import MalSimulator
from malsim.scenario import load_scenario
from malsim.agents.searchers import BreadthFirstAttacker, DepthFirstAttacker


def run_simulation_n_steps_assert_attacker_actions(
        sim, attacker_agent, seed, expected_actions, n_steps=10):
    """Run simulation n steps with attacker agent taking actions"""

    attacker_agent_id = next(iter(sim.get_attacker_agents()))
    defender_agent_id = next(iter(sim.get_defender_agents()))

    obs, infos = sim.reset(seed=seed)
    chosen_actions = []
    for _ in range(n_steps):
        attacker_action = attacker_agent.compute_action_from_dict(
            obs[attacker_agent_id],
            infos[attacker_agent_id]['action_mask']
        )
        chosen_actions.append(attacker_action)
        actions = {
            attacker_agent_id: attacker_action,
            defender_agent_id: (0, None)}

        obs, _, _, _, infos = sim.step(actions)

    for i, action in enumerate(chosen_actions):
        assert action == expected_actions[i]


def test_bfs_attacker_actions_seed():
    """Make sure MalSimulator bfs agent with seed is deterministic"""
    attack_graph, _ = load_scenario(
        'tests/testdata/scenarios/simple_scenario.yml')

    sim = MalSimulator(
        attack_graph.lang_graph, attack_graph.model, attack_graph)

    attacker_agent_id = "attacker"
    defender_agent_id = "defender"
    sim.register_attacker(attacker_agent_id, 0)
    sim.register_defender(defender_agent_id)

    seed = 1337
    # Seed should yield these actions every time
    expected_actions_seed_1337 = [
        (1, 329), (1, 353), (1, 330), (1, 354), (1, 355),
        (1, 356), (1, 357), (1, 331), (1, 358), (1, 283)
    ]
    for _ in range(10):
        bfs_attacker = BreadthFirstAttacker({'randomize': True, 'seed': seed})
        run_simulation_n_steps_assert_attacker_actions(
            sim, bfs_attacker, seed, expected_actions_seed_1337)

    # Different seed gives different actions
    seed = 1857
    bfs_attacker = BreadthFirstAttacker({'randomize': True, 'seed': seed})
    with pytest.raises(AssertionError):
        run_simulation_n_steps_assert_attacker_actions(
            sim, bfs_attacker, seed, expected_actions_seed_1337)


def test_dfs_attacker_actions_seed():
    """Make sure MalSimulator dfs agent with seed is deterministic"""

    attack_graph, _ = load_scenario(
        'tests/testdata/scenarios/simple_scenario.yml')

    sim = MalSimulator(
        attack_graph.lang_graph, attack_graph.model, attack_graph)

    attacker_agent_id = "attacker"
    defender_agent_id = "defender"
    sim.register_attacker(attacker_agent_id, 0)
    sim.register_defender(defender_agent_id)

    seed = 1337
    # Seed should yield these actions every time
    expected_actions_seed_1337 = [
        (1, 353), (1, 354), (1, 355), (1, 356), (1, 357),
        (1, 358), (1, 460), (1, 461), (1, 400), (1, 401)
    ]
    for _ in range(10):
        bfs_attacker = DepthFirstAttacker({'randomize': True, 'seed': seed})
        run_simulation_n_steps_assert_attacker_actions(
            sim, bfs_attacker, seed, expected_actions_seed_1337)

    # Different seed gives different actions
    seed = 1697
    bfs_attacker = DepthFirstAttacker({'randomize': True, 'seed': seed})
    with pytest.raises(AssertionError):
        run_simulation_n_steps_assert_attacker_actions(
            sim, bfs_attacker, seed, expected_actions_seed_1337)
