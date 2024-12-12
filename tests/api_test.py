import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest

import logging
import numpy as np
import gymnasium as gym
from gymnasium.utils import env_checker
from pettingzoo.test import parallel_api_test

from malsim.envs.mal_sim_parallel_env import MalSimParallelEnv, MalSimulator
from malsim.wrappers.gym_wrapper import AttackerEnv, DefenderEnv, MaskingWrapper
from malsim.agents import BreadthFirstAttacker, DepthFirstAttacker, PassiveAttacker, PassiveDefender

logger = logging.getLogger(__name__)

AGENT_ATTACKER = 'attacker'
AGENT_DEFENDER = 'defender'
ACTION_TERMINATE = 'terminate'
ACTION_WAIT = 'wait'

scenario_file = 'tests/testdata/scenarios/simple_scenario.yml'
scenario_file_no_defender = 'tests/testdata/scenarios/no_defender_agent_scenario.yml'


def register_gym_agent(agent_id, entry_point):
    if agent_id not in gym.envs.registry.keys():
        gym.register(agent_id, entry_point=entry_point)


def test_pz(env: MalSimParallelEnv):
    logger.debug('Run Parrallel API test.')
    parallel_api_test(env)


# Check that an environment follows Gym API
def test_gym():
    logger.debug('Run Gym Test.')
    register_gym_agent('MALDefenderEnv-v0', entry_point=DefenderEnv)
    env = gym.make(
        'MALDefenderEnv-v0',
        scenario_file=scenario_file,
    )
    env_checker.check_env(env.unwrapped)
    # register_gym_agent('MALAttackerEnv-v0', entry_point=AttackerEnv)
    # env = gym.make(
    #     'MALAttackerEnv-v0',
    #     scenario_file=scenario_file_no_defender,
    # )
    # env_checker.check_env(env.unwrapped)


def test_random_defender_actions():
    register_gym_agent('MALDefenderEnv-v0', entry_point=DefenderEnv)
    env = gym.make(
        'MALDefenderEnv-v0',
        scenario_file=scenario_file,
    )

    def available_steps(x):
        np.flatnonzero(x['action_mask'][1])

    def available_actions(x):
        np.flatnonzero(x['action_mask'][0])

    done = False
    _, info = env.reset()

    while not done:
        # TODO: fix test - these functions don't return anything
        available_s = available_steps(info)
        defense = np.random.choice(1, available_s)
        available_a = available_actions(info)
        action = np.random.choice(1, available_a)
        _, _, term, trunc, info = env.step((action, defense))
        done = term or trunc

def test_episode():
    logger.debug('Run Episode Test.')
    register_gym_agent('MALDefenderEnv-v0', entry_point=DefenderEnv)
    env = gym.make(
        'MALDefenderEnv-v0',
        scenario_file=scenario_file,
    )

    done = False
    obs, info = env.reset()
    step = 0
    _return = 0.0
    while not done:
        obs, reward, term, trunc, info = env.step((0, None))
        done = term or trunc
        logger.debug(f'Step {step}:{obs}, {reward}, {done}, {info}')
        step += 1
        _return += reward

    assert done
    # assert _return < 0.0 # If the defender does nothing then it will get a penalty for being attacked


def test_mask():
    gym.register('MALDefenderEnv-v0', entry_point=DefenderEnv)
    env = gym.make(
        'MALDefenderEnv-v0',
        scenario_file='tests/testdata/scenarios/simple_scenario.yml',
    )
    env_checker.check_env(env.unwrapped)

    env = MaskingWrapper(env)

    obs, info = env.reset()

    print(obs)


def test_defender_penalty():
    register_gym_agent('MALDefenderEnv-v0', entry_point=DefenderEnv)
    env = gym.make(
        'MALDefenderEnv-v0',
        scenario_file=scenario_file,
    )

    _, info = env.reset()
    possible_defense_steps = np.flatnonzero(info['action_mask'][1])
    step = np.random.choice(possible_defense_steps)
    _, reward, _, _, info = env.step((1, step))
    # assert reward < 0 # All defense steps cost something


def test_action_mask():
    register_gym_agent('MALDefenderEnv-v0', entry_point=DefenderEnv)
    env = gym.make(
        'MALDefenderEnv-v0',
        scenario_file=scenario_file,
    )

    _, info = env.reset()

    num_defenses = len(np.flatnonzero(info['action_mask'][1]))
    terminated = False

    while num_defenses > 1 and not terminated:
        action = env.action_space.sample(info['action_mask'])
        p, o = action
        _, _, terminated, _, info = env.step((1, o))
        new_num_defenses = len(np.flatnonzero(info['action_mask'][1]))
        assert new_num_defenses == num_defenses - 1
        num_defenses = new_num_defenses

    pass
    # assert reward < 0 # All defense steps cost something


def test_env_step(env: MalSimParallelEnv) -> None:
    obs, info = env.reset()

    env.register_agent(
        PassiveAttacker('attacker', env.attack_graph.attackers[0].id)
    )
    env.register_agent(
        PassiveDefender('defender')
    )

    attacker_action = env.serialized_action_to_node(
        env.action_space('attacker').sample()
    )

    defender_action = env.serialized_action_to_node(
        env.action_space('defender').sample()
    )

    action = {
        AGENT_ATTACKER: attacker_action,
        AGENT_DEFENDER: defender_action
    }

    obs, reward, terminated, truncated, info = env.step(action)

    assert 'attacker' in obs
    assert 'defender' in obs


def test_check_space_env(env: MalSimParallelEnv) -> None:
    attacker_space = env.observation_space('attacker')
    defender_space = env.observation_space('defender')

    def check_space(space, obs):
        for k, v in obs.items():
            assert k in space.spaces, f'{k} not in {space.spaces}'
            assert space.spaces[k].contains(v), f'{k} {v} not in {space.spaces[k]}'

        assert space.contains(obs)

    obs, _ = env.reset()

    attacker_obs = obs[AGENT_ATTACKER]
    defender_obs = obs[AGENT_DEFENDER]

    check_space(attacker_space, attacker_obs)
    check_space(defender_space, defender_obs)

    obs, *_ = env.step({
        AGENT_ATTACKER: env.serialized_action_to_node((0, 0)),
        AGENT_DEFENDER: env.serialized_action_to_node((0, 0))
    })

    attacker_obs = obs[AGENT_ATTACKER]
    defender_obs = obs[AGENT_DEFENDER]

    check_space(attacker_space, attacker_obs)
    check_space(defender_space, defender_obs)


@pytest.mark.parametrize(
    'attacker_class',
    [
        BreadthFirstAttacker,
        DepthFirstAttacker,
    ],
)
def test_attacker(env: MalSimParallelEnv, attacker_class) -> None:
    """Register attacker, run it"""
    attacker = env.attack_graph.attackers[0]

    attacker_agent_name = attacker_class.__name__
    attacker_agent = attacker_class(
        attacker_agent_name,
        attacker_id=attacker.id,
        agent_config=dict(
            seed=16,
        ),
    )

    env.register_agent(attacker_agent)

    steps, sum_rewards = 0, 0
    step_limit = 1000000
    done = False
    while not done and steps < step_limit:
        attacker_action = attacker_agent.get_next_action()
        _, rewards, terminated, truncated, _ = env.step(
            {
                attacker_agent_name: attacker_action,
                AGENT_DEFENDER: env.serialized_action_to_node((0, None))
            }
        )
        sum_rewards += rewards[attacker_agent_name]
        done = terminated[attacker_agent_name] or truncated[attacker_agent_name]
        steps += 1

    assert done, 'Attacker failed to explore attack steps'


def test_env_multiple_steps(env: MalSimParallelEnv) -> None:
    obs, info = env.reset()
    for _ in range(100):
        attacker_action = env.serialized_action_to_node(
            env.action_space('attacker').sample()
        )
        defender_action = env.serialized_action_to_node(
            env.action_space('defender').sample()
        )
        action = {
            AGENT_ATTACKER: attacker_action,
            AGENT_DEFENDER: defender_action
        }
        obs, reward, terminated, truncated, info = env.step(action)
        assert 'attacker' in obs
        assert 'defender' in obs
