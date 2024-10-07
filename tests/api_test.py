import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest

import logging
import numpy as np
import gymnasium as gym
from gymnasium.utils import env_checker
from pettingzoo.test import parallel_api_test

from malsim.sims.mal_simulator import MalSimulator
from malsim.wrappers.gym_wrapper import AttackerEnv, DefenderEnv
from malsim.agents.searchers import BreadthFirstAttacker, DepthFirstAttacker

logger = logging.getLogger(__name__)

AGENT_ATTACKER = "attacker"
AGENT_DEFENDER = "defender"
ACTION_TERMINATE = "terminate"
ACTION_WAIT = "wait"

scenario_file='tests/testdata/scenarios/simple_scenario.yml'
scenario_file_no_defender='tests/testdata/scenarios/no_defender_agent_scenario.yml'

def register_gym_agent(agent_id, entry_point):
    if agent_id not in gym.envs.registry.keys():
        gym.register(agent_id, entry_point=entry_point)

def test_pz(env: MalSimulator):
    logger.debug("Run Parrallel API test.")
    parallel_api_test(env)

#Check that an environment follows Gym API
def test_gym():
    logger.debug("Run Gym Test.")
    register_gym_agent("MALDefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make(
        "MALDefenderEnv-v0",
        scenario_file=scenario_file,
        unholy=False,
    )
    env_checker.check_env(env.unwrapped)
    register_gym_agent("MALAttackerEnv-v0", entry_point=AttackerEnv)
    env = gym.make(
        "MALAttackerEnv-v0",
        scenario_file=scenario_file_no_defender,
    )
    env_checker.check_env(env.unwrapped)


def test_random_defender_actions():
    register_gym_agent("MALDefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make(
        "MALDefenderEnv-v0",
        scenario_file=scenario_file,
    )
    def available_steps(x):
        np.flatnonzero(x["action_mask"][1])

    def available_actions(x):
        np.flatnonzero(x["action_mask"][0])
    done = False
    _, info = env.reset()

    while not done:
        available_s = available_steps(info)
        defense = np.random.choice(1, available_s)
        available_a = available_actions(info)
        action = np.random.choice(1, available_a)

        _, _, term, trunc, info = env.step((action, defense))
        done = term or trunc


def test_episode():
    logger.debug("Run Episode Test.")
    register_gym_agent("MALDefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make(
        "MALDefenderEnv-v0",
        scenario_file=scenario_file,
        unholy=False,
    )

    done = False
    obs, info = env.reset()
    step = 0
    _return = 0.0
    while not done:
        obs, reward, term, trunc, info = env.step((0, None))
        done = term or trunc
        logger.debug(f"Step {step}:{obs}, {reward}, {done}, {info}")
        step += 1
        _return += reward

    assert done
    #assert _return < 0.0 # If the defender does nothing then it will get a penalty for being attacked


def test_defender_penalty():
    register_gym_agent("MALDefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make(
        "MALDefenderEnv-v0",
        scenario_file=scenario_file,
        unholy=False,
    )

    _, info = env.reset()
    possible_defense_steps = np.flatnonzero(info['action_mask'][1])
    step = np.random.choice(possible_defense_steps)
    _, reward, _, _, info = env.step((1, step))
    #assert reward < 0 # All defense steps cost something


def test_env_step(env: MalSimulator) -> None:
    obs, info = env.reset()
    attacker_action = env.action_space("attacker").sample()
    defender_action = env.action_space("defender").sample()
    action = {AGENT_ATTACKER: attacker_action, AGENT_DEFENDER: defender_action}
    obs, reward, terminated, truncated, info = env.step(action)

    assert "attacker" in obs
    assert "defender" in obs


def test_check_space_env(env : MalSimulator) -> None:
    attacker_space = env.observation_space("attacker")
    defender_space = env.observation_space("defender")

    def check_space(space, obs):
        for k, v in obs.items():
            assert k in space.spaces, f"{k} not in {space.spaces}"
            assert space.spaces[k].contains(v), f"{k} {v} not in {space.spaces[k]}"

        assert space.contains(obs)


    obs, _ = env.reset()

    attacker_obs = obs[AGENT_ATTACKER]
    defender_obs = obs[AGENT_DEFENDER]

    check_space(attacker_space, attacker_obs)
    check_space(defender_space, defender_obs)

    obs, *_ = env.step({AGENT_ATTACKER: (0, 0), AGENT_DEFENDER: (0, 0)})

    attacker_obs = obs[AGENT_ATTACKER]
    defender_obs = obs[AGENT_DEFENDER]

    check_space(attacker_space, attacker_obs)
    check_space(defender_space, defender_obs)


@pytest.mark.parametrize("attacker_class", [BreadthFirstAttacker,DepthFirstAttacker,],)
def test_attacker(env: MalSimulator, attacker_class)-> None :
    obs, info = env.reset()
    attacker = attacker_class(
        dict(
            seed=16,
        )
    )

    steps ,sum_rewards = 0,0
    step_limit = 1000000
    done = False
    while not done and steps < step_limit:
        action = attacker.compute_action_from_dict(obs[AGENT_ATTACKER], info[AGENT_ATTACKER]["action_mask"])
        assert action != ACTION_TERMINATE
        assert action != ACTION_WAIT
        obs, rewards, terminated, truncated, info = env.step({AGENT_ATTACKER: action, AGENT_DEFENDER: [0]})
        sum_rewards += rewards[AGENT_ATTACKER]
        done = terminated[AGENT_ATTACKER] or truncated[AGENT_ATTACKER]
        steps += 1

    assert done, "Attacker failed to explore attack steps"


def test_env_multiple_steps(env: MalSimulator) -> None:
    obs, info = env.reset()
    for _ in range(100):
        attacker_action = env.action_space("attacker").sample()
        defender_action = env.action_space("defender").sample()
        action = {AGENT_ATTACKER: attacker_action, AGENT_DEFENDER: defender_action}
        obs, reward, terminated, truncated, info = env.step(action)
        assert "attacker" in obs
        assert "defender" in obs

