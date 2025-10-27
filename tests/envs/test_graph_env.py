from malsim.envs.graph_env import GraphAttackerEnv, register_envs
from malsim.scenario import Scenario
from malsim.mal_simulator import MalSimulatorSettings, TTCMode, RewardMode
from gymnasium.utils.env_checker import check_env
import numpy as np
from malsim.envs.mal_spaces import MALObs

def test_graph_env() -> None:
    scenario_file = (
        "tests/testdata/scenarios/simple_scenario.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    register_envs(scenario)
    env = GraphAttackerEnv(scenario, use_logic_gates=False, sim_settings=MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface_skip_unnecessary=False,
            attacker_reward_mode=RewardMode.ONE_OFF,
        ))
    check_env(env, skip_render_check=True, skip_close_check=True)


def test_obs_in_space() -> None:
    scenario_file = (
        "tests/testdata/scenarios/simple_scenario.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    register_envs(scenario)
    env = GraphAttackerEnv(scenario, use_logic_gates=False, sim_settings=MalSimulatorSettings(
        ttc_mode=TTCMode.DISABLED,
        run_defense_step_bernoullis=False,
        run_attack_step_bernoullis=False,
        attack_surface_skip_unnecessary=False,
        attacker_reward_mode=RewardMode.ONE_OFF,
    ))
    obs_space: MALObs = env.observation_space

    done = False
    obs, info = env.reset()
    while not done:
        traversable = obs.attack_steps.traversable
        valid_actions = np.where(traversable)[0]
        if len(valid_actions) > 0:
            action = obs.attack_steps.id[np.random.choice(valid_actions)]
        else:
            action = env.action_space.sample()
            done = True
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.time in obs_space.time
        assert all(obs.assets.type[i] in obs_space.asset_type for i in range(len(obs.assets.type)))
        assert all(obs.attack_steps.type[i] in obs_space.attack_step_type for i in range(len(obs.attack_steps.type)))
        assert all(obs.attack_steps.logic_class[i] in obs_space.attack_step_class for i in range(len(obs.attack_steps.logic_class)))
        assert all(obs.attack_steps.tags[i] in obs_space.attack_step_tags for i in range(len(obs.attack_steps.tags)))
        assert all(obs.attack_steps.compromised[i] in obs_space.attack_step_compromised for i in range(len(obs.attack_steps.compromised)))
        assert all(obs.attack_steps.attempts[i] in obs_space.attack_step_attempts for i in range(len(obs.attack_steps.attempts)))
        assert all(obs.attack_steps.traversable[i] in obs_space.attack_step_traversable for i in range(len(obs.attack_steps.traversable)))
        if obs.associations is not None:
            assert all(obs.associations.type[i] in obs_space.association_type for i in range(len(obs.associations.type)))
        if env.use_logic_gates and obs.logic_gates is not None:
            assert all(obs.logic_gates.type[i] in obs_space.logic_gate_type for i in range(len(obs.logic_gates.type)))
        done = terminated or truncated

    env.close()

def test_obs_in_space_w_logic_gates() -> None:
    scenario_file = (
        "tests/testdata/scenarios/simple_scenario.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    register_envs(scenario)
    env = GraphAttackerEnv(scenario, use_logic_gates=True, sim_settings=MalSimulatorSettings(
        ttc_mode=TTCMode.DISABLED,
        run_defense_step_bernoullis=False,
        run_attack_step_bernoullis=False,
        attack_surface_skip_unnecessary=False,
        attacker_reward_mode=RewardMode.ONE_OFF,
    ))
    obs_space: MALObs = env.observation_space

    done = False
    obs, info = env.reset()
    while not done:
        traversable = obs.attack_steps.traversable
        valid_actions = np.where(traversable)[0]
        if len(valid_actions) > 0:
            action = obs.attack_steps.id[np.random.choice(valid_actions)]
        else:
            action = env.action_space.sample()
            done = True
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.time in obs_space.time
        assert all(obs.assets.type[i] in obs_space.asset_type for i in range(len(obs.assets.type)))
        assert all(obs.attack_steps.type[i] in obs_space.attack_step_type for i in range(len(obs.attack_steps.type)))
        assert all(obs.attack_steps.logic_class[i] in obs_space.attack_step_class for i in range(len(obs.attack_steps.logic_class)))
        assert all(obs.attack_steps.tags[i] in obs_space.attack_step_tags for i in range(len(obs.attack_steps.tags)))
        assert all(obs.attack_steps.compromised[i] in obs_space.attack_step_compromised for i in range(len(obs.attack_steps.compromised)))
        assert all(obs.attack_steps.attempts[i] in obs_space.attack_step_attempts for i in range(len(obs.attack_steps.attempts)))
        assert all(obs.attack_steps.traversable[i] in obs_space.attack_step_traversable for i in range(len(obs.attack_steps.traversable)))
        if obs.associations is not None:
            assert all(obs.associations.type[i] in obs_space.association_type for i in range(len(obs.associations.type)))
        if env.use_logic_gates and obs.logic_gates is not None:
            assert all(obs.logic_gates.type[i] in obs_space.logic_gate_type for i in range(len(obs.logic_gates.type)))
        done = terminated or truncated

    env.close()