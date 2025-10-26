from malsim.envs.graph_env import GraphAttackerEnv, register_envs
from malsim.scenario import Scenario
from malsim.mal_simulator import MalSimulatorSettings, TTCMode, RewardMode
from gymnasium.utils.env_checker import check_env

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
    check_env(env)