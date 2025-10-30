from malsim.envs.graph.graph_env import (
    GraphAttackerEnv, GraphDefenderEnv, register_graph_envs
)
from malsim.scenario import Scenario
from malsim.mal_simulator import MalSimulatorSettings, TTCMode, RewardMode
from gymnasium.utils.env_checker import check_env

import pytest

@pytest.mark.parametrize(
    ("sim_settings", "use_logic_gates"),
    [
        (
            MalSimulatorSettings(
                ttc_mode=TTCMode.PER_STEP_SAMPLE,
                run_defense_step_bernoullis=False,
                run_attack_step_bernoullis=False,
                attack_surface_skip_unnecessary=False,
                attacker_reward_mode=RewardMode.ONE_OFF,
            ),
            True
        ),
        (
            MalSimulatorSettings(
                ttc_mode=TTCMode.PER_STEP_SAMPLE,
                run_defense_step_bernoullis=False,
                run_attack_step_bernoullis=False,
                attack_surface_skip_unnecessary=False,
                attacker_reward_mode=RewardMode.ONE_OFF,
            ),
            False
        ),
        (
            MalSimulatorSettings(
                ttc_mode=TTCMode.PRE_SAMPLE,
                run_defense_step_bernoullis=True,
                run_attack_step_bernoullis=True,
                attack_surface_skip_unnecessary=True,
                attacker_reward_mode=RewardMode.CUMULATIVE,
            ),
            True
        ),
        (
            MalSimulatorSettings(
                ttc_mode=TTCMode.PRE_SAMPLE,
                run_defense_step_bernoullis=True,
                run_attack_step_bernoullis=True,
                attack_surface_skip_unnecessary=True,
                attacker_reward_mode=RewardMode.CUMULATIVE,
            ),
            False
        ),
        (
            MalSimulatorSettings(
                ttc_mode=TTCMode.EXPECTED_VALUE,
                run_defense_step_bernoullis=False,
                run_attack_step_bernoullis=True,
                attack_surface_skip_unnecessary=True,
                attacker_reward_mode=RewardMode.ONE_OFF,
            ),
            True
        ),
        (
            MalSimulatorSettings(
                ttc_mode=TTCMode.EXPECTED_VALUE,
                run_defense_step_bernoullis=False,
                run_attack_step_bernoullis=True,
                attack_surface_skip_unnecessary=True,
                attacker_reward_mode=RewardMode.ONE_OFF,
            ),
            False
        ),
        (
            MalSimulatorSettings(
                ttc_mode=TTCMode.DISABLED,
                run_defense_step_bernoullis=False,
                run_attack_step_bernoullis=False,
                attack_surface_skip_unnecessary=False,
                attack_surface_skip_unviable=False,
            ),
            True
        ),
        (
            MalSimulatorSettings(
                ttc_mode=TTCMode.DISABLED,
                run_defense_step_bernoullis=False,
                run_attack_step_bernoullis=False,
                attack_surface_skip_unnecessary=False,
                attack_surface_skip_unviable=False,
            ),
            False
        ),
    ]
)
def test_check_graph_env(sim_settings, use_logic_gates) -> None:
    scenario_file = (
        "tests/testdata/scenarios/simple_scenario.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    register_graph_envs(scenario, use_logic_gates=use_logic_gates, sim_settings=sim_settings)
    env = GraphAttackerEnv(scenario, use_logic_gates=use_logic_gates, sim_settings=sim_settings)
    check_env(env, skip_render_check=True, skip_close_check=True)

    env = GraphDefenderEnv(
        scenario,
        use_logic_gates=use_logic_gates,
        sim_settings=sim_settings
    )
    check_env(env, skip_render_check=True, skip_close_check=True)

