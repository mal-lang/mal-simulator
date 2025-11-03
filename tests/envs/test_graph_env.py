from malsim.envs.graph.graph_env import (
    GraphAttackerEnv, GraphDefenderEnv, register_graph_envs
)
from malsim.scenario import Scenario
from typing import Any
from malsim.mal_simulator import MalSimulatorSettings, TTCMode, RewardMode, MalSimAttackerState
from gymnasium.utils.env_checker import check_env
import numpy as np
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
def test_check_graph_env(sim_settings: MalSimulatorSettings, use_logic_gates: bool) -> None:
    scenario_file = (
        "tests/testdata/scenarios/simple_scenario.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    register_graph_envs(scenario, use_logic_gates=use_logic_gates, sim_settings=sim_settings)
    env: Any = GraphAttackerEnv(scenario, use_logic_gates=use_logic_gates, sim_settings=sim_settings)
    check_env(env, skip_render_check=True, skip_close_check=True)

    env = GraphDefenderEnv(
        scenario,
        use_logic_gates=use_logic_gates,
        sim_settings=sim_settings
    )
    check_env(env, skip_render_check=True, skip_close_check=True)


def test_attacker_episode() -> None:
    scenario_file = (
        "tests/testdata/scenarios/traininglang_scenario_with_model.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    attacker_env = GraphAttackerEnv(scenario, use_logic_gates=True, sim_settings=MalSimulatorSettings(
        ttc_mode=TTCMode.PER_STEP_SAMPLE,
        run_defense_step_bernoullis=False,
        run_attack_step_bernoullis=False,
        attack_surface_skip_unnecessary=False,
        attacker_reward_mode=RewardMode.ONE_OFF,
    ))
    ser = attacker_env.lang_serializer

    done = False
    obs, info = attacker_env.reset()
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = attacker_env.step(attacker_env.action_space.sample(obs.steps.action_mask))
        state = info["state"]
        assert isinstance(state, MalSimAttackerState)
        visible_assets = {node.model_asset for node in state.action_surface} | {node.model_asset for node in state.performed_nodes}
        visible_steps = {node for node in state.sim.attack_graph.nodes.values() if node.model_asset in visible_assets and node.type in ('and', 'or')}
        for node in visible_steps:
            assert node.id in obs.steps.id
            node_idx = np.where(obs.steps.id == node.id)[0][0]
            assert node.id == obs.steps.id[node_idx]
            if ser.split_attack_step_types and node.model_asset:
                assert ser.attack_step_type[(node.model_asset.type, node.name)] == obs.steps.type[node_idx]
            else:
                assert ser.attack_step_type[(node.name,)] == obs.steps.type[node_idx]
            assert ser.attack_step_class[node.type] == obs.steps.logic_class[node_idx]
            assert ser.attack_step_tag[node.tags[0] if len(node.tags) > 0 else None] == obs.steps.tags[node_idx]
            assert state.sim.node_is_compromised(node) == obs.steps.compromised[node_idx]
            assert obs.steps.attempts is not None and state.num_attempts.get(node, 0) == obs.steps.attempts[node_idx]
            assert state.sim.node_is_traversable(state.performed_nodes, node) == obs.steps.action_mask[node_idx]

        steps += 1
        done = terminated or truncated or (steps > 10_000)

    done = False
    obs, info = attacker_env.reset()
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = attacker_env.step(attacker_env.action_space.sample(obs.steps.action_mask))
        state = info["state"]
        assert isinstance(state, MalSimAttackerState)
        visible_assets = {node.model_asset for node in state.action_surface} | {node.model_asset for node in state.performed_nodes}
        visible_steps = {node for node in state.sim.attack_graph.nodes.values() if node.model_asset in visible_assets and node.type in ('and', 'or')}
        for node in visible_steps:
            assert node.id in obs.steps.id
            node_idx = np.where(obs.steps.id == node.id)[0][0]
            assert node.id == obs.steps.id[node_idx]
            if ser.split_attack_step_types and node.model_asset:
                assert ser.attack_step_type[(node.model_asset.type, node.name)] == obs.steps.type[node_idx]
            else:
                assert ser.attack_step_type[(node.name,)] == obs.steps.type[node_idx]
            assert ser.attack_step_class[node.type] == obs.steps.logic_class[node_idx]
            assert ser.attack_step_tag[node.tags[0] if len(node.tags) > 0 else None] == obs.steps.tags[node_idx]
            assert state.sim.node_is_compromised(node) == obs.steps.compromised[node_idx]
            assert obs.steps.attempts is not None and state.num_attempts.get(node, 0) == obs.steps.attempts[node_idx]
            assert state.sim.node_is_traversable(state.performed_nodes, node) == obs.steps.action_mask[node_idx]

        steps += 1
        done = terminated or truncated or (steps > 10_000)

def test_defender_episode() -> None:
    scenario_file = (
        "tests/testdata/scenarios/traininglang_scenario_with_model.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    defender_env = GraphDefenderEnv(scenario, use_logic_gates=True, sim_settings=MalSimulatorSettings(
        ttc_mode=TTCMode.PER_STEP_SAMPLE,
        run_defense_step_bernoullis=False,
        run_attack_step_bernoullis=False,
        attack_surface_skip_unnecessary=False,
        attacker_reward_mode=RewardMode.ONE_OFF,
    ))

    done = False
    obs, info = defender_env.reset()
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = defender_env.step(defender_env.action_space.sample(obs.steps.action_mask))
        steps += 1
        done = terminated or truncated or (steps > 10_000)

    done = False
    obs, info = defender_env.reset()
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = defender_env.step(defender_env.action_space.sample(obs.steps.action_mask))
        steps += 1
        done = terminated or truncated or (steps > 10_000)