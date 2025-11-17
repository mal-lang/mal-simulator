from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from malsim import MalSimulator
from malsim.envs.graph.graph_env import (
    MalSimAttackerGraph,
    MalSimDefenderGraph,
    MalSimGraph,
    register_graph_envs,
)
from malsim.envs.graph.mal_spaces import (
    AssetThenAttackerAction,
    AssetThenDefenderAction,
    AttackerActionThenAsset,
    DefenderActionThenAsset,
)
from malsim.scenario import Scenario
from typing import Any
from malsim.mal_simulator import (
    MalSimulatorSettings,
    TTCMode,
    RewardMode,
    MalSimAttackerState,
)
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest
from pettingzoo.test import parallel_api_test
from malsim.envs.graph.wrapper import ActionThenAssetWrapper, AssetThenActionWrapper


@pytest.mark.parametrize(
    'sim_settings',
    [
        MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface_skip_unnecessary=False,
            attacker_reward_mode=RewardMode.ONE_OFF,
        ),
        MalSimulatorSettings(
            ttc_mode=TTCMode.PRE_SAMPLE,
            run_defense_step_bernoullis=True,
            run_attack_step_bernoullis=True,
            attack_surface_skip_unnecessary=True,
            attacker_reward_mode=RewardMode.CUMULATIVE,
        ),
        MalSimulatorSettings(
            ttc_mode=TTCMode.EXPECTED_VALUE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=True,
            attack_surface_skip_unnecessary=True,
            attacker_reward_mode=RewardMode.ONE_OFF,
        ),
        MalSimulatorSettings(
            ttc_mode=TTCMode.DISABLED,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface_skip_unnecessary=False,
            attack_surface_skip_unviable=False,
        ),
    ],
)
def test_check_graph_env(sim_settings: MalSimulatorSettings) -> None:
    scenario_file = 'tests/testdata/scenarios/simple_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    register_graph_envs(scenario, sim_settings=sim_settings)
    env: Any = MalSimAttackerGraph(scenario, sim_settings=sim_settings)
    check_env(env, skip_render_check=True, skip_close_check=True)

    env = MalSimDefenderGraph(scenario, sim_settings=sim_settings)
    check_env(env, skip_render_check=True, skip_close_check=True)


def test_attacker_episode() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_scenario_with_model.yml'
    scenario = Scenario.load_from_file(scenario_file)
    attacker_env = MalSimAttackerGraph(
        scenario,
        sim_settings=MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface_skip_unnecessary=False,
            attacker_reward_mode=RewardMode.ONE_OFF,
        ),
    )
    ser = attacker_env.multi_env.lang_serializer

    done = False
    obs, info = attacker_env.reset()
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = attacker_env.step(
            attacker_env.action_space.sample(obs.steps.action_mask)
        )
        state = info['state']
        assert isinstance(state, MalSimAttackerState)
        visible_assets = {node.model_asset for node in state.action_surface} | {
            node.model_asset for node in state.performed_nodes
        }
        visible_steps = {
            node
            for node in state.sim.attack_graph.nodes.values()
            if node.model_asset in visible_assets and node.type in ('and', 'or')
        }
        for node in visible_steps:
            assert node.id in obs.steps.id
            node_idx = np.where(obs.steps.id == node.id)[0][0]
            assert node.id == obs.steps.id[node_idx]
            if ser.split_step_types and node.model_asset:
                assert (
                    ser.attacker_step_type[(node.model_asset.type, node.name)]
                    == obs.steps.type[node_idx]
                )
            else:
                assert ser.attacker_step_type[(node.name,)] == obs.steps.type[node_idx]
            assert ser.step_class[node.type] == obs.steps.logic_class[node_idx]
            assert (
                ser.step_tag[node.tags[0] if len(node.tags) > 0 else None]
                == obs.steps.tags[node_idx]
            )
            assert (
                state.sim.node_is_compromised(node) == obs.steps.compromised[node_idx]
            )
            assert (
                obs.steps.attempts is not None
                and state.num_attempts.get(node, 0) == obs.steps.attempts[node_idx]
            )
            assert (
                state.sim.node_is_traversable(state.performed_nodes, node)
                == obs.steps.action_mask[node_idx]
            )

        steps += 1
        done = terminated or truncated or (steps > 10_000)

    done = False
    obs, info = attacker_env.reset()
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = attacker_env.step(
            attacker_env.action_space.sample(obs.steps.action_mask)
        )
        state = info['state']
        assert isinstance(state, MalSimAttackerState)
        visible_assets = {node.model_asset for node in state.action_surface} | {
            node.model_asset for node in state.performed_nodes
        }
        visible_steps = {
            node
            for node in state.sim.attack_graph.nodes.values()
            if node.model_asset in visible_assets and node.type in ('and', 'or')
        }
        for node in visible_steps:
            assert node.id in obs.steps.id
            node_idx = np.where(obs.steps.id == node.id)[0][0]
            assert node.id == obs.steps.id[node_idx]
            if ser.split_step_types and node.model_asset:
                assert (
                    ser.attacker_step_type[(node.model_asset.type, node.name)]
                    == obs.steps.type[node_idx]
                )
            else:
                assert ser.attacker_step_type[(node.name,)] == obs.steps.type[node_idx]
            assert ser.step_class[node.type] == obs.steps.logic_class[node_idx]
            assert (
                ser.step_tag[node.tags[0] if len(node.tags) > 0 else None]
                == obs.steps.tags[node_idx]
            )
            assert (
                state.sim.node_is_compromised(node) == obs.steps.compromised[node_idx]
            )
            assert (
                obs.steps.attempts is not None
                and state.num_attempts.get(node, 0) == obs.steps.attempts[node_idx]
            )
            assert (
                state.sim.node_is_traversable(state.performed_nodes, node)
                == obs.steps.action_mask[node_idx]
            )

        steps += 1
        done = terminated or truncated or (steps > 10_000)


def test_defender_episode() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_scenario_with_model.yml'
    scenario = Scenario.load_from_file(scenario_file)
    defender_env = MalSimDefenderGraph(
        scenario,
        sim_settings=MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface_skip_unnecessary=False,
            attacker_reward_mode=RewardMode.ONE_OFF,
        ),
    )

    done = False
    obs, info = defender_env.reset()
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = defender_env.step(
            defender_env.action_space.sample(obs.steps.action_mask)
        )
        steps += 1
        done = terminated or truncated or (steps > 10_000)

    done = False
    obs, info = defender_env.reset()
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = defender_env.step(
            defender_env.action_space.sample(obs.steps.action_mask)
        )
        steps += 1
        done = terminated or truncated or (steps > 10_000)


def test_pettingzoo_api_check() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_scenario_with_model.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim = MalSimulator.from_scenario(scenario)
    env = MalSimGraph(sim)
    parallel_api_test(env)


def test_asset_then_action_wrapper() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_scenario_with_model.yml'
    scenario = Scenario.load_from_file(scenario_file)
    attacker_env = MalSimAttackerGraph(
        scenario,
        MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface_skip_unnecessary=False,
            attacker_reward_mode=RewardMode.ONE_OFF,
        ),
    )
    assert scenario.attack_graph.model is not None, (
        'Attack graph needs to have a model attached to it'
    )
    wrapped_env = AssetThenActionWrapper(
        attacker_env,
        scenario.attack_graph.model,
        attacker_env.multi_env.lang_serializer,
    )
    assert isinstance(wrapped_env.action_space, AssetThenAttackerAction)

    i = 0
    done = False
    obs, info = wrapped_env.reset()
    while not done and i < 100:
        asset_mask, action_mask = info['asset_mask'], info['action_mask']
        action = wrapped_env.action_space.sample(mask=(asset_mask, action_mask))
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        i += 1
        done = terminated or truncated

    defender_env = MalSimDefenderGraph(
        scenario,
        MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface_skip_unnecessary=False,
            attacker_reward_mode=RewardMode.ONE_OFF,
        ),
    )
    assert scenario.attack_graph.model is not None, (
        'Attack graph needs to have a model attached to it'
    )
    wrapped_env = AssetThenActionWrapper(
        defender_env,
        scenario.attack_graph.model,
        defender_env.multi_env.lang_serializer,
    )
    assert isinstance(wrapped_env.action_space, AssetThenDefenderAction)
    i = 0
    done = False
    obs, info = wrapped_env.reset()
    while not done and i < 100:
        action_mask, asset_mask = info['action_mask'], info['asset_mask']
        action = wrapped_env.action_space.sample(mask=(asset_mask, action_mask))
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        i += 1
        done = terminated or truncated


def test_action_then_asset_wrapper() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_scenario_with_model.yml'
    scenario = Scenario.load_from_file(scenario_file)
    attacker_env = MalSimAttackerGraph(
        scenario,
        MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface_skip_unnecessary=False,
            attacker_reward_mode=RewardMode.ONE_OFF,
        ),
    )
    assert scenario.attack_graph.model is not None, (
        'Attack graph needs to have a model attached to it'
    )
    wrapped_env = ActionThenAssetWrapper(
        attacker_env,
        scenario.attack_graph.model,
        attacker_env.multi_env.lang_serializer,
    )
    assert isinstance(wrapped_env.action_space, AttackerActionThenAsset)
    i = 0
    done = False
    obs, info = wrapped_env.reset()
    while not done and i < 100:
        action_mask, asset_mask = info['action_mask'], info['asset_mask']
        action = wrapped_env.action_space.sample(mask=(action_mask, asset_mask))
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        i += 1
        done = terminated or truncated

    defender_env = MalSimDefenderGraph(
        scenario,
        MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface_skip_unnecessary=False,
            attacker_reward_mode=RewardMode.ONE_OFF,
        ),
    )
    wrapped_env = ActionThenAssetWrapper(
        defender_env,
        scenario.attack_graph.model,
        defender_env.multi_env.lang_serializer,
    )
    assert isinstance(wrapped_env.action_space, DefenderActionThenAsset)
    i = 0
    done = False
    obs, info = wrapped_env.reset()
    while not done and i < 100:
        action_mask, asset_mask = info['action_mask'], info['asset_mask']
        action = wrapped_env.action_space.sample(mask=(action_mask, asset_mask))
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        i += 1
        done = terminated or truncated


