from typing import Any

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
from numpy.typing import NDArray

from malsim.config.agent_settings import AttackerSettings
from malsim.config.sim_settings import AttackSurfaceSettings
from malsim.envs.graph.graph_env import AttackerGraphEnv, DefenderGraphEnv
from malsim.envs.graph.mal_spaces import (
    AssetThenAttackerAction,
    AssetThenDefenderAction,
    AttackerActionThenAsset,
    DefenderActionThenAsset,
    MALObsAttackStepSpace,
    MALObsDefenseStepSpace,
    MALObsInstance,
)
from malsim.envs.graph.wrapper import (
    ActionThenAssetWrapper,
    AssetThenActionWrapper,
    PadActionSpaceWrapper,
)
from malsim.mal_simulator import MalSimulatorSettings, TTCMode
from malsim.scenario.scenario import Scenario


def test_asset_then_action_wrapper() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_ai_scenario_with_model.yml'
    scenario = Scenario.load_from_file(scenario_file)
    attacker_env = AttackerGraphEnv(
        scenario,
        MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
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
    _, info = wrapped_env.reset()
    while not done and i < 100:
        asset_mask, action_mask = info['asset_mask'], info['action_mask']
        action = wrapped_env.action_space.sample(mask=(asset_mask, action_mask))
        _, _, terminated, truncated, info = wrapped_env.step(action)
        i += 1
        done = terminated or truncated

    defender_env = DefenderGraphEnv(
        scenario,
        MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
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
    _, info = wrapped_env.reset()
    while not done and i < 100:
        action_mask, asset_mask = info['action_mask'], info['asset_mask']
        action = wrapped_env.action_space.sample(mask=(asset_mask, action_mask))
        _, _, terminated, truncated, info = wrapped_env.step(action)
        i += 1
        done = terminated or truncated


def test_action_then_asset_wrapper() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_ai_scenario_with_model.yml'
    scenario = Scenario.load_from_file(scenario_file)
    attacker_env = AttackerGraphEnv(
        scenario,
        MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
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
    _, info = wrapped_env.reset()
    while not done and i < 100:
        action_mask, asset_mask = info['action_mask'], info['asset_mask']
        action = wrapped_env.action_space.sample(mask=(action_mask, asset_mask))
        _, _, terminated, truncated, info = wrapped_env.step(action)
        i += 1
        done = terminated or truncated

    defender_env = DefenderGraphEnv(
        scenario,
        MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
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
    _, info = wrapped_env.reset()
    while not done and i < 100:
        action_mask, asset_mask = info['action_mask'], info['asset_mask']
        action = wrapped_env.action_space.sample(mask=(action_mask, asset_mask))
        _, _, terminated, truncated, info = wrapped_env.step(action)
        i += 1
        done = terminated or truncated


def test_pad_action_space() -> None:
    scenario_files = [
        'tests/testdata/scenarios/bfs_vs_bfs_network_app_data_scenario.yml',
        'tests/testdata/scenarios/bfs_vs_bfs_scenario_multiple_entrypoint_sets.yml',
        'tests/testdata/scenarios/bfs_vs_bfs_scenario.yml'
    ]
    scenarios = [Scenario.load_from_file(file) for file in scenario_files]
    for scenario in scenarios:
        try:
            attacker_name = next(iter(scenario.attacker_settings))
            scenario.attacker_settings[attacker_name].policy = None
        except StopIteration:
            raise ValueError("No attacker settings found in scenario")

    sim_settings = MalSimulatorSettings(
        ttc_mode=TTCMode.PER_STEP_SAMPLE,
        run_defense_step_bernoullis=False,
        run_attack_step_bernoullis=False,
        attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
    )

    attacker_envs = [
        lambda: AttackerGraphEnv(scenario, sim_settings=sim_settings) for scenario in scenarios
    ]
    max_action = max(thunk().action_space.n for thunk in attacker_envs)
    thunks = [lambda: PadActionSpaceWrapper(AttackerGraphEnv(scenario, sim_settings=sim_settings), (max_action,)) for scenario in scenarios]
    sync_env = SyncVectorEnv(thunks)

    _obs, _info = sync_env.reset()
    for _ in range(10):
        _obs, _reward, _terminated, _truncated, _ = sync_env.step(sync_env.action_space.sample())