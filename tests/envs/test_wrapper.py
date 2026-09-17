from typing import Any

import pytest
from gymnasium import spaces

from malsim.config.sim_settings import AttackSurfaceSettings
from malsim.envs.graph.graph_env import AttackerGraphEnv, DefenderGraphEnv
from malsim.envs.graph.mal_spaces import (
    AssetThenAttackerAction,
    AssetThenDefenderAction,
    AttackerActionThenAsset,
    DefenderActionThenAsset,
    MALObsAttackStepSpace,
    MALObsDefenseStepSpace,
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


def test_pad_action_space_wrapper_step_space() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_ai_scenario_with_model.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim_settings = MalSimulatorSettings(
        ttc_mode=TTCMode.PER_STEP_SAMPLE,
        run_defense_step_bernoullis=False,
        run_attack_step_bernoullis=False,
        attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
    )

    attacker_env: Any = AttackerGraphEnv(scenario, sim_settings=sim_settings)
    assert isinstance(attacker_env.action_space, MALObsAttackStepSpace)
    padded_n = int(attacker_env.action_space.n) + 10
    wrapped_env = PadActionSpaceWrapper(attacker_env, max_action=(padded_n,))
    assert wrapped_env.orig_action_space is attacker_env.action_space
    assert isinstance(wrapped_env.action_space, spaces.Discrete)
    assert wrapped_env.action_space.n == padded_n

    obs, _info = wrapped_env.reset()
    done = False
    steps = 0
    while not done and steps < 100:
        action = attacker_env.action_space.sample(obs.steps.action_mask)
        obs, _, terminated, truncated, _info = wrapped_env.step(action)
        done = terminated or truncated
        steps += 1

    defender_env: Any = DefenderGraphEnv(scenario, sim_settings=sim_settings)
    assert isinstance(defender_env.action_space, MALObsDefenseStepSpace)
    padded_n = int(defender_env.action_space.n) + 10
    wrapped_env = PadActionSpaceWrapper(defender_env, max_action=(padded_n,))
    assert isinstance(wrapped_env.action_space, spaces.Discrete)
    assert wrapped_env.action_space.n == padded_n

    obs, _info = wrapped_env.reset()
    done = False
    steps = 0
    while not done and steps < 100:
        action = defender_env.action_space.sample(obs.steps.action_mask)
        obs, _, terminated, truncated, _info = wrapped_env.step(action)
        done = terminated or truncated
        steps += 1


def test_pad_action_space_wrapper_action_then_asset() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_ai_scenario_with_model.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim_settings = MalSimulatorSettings(
        ttc_mode=TTCMode.PER_STEP_SAMPLE,
        run_defense_step_bernoullis=False,
        run_attack_step_bernoullis=False,
        attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
    )
    assert scenario.attack_graph.model is not None, (
        'Attack graph needs to have a model attached to it'
    )

    attacker_env = AttackerGraphEnv(scenario, sim_settings=sim_settings)
    action_then_asset_env: Any = ActionThenAssetWrapper(
        attacker_env,
        scenario.attack_graph.model,
        attacker_env.multi_env.lang_serializer,
    )
    inner_space = action_then_asset_env.action_space
    assert isinstance(inner_space, AttackerActionThenAsset)

    padded_action_n = int(inner_space.action.n) + 5
    padded_asset_n = int(inner_space.asset.n) + 5
    wrapped_env = PadActionSpaceWrapper(
        action_then_asset_env, max_action=(padded_action_n, padded_asset_n)
    )
    assert isinstance(wrapped_env.action_space, spaces.Tuple)
    padded_action_space, padded_asset_space = wrapped_env.action_space.spaces
    assert isinstance(padded_action_space, spaces.Discrete)
    assert isinstance(padded_asset_space, spaces.Discrete)
    assert padded_action_space.n == padded_action_n
    assert padded_asset_space.n == padded_asset_n

    i = 0
    done = False
    _, info = wrapped_env.reset()
    while not done and i < 100:
        action_mask, asset_mask = info['action_mask'], info['asset_mask']
        action = inner_space.sample(mask=(action_mask, asset_mask))
        _, _, terminated, truncated, info = wrapped_env.step(action)
        i += 1
        done = terminated or truncated


def test_pad_action_space_wrapper_asset_then_action() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_ai_scenario_with_model.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim_settings = MalSimulatorSettings(
        ttc_mode=TTCMode.PER_STEP_SAMPLE,
        run_defense_step_bernoullis=False,
        run_attack_step_bernoullis=False,
        attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
    )
    assert scenario.attack_graph.model is not None, (
        'Attack graph needs to have a model attached to it'
    )

    attacker_env = AttackerGraphEnv(scenario, sim_settings=sim_settings)
    asset_then_action_env: Any = AssetThenActionWrapper(
        attacker_env,
        scenario.attack_graph.model,
        attacker_env.multi_env.lang_serializer,
    )
    inner_space = asset_then_action_env.action_space
    assert isinstance(inner_space, AssetThenAttackerAction)

    padded_asset_n = int(inner_space.asset.n) + 5
    padded_action_n = int(inner_space.action.n) + 5
    wrapped_env = PadActionSpaceWrapper(
        asset_then_action_env, max_action=(padded_asset_n, padded_action_n)
    )
    assert isinstance(wrapped_env.action_space, spaces.Tuple)
    padded_asset_space, padded_action_space = wrapped_env.action_space.spaces
    assert isinstance(padded_asset_space, spaces.Discrete)
    assert isinstance(padded_action_space, spaces.Discrete)
    assert padded_asset_space.n == padded_asset_n
    assert padded_action_space.n == padded_action_n

    i = 0
    done = False
    _, info = wrapped_env.reset()
    while not done and i < 100:
        asset_mask, action_mask = info['asset_mask'], info['action_mask']
        action = inner_space.sample(mask=(asset_mask, action_mask))
        _, _, terminated, truncated, info = wrapped_env.step(action)
        i += 1
        done = terminated or truncated


def test_pad_action_space_wrapper_invalid_max_action_length() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_ai_scenario_with_model.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim_settings = MalSimulatorSettings(
        ttc_mode=TTCMode.PER_STEP_SAMPLE,
        run_defense_step_bernoullis=False,
        run_attack_step_bernoullis=False,
        attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
    )
    assert scenario.attack_graph.model is not None, (
        'Attack graph needs to have a model attached to it'
    )

    attacker_env: Any = AttackerGraphEnv(scenario, sim_settings=sim_settings)
    with pytest.raises(AssertionError):
        PadActionSpaceWrapper(attacker_env, max_action=(10, 10))

    action_then_asset_env: Any = ActionThenAssetWrapper(
        attacker_env,
        scenario.attack_graph.model,
        attacker_env.multi_env.lang_serializer,
    )
    with pytest.raises(AssertionError):
        PadActionSpaceWrapper(action_then_asset_env, max_action=(10,))


def test_pad_action_space_wrapper_unsupported_action_space() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_ai_scenario_with_model.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim_settings = MalSimulatorSettings(
        ttc_mode=TTCMode.PER_STEP_SAMPLE,
        run_defense_step_bernoullis=False,
        run_attack_step_bernoullis=False,
        attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
    )
    attacker_env: Any = AttackerGraphEnv(scenario, sim_settings=sim_settings)
    attacker_env.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,))
    with pytest.raises(ValueError):
        PadActionSpaceWrapper(attacker_env, max_action=(10,))
