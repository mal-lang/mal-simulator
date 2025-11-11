"""Gymnasium wrappers for modifying action spaces.

This module provides wrappers for Gymnasium Env that transform actions before
passing them to the base environment, following the pattern of Gymnasium's
ActionWrapper (https://gymnasium.farama.org/api/wrappers/action_wrappers/).
"""

from typing import Any, Callable, SupportsFloat
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.core import ActType, ObsType, Env, WrapperActType
from gymnasium.spaces import Space
from malsim.envs.graph.mal_spaces import AssetThenDefenderAction, AttackerActionThenAsset, DefenderActionThenAsset, MALObsAttackStepSpace, AssetThenAttackerAction, MALObsDefenseStepSpace
from malsim.envs.graph.mal_spaces import MALObsInstance
from maltoolbox.model import Model
from malsim.envs.graph.serialization import LangSerializer


class AssetThenActionWrapper(Wrapper[MALObsInstance, tuple[np.int64, np.int64], MALObsInstance, np.int64]):
    """Wrapper that transforms the action space to be over (asset, lang action) instead of step index.
    """

    def __init__(self, env: Env[MALObsInstance, np.int64], model: Model, lang_serializer: LangSerializer):
        """

        Args:
            env: Environment to be wrapped.
            model: Model to use
            lang_serializer: Language serializer to use
        """
        Wrapper.__init__(self, env)
        self.model = model
        self.serializer = lang_serializer
        action_space: AssetThenAttackerAction | AssetThenDefenderAction
        if isinstance(env.action_space, MALObsAttackStepSpace):
            action_space = AssetThenAttackerAction(model, lang_serializer)
            self.action_space = action_space
        elif isinstance(env.action_space, MALObsDefenseStepSpace):
            action_space = AssetThenDefenderAction(model, lang_serializer)
            self.action_space = action_space
        else:
            raise ValueError(f"Unsupported action space: {env.action_space}")

        self.mask = action_space.mask

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[MALObsInstance, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs = obs
        return obs, info

    def step(
        self, action: tuple[np.int64, np.int64]
    ) -> tuple[MALObsInstance, SupportsFloat, bool, bool, dict[str, Any]]:
        asset_idx, action_type = action
        step_idx = self._obs.step2asset[0, np.where(self._obs.step2asset[1] == asset_idx)]
        step_idx = step_idx[self._obs.steps.type[step_idx] == action_type]

        obs, reward, terminated, truncated, info = self.env.step(step_idx[0])
        self._obs = obs

        return obs, reward, terminated, truncated, info

class ActionThenAssetWrapper(Wrapper[MALObsInstance, tuple[np.int64, np.int64], MALObsInstance, np.int64]):
    """Wrapper that transforms the action space to be over (lang action, asset) instead of step index.
    """

    def __init__(self, env: Env[MALObsInstance, np.int64], model: Model, lang_serializer: LangSerializer):
        """

        Args:
            env: Environment to be wrapped. Needs to use MALObsAttackStepSpace or MALObsDefenseStepSpace as action space.
            model: Model to use
            lang_serializer: Language serializer to use
        """
        Wrapper.__init__(self, env)
        self.model = model
        self.serializer = lang_serializer
        action_space: AttackerActionThenAsset | DefenderActionThenAsset
        if isinstance(env.action_space, MALObsAttackStepSpace):
            action_space = AttackerActionThenAsset(model, lang_serializer)
            self._action_space = action_space
        elif isinstance(env.action_space, MALObsDefenseStepSpace):
            action_space = DefenderActionThenAsset(model, lang_serializer)
            self._action_space = action_space
        else:
            raise ValueError(f"Unsupported action space: {env.action_space}")

        self.mask = action_space.mask

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[MALObsInstance, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs = obs
        return obs, info

    def step(
        self, action: tuple[np.int64, np.int64]
    ) -> tuple[MALObsInstance, SupportsFloat, bool, bool, dict[str, Any]]:
        action_type, asset_idx = action
        step_idx = np.where(self._obs.steps.type == action_type)[0]
        step_idx = self._obs.step2asset[0, np.isin(self._obs.step2asset[0], step_idx) & np.isin(self._obs.step2asset[1], asset_idx)]

        obs, reward, terminated, truncated, info = self.env.step(step_idx[0])
        self._obs = obs

        return obs, reward, terminated, truncated, info