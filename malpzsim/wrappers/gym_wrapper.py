from typing import Any, Dict, SupportsFloat

import gymnasium as gym
import gymnasium.utils.env_checker as env_checker
from gymnasium import Wrapper
from gymnasium import spaces
from gymnasium.core import RenderFrame
import numpy as np

from malpzsim.wrappers.wrapper import LazyWrapper
from malpzsim.agents import searchers


AGENT_ATTACKER = "attacker"
AGENT_DEFENDER = "defender"


class AttackerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **kwargs: Any) -> None:
        self.render_mode = kwargs.pop("render_mode", None)
        agents = {AGENT_ATTACKER: AGENT_ATTACKER}
        self.env = LazyWrapper(agents=agents, **kwargs)
        self.observation_space = self.env.observation_space(AGENT_ATTACKER)
        self.action_space = self.env.action_space(AGENT_ATTACKER)
        super().__init__()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        obs, info = self.env.reset(seed=seed, options=options)
        return obs[AGENT_ATTACKER], info[AGENT_ATTACKER]

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs: Dict[str, Any]
        obs, rewards, terminated, truncated, infos = self.env.step(
            {AGENT_ATTACKER: action}
        )
        return (
            obs[AGENT_ATTACKER],
            rewards[AGENT_ATTACKER],
            terminated[AGENT_ATTACKER],
            truncated[AGENT_ATTACKER],
            infos[AGENT_ATTACKER],
        )

    def render(self):
        return self.env.render()

    @property
    def num_assets(self):
        return self.env.sim.num_assets

    @property
    def num_step_names(self):
        return self.env.sim.num_step_names


class DefenderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        attacker_class: str = str(kwargs.pop("attacker_class", "BreadthFirstAttacker"))
        self.randomize = kwargs.pop("randomize_attacker_behavior", False)
        self.render_mode = kwargs.pop("render_mode", None)
        agents = {AGENT_ATTACKER: AGENT_ATTACKER, AGENT_DEFENDER: AGENT_DEFENDER}
        self.env = LazyWrapper(agents=agents, **kwargs)
        self.attacker_class = searchers.AGENTS[attacker_class]
        self.observation_space = self.env.observation_space(AGENT_DEFENDER)
        self.action_space = self.env.action_space(AGENT_DEFENDER)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.attacker = self.attacker_class({"seed": seed, "randomize": self.randomize})
        obs, info = self.env.reset(seed=seed, options=options)

        self.attacker_obs = obs[AGENT_ATTACKER]
        self.attacker_mask = info[AGENT_ATTACKER]["action_mask"]

        return obs[AGENT_DEFENDER], info[AGENT_DEFENDER]

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        attacker_action = self.attacker.compute_action_from_dict(
            self.attacker_obs, self.attacker_mask
        )
        obs: Dict[str, Any]
        obs, rewards, terminated, truncated, infos = self.env.step(
            {AGENT_DEFENDER: action, AGENT_ATTACKER: attacker_action}
        )
        self.attacker_obs = obs[AGENT_ATTACKER]
        self.attacker_mask = infos[AGENT_ATTACKER]["action_mask"]
        return (
            obs[AGENT_DEFENDER],
            rewards[AGENT_DEFENDER],
            terminated[AGENT_DEFENDER],
            truncated[AGENT_DEFENDER],
            infos[AGENT_DEFENDER],
        )

    def render(self):
        return self.env.render()

    @staticmethod
    def add_reverse_edges(edges: np.ndarray, defense_steps: set) -> np.ndarray:
        # Add reverse edges from the defense steps children to the defense steps
        # themselves
        if defense_steps is not None:
            for p, c in zip(edges[0, :], edges[1, :]):
                if p in defense_steps:
                    new_edge = np.array([c, p]).reshape((2, 1))
                    edges = np.concatenate((edges, new_edge), axis=1)
        return edges

    @property
    def num_assets(self):
        return self.env.sim.num_assets

    @property
    def num_step_names(self):
        return self.env.sim.num_step_names


def _to_binary(val, max_val):
    return np.array(
        list(np.binary_repr(val, width=max_val.bit_length())), dtype=np.int64
    )


def vec_to_binary(vec, max_val):
    return np.array([_to_binary(val, max_val) for val in vec])


class LabeledGraphWrapper(Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        self.num_assets = self.env.unwrapped.num_assets
        self.num_steps = self.env.unwrapped.num_step_names
        num_nodes = self.env.observation_space["observed_state"].shape[0]
        num_commands = 2
        self.observation_space = spaces.Dict(
            {
                "nodes": spaces.Box(
                    0,
                    1,
                    shape=(
                        num_nodes,
                        (3).bit_length()
                        + self.num_assets.bit_length()
                        + self.num_steps.bit_length(),
                    ),
                    dtype=np.int8,
                ),
                "edges": self.env.observation_space["edges"],
                "mask_0": spaces.Box(0, 1, shape=(num_commands,), dtype=np.int8),
                "mask_1": spaces.Box(0, 1, shape=(num_nodes,), dtype=np.int8),
            }
        )
        pass

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env.render()

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._to_graph(obs, info), info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_graph(obs, info), reward, terminated, truncated, info

    def _to_graph(self, obs: dict[str, Any], info) -> Dict[str, Any]:
        nodes = np.concatenate(
            [
                vec_to_binary(obs["observed_state"] + 1, 3),
                vec_to_binary(obs["asset_type"], self.num_assets),
                vec_to_binary(obs["step_name"], self.num_steps),
            ],
            axis=1,
        )
        return {
            "nodes": nodes,
            "edges": obs["edges"],
            "mask_0": info["action_mask"][0],
            "mask_1": info["action_mask"][1],
        }


def register_envs():
    gym.register("MALDefenderEnv-v0", entry_point=DefenderEnv)
    gym.register("MALAttackerEnv-v0", entry_point=AttackerEnv)


if __name__ == "__main__":
    gym.register("MALDefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make(
        "MALDefenderEnv-v0",
        model_file="/storage/GitHub/mal-petting-zoo-simulator/tests/example_model.json",
        lang_file="/storage/GitHub/mal-petting-zoo-simulator/tests/org.mal-lang.coreLang-1.0.0.mar",
        unholy=True,
    )
    env_checker.check_env(env.unwrapped)

    gym.register("MALAttackerEnv-v0", entry_point=AttackerEnv)
    env = gym.make(
        "MALAttackerEnv-v0",
        model_file="/storage/GitHub/mal-petting-zoo-simulator/tests/example_model.json",
        lang_file="/storage/GitHub/mal-petting-zoo-simulator/tests/org.mal-lang.coreLang-1.0.0.mar",
    )
    env_checker.check_env(env.unwrapped)
