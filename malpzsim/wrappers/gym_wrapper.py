from typing import Any, Dict, SupportsFloat

import gymnasium as gym
from gymnasium.core import RenderFrame
import gymnasium.utils.env_checker as env_checker

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


if __name__ == "__main__":
    gym.register("DefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make(
        "DefenderEnv-v0",
        model_file="/storage/GitHub/mal-petting-zoo-simulator/tests/example_model.json",
        lang_file="/storage/GitHub/mal-petting-zoo-simulator/tests/org.mal-lang.coreLang-1.0.0.mar",
    )
    env_checker.check_env(env.unwrapped)

    gym.register("AttackerEnv-v0", entry_point=AttackerEnv)
    env = gym.make(
        "AttackerEnv-v0",
        model_file="/storage/GitHub/mal-petting-zoo-simulator/tests/example_model.json",
        lang_file="/storage/GitHub/mal-petting-zoo-simulator/tests/org.mal-lang.coreLang-1.0.0.mar",
    )
    env_checker.check_env(env.unwrapped)
