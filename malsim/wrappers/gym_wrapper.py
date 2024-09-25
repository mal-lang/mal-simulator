from typing import Any, Dict, SupportsFloat

import gymnasium as gym
import gymnasium.utils.env_checker as env_checker
from gymnasium import Wrapper
from gymnasium import spaces
from gymnasium.core import RenderFrame
import numpy as np

from ..scenario import create_simulator_from_scenario


class AttackerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario_file: str, **kwargs) -> None:
        """
        Params:
        - scenario_file: the scenario that should be loaded
                         and made into a MalSimulator
        """

        self.render_mode = kwargs.pop("render_mode", None)

        # Create a simulator from the scenario given
        self.sim, _ = create_simulator_from_scenario(scenario_file, **kwargs)

        # Use first attacker as attacker agent in simulation
        # since only single agent is currently supported
        self.attacker_agent_id = list(self.sim.get_attacker_agents().keys())[0]
        self.observation_space = self.sim.observation_space(self.attacker_agent_id)
        self.action_space = self.sim.action_space(self.attacker_agent_id)
        super().__init__()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        # TODO: params not used by method, find out if we need to send them
        obs, info = self.sim.reset(seed=seed, options=options)
        return obs[self.attacker_agent_id], info[self.attacker_agent_id]

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs: Dict[str, Any]

        # TODO: Add potential defender and give defender action if it exists
        actions = {
            self.attacker_agent_id: action,
        }
        obs, rewards, terminated, truncated, infos = self.sim.step(actions)
        return (
            obs[self.attacker_agent_id],
            rewards[self.attacker_agent_id],
            terminated[self.attacker_agent_id],
            truncated[self.attacker_agent_id],
            infos[self.attacker_agent_id],
        )

    def render(self):
        return self.sim.render()

    @property
    def num_assets(self):
        return self.sim.num_assets

    @property
    def num_step_names(self):
        return self.sim.num_step_names


class DefenderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario_file, **kwargs) -> None:
        self.randomize = kwargs.pop("randomize_attacker_behavior", False)
        self.render_mode = kwargs.pop("render_mode", None)

        self.sim, conf = create_simulator_from_scenario(scenario_file, **kwargs)

        # Select first attacker and first defender for the simulation
        # currently only one of each agent is supported
        self.attacker_agent_id = list(self.sim.get_attacker_agents().keys())[0]
        self.defender_agent_id = list(self.sim.get_defender_agents().keys())[0]

        self.attacker_class = conf["agents"][self.attacker_agent_id]["agent_class"]
        self.attacker = self.attacker_class({})

        self.observation_space = self.sim.observation_space(self.defender_agent_id)
        self.action_space = self.sim.action_space(self.defender_agent_id)

        self.attacker_obs = None
        self.attacker_mask = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.attacker = self.attacker_class({"seed": seed, "randomize": self.randomize})

        # TODO: params not used by method, find out if we need to send them
        obs, info = self.sim.reset(seed=seed, options=options)

        self.attacker_obs = obs[self.attacker_agent_id]
        self.attacker_mask = info[self.attacker_agent_id]["action_mask"]

        return obs[self.defender_agent_id], info[self.defender_agent_id]

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        attacker_action = self.attacker.compute_action_from_dict(
            self.attacker_obs, self.attacker_mask
        )

        actions = {
            self.defender_agent_id: action,
            self.attacker_agent_id: attacker_action,
        }
        obs, rewards, terminated, truncated, infos = self.sim.step(actions)

        self.attacker_obs = obs[self.attacker_agent_id]
        self.attacker_mask = infos[self.attacker_agent_id]["action_mask"]

        return (
            obs[self.defender_agent_id],
            rewards[self.defender_agent_id],
            terminated[self.defender_agent_id],
            truncated[self.defender_agent_id],
            infos[self.defender_agent_id],
        )

    def render(self):
        return self.sim.render()

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
        return len(self.sim.unwrapped._index_to_asset_type)

    @property
    def num_step_names(self):
        return len(self.sim.unwrapped._index_to_step_name)


def _to_binary(val, max_val):
    return np.array(
        list(np.binary_repr(val, width=max_val.bit_length())), dtype=np.int64
    )


def vec_to_binary(vec, max_val):
    return np.array([_to_binary(val, max_val) for val in vec])


class LabeledGraphWrapper(Wrapper):
    def __init__(self, env: gym.Env[spaces.Dict, spaces.MultiDiscrete]) -> None:
        super().__init__(env)

        self.num_assets: int = self.env.unwrapped.num_assets
        self.num_steps: int = self.env.unwrapped.num_step_names
        num_nodes: int = self.env.observation_space["observed_state"].shape[0]
        num_commands = 2
        node_shape: tuple[int, int] = (
            num_nodes,
            (3).bit_length()
            + self.num_assets.bit_length()
            + self.num_steps.bit_length(),
        )
        edge_space: spaces.Box = self.env.observation_space["attack_graph_edges"]
        self.observation_space = spaces.Dict(
            {
                "nodes": spaces.Box(
                    0,
                    1,
                    shape=node_shape,
                    dtype=np.int8,
                ),
                "edges": edge_space,
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

    def _to_graph(self, obs: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
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
            "edges": obs["attack_graph_edges"],
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
        scenario_file="tests/testdata/scenarios/simple_scenario.yml",
    )
    env_checker.check_env(env.unwrapped)

    gym.register("MALAttackerEnv-v0", entry_point=AttackerEnv)
    env = gym.make(
        "MALAttackerEnv-v0",
        scenario_file="tests/testdata/scenarios/no_defender_agent_scenario.yml",
    )
    env_checker.check_env(env.unwrapped)
