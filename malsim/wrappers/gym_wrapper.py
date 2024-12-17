from typing import Any, Dict, SupportsFloat

import gymnasium as gym
import gymnasium.utils.env_checker as env_checker
from gymnasium import Wrapper
from gymnasium import spaces
from gymnasium.core import RenderFrame
import numpy as np

from ..scenario import load_scenario
from ..agents import PassiveAttacker, PassiveDefender, AgentType
from ..envs.mal_sim_parallel_env import VectorizedObsMalSimulator


class AttackerEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, scenario_file: str, **kwargs) -> None:
        """
        Params:
        - scenario_file: the scenario that should be loaded
                         and made into a MalSimulator
        """

        self.render_mode = kwargs.pop('render_mode', None)

        # Create a simulator from the scenario given
        attack_graph, _ = load_scenario(scenario_file, **kwargs)
        attacker_id = attack_graph.attackers[0].id
        self.sim = VectorizedObsMalSimulator(attack_graph)

        self.attacker_agent = PassiveAttacker("attacker", attacker_id)
        self.sim.register_agent(self.attacker_agent)

        self.sim.reset()

        # Use first attacker as attacker agent in simulation
        # since only single agent is currently supported
        self.attacker_agent_id = self.attacker_agent.name
        self.observation_space = self.sim.observation_space(self.attacker_agent.name)
        self.action_space = self.sim.action_space(self.attacker_agent.name)
        super().__init__()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        # TODO: params not used by method, find out if we need to send them
        self.sim.reset(seed=seed, options=options)
        return self.attacker_agent.info, self.attacker_agent.info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs: Dict[str, Any]

        # TODO: Add potential defender and give defender action if it exists
        actions = {
            self.attacker_agent_id: action,
        }
        self.sim.step(actions)

        return (
            self.attacker_agent.info,
            self.attacker_agent.reward,
            len(self.action_space) == 0,
            self.attacker_agent.info,
            {},
        )

    def render(self):
        return self.sim.render()

    @property
    def num_assets(self):
        return len(self.sim._index_to_asset_type)

    @property
    def num_step_names(self):
        return len(self.sim._index_to_step_name)

class DefenderEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, scenario_file, **kwargs) -> None:
        self.randomize = kwargs.pop('randomize_attacker_behavior', False)
        self.render_mode = kwargs.pop('render_mode', None)

        ag, agents = load_scenario(scenario_file)
        self.sim = VectorizedObsMalSimulator(ag, **kwargs)

        # Register attacker agents from scenario
        self.attacker_agents = [agent for agent in agents.values()
                                if agent.type == AgentType.ATTACKER]

        for agent in self.attacker_agents:
            self.sim.register_agent(agent)

        # Register defender agent (Does not matter which type)
        self.defender_agent = PassiveDefender("defender")
        self.sim.register_agent(self.defender_agent)
        self.sim.reset()

        self.observation_space = \
            self.sim.observation_space(self.defender_agent.name)
        self.action_space = \
            self.sim.action_space(self.defender_agent.name)

    def reset(
        self, *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:

        super().reset(seed=seed, options=options)
        obs, infos = self.sim.reset(seed=seed, options=options)

        return (
            obs[self.defender_agent.name],
            infos[self.defender_agent.name]
        )

    def step(
        self, action: tuple[int, int]
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:

        actions = {}

        # Get actions from scenario attackers
        for agent in self.attacker_agents:
            actions[agent.name] = agent.get_next_action()

        # Action from defender agent needs to be converted to nodes
        actions[self.defender_agent.name] = \
            self.sim.serialized_action_to_node(action)

        obs, rewards, terminated, truncated, infos = \
            self.sim.step(actions)

        return (
            obs[self.defender_agent.name],
            rewards[self.defender_agent.name],
            terminated[self.defender_agent.name],
            truncated[self.defender_agent.name],
            infos[self.defender_agent.name],
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
        return len(self.sim._index_to_asset_type)

    @property
    def num_step_names(self):
        return len(self.sim._index_to_step_name)


def _to_binary(val, max_val):
    return np.array(
        list(np.binary_repr(val, width=max_val.bit_length())), dtype=np.int64
    )


def vec_to_binary(vec, max_val):
    return np.array([_to_binary(val, max_val) for val in vec])


def vec_to_one_hot(vec, num_vals):
    return np.eye(num_vals, dtype=np.int8)[vec]


class MaskingWrapper(Wrapper):
    def _apply_mask(
        self, obs: dict[str, Any], info: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs['observed_state'] = obs['observed_state'] * obs['is_observable']
        info['action_mask'] = (
            info['action_mask'][0],
            info['action_mask'][1] * obs['is_actionable'],
        )

        if np.nonzero(info['action_mask'][1])[0].size == 0:
            info['action_mask'][0][1] = 0

        return obs, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        obs, info = self._apply_mask(obs, info)

        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        obs, info = self._apply_mask(obs, info)

        return obs, reward, terminated, truncated, info


class LabeledGraphWrapper(Wrapper):
    def __init__(self, env: gym.Env[spaces.Dict, spaces.MultiDiscrete]) -> None:
        super().__init__(env)

        self.num_steps: int = self.env.unwrapped.num_step_names
        num_nodes: int = self.env.observation_space['observed_state'].shape[0]
        num_commands = 2
        node_shape: tuple[int, int] = (
            num_nodes,
            3 + self.num_steps,
        )
        edge_space: spaces.Box = self.env.observation_space['attack_graph_edges']
        self.observation_space = spaces.Dict(
            {
                'nodes': spaces.Box(
                    0,
                    1,
                    shape=node_shape,
                    dtype=np.int8,
                ),
                'edge_index': edge_space,
                'mask_0': spaces.Box(0, 1, shape=(num_commands,), dtype=np.int8),
                'mask_1': spaces.Box(0, 1, shape=(num_nodes,), dtype=np.int8),
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
        return to_graph(obs, info, self.num_steps)


def to_graph(obs: dict[str, Any], info: dict[str, Any], num_steps) -> dict[str, Any]:
    nodes = np.concatenate(
        [
            vec_to_one_hot(obs['observed_state'] + 1, 3),
            vec_to_one_hot(obs['step_name'], num_steps),
        ],
        axis=1,
    )
    return {
        'nodes': nodes,
        'edge_index': obs['attack_graph_edges'],
        'mask_0': info['action_mask'][0],
        'mask_1': info['action_mask'][1],
    }


def register_envs():
    gym.register('MALDefenderEnv-v0', entry_point=DefenderEnv)
    gym.register('MALAttackerEnv-v0', entry_point=AttackerEnv)


if __name__ == '__main__':
    gym.register('MALDefenderEnv-v0', entry_point=DefenderEnv)
    env = gym.make(
        'MALDefenderEnv-v0',
        scenario_file='tests/testdata/scenarios/simple_scenario.yml',
    )
    env_checker.check_env(env.unwrapped)

    gym.register('MALAttackerEnv-v0', entry_point=AttackerEnv)
    env = gym.make(
        'MALAttackerEnv-v0',
        scenario_file='tests/testdata/scenarios/no_defender_agent_scenario.yml',
    )
    env_checker.check_env(env.unwrapped)
