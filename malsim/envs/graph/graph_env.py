import gymnasium as gym
from typing import Any

from .mal_spaces import (
    MALObsAttackStepSpace,
    MALObsDefenseStepSpace,
    MALObs,
    MALObsInstance,
    MALAttackerObs,
    MALDefenderObs,
)
from .utils import create_full_obs, full_obs2attacker_obs, full_obs2defender_obs
from os import PathLike
from malsim.scenario import AgentType, Scenario
from malsim.mal_simulator import (
    MalSimulator,
    MalSimulatorSettings,
    TTCMode,
    RewardMode,
    MalSimAttackerState,
)
from .serialization import LangSerializer
from gymnasium.envs.registration import EnvSpec
import numpy as np
from typing import SupportsFloat
import logging
from pettingzoo import ParallelEnv

logger = logging.getLogger(__name__)


def register_graph_envs(
    scenario: Scenario | PathLike[str],
    sim_settings: MalSimulatorSettings = MalSimulatorSettings(
        ttc_mode=TTCMode.PER_STEP_SAMPLE,
        run_defense_step_bernoullis=False,
        run_attack_step_bernoullis=False,
        attack_surface_skip_unnecessary=False,
        attacker_reward_mode=RewardMode.ONE_OFF,
    ),
) -> None:
    gym.register(
        id='GraphAttackerEnv-v0',
        entry_point='malsim.envs.graph.graph_env:GraphAttackerEnv',
        kwargs={
            'scenario': scenario,
            'sim_settings': sim_settings,
        },
    )

    gym.register(
        id='GraphDefenderEnv-v0',
        entry_point='malsim.envs.graph.graph_env:GraphDefenderEnv',
        kwargs={
            'scenario': scenario,
            'sim_settings': sim_settings,
        },
    )


class MalSimAttackerGraph(gym.Env[MALObsInstance, np.int64]):
    metadata = {'render_modes': []}

    spec: EnvSpec = EnvSpec(
        id='GraphAttackerEnv-v0',
        entry_point='malsim.envs.graph.graph_env:GraphAttackerEnv',
        nondeterministic=True,
        kwargs={
            'sim_settings': MalSimulatorSettings(
                ttc_mode=TTCMode.PER_STEP_SAMPLE,
                run_defense_step_bernoullis=False,
                run_attack_step_bernoullis=False,
                attack_surface_skip_unnecessary=False,
                attacker_reward_mode=RewardMode.ONE_OFF,
            ),
        },
    )

    def __init__(
        self,
        scenario: PathLike[str] | Scenario,
        sim_settings: MalSimulatorSettings,
        **kwargs: dict[str, Any],
    ) -> None:
        self.render_mode: Any | None = kwargs.pop('render_mode', None)

        if not isinstance(scenario, Scenario):
            scenario = Scenario.load_from_file(str(scenario))

        self.scenario = scenario
        self.sim = MalSimulator.from_scenario(scenario, sim_settings)
        self.multi_env = MalSimGraph(self.sim, attacker_visible_defense_steps=False)
        self.agent_name = get_agent_name(scenario, AgentType.ATTACKER)
        self.observation_space = self.multi_env.observation_space(self.agent_name)
        self.action_space = self.multi_env.action_space(self.agent_name)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[MALObsInstance, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        obs, info = self.multi_env.reset(seed=seed, options=options)
        return obs[self.agent_name], info[self.agent_name]

    def step(
        self, action: np.int64
    ) -> tuple[MALObsInstance, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.multi_env.step(
            {self.agent_name: action}
        )
        return (
            obs[self.agent_name],
            reward[self.agent_name],
            terminated[self.agent_name],
            truncated[self.agent_name],
            info[self.agent_name],
        )

    def render(self) -> None:
        return self.multi_env.render()

    def close(self) -> None:
        return self.multi_env.close()


class MalSimDefenderGraph(gym.Env[MALObsInstance, np.int64]):
    metadata = {'render_modes': []}

    spec: EnvSpec = EnvSpec(
        id='GraphDefenderEnv-v0',
        entry_point='malsim.envs.graph.graph_env:GraphDefenderEnv',
        nondeterministic=True,
        kwargs={
            'sim_settings': MalSimulatorSettings(
                ttc_mode=TTCMode.PER_STEP_SAMPLE,
                run_defense_step_bernoullis=False,
                run_attack_step_bernoullis=False,
                attack_surface_skip_unnecessary=False,
                attacker_reward_mode=RewardMode.ONE_OFF,
            ),
        },
    )

    def __init__(
        self,
        scenario: PathLike[str] | Scenario,
        sim_settings: MalSimulatorSettings,
        **kwargs: dict[str, Any],
    ) -> None:
        self.render_mode: Any | None = kwargs.pop('render_mode', None)

        if not isinstance(scenario, Scenario):
            scenario = Scenario.load_from_file(str(scenario))

        self.scenario = scenario
        self.sim = MalSimulator.from_scenario(scenario, sim_settings)
        self.multi_env = MalSimGraph(self.sim, attacker_visible_defense_steps=True)
        self.agent_name = get_agent_name(scenario, AgentType.DEFENDER)
        self.observation_space = self.multi_env.observation_space(self.agent_name)
        self.action_space = self.multi_env.action_space(self.agent_name)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[MALObsInstance, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        obs, info = self.multi_env.reset(seed=seed, options=options)
        return obs[self.agent_name], info[self.agent_name]

    def step(
        self, action: np.int64
    ) -> tuple[MALObsInstance, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.multi_env.step(
            {self.agent_name: action}
        )
        return (
            obs[self.agent_name],
            reward[self.agent_name],
            terminated[self.agent_name],
            truncated[self.agent_name],
            info[self.agent_name],
        )

    def render(self) -> None:
        return self.multi_env.render()

    def close(self) -> None:
        return self.multi_env.close()


def get_agent_name(scenario: Scenario, type: AgentType) -> str:
    agents = [agent for agent in scenario.agents if agent['type'] == type]
    assert len(agents) == 1, (
        f'Expected exactly one agent of type {type}, got {len(agents)} agents'
    )
    agent_name = agents[0]['name']
    return str(agent_name)


class MalSimGraph(ParallelEnv[str, MALObsInstance, np.int64]):
    metadata = {
        'name': 'GraphEnv-v0',
    }

    def __init__(
        self, simulator: MalSimulator, attacker_visible_defense_steps: bool = False
    ):
        self.see_def_steps = attacker_visible_defense_steps
        self.sim = simulator
        self.attack_graph = self.sim.attack_graph
        self.lang_serializer = LangSerializer(
            self.attack_graph.lang_graph, split_assoc_types=False, split_step_types=True
        )
        self.observation_spaces: dict[str, MALObs] = {
            'attacker': MALAttackerObs(self.lang_serializer),
            'defender': MALDefenderObs(self.lang_serializer),
        }
        self.action_spaces: dict[
            str, MALObsAttackStepSpace | MALObsDefenseStepSpace
        ] = {
            'attacker': MALObsAttackStepSpace(self.sim),
            'defender': MALObsDefenseStepSpace(self.sim),
        }
        self._full_obs = create_full_obs(self.sim, self.lang_serializer)
        self.possible_agents = [name for name in self.sim.agent_states.keys()]
        self.agents = list(self.sim._alive_agents)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, MALObsInstance], dict[str, dict[str, Any]]]:
        states = self.sim.reset()
        self._obs = {
            agent_name: (
                full_obs2attacker_obs(
                    self._full_obs,
                    state,
                    self.lang_serializer,
                    see_defense_steps=self.see_def_steps,
                )
                if isinstance(state, MalSimAttackerState)
                else full_obs2defender_obs(self._full_obs, state, self.lang_serializer)
            )
            for agent_name, state in states.items()
        }
        for agent_name, obs in self._obs.items():
            self.action_space(agent_name)._mask = obs.steps.action_mask
        return self._obs, {
            agent_name: {'state': state} for agent_name, state in states.items()
        }

    def step(
        self, actions: dict[str, np.int64]
    ) -> tuple[
        dict[str, MALObsInstance],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        # Check if actions are valid
        for agent_name, action_idx in actions.items():
            if self._obs[agent_name].steps.type.shape[0] < action_idx:
                logger.error(
                    f'Action {action_idx} is not valid'
                    f' for observation {self._obs[agent_name]}'
                )

        # Convert observation indicies to action nodes
        action_nodes = {
            agent_name: [
                self.attack_graph.nodes[self._obs[agent_name].steps.id[action_idx]]
            ]
            if not (self._obs[agent_name].steps.type.shape[0] < action_idx)
            else []
            for agent_name, action_idx in actions.items()
        }
        states = self.sim.step(action_nodes)
        self.agents = list(self.sim._alive_agents)
        self._obs = {
            agent_name: (
                full_obs2attacker_obs(
                    self._full_obs,
                    state,
                    self.lang_serializer,
                    see_defense_steps=self.see_def_steps,
                )
                if isinstance(state, MalSimAttackerState)
                else full_obs2defender_obs(self._full_obs, state, self.lang_serializer)
            )
            for agent_name, state in states.items()
        }
        for agent_name, obs in self._obs.items():
            self.action_space(agent_name)._mask = obs.steps.action_mask
        rewards = {
            agent_name: self.sim.agent_reward(agent_name)
            for agent_name in states.keys()
        }
        terminations = {
            agent_name: self.sim.agent_is_terminated(agent_name)
            for agent_name in states.keys()
        }
        truncations = {agent_name: self.sim.done() for agent_name in states.keys()}
        return (
            self._obs,
            rewards,
            terminations,
            truncations,
            {agent_name: {'state': state} for agent_name, state in states.items()},
        )

    def render(self) -> None:
        raise NotImplementedError('Render not implemented')

    def close(self) -> None:
        return

    def observation_space(self, agent: str) -> MALObs:
        return self.observation_spaces[
            'attacker'
            if isinstance(self.sim.agent_states[agent], MalSimAttackerState)
            else 'defender'
        ]

    def action_space(
        self, agent: str
    ) -> MALObsAttackStepSpace | MALObsDefenseStepSpace:
        return self.action_spaces[
            'attacker'
            if isinstance(self.sim.agent_states[agent], MalSimAttackerState)
            else 'defender'
        ]
