import gymnasium as gym
from typing import Any

from .mal_spaces import AttackGraphNodeSpace, MALObs, MALObsInstance, attacker_state2graph
from os import PathLike
from malsim.scenario import AgentType, Scenario
from malsim.mal_simulator import (
    MalSimulator,
    MalSimulatorSettings,
    TTCMode,
    RewardMode,
    MalSimAttackerState,
)
from malsim.envs.serialization import LangSerializer
from gymnasium.envs.registration import EnvSpec
import numpy as np
from typing import SupportsFloat

def register_graph_envs(scenario: Scenario | PathLike[str]) -> None:

    gym.register(
        id='GraphAttackerEnv-v0',
        entry_point='malsim.envs.graph_env:GraphAttackerEnv',
        kwargs={
            "scenario": scenario,
            "use_logic_gates": False,
            'sim_settings': MalSimulatorSettings(
                ttc_mode=TTCMode.PER_STEP_SAMPLE,
                run_defense_step_bernoullis=False,
                run_attack_step_bernoullis=False,
                attack_surface_skip_unnecessary=False,
                attacker_reward_mode=RewardMode.ONE_OFF,
            ),
        }
    )

class GraphAttackerEnv(gym.Env[MALObsInstance, np.int64]):
    metadata = {'render_modes': []}

    spec: EnvSpec = EnvSpec(
        id='GraphAttackerEnv-v0',
        entry_point='malsim.envs.graph_env:GraphAttackerEnv',
        nondeterministic=True,
        kwargs={
            "use_logic_gates": False,
            'sim_settings': MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE,
            run_defense_step_bernoullis=False,
            run_attack_step_bernoullis=False,
            attack_surface_skip_unnecessary=False,
            attacker_reward_mode=RewardMode.ONE_OFF,
        ),
        }
    )

    def __init__(
        self,
        scenario: PathLike[str] | Scenario,
        use_logic_gates: bool,
        sim_settings: MalSimulatorSettings,
        **kwargs: dict[str, Any]
    ) -> None:
        self.render_mode: Any | None = kwargs.pop('render_mode', None)

        if not isinstance(scenario, Scenario):
            scenario = Scenario.load_from_file(str(scenario))

        self.scenario = scenario
        self.use_logic_gates = use_logic_gates
        self.sim = MalSimulator.from_scenario(scenario, sim_settings)
        self.agent_name = get_agent_name(scenario, AgentType.ATTACKER)
        self.attack_graph = self.sim.attack_graph
        self.model = self.attack_graph.model
        assert self.model, "Attack graph must have a model"
        self.lang_serializer = LangSerializer(self.attack_graph.lang_graph)
        self.observation_space = MALObs(self.lang_serializer, self.use_logic_gates, sim_settings.seed)
        self.action_space = AttackGraphNodeSpace(self.attack_graph)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[MALObsInstance, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        state = self.sim.reset()[self.agent_name]
        assert isinstance(state, MalSimAttackerState)
        obs = attacker_state2graph(state, self.lang_serializer, self.use_logic_gates)
        info = {"state": state}
        return obs, info

    def step(self, action: np.int64) -> tuple[MALObsInstance, SupportsFloat, bool, bool, dict[str, Any]]:
        action_node = self.attack_graph.nodes[int(action)]
        state = self.sim.step({self.agent_name: [action_node]})[self.agent_name]
        assert isinstance(state, MalSimAttackerState)
        obs = attacker_state2graph(state, self.lang_serializer, self.use_logic_gates)
        terminated = self.sim.agent_is_terminated(self.agent_name)
        reward = self.sim.agent_reward(self.agent_name)
        info = {"state": state}
        return obs, reward, terminated, False, info

    def render(self) -> None:
        raise NotImplementedError("Render not implemented")

    def close(self) -> None:
        return

def get_agent_name(scenario: Scenario, type: AgentType) -> str:
    agents = [agent for agent in scenario.agents if agent["type"] == type]
    assert len(agents) == 1, (
        f"Expected exactly one agent of type {type}, got {len(agents)} agents"
    )
    agent_name = agents[0]["name"]
    return str(agent_name)