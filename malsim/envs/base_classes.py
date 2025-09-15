from abc import ABC, abstractmethod
from typing import Any, Optional

from maltoolbox.attackgraph import AttackGraphNode
from ..mal_simulator import MalSimulator, MalSimAgentState

class MalSimEnv(ABC):

    def __init__(self, sim: MalSimulator):
        self.sim = sim

    @abstractmethod
    def step(self, actions: Any) -> Any:
        ...

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None
    ) -> None:
        if seed is not None:
            self.sim.sim_settings.seed = seed
        self.sim.reset(options = options)

    def register_attacker(
            self, attacker_name: str, entry_points: set[AttackGraphNode]
        ) -> None:
        self.sim.register_attacker(attacker_name, entry_points)

    def register_defender(
            self, defender_name: str
        ) -> None:
        self.sim.register_defender(defender_name)

    def get_agent_state(self, agent_name: str) -> MalSimAgentState:
        return self.sim.agent_states[agent_name]

    def render(self) -> None:
        pass
