from abc import ABC, abstractmethod
from collections.abc import Set
from typing import Any


from ..mal_simulator.defender_state import MalSimDefenderState
from ..mal_simulator.attacker_state import MalSimAttackerState
from ..mal_simulator import MalSimulator, MalSimAgentState


class MalSimEnv(ABC):
    def __init__(self, sim: MalSimulator):
        self.sim = sim

    @abstractmethod
    def step(self, actions: Any) -> Any: ...

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> dict[str, MalSimAttackerState | MalSimDefenderState]:
        if seed is not None:
            self.sim.sim_settings.seed = seed
        return self.sim.reset()

    def get_agent_state(self, agent_name: str) -> MalSimAgentState:
        return self.sim.agent_states[agent_name]

    @abstractmethod
    def render(self) -> None:
        pass
