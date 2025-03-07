from abc import ABC, abstractmethod
from ..mal_simulator import AgentState, MalSimulator

class MalSimEnv(ABC):

    def __init__(self, sim: MalSimulator):
        self.sim = sim

    @abstractmethod
    def step(self, actions):
        ...

    def reset(self, seed=None, options=None):
        return self.sim.reset(seed=seed, options=options)

    def register_attacker(
            self, attacker_name: str, attacker_id: int
        ):
        return self.sim.register_attacker(attacker_name, attacker_id)

    def register_defender(
            self, defender_name: str
        ):
        return self.sim.register_defender(defender_name)

    def get_agent_state(self, agent_name: str) -> AgentState:
        return self.sim.agent_states[agent_name]

    def render(self):
        pass
