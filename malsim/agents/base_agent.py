"""A decision agent is a heuristic agent"""

from abc import ABC, abstractmethod
from ..sims import MalSimAgent

class DecisionAgent(ABC):

    @abstractmethod
    def get_next_action(
        self,
        agent: MalSimAgent,
        **kwargs
    ) -> tuple[int, int]: ...

class PassiveAgent(DecisionAgent):
    def __init__(self, info):
        return

    def get_next_action(
        self,
        agent: MalSimAgent,
        **kwargs
    ) -> tuple[int, int]:
        return (0, None)
