"""A decision agent is a heuristic agent"""

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..sims import MalSimAgentView

class DecisionAgent(ABC):

    @abstractmethod
    def get_next_action(
        self,
        agent: MalSimAgentView,
        **kwargs
    ) -> tuple[int, int]: ...

class PassiveAgent(DecisionAgent):
    def __init__(self, info):
        return

    def get_next_action(
        self,
        agent: MalSimAgentView,
        **kwargs
    ) -> tuple[int, int]:
        return (0, None)
