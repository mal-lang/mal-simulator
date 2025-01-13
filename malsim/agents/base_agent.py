"""A decision agent is a heuristic agent"""

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
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
