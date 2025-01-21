"""A decision agent is a heuristic agent"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..sims import MalSimAgentStateView
    from maltoolbox.attackgraph import AttackGraphNode

class DecisionAgent(ABC):

    @abstractmethod
    def get_next_action(
        self,
        agent: MalSimAgentStateView,
        **kwargs
    ) -> Optional[AttackGraphNode]: ...

class PassiveAgent(DecisionAgent):
    def __init__(self, info):
        return

    def get_next_action(
        self,
        agent: MalSimAgentStateView,
        **kwargs
    ) -> Optional[AttackGraphNode]:
        return None
