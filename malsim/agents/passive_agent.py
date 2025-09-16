"""A passive agent that always choose to do nothing"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any

from .decision_agent import DecisionAgent

if TYPE_CHECKING:
    from ..mal_simulator import MalSimAgentState
    from maltoolbox.attackgraph import AttackGraphNode

class PassiveAgent(DecisionAgent):
    def __init__(self, *args: Any, **kwargs: Any):
        ...

    def get_next_action(
        self,
        agent_state: MalSimAgentState,
        **kwargs: Any
    ) -> Optional[AttackGraphNode]:
        # A passive agent never does anything
        return None
