"""A passive agent that always choose to do nothing"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from .decision_agent import DecisionAgent
from ..mal_simulator import AgentState

if TYPE_CHECKING:
    from ..mal_simulator import AgentState
    from maltoolbox.attackgraph import AttackGraphNode

class PassiveAgent(DecisionAgent):
    def __init__(self, *args, **kwargs):
        ...

    def get_next_action(
        self,
        agent_state: AgentState,
        **kwargs
    ) -> Optional[AttackGraphNode]:
        # A passive agent never does anything
        return None
