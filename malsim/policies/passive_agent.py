"""A passive agent that always choose to do nothing"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .decision_agent import DecisionAgent

if TYPE_CHECKING:
    from ..mal_simulator import AgentState
    from maltoolbox.attackgraph import AttackGraphNode


class PassiveAgent(DecisionAgent):
    def __init__(self, *args: Any, **kwargs: Any): ...

    def get_next_action(
        self, agent_state: AgentState, **kwargs: Any
    ) -> AttackGraphNode | None:
        # A passive agent never does anything
        return None
