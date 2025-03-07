"""A passive agent that always choose to do nothing"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from .decision_agent import DecisionAgent
from ..mal_simulator import MalSimAgentStateView

if TYPE_CHECKING:
    from ..mal_simulator import MalSimAgentStateView
    from maltoolbox.attackgraph import AttackGraphNode

class PassiveAgent(DecisionAgent):
    def __init__(self, *args, **kwargs):
        ...

    def get_next_action(
        self,
        agent_state: MalSimAgentStateView,
        **kwargs
    ) -> Optional[AttackGraphNode]:
        # A passive agent never does anything
        return None
