"""A decision agent is a heuristic agent"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..mal_simulator import MalSimAgentState
    from maltoolbox.attackgraph import AttackGraphNode


class DecisionAgent(ABC):
    @abstractmethod
    def get_next_action(
        self, agent_state: MalSimAgentState, **kwargs: Any
    ) -> AttackGraphNode | None:
        """
        Select next action the agent will work with.

        Attributes:
            agent: Current state of and other info about the agent from the simulator

        Returns:
            The selected action or None if there are no actions to select from.
        """
        ...
