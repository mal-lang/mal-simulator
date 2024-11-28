"""Abstract base class for agents"""

from abc import ABC, abstractmethod
from typing import Optional

class MalSimAgent(ABC):
    """Base class for an action selecting agent for the MalSimulator"""

    @abstractmethod
    def __init__(self, agent_config: dict, **kwargs):
        raise NotImplementedError(
            "__init__ must be implemented by inheriting class"
        )

    @abstractmethod
    def compute_next_action(self, action_surface) -> tuple[int, Optional[int]]:
        """From observation and mask, find the next action for agent"""

        raise NotImplementedError(
            "compute_next_action must be implemented by inheriting class"
        )

class MalSimAttackerAgent(MalSimAgent):
    """Base class for an action selecting agent for the MalSimulator"""

    def __init__(self, agent_config: dict, **kwargs):
        self.attacker = kwargs.get('attacker')

class MalSimDefenderAgent(MalSimAgent):
    """Base class for an action selecting agent for the MalSimulator"""

    def __init__(self, agent_config: dict, **kwargs):
        self.simulator = kwargs.get('simulator')
