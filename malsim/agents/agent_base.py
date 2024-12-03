"""Abstract base class for agents"""

from abc import ABC, abstractmethod
from typing import Optional

class MalSimAgent(ABC):
    """Base class for an action selecting agent for the MalSimulator"""

    @abstractmethod
    def __init__(self, agent_config: dict, **kwargs):
        self.simulator = kwargs.get('simulator')

    @abstractmethod
    def get_next_action(
        self, action_surface, **kwargs
    ) -> tuple[int, Optional[int]]:
        """From observation and mask, find the next action for agent"""

        raise NotImplementedError(
            "get_next_action must be implemented by inheriting class"
        )

class MalSimAttackerAgent(MalSimAgent):
    """Base class for an action selecting agent for the MalSimulator"""

    def __init__(self, agent_config: dict, **kwargs):
        self.attacker = kwargs.get('attacker')
        super().__init__(agent_config, **kwargs)

class MalSimDefenderAgent(MalSimAgent):
    """Base class for an action selecting agent for the MalSimulator"""

    def __init__(self, agent_config: dict, **kwargs):
        super().__init__(agent_config, **kwargs)


class PassiveAgent(MalSimAgent):
    """An agent that does nothing"""
    def get_next_action(self, _) -> list:
        return []
