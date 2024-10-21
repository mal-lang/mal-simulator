"""Abstract base class for agents"""

from abc import ABC, abstractmethod
from typing import Optional

class MalSimulatorAgent(ABC):
    """Base class for an action selecting agent for the MalSimulator"""

    @abstractmethod
    def __init__(self, agent_config: dict, **kwargs):
        raise NotImplementedError(
            "__init__ must be implemented by inheriting class"
        )

    @abstractmethod
    def compute_action_from_dict(
        self, observation: dict, mask: dict) -> tuple[int, Optional[int]]:
        """From observation and mask, find the next action for agent"""

        raise NotImplementedError(
            "compute_action_from_dict must be implemented by inheriting class"
        )
