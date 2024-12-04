"""Abstract base class for agents"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from maltoolbox.attackgraph import AttackGraphNode

class AgentType(Enum):
    """Enum for agent types"""
    ATTACKER = 'attacker'
    DEFENDER = 'defender'

class MalSimAgent(ABC):
    """Stores the state of an agent in the simulator"""

    def __init__(self, name: str, agent_type: AgentType):
        self.name = name
        self.type = agent_type

        self.observation = {}
        self.action_surface = []
        self.reward = 0
        self.done = False

    @abstractmethod
    def update_state(self, performed_steps: list[AttackGraphNode]):
        """An abstract method that is supposed to update whatever state
        the agent builds up for it to be able to take next action"""

        raise NotImplementedError(
            "Agent class need to implement 'update_state'"
        )

    @abstractmethod
    def get_next_action(
        self, action_surface, **kwargs
        ) -> list[AttackGraphNode]:
        """From the current state, find the next action for agent"""

        raise NotImplementedError(
            "Agent class need to implement 'get_next_action'"
        )

class MalSimDefender(MalSimAgent):
    """A defender agent"""
    type = AgentType.DEFENDER

    def __init__(self, name: str):
        super().__init__(name, self.type)

class MalSimAttacker(MalSimAgent):
    """A defender agent"""
    type = AgentType.ATTACKER

    def __init__(self, name: str, attacker_id: int):
        self.attacker_id = attacker_id
        super().__init__(name, self.type)


class PassiveAttacker(MalSimAttacker):
    """An agent that does nothing"""

    def update_state(self, performed_steps: list): ...

    def get_next_action(
            self, action_surface, **kwargs
        ) -> list[AttackGraphNode]:
        return []

class PassiveDefender(MalSimDefender):
    """An agent that does nothing"""

    def update_state(self, performed_steps: list): ...

    def get_next_action(
            self, action_surface, **kwargs
        ) -> list[AttackGraphNode]:
        return []
