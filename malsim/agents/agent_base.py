"""Abstract base class for agents"""

from abc import ABC, abstractmethod
from enum import Enum

from maltoolbox.attackgraph import AttackGraphNode

class AgentType(Enum):
    """Enum for agent types"""
    ATTACKER = 'attacker'
    DEFENDER = 'defender'


class MalSimAgent(ABC):
    """Stores the state of an agent in the simulator"""

    def __init__(
            self,
            name: str,
            agent_type: AgentType,
            **kwargs
        ):

        self.name = name
        self.type = agent_type

        self.observation = {}
        self.action_surface = []
        self.reward = 0
        self.done = False

    def update_obs(self, performed_steps: list[AttackGraphNode]):
        """An abstract method that is supposed to update whatever state
        the agent builds up for it to be able to take next action"""

    @abstractmethod
    def get_next_action(
        self, action_surface, **kwargs
        ) -> list[AttackGraphNode]:
        """From the current state, find the next action for agent"""

        raise NotImplementedError(
            "Agent class need to implement 'get_next_action'"
        )


class MalSimDefender(MalSimAgent):
    """A defender agent to use in MAL simulator"""

    def __init__(self, name: str, **kwargs):
        super().__init__(name, AgentType.DEFENDER, **kwargs)


class MalSimAttacker(MalSimAgent):
    """An attacker agent to use in MAL Simulator"""

    def __init__(self, name: str, attacker_id: int, **kwargs):
        super().__init__(
            name, AgentType.ATTACKER, **kwargs
        )
        self.attacker_id = attacker_id


class PassiveAttacker(MalSimAttacker):
    """An agent that does nothing"""

    def get_next_action(
            self, action_surface, **kwargs
        ) -> list[AttackGraphNode]:
        return []


class PassiveDefender(MalSimDefender):
    """An agent that does nothing"""

    def get_next_action(
            self, action_surface, **kwargs
        ) -> list[AttackGraphNode]:
        return []
