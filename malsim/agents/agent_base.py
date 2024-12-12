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

        # Identifier of the agent, used in MalSimulator for lookup
        self.name = name

        # AgentType.ATTACKER / AgentType.DEFENDER
        self.type = agent_type

        # Contains reward in last step of MalSimulator
        self.reward = 0

        # Contains action mask (only for ParallelEnv)
        self.observation = {}

        # Contains action mask (only for ParallelEnv)
        self.info = {}

        # Agent is truncated if Malsimulator has run for
        # more than `max_iter` iterations
        self.truncated = False

        # Attacker agent is terminated if it has nothing left to do
        # Defender agent is terminated if all attacker agents are terminated
        self.terminated = False

        # Contains list of potential actions (nodes
        # for an agent in next step of MalSimulator
        self.action_surface: list[AttackGraphNode] = []

    @abstractmethod
    def get_next_action(
        self, **kwargs
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
            self, **kwargs
        ) -> list[AttackGraphNode]:
        return []


class PassiveDefender(MalSimDefender):
    """An agent that does nothing"""

    def get_next_action(
            self, **kwargs
        ) -> list[AttackGraphNode]:
        return []
