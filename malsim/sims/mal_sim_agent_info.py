"""Abstract base class for agents"""

from abc import ABC
from enum import Enum

from maltoolbox.attackgraph import AttackGraphNode

class AgentType(Enum):
    """Enum for agent types"""
    ATTACKER = 'attacker'
    DEFENDER = 'defender'


class AgentInfo(ABC):
    """Stores the state of an agent in the simulator"""

    def __init__(
            self,
            name: str,
            agent_type: AgentType
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


class AttackerAgentInfo(AgentInfo):
    """Stores the state of an attacker in the simulator"""

    def __init__(self, name, attacker_id: int):
        super().__init__(name, AgentType.ATTACKER)
        self.attacker_id = attacker_id


class DefenderAgentInfo(AgentInfo):
    """Stores the state of an attacker in the simulator"""

    def __init__(self, name):
        super().__init__(name, AgentType.DEFENDER)
