"""MalSimAgent stores the state of an agent inside the MalSimulator"""

from __future__ import annotations
from typing import TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode


class AgentType(Enum):
    """Enum for agent types"""
    ATTACKER = 'attacker'
    DEFENDER = 'defender'


@dataclass
class MalSimAgent:
    """Stores the state of an agent in the simulator"""

    # Identifier of the agent, used in MalSimulator for lookup
    name: str
    type: AgentType

    # Contains current agent reward in the simulation
    # Attackers get positive rewards, defenders negative
    reward: int = 0

    # Contains possible actions for the agent in the next step
    action_surface: list[AttackGraphNode] = field(default_factory=list)

    # Fields that tell if agent is 'dead' / disabled
    truncated: bool = False
    terminated: bool = False

    # Fields that are not used by 'base' MalSimulator
    # but can be useful in wrappers/envs
    observation: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

class MalSimAttacker(MalSimAgent):
    """Stores the state of an attacker in the simulator"""
    def __init__(self, name: str, attacker_id: int):
        super().__init__(name, AgentType.ATTACKER)
        self.attacker_id = attacker_id

class MalSimDefender(MalSimAgent):
    """Stores the state of a defender in the simulator"""
    def __init__(self, name: str):
        super().__init__(name, AgentType.DEFENDER)


class MalSimAgentView:
    """Read-only interface to MalSimAgent."""
    def __init__(self, agent: MalSimAgent):
        self._agent = agent

    @property
    def name(self) -> str:
        return self._agent.name

    @property
    def type(self) -> AgentType:
        return self._agent.type

    @property
    def reward(self) -> int:
        return self._agent.reward

    @property
    def truncated(self) -> bool:
        return self._agent.truncated

    @property
    def terminated(self) -> bool:
        return self._agent.terminated

    @property
    def action_surface(self) -> list[AttackGraphNode]:
        return self._agent.action_surface

    @property
    def attacker_id(self) -> int:
        assert isinstance(self._agent, MalSimAttacker), \
               "Only MalSimAttackers have property attacker_id"
        return self._agent.attacker_id
