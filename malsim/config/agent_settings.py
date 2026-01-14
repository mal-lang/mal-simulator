from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from maltoolbox.attackgraph import AttackGraphNode
from malsim.config.node_property_rule import NodePropertyRule

from malsim.policies import (
    BreadthFirstAttacker,
    DepthFirstAttacker,
    KeyboardAgent,
    PassiveAgent,
    DefendCompromisedDefender,
    DefendFutureCompromisedDefender,
    RandomAgent,
    TTCSoftMinAttacker,
    ShortestPathAttacker,
)

policy_name_to_class = {
    'KeyboardAgent': KeyboardAgent,
    'PassiveAgent': PassiveAgent,
    'DepthFirstAttacker': DepthFirstAttacker,
    'BreadthFirstAttacker': BreadthFirstAttacker,
    'TTCSoftMinAttacker': TTCSoftMinAttacker,
    'ShortestPathAttacker': ShortestPathAttacker,
    'DefendCompromisedDefender': DefendCompromisedDefender,
    'DefendFutureCompromisedDefender': DefendFutureCompromisedDefender,
    'RandomAgent': RandomAgent,
}


class AgentType(Enum):
    """Enum for agent types"""

    ATTACKER = 'attacker'
    DEFENDER = 'defender'


class AgentRuntimeMixin:
    """Provides agent construction for attacker and defender."""

    policy: Optional[type]
    config: dict[str, Any]
    _agent: Optional[Any] = None

    def _create_agent(self) -> Optional[Any]:
        if self.policy:
            return self.policy(self.config)
        return None

    def reset_agent(self) -> None:
        self._agent = self._create_agent()

    @property
    def agent(self) -> Optional[Any]:
        if self._agent:
            return self._agent
        if self.policy is None:
            return None
        self._agent = self._create_agent()
        return self._agent


@dataclass
class AttackerSettings(AgentRuntimeMixin):
    """Settings for an attacker in a scenario."""

    name: str
    entry_points: set[str] | set[AttackGraphNode]
    goals: set[str] | set[AttackGraphNode] = field(default_factory=set)
    policy: Optional[type] = None
    actionable_steps: Optional[NodePropertyRule] = None
    rewards: Optional[NodePropertyRule] = None
    ttc_overrides: Optional[NodePropertyRule] = None
    config: dict[str, Any] = field(default_factory=dict)
    type: AgentType = AgentType.ATTACKER

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            'name': self.name,
            'type': AgentType.ATTACKER.value,
            'entry_points': {
                n.full_name if isinstance(n, AttackGraphNode) else n
                for n in self.entry_points
            },
        }
        if self.goals:
            d['goals'] = {
                n.full_name if isinstance(n, AttackGraphNode) else n for n in self.goals
            }
        if self.policy:
            d['policy'] = self.policy.__name__
        if self.actionable_steps:
            d['actionable_steps'] = self.actionable_steps.to_dict()
        if self.rewards:
            d['rewards'] = self.rewards.to_dict()
        if self.config:
            d['config'] = self.config
        return d


@dataclass
class DefenderSettings(AgentRuntimeMixin):
    """Settings for a defender in a scenario."""

    name: str
    policy: Optional[type] = None
    observable_steps: Optional[NodePropertyRule] = None
    actionable_steps: Optional[NodePropertyRule] = None
    rewards: Optional[NodePropertyRule] = None
    false_positive_rates: Optional[NodePropertyRule] = None
    false_negative_rates: Optional[NodePropertyRule] = None
    config: dict[str, Any] = field(default_factory=dict)
    type: AgentType = AgentType.DEFENDER

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            'name': self.name,
            'type': AgentType.DEFENDER.value,
        }
        if self.policy:
            d['policy'] = self.policy.__name__
        if self.observable_steps:
            d['observable_steps'] = self.observable_steps.to_dict()
        if self.actionable_steps:
            d['actionable_steps'] = self.actionable_steps.to_dict()
        if self.rewards:
            d['rewards'] = self.rewards.to_dict()
        if self.false_positive_rates:
            d['false_positive_rates'] = self.false_positive_rates.to_dict()
        if self.false_negative_rates:
            d['false_negative_rates'] = self.false_negative_rates.to_dict()
        if self.config:
            d['config'] = self.config
        return d


def agent_settings_from_dict(
    name: str,
    d: dict[str, Any],
) -> AttackerSettings | DefenderSettings:
    """Load agent settings from a dict"""

    agent_type = AgentType(d['type'])

    # Resolve policy class if provided
    policy = None
    policy_name = d.get('policy') or d.get('agent_class')
    if policy_name:
        if policy_name not in policy_name_to_class:
            raise LookupError(
                f"Policy class '{policy_name}' not supported. "
                f'Must be one of: {list(policy_name_to_class.keys())}'
            )
        policy = policy_name_to_class[policy_name]

    config = d.get('config', {})

    if agent_type == AgentType.ATTACKER:
        return AttackerSettings(
            name=name,
            entry_points=set(d['entry_points']),
            goals=set(d.get('goals', [])),
            ttc_overrides=NodePropertyRule.from_optional_dict(d.get('ttc_overrides')),
            policy=policy,
            actionable_steps=NodePropertyRule.from_optional_dict(
                d.get('actionable_steps')
            ),
            rewards=NodePropertyRule.from_optional_dict(d.get('rewards')),
            config=config,
        )

    # Defender
    return DefenderSettings(
        name=name,
        policy=policy,
        observable_steps=NodePropertyRule.from_optional_dict(d.get('observable_steps')),
        actionable_steps=NodePropertyRule.from_optional_dict(d.get('actionable_steps')),
        rewards=NodePropertyRule.from_optional_dict(d.get('rewards')),
        false_positive_rates=NodePropertyRule.from_optional_dict(
            d.get('false_positive_rates')
        ),
        false_negative_rates=NodePropertyRule.from_optional_dict(
            d.get('false_negative_rates')
        ),
        config=config,
    )

AgentSettings = dict[str, AttackerSettings | DefenderSettings]
