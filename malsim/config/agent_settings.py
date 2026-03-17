from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from malsim.config.node_property_rule import NodePropertyRule
from malsim.config.sim_settings import RewardMode
from malsim.mal_simulator.node_getters import full_name_or_node_to_node
from malsim.mal_simulator.ttc_utils import TTCDist


class AgentType(Enum):
    """Enum for agent types"""

    ATTACKER = 'attacker'
    DEFENDER = 'defender'


class AgentRuntimeMixin:
    """Provides agent construction for attacker and defender."""

    policy: type | None
    config: dict[str, Any]
    _agent: Any | None = None

    def _create_agent(self) -> Any | None:
        if self.policy:
            return self.policy(self.config)
        return None

    def reset_agent(self) -> None:
        self._agent = self._create_agent()

    @property
    def agent(self) -> Any | None:
        if self._agent:
            return self._agent
        if self.policy is None:
            return None
        self._agent = self._create_agent()
        return self._agent


T = TypeVar('T', bound=AttackGraphNode | str, covariant=True)


@dataclass
class AttackerSettings(AgentRuntimeMixin, Generic[T]):
    """Settings for an attacker in a scenario."""

    name: str
    entry_points: Set[T]
    policy: type | None = None
    actionable_steps: NodePropertyRule[bool] | None = None
    rewards: NodePropertyRule[float] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    type: AgentType = AgentType.ATTACKER
    reward_mode: RewardMode = RewardMode.CUMULATIVE
    # Goals affect simulation termination but is optional
    goals: Set[T] = field(default_factory=frozenset)
    # TTC distributions that override TTCs set in language
    ttc_dists: NodePropertyRule[TTCDist] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
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

    def __post_init__(self) -> None:
        if isinstance(self.reward_mode, str):
            self.reward_mode = RewardMode[self.reward_mode]

    def convert_to_attack_graph_nodes(
        self, attack_graph: AttackGraph
    ) -> AttackerSettings[AttackGraphNode]:
        """Convert entry points, goals, and impossible steps from full names to nodes"""
        return AttackerSettings(
            goals=frozenset(
                full_name_or_node_to_node(attack_graph, g) for g in self.goals
            ),
            entry_points=frozenset(
                full_name_or_node_to_node(attack_graph, ep) for ep in self.entry_points
            ),
            name=self.name,
            policy=self.policy,
            actionable_steps=self.actionable_steps,
            rewards=self.rewards,
            config=self.config,
            type=self.type,
            reward_mode=self.reward_mode,
            ttc_dists=self.ttc_dists,
        )


@dataclass
class DefenderSettings(AgentRuntimeMixin):
    """Settings for a defender in a scenario."""

    name: str
    policy: type | None = None
    observable_steps: NodePropertyRule[bool] | None = None
    actionable_steps: NodePropertyRule[bool] | None = None
    rewards: NodePropertyRule[float] | None = None
    false_positive_rates: NodePropertyRule[float] | None = None
    false_negative_rates: NodePropertyRule[float] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    type: AgentType = AgentType.DEFENDER
    reward_mode: RewardMode = RewardMode.CUMULATIVE

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
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

    def __post_init__(self) -> None:
        if isinstance(self.reward_mode, str):
            self.reward_mode = RewardMode[self.reward_mode]


def get_defender_settings(
    agent_settings: dict[str, DefenderSettings | AttackerSettings[AttackGraphNode]],
) -> dict[str, DefenderSettings]:
    """Return the defender settings from agent_settings dict"""
    return {k: v for k, v in agent_settings.items() if isinstance(v, DefenderSettings)}


def get_attacker_settings(
    agent_settings: dict[str, DefenderSettings | AttackerSettings[AttackGraphNode]],
) -> dict[str, AttackerSettings[AttackGraphNode]]:
    """Return the attacker settings from agent_settings dict"""
    return {k: v for k, v in agent_settings.items() if isinstance(v, AttackerSettings)}


AgentSettings = dict[str, AttackerSettings[AttackGraphNode] | DefenderSettings]


def defender_settings(agent_settings: AgentSettings) -> dict[str, DefenderSettings]:
    return {
        name: settings
        for name, settings in agent_settings.items()
        if isinstance(settings, DefenderSettings)
    }


def attacker_settings(
    agent_settings: AgentSettings,
) -> dict[str, AttackerSettings[AttackGraphNode]]:
    return {
        name: settings
        for name, settings in agent_settings.items()
        if isinstance(settings, AttackerSettings)
    }
