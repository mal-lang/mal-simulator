"""
A collection of functions used to load 'scenarios'.

A scenario is a combination of:
    - a MAL language
    - a MAL model
    - optionally defined rewards
    - optionally defined attacker entrypoints
    - Additional simulation configurations
        - attacker_class
        - defender_class
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Optional, TextIO
from enum import Enum
import logging

import yaml

from maltoolbox.model import Model
from maltoolbox.language import LanguageGraph
from maltoolbox.attackgraph import AttackGraphNode, AttackGraph, create_attack_graph

from .agents import (
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

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Enum for agent types"""

    ATTACKER = 'attacker'
    DEFENDER = 'defender'


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

deprecated_fields = [
    'attacker_agent_class',
    'defender_agent_class',
    'attacker_entry_points',
]

# All required fields in scenario yml file
# Tuple indicates one of the fields in the tuple is required
required_fields: list[str | tuple[str, str]] = [
    ('agents', 'agent_settings'),
    'lang_file',
    ('model_file', 'model'),
]

# All allowed fields in scenario yml fild
allowed_fields = required_fields + [
    'rewards',
    'observable_steps',
    'actionable_steps',
    'false_positive_rates',
    'false_negative_rates',
]


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


@dataclass
class NodePropertyRule:
    """
    Defines a mapping from nodes to values based on:
    - asset_type filters
    - asset_name filters
    """

    by_asset_type: dict[str, Any] = field(default_factory=dict)
    by_asset_name: dict[str, Any] = field(default_factory=dict)
    default: Any = None

    def __post_init__(self) -> None:
        if not self.by_asset_type and not self.by_asset_name:
            raise ValueError('Expected either "by_asset_type" or "by_asset_name"')

    def value(self, node: AttackGraphNode, default: Any = None) -> Any:
        """Get value for `node` based on this node property config"""
        if not node.model_asset:
            return default

        asset_name = node.model_asset.name
        asset_type = node.model_asset.type

        # precedence: asset_name > asset_type > default
        by_asset_name = self.by_asset_name.get(asset_name, {})
        if node.name in by_asset_name:
            if isinstance(by_asset_name, list):
                return True
            return by_asset_name[node.name]
        by_asset_type = self.by_asset_type.get(asset_type, {})
        if node.name in by_asset_type:
            if isinstance(by_asset_type, list):
                return True
            return by_asset_type[node.name]

        return default

    def per_node(self, attack_graph: AttackGraph) -> dict[str, Any]:
        """Return a dict mapping from each step full name to value given by config"""
        per_node_dict = dict()
        for n in attack_graph.nodes.values():
            value = self.value(n)
            if value is not None:
                per_node_dict[n.full_name] = value
        return per_node_dict

    @classmethod
    def _validate_dict(cls, node_property_dict: dict[str, Any]) -> None:
        allowed_fields = {'by_asset_type', 'by_asset_name'}
        present_allowed_fields = allowed_fields & node_property_dict.keys()
        forbidden_fields = node_property_dict.keys() - allowed_fields
        if not present_allowed_fields:
            raise ValueError(
                "Node property dict need at least 'by_asset_type' or 'by_asset_name'"
            )
        if forbidden_fields:
            raise ValueError(f'Node property fields not allowed: {forbidden_fields}')

    @classmethod
    def from_optional_dict(
        cls, node_property_dict: dict[str, dict[str, Any]] | None
    ) -> Optional[NodePropertyRule]:
        if node_property_dict is None:
            return None

        cls._validate_dict(node_property_dict)
        return NodePropertyRule(
            node_property_dict.get('by_asset_type', {}),
            node_property_dict.get('by_asset_name', {}),
        )

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            'by_asset_type': self.by_asset_type,
            'by_asset_name': self.by_asset_name,
        }


@dataclass
class Scenario:
    """Scenarios define everything needed to run a simulation"""

    def __init__(
        self,
        lang_file: str,
        model: Model | dict[str, Any] | str,
        agent_settings: dict[str, AttackerSettings | DefenderSettings],
        rewards: Optional[dict[str, Any]] = None,
        false_positive_rates: Optional[dict[str, Any]] = None,
        false_negative_rates: Optional[dict[str, Any]] = None,
        observable_steps: Optional[dict[str, Any]] = None,
        actionable_steps: Optional[dict[str, Any]] = None,
    ):
        # Lang file is required
        self._lang_file = lang_file
        self.lang_graph = LanguageGraph.load_from_file(self._lang_file)

        self._model_file = None
        if isinstance(model, str):
            self._model_file = model
            self.model = Model.load_from_file(self._model_file, self.lang_graph)
        elif isinstance(model, dict):
            self.model = Model._from_dict(model, self.lang_graph)
        elif isinstance(model, Model):
            self.model = model
        else:
            raise ValueError('`model` must be Model, dict, or str (file path)')

        self.attack_graph = create_attack_graph(self.lang_graph, self.model)
        self.agent_settings = agent_settings

        # Wrap dicts in NodePropertyRule if given
        self.rewards = NodePropertyRule.from_optional_dict(rewards)
        self.false_positive_rates = NodePropertyRule.from_optional_dict(
            false_positive_rates
        )
        self.false_negative_rates = NodePropertyRule.from_optional_dict(
            false_negative_rates
        )
        self.is_observable = NodePropertyRule.from_optional_dict(observable_steps)
        self.is_actionable = NodePropertyRule.from_optional_dict(actionable_steps)

    def to_dict(self) -> dict[str, Any]:
        assert self._lang_file, (
            'Can not save scenario to file if lang file was not given'
        )
        scenario_dict = {
            # 'version': ?
            'lang_file': self._lang_file,
            'agents': {
                name: agent.to_dict() for name, agent in self.agent_settings.items()
            },
        }

        if self.rewards:
            scenario_dict['rewards'] = self.rewards.to_dict()
        if self.false_positive_rates:
            scenario_dict['false_positive_rates'] = self.false_positive_rates.to_dict()
        if self.false_negative_rates:
            scenario_dict['false_negative_rates'] = self.false_negative_rates.to_dict()
        if self.is_observable:
            scenario_dict['observable_steps'] = self.is_observable.to_dict()
        if self.is_actionable:
            scenario_dict['actionable_steps'] = self.is_actionable.to_dict()
        if self._model_file:
            # Use model file name instead of full model if model file was given at init
            scenario_dict['model_file'] = self._model_file
        else:
            scenario_dict['model'] = self.model.to_dict()

        return scenario_dict

    def save_to_file(self, file_path: str) -> None:
        """Save scenario to a yaml-file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, scenario_dict: dict[str, Any]) -> Scenario:
        """Create a scenario object from a scenario dictionary"""
        _validate_scenario_dict(scenario_dict)

        agent_settings = {}
        # Load agent settings from dict
        for name, agent_settings_dict in scenario_dict['agents'].items():
            agent_settings[str(name)] = agent_settings_from_dict(
                name, agent_settings_dict
            )

        model_or_model_file = scenario_dict.get('model') or scenario_dict['model_file']
        return Scenario(
            lang_file=scenario_dict['lang_file'],
            agent_settings=agent_settings,
            model=model_or_model_file,
            rewards=scenario_dict.get('rewards'),
            false_positive_rates=scenario_dict.get('false_positive_rates'),
            false_negative_rates=scenario_dict.get('false_negative_rates'),
            observable_steps=scenario_dict.get('observable_steps'),
            actionable_steps=scenario_dict.get('actionable_steps'),
        )

    @classmethod
    def load_from_file(cls, scenario_file: str) -> Scenario:
        scenario_dict = load_scenario_dict(scenario_file)
        return cls.from_dict(scenario_dict)


def _validate_scenario_dict(scenario_dict: dict[str, Any]) -> None:
    """Verify scenario file keys"""

    # Unpack tuples from allowed fields
    allowed_fields_flattened = []
    for f in allowed_fields:
        if isinstance(f, str):
            allowed_fields_flattened.append(f)
        elif isinstance(f, tuple):
            for sf in f:
                allowed_fields_flattened.append(sf)

    # Verify that all keys in dict are supported
    for key in scenario_dict.keys():
        if key in deprecated_fields:
            raise SyntaxError(
                f"Scenario setting '{key}' is deprecated, see "
                'README or ./tests/testdata/scenarios'
            )
        if key not in allowed_fields_flattened:
            raise SyntaxError(f"Scenario setting '{key}' is not supported")

    # Verify that all required fields are in scenario file
    for key_or_keys in required_fields:
        if isinstance(key_or_keys, tuple):
            if not any(k in scenario_dict for k in key_or_keys):
                raise RuntimeError(
                    f"One of '{key_or_keys}' is required in scenario file"
                )
            if all(k in scenario_dict for k in key_or_keys):
                raise RuntimeError(
                    f"Only one of '{key_or_keys}' is allowed in scenario file"
                )
        elif isinstance(key_or_keys, str):
            if key_or_keys not in scenario_dict:
                raise RuntimeError(f"Setting '{key_or_keys}' required in scenario file")


def path_relative_to_file_dir(rel_path: str, file: TextIO) -> str:
    """Returns the absolute path of a relative path in a second file

    Arguments:
    rel_path    - relative path to append to dir of `file`
    file        - the file of which directory to evaluate the path from
    """

    file_dir_path = os.path.dirname(os.path.realpath(file.name))
    return os.path.join(file_dir_path, rel_path)


def _extend_scenario(
    original_scenario_path: str, overriding_scenario: dict[str, Any]
) -> dict[str, Any]:
    """
    Override settings in `original_scenario_path` with settings
    in `overriding_scenario` and return the result.
    """

    original_scenario: dict[str, Any] = load_scenario_dict(original_scenario_path)
    resulting_scenario = original_scenario.copy()
    for key, value in overriding_scenario.items():
        # Override the original scenario with the
        # overriding scenario key,value pairs
        if key == 'extends':
            # The 'extends' key is not needed after extend is done
            continue
        resulting_scenario[key] = value
    return resulting_scenario


def load_scenario_dict(scenario_file: str) -> dict[str, Any]:
    """From a scenario file, load a scenario dict.

    Extend it with other scenario if `extend` keyword is used.
    """
    with open(scenario_file, 'r', encoding='utf-8') as s_file:
        scenario: dict[str, Any] = yaml.safe_load(s_file)

        if 'extends' in scenario:
            original_scenario_path = path_relative_to_file_dir(
                scenario['extends'], s_file
            )
            scenario = _extend_scenario(original_scenario_path, scenario)

        # Convert path relative to scenario file
        scenario['lang_file'] = path_relative_to_file_dir(scenario['lang_file'], s_file)

        if 'model_file' in scenario:
            scenario['model_file'] = path_relative_to_file_dir(
                scenario['model_file'], s_file
            )
    return scenario
