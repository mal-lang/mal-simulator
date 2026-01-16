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
from dataclasses import dataclass
from typing import Any, Optional, TextIO
import logging

import yaml

from maltoolbox.model import Model
from maltoolbox.language import LanguageGraph
from maltoolbox.attackgraph import create_attack_graph

from malsim.config.agent_settings import (
    DefenderSettings,
    AttackerSettings,
    agent_settings_from_dict,
)
from malsim.config.node_property_rule import NodePropertyRule


logger = logging.getLogger(__name__)

deprecated_fields = [
    'attacker_agent_class',
    'defender_agent_class',
    'attacker_entry_points',
]

# All required fields in scenario yml file
# Tuple indicates one of the fields in the tuple is required
required_fields: list[str, tuple[str, str]] = [
    ('agents', 'agent_settings'),
    'lang_file',
    ('model_file', 'model'),
]

# All allowed fields in scenario yml fild
allowed_fields = [
    *required_fields,
    'rewards',
    'observable_steps',
    'actionable_steps',
    'false_positive_rates',
    'false_negative_rates',
]


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
            allowed_fields_flattened.extend(f)

    # Verify that all keys in dict are supported
    for key in scenario_dict:
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
