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
from enum import Enum
import logging

import yaml

from maltoolbox.model import Model
from maltoolbox.language import LanguageGraph
from maltoolbox.attackgraph import (
    AttackGraph,
    AttackGraphNode,
    create_attack_graph
)

from .agents import (
    BreadthFirstAttacker,
    DepthFirstAttacker,
    KeyboardAgent,
    PassiveAgent,
    DefendCompromisedDefender,
    DefendFutureCompromisedDefender,
    RandomAgent,
    TTCSoftMinAttacker,
    ShortestPathAttacker
)

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Enum for agent types"""
    ATTACKER = 'attacker'
    DEFENDER = 'defender'

agent_class_name_to_class = {
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
    'attacker_entry_points'
]

# All required fields in scenario yml file
# Tuple indicates one of the fields in the tuple is required
required_fields: list[str | tuple[str, str]] = [
    'agents',
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

@dataclass
class AgentConfig:
    # Will be used for agents in the future instead of dicts
    name: str
    agent_class: Any
    agent: Any
    policy: Any
    config: dict[str, Any]
    type: AgentType

@dataclass
class AttackerAgentConfig(AgentConfig):
    """Agent configuration for the scenario"""
    entry_points: set[AttackGraphNode]
    goals: set[AttackGraphNode]
    ...

@dataclass
class DefenderAgentConfig(AgentConfig):
    """Agent configuration for the scenario"""
    ...

@dataclass
class Scenario:
    """Scenarios defines everything needed to run a simulation"""

    def __init__(
        self,
        lang_file: str,
        agents: dict[str, Any],
        model_dict: Optional[dict[str, Any]] = None,
        model_file: Optional[str] = None,
        model: Optional[Model] = None,
        rewards: Optional[dict[str, Any]] = None,
        false_positive_rates: Optional[dict[str, Any]] = None,
        false_negative_rates: Optional[dict[str, Any]] = None,
        is_observable: Optional[dict[str, Any]] = None,
        is_actionable: Optional[dict[str, Any]] = None,
    ):

        # Lang file is required
        self._lang_file = lang_file
        self.lang_graph = LanguageGraph.load_from_file(self._lang_file)

        # Model file is optional
        self._model_file = model_file

        # User must give one of model, model_dict, model_file
        # and we load the model accordingly.
        if model and not (model_dict or model_file):
            self.model = model
        elif model_dict and not (model or model_file):
            self.model = Model._from_dict(model_dict, self.lang_graph)
        elif model_file and not (model or model_dict):
            self.model = Model.load_from_file(model_file, self.lang_graph)
        else:
            raise ValueError(
                "Scenario needs exactly one of model_dict, model or model_file set"
            )

        # Attack graph is created from given lang and model
        self.attack_graph = create_attack_graph(self.lang_graph, self.model)

        # Agents are generated from agents dict (in scenario file)
        self._agents_dict = agents
        self.agents = load_simulator_agents(self.attack_graph, agents)

        self.rewards = apply_scenario_node_property(
            self.attack_graph, rewards or {}
        )
        self.false_positive_rates = apply_scenario_node_property(
            self.attack_graph, false_positive_rates or {}
        )
        self.false_negative_rates = apply_scenario_node_property(
            self.attack_graph, false_negative_rates or {}
        )
        self.is_observable = apply_scenario_node_property(
            self.attack_graph, is_observable or {}, default_value=False
        )
        self.is_actionable = apply_scenario_node_property(
            self.attack_graph, is_actionable or {}, default_value=False
        )

    def to_dict(self) -> dict[str, Any]:
        assert self._lang_file, "Can not save scenario to file if lang file was not given"
        scenario_dict = {
            # 'version': ?
            'lang_file': self._lang_file,
            'agents': self._agents_dict,
            'rewards': {},
            'false_positive_rates': {},
            'false_negative_rates': {},
            'is_observable': {},
            'is_actionable': {}
        }

        if self._model_file:
            scenario_dict['model_file'] = self._model_file
        elif self.model:
            scenario_dict['model'] = self.model._to_dict()

        return scenario_dict

    def save_to_file(self, file_path: str) -> None:
        save_scenario_dict(self.to_dict(), file_path)

    @classmethod
    def from_dict(cls, scenario_dict: dict[str, Any]) -> Scenario:
        return Scenario(
            lang_file=scenario_dict['lang_file'],
            agents=scenario_dict['agents'],
            model_dict=scenario_dict.get('model'),
            model_file=scenario_dict.get('model_file'),
            rewards=scenario_dict.get('rewards'),
            false_positive_rates=scenario_dict.get('false_positive_rates'),
            false_negative_rates=scenario_dict.get('false_negative_rates'),
            is_observable=scenario_dict.get('observable_steps'),
            is_actionable=scenario_dict.get('actionable_steps'),
        )

    @classmethod
    def load_from_file(cls, scenario_file: str) -> Scenario:
        scenario_dict = load_scenario_dict(scenario_file)
        _validate_scenario_dict(scenario_dict)
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
            raise SyntaxError(f"Scenario setting '{key}' is deprecated, see "
                               "README or ./tests/testdata/scenarios")
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
                raise RuntimeError(
                    f"Setting '{key_or_keys}' required in scenario file"
                )


def path_relative_to_file_dir(rel_path: str, file: TextIO) -> str:
    """Returns the absolute path of a relative path in a second file

    Arguments:
    rel_path    - relative path to append to dir of `file`
    file        - the file of which directory to evaluate the path from
    """

    file_dir_path = os.path.dirname(
        os.path.realpath(file.name)
    )
    return os.path.join(file_dir_path, rel_path)


def _validate_scenario_node_property_config(
        graph: AttackGraph, prop_config: dict[str, dict[str, Any]]) -> None:
    """Verify that node property configurations in a scenario contains only
    valid assets, asset types and step names"""

    # a way to lookup attack steps for asset types
    asset_type_step_names = {
        asset_type_name: [step_name for step_name in asset_type.attack_steps]
        for asset_type_name, asset_type in graph.lang_graph.assets.items()
    }

    if not prop_config:
        # Property configurations can be empty
        return

    assert 'by_asset_type' in prop_config or 'by_asset_name' in prop_config, (
        "Node property config in scenario file must  contain"
        "either 'by_asset_type' or 'by_asset_name' as keys"
    )

    for asset_type in prop_config.get('by_asset_type', []):
        # Make sure each specified asset type exists
        assert asset_type in asset_type_step_names.keys(), (
            f"Failed to find asset type '{asset_type}' in language "
            "when applying node property configuration")

        for step_name in prop_config['by_asset_type'][asset_type]:
            # Make sure each specified attack step name
            # exists for the specified asset type
            assert step_name in asset_type_step_names[asset_type], (
                f"Step '{step_name}' not found for asset type "
                f"'{asset_type}' in language when applying "
                "node property configuration"
            )

    assert graph.model, (
        "Attack graph in scenario needs to have a model attached to it"
    )
    asset_names = set(a.name for a in graph.model.assets.values())
    for asset_name in prop_config.get('by_asset_name', []):
        # Make sure each specified asset exist
        assert asset_name in asset_names, (
            f"Failed to find asset name '{asset_name}' in model "
            f"'{graph.model.name}' when applying node property "
            "configurations"
        )

        for step_name in prop_config['by_asset_name'][asset_name]:
            # Make sure each specified attack step name exists
            # for the specified asset
            expected_full_name = f"{asset_name}:{step_name}"
            assert graph.get_node_by_full_name(expected_full_name), (
                f"Attack step '{step_name}' not found for asset "
                f"'{asset_name}' when applying node property configurations"
            )


def apply_scenario_node_property(
        attack_graph: AttackGraph,
        prop_config: dict[str, dict[str, Any]],
        default_value: Optional[Any] = None,
) -> dict[AttackGraphNode, Any]:
    """Apply node property values from scenario configuration.

    Note: Property values provided 'by_asset_name' will take precedence over
    those provided 'by_asset_type' as they are more specific.

    Arguments:
    - attack_graph:     The attack graph to apply the settings to
    - node_prop:        Property name in string format (i.e. 'false_positive_rate')
    - prop_config:      Settings from scenario file with keys `by_asset_name`
                        and/or `by_asset_type`
    - default_value:    The default value to set for the property for nodes
                        where no value is given in the configuration. If None
                        no values will be set. This is only relevant if the
                        property is included in the scenario configuration.

    Return dict mapping from attack graph node to the value it is given
    for the requested property.
    """

    def _extract_value_from_entries(
            entries: dict[str, Any] | list[str], step_name: str
        ) -> Any:
        """
        Return the property value matching the step name in the provided
        entries.

        Arguments:
        - entries:      A list or dictionary representing the property entries
        - step_name:    The attack step name to look up

        Returns:
        - The value of the matching property or None if no match is found
        """
        if isinstance(entries, dict):
            # If a value is provided in a dictionary we assign it to the node
            return entries.get(step_name)
        elif isinstance(entries, list):
            # If a list of attack steps is provided we interpret it as a
            # binary property and set it to 1 if the attack step is in the
            # list
            value = 1 if step_name in entries else None
            return value
        else:
            raise ValueError('Error! Scenario node property configuration '
                'is neither dictionary, nor list!')


    property_dict: dict[AttackGraphNode, Any] = {}

    if not prop_config:
        return property_dict

    _validate_scenario_node_property_config(attack_graph, prop_config)

    for step in attack_graph.nodes.values():

        if default_value is not None:
            property_dict[step] = default_value

        # Check for matching asset type property configuration entry
        prop_asset_type_entries = (
            prop_config.get('by_asset_type', {})
            .get(step.lg_attack_step.asset.name, {})
        )
        prop_value_from_asset_type = _extract_value_from_entries(
            prop_asset_type_entries,
            step.name
        )

        # Check for matching specific asset(given by name) property
        # configuration entry
        assert step.model_asset, (
            f"Attack step {step} missing connection to model"
        )
        prop_specific_asset_entries = (
            prop_config.get('by_asset_name', {})
            .get(step.model_asset.name, {})
        )
        prop_value_from_specific_asset = _extract_value_from_entries(
            prop_specific_asset_entries,
            step.name
        )

        # Asset type values are applied first
        if prop_value_from_asset_type:
            property_dict[step] = prop_value_from_asset_type

        # Specific asset defined values override asset type values
        if prop_value_from_specific_asset:
            property_dict[step] = prop_value_from_specific_asset

    return property_dict


def get_entry_point_nodes(
    attack_graph: AttackGraph, entry_point_names: list[str]
) -> set[AttackGraphNode]:
    """Get entry point nodes from attackgraph

    Args:
    - attack_graph: the attack graph to the nodes from
    - entry_points: the entry points names to look for

    Returns:
    - the set of entry point nodes from the attack graph
    """

    entry_points = set()

    for entry_point_name in entry_point_names:
        entry_point = attack_graph.get_node_by_full_name(entry_point_name)
        if not entry_point:
            raise LookupError(f'Node {entry_point_name} does not exist')
        entry_points.add(entry_point)

    return entry_points


def load_simulator_agents(
        attack_graph: AttackGraph, scenario_agents: dict[str, Any]
    ) -> list[dict[str, Any]]:
    """Load agents to be registered in MALSimulator

    Create the agents from the specified classes,
    register entrypoints for attackers.

    Args:
    - attack_graph: the attack graph
    - scenario: the scenario in question as a dict
    Return:
    - agents: a list of agent configurations (dicts)
    """

    # Create list of agents dicts
    agents = []

    for agent_name, agent_info in scenario_agents.items():
        agent_type = AgentType(agent_info.get('type'))
        agent_class_name = agent_info.get('agent_class')
        agent_class = agent_class_name_to_class.get(agent_class_name)
        agent_config = agent_info.get('config', {})
        agent = agent_class(agent_config) if agent_class else None
        policy = agent_class(agent_config) if agent_class else None

        if agent_class_name and agent_class_name not in agent_class_name_to_class:
            raise LookupError(
                f"Agent class '{agent_class_name}' not supported.\n"
                f"Must be one of: {agent_class_name_to_class.values()}"
            )

        if agent_type == AgentType.ATTACKER:
            agent_config = {
                'name': agent_name,
                'agent_class': agent_class,
                'agent': agent,
                'policy': policy,
                'config': agent_config,
                'entry_points': get_entry_point_nodes(
                    attack_graph, agent_info['entry_points'] # Required
                ),
                'goals': get_entry_point_nodes(
                    attack_graph, agent_info.get('goals', []) # Optional
                ),
                'type': AgentType.ATTACKER
            }
        elif agent_type == AgentType.DEFENDER:
            agent_config = {
                'name': agent_name,
                'agent_class': agent_class,
                'agent': agent,
                'policy': policy,
                'config': agent_config,
                'type':AgentType.DEFENDER
            }

        agents.append(agent_config)

    return agents


def _extend_scenario(
        original_scenario_path: str, overriding_scenario: dict[str, Any]
    ) -> dict[str, Any]:
    """
    Override settings in `original_scenario_path` with settings
    in `overriding_scenario` and return the result.
    """

    original_scenario: dict[str, Any] = (
        load_scenario_dict(original_scenario_path)
    )
    resulting_scenario = original_scenario.copy()

    for key, value in overriding_scenario.items():
        # Override the original scenario with the
        # overriding scenario key,value pairs
        if key == "extends":
            # The 'extends' key is not needed after extend is done
            continue
        resulting_scenario[key] = value

    return resulting_scenario

def load_scenario_dict(scenario_file: str) -> dict[str, Any]:
    with open(scenario_file, 'r', encoding='utf-8') as s_file:
        scenario: dict[str, Any] = yaml.safe_load(s_file)

        if "extends" in scenario:
            original_scenario_path = (
                path_relative_to_file_dir(scenario['extends'], s_file)
            )
            scenario = _extend_scenario(original_scenario_path, scenario)

        # Convert path relative to scenario file
        scenario['lang_file'] = path_relative_to_file_dir(
            scenario['lang_file'], s_file
        )

        if 'model_file' in scenario:
            scenario['model_file'] = path_relative_to_file_dir(
                scenario['model_file'], s_file
            )

    return scenario


def load_scenario(scenario_file: str) -> Scenario:
    """Load a Scenario from a scenario file"""
    msg = (
        "'load_scenario' will be deprecated in mal-simulator 1.1.0, "
        "please use Scenario.load_from_file"
    )
    print(msg)
    logger.warning(msg)
    return Scenario.load_from_file(scenario_file)


def create_simulator_from_scenario(
        scenario_file: str,
        **kwargs: Any,
    ) -> None:
    """Deprecated"""
    raise DeprecationWarning("Use MalSimulator.from_scenario instead")


def create_scenario_dict(
    lang_file: str,
    model: str | Model,
    agents: dict[str, Any],
    settings: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Create a scenario dict from given settings"""
    raise DeprecationWarning("Use Scenario.to_dict")


def save_scenario_dict(
    scenario_dict: dict[str, Any], file_path: str
) -> None:
    """Save scenario to file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(scenario_dict, f, sort_keys=False)
