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

import os
from typing import Any, Optional

import yaml

from maltoolbox.attackgraph import (
    AttackGraph,
    AttackGraphNode,
    Attacker,
    create_attack_graph
)

from .agents import (
    BreadthFirstAttacker,
    DepthFirstAttacker,
    KeyboardAgent,
    PassiveAgent,
    DefendCompromisedDefender,
    DefendFutureCompromisedDefender
)

from .mal_simulator import AgentType, MalSimulator

agent_class_name_to_class = {
    'DepthFirstAttacker': DepthFirstAttacker,
    'BreadthFirstAttacker': BreadthFirstAttacker,
    'KeyboardAgent': KeyboardAgent,
    'PassiveAgent': PassiveAgent,
    'DefendCompromisedDefender': DefendCompromisedDefender,
    'DefendFutureCompromisedDefender': DefendFutureCompromisedDefender
}

deprecated_fields = [
    'attacker_agent_class',
    'defender_agent_class',
    'attacker_entry_points'
]

# All required fields in scenario yml file
required_fields = [
    'agents',
    'lang_file',
    'model_file',
]

# All allowed fields in scenario yml fild
allowed_fields = required_fields + [
    'rewards',
    'observable_steps',
    'actionable_steps',
    'false_positive_rates',
    'false_negative_rates',
]


def validate_scenario(scenario_dict):
    """Verify scenario file keys"""

    # Verify that all keys in dict are supported
    for key in scenario_dict.keys():
        if key in deprecated_fields:
            raise SyntaxError(f"Scenario setting '{key}' is deprecated, see "
                               "README or ./tests/testdata/scenarios")
        if key not in allowed_fields:
            raise SyntaxError(f"Scenario setting '{key}' is not supported")

    # Verify that all required fields are in scenario file
    for key in required_fields:
        if key not in scenario_dict:
            raise RuntimeError(f"Setting '{key}' missing from scenario file")


def path_relative_to_file_dir(rel_path, file):
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
        graph: AttackGraph, prop_config: dict):
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

    # TODO: revisit this variable once LookupDicts are merged
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
        node_prop: str,
        prop_config: dict,
        assumed_value: Optional[Any] = None,
        default_value: Optional[Any] = None,
        set_as_extras: bool = True
):
    """Apply node property values from scenario configuration.

    Note: Property values provided 'by_asset_name' will take precedence over
    those provided 'by_asset_type' as they are more specific.

    Arguments:
    - attack_graph:     The attack graph to apply the settings to
    - node_prop:        Property name in string format (i.e. 'false_positive_rate')
    - prop_config:      Settings from scenario file with keys `by_asset_name`
                        and/or `by_asset_type`
    - assumed_value:    The assumed value to set for the property for all
                        nodes if property is entirely omitted in the
                        configuration. If None no values will be set.
    - default_value:    The default value to set for the property for nodes
                        where no value is given in the configuration. If None
                        no values will be set. This is only relevant if the
                        property is included in the scenario configuration.
    - set_as_extras:    Whether or not to save the property values in the
                        extras field or set them as a property of the nodes
                        themselves.
    """

    def _extract_value_from_entries(entries: dict|list, step_name: str) -> Any:
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

    def _set_value(step: AttackGraphNode, node_prop: str, value: Any,
        set_as_extras:bool):
        """
        Set the value of the node property to the value provided

        Arguments:
        - step:             The attack graph step to set the property for
        - node_prop:        Property name in string format
        - value:            The value to set the property to
        - set_as_extras:    Whether or not to save the property value in the
                            extras field or set it as a property of the node
                            itself.

        """
        if set_as_extras:
            step.extras[node_prop] = value
        else:
            setattr(step, node_prop, value)


    _validate_scenario_node_property_config(attack_graph, prop_config)

    if not prop_config:
        # If the property is not present in the configuration at all apply the
        # default to all nodes if provided.
        if assumed_value is not None:
            for step in attack_graph.nodes.values():
                _set_value(step, node_prop, assumed_value, set_as_extras)
        return
    else:
        if default_value is not None:
            for step in attack_graph.nodes.values():
                _set_value(step, node_prop, default_value, set_as_extras)

    for step in attack_graph.nodes.values():
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
            _set_value(step, node_prop, prop_value_from_asset_type,
                set_as_extras)

        # Specific asset defined values override asset type values
        if prop_value_from_specific_asset:
            _set_value(step, node_prop, prop_value_from_specific_asset,
                set_as_extras)


def create_scenario_attacker(
    attack_graph: AttackGraph, attacker_name: str, entry_point_names: list[str]
) -> Attacker:
    """Apply attacker entrypoints to attackgraph from scenario

    Creater attacker, add entrypoints to it and compromise them.

    Args:
    - attack_graph: the attack graph to apply entry points to
    - attacker_name: the name to give the attacker
    - entry_points: the entry points to apply for the attacker

    Returns:
    - the Attacker with the relevant entrypoints
    """

    entry_points = []
    reached_attack_steps = []

    for entry_point_name in entry_point_names:
        entry_point = attack_graph.get_node_by_full_name(entry_point_name)
        if not entry_point:
            raise LookupError(f'Node {entry_point_name} does not exist')
        entry_points.append(entry_point)
        reached_attack_steps.append(entry_point)

    attacker = Attacker(
        name=attacker_name,
        entry_points=set(entry_points),
        reached_attack_steps=set(reached_attack_steps),
    )
    attack_graph.add_attacker(attacker)

    # Compromise the entry points
    for entry_point in entry_points:
        attacker.compromise(entry_point)

    return attacker


def load_simulator_agents(
        attack_graph: AttackGraph, scenario: dict
    ) -> list[dict[str, Any]]:
    """Load agents to be registered in MALSimulator

    Create the agents from the specified classes,
    register entrypoints for attackers.

    Args:
    - attack_graph: the attack graph
    - scenario: the scenario in question as a dict
    Return:
    - agents: a dict containing agents and their settings
    """

    # Create list of agents dicts
    agents = []
    scenario_agents = scenario.get('agents', {})

    # Override attackers in attack graph / model if
    # attacker entry points are defined in scenario
    all_attacker_entry_points = [
        entry_point
        for agent_info in scenario_agents.values()
        for entry_point in agent_info.get('entry_points', [])
        if AgentType(agent_info.get('type')) == AgentType.ATTACKER
    ]
    if len(all_attacker_entry_points) > 0:
        all_attackers = list(attack_graph.attackers.values())
        for attacker in all_attackers:
            attack_graph.remove_attacker(attacker)

    for agent_name, agent_info in scenario.get('agents', {}).items():
        class_name = agent_info.get('agent_class')
        agent_type = AgentType(agent_info.get('type'))
        agent_dict = {'name': agent_name, 'type': agent_type}
        agent_config = agent_info.get('config', {})

        if agent_type == AgentType.ATTACKER:
            # Attacker has entrypoints
            entry_points = agent_info.get('entry_points')
            attacker = create_scenario_attacker(attack_graph, agent_name, entry_points)
            agent_dict['attacker_id'] = attacker.id

        if class_name is None:
            # No class name - no agent object created
            agents.append(agent_dict)
            continue

        if class_name not in agent_class_name_to_class:
            # Illegal class agent
            raise LookupError(
                f"Agent class '{class_name}' not supported"
            )

        # Initialize the agent object
        agent_class = agent_class_name_to_class[class_name]
        agent = agent_class(agent_config)
        agent_dict['agent_class'] = agent_class
        agent_dict['agent'] = agent
        agents.append(agent_dict)

    return agents


def apply_scenario_to_attack_graph(
        attack_graph: AttackGraph, scenario: dict):
    """Update attack graph according to scenario configuration

    Apply scenario configurations from a loaded scenario file
    to an attack graph

    Arguments:
    - attack_graph: The attack graph to apply scenario to
    - scenario: The scenario file loaded into a dict
    """

    # Validate that all necessary keys are in there
    validate_scenario(scenario)

    # Apply properties to attack graph nodes
    apply_scenario_node_property(
        attack_graph,
        'observable',
        scenario.get('observable_steps', {}),
        assumed_value = 1,
        default_value = 0
    )
    apply_scenario_node_property(
        attack_graph,
        'actionable',
        scenario.get('actionable_steps', {}),
        assumed_value = 1,
        default_value = 0
    )
    apply_scenario_node_property(
        attack_graph,
        'false_positive_rate',
        scenario.get('false_positive_rates', {})
    )
    apply_scenario_node_property(
        attack_graph,
        'false_negative_rate',
        scenario.get('false_negative_rates', {})
    )
    apply_scenario_node_property(
        attack_graph,
        'reward',
        scenario.get('rewards', {})
    )


def load_scenario(scenario_file: str) -> tuple[AttackGraph, list[dict[str, Any]]]:
    """Load a scenario from a scenario file to an AttackGraph"""

    with open(scenario_file, 'r', encoding='utf-8') as s_file:
        scenario = yaml.safe_load(s_file)

        lang_file = path_relative_to_file_dir(scenario['lang_file'], s_file)
        model_file = path_relative_to_file_dir(scenario['model_file'], s_file)

        # Create the attack graph from model + lang and apply scenario
        attack_graph = create_attack_graph(lang_file, model_file)
        apply_scenario_to_attack_graph(attack_graph, scenario)

        # Load the scenario configuration
        scenario_agents = load_simulator_agents(attack_graph, scenario)

        return attack_graph, scenario_agents


def create_simulator_from_scenario(
        scenario_file: str,
        sim_class=MalSimulator,
        **kwargs,
    ) -> tuple[MalSimulator, list[dict[str, Any]]]:
    """Creates and returns a MalSimulator created according to scenario file

    A wrapper that loads the graph and configuration from the scenario file
    and returns a MalSimulator object with registered agents according to the
    configuration.

    Args:
    - scenario_file: the file name of the scenario

    Returns:
    - sim: the resulting simulator
    - agents: the agent infos as a list of dicts
    """

    attack_graph, scenario_agents = load_scenario(scenario_file)

    sim = sim_class(attack_graph, **kwargs)

    # Register agents in simulator
    for agent_dict in scenario_agents:
        if agent_dict['type'] == AgentType.ATTACKER:
            sim.register_attacker(
                agent_dict['name'],
                agent_dict['attacker_id']
            )
        elif agent_dict['type'] == AgentType.DEFENDER:
            sim.register_defender(
                agent_dict['name']
            )

    return sim, scenario_agents
