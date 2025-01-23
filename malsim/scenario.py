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
from typing import Optional, Any

import yaml

from maltoolbox.attackgraph import AttackGraph, Attacker, create_attack_graph

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
    'actionable_steps'
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


def apply_scenario_rewards(
        attack_graph: AttackGraph, rewards: dict[str, float]
    ) -> None:
    """Go through rewards, add them to referenced nodes in the AttackGraph"""

    # Set the rewards according to scenario rewards
    for attack_step_full_name, reward in rewards.items():
        node = attack_graph.get_node_by_full_name(attack_step_full_name)
        if node is None:
            raise LookupError(
                f"Could not set reward to node {attack_step_full_name}"
                " since it was not found in the attack graph"
            )
        node.extras['reward'] = reward


def _validate_scenario_property_rules(
        graph: AttackGraph, rules: dict):
    """Verify that observability/actionability rules in a scenario contains
    only valid assets, asset types and step names"""

    # a way to lookup attack steps for asset types
    asset_type_step_names = {
        asset_type_name: [step_name for step_name in asset_type.attack_steps]
        for asset_type_name, asset_type in graph.lang_graph.assets.items()
    }

    if rules is None:
        # Rules are allowed to be empty
        return

    assert 'by_asset_type' in rules or 'by_asset_name' in rules, (
        "Observability/Actionability rules in scenario file must "
        "contain either 'by_asset_type' or 'by_asset_name' as keys"
    )

    for asset_type in rules.get('by_asset_type', []):
        # Make sure each specified asset type exists
        assert asset_type in asset_type_step_names.keys(), (
            f"Failed to find asset type '{asset_type}' in language "
            "when applying scenario observability/actionability rules")

        for step_name in rules['by_asset_type'][asset_type]:
            # Make sure each specified attack step name
            # exists for the specified asset type
            assert step_name in asset_type_step_names[asset_type], (
                f"Step '{step_name}' not found for asset type "
                f"'{asset_type}' in language when applying scenario "
                "observability/actionability rules"
            )

    # TODO: revisit this variable once LookupDicts are merged
    asset_names = set(a.name for a in graph.model.assets.values())
    for asset_name in rules.get('by_asset_name', []):
        # Make sure each specified asset exist
        assert asset_name in asset_names, (
            f"Failed to find asset name '{asset_name}' in model "
            f"'{graph.model.name}' when applying scenario" 
            "observability/actionability rules")

        for step_name in rules['by_asset_name'][asset_name]:
            # Make sure each specified attack step name exists
            # for the specified asset
            expected_full_name = f"{asset_name}:{step_name}"
            assert graph.get_node_by_full_name(expected_full_name), (
                f"Attack step '{step_name}' not found for asset "
                f"'{asset_name}' when applying scenario"
                "observability/actionability rules"
            )


def apply_scenario_node_property_rules(
        attack_graph: AttackGraph,
        node_prop: str,
        rules: Optional[dict]
    ):
    """Apply the observability/actionability rules from a scenario
    configuration.

    If no rules are given in the scenarios file, make all steps
    observable/actionable

    If rules are given, make all specified steps observable/actionable,
    and all other steps unobservable/unactinable

    Arguments:
    - attack_graph: The attack graph to apply the settings to
    - node_prop: property name in string format('actionable' or 'observable')
    - rules: settings from scenario file with keys `by_asset_name`
             and/or `by_asset_type`
    """

    _validate_scenario_property_rules(attack_graph, rules)

    if not rules:
        # If no rules are given, make all steps as observable/actionable
        for step in attack_graph.nodes.values():
            step.extras[node_prop] = 1
    else:
        # If observability/actionability rules are given
        # make the matching steps observable/actionable,
        # and all other unobservable/unactionable
        for step in attack_graph.nodes.values():
            node_prop_rule_step_names = (
                rules.get('by_asset_type', {}).get(step.lg_attack_step.asset.name, []) +
                rules.get('by_asset_name', {}).get(step.model_asset.name, [])
            )

            if step.name in node_prop_rule_step_names:
                step.extras[node_prop] = 1
            else:
                step.extras[node_prop] = 0


def add_attacker_entrypoints(
        attack_graph: AttackGraph, attacker_name: str, entry_points: dict
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

    if entry_points:
        # Override attackers in attack graph / model if
        # entry points are defined in scenario
        all_attackers = list(attack_graph.attackers.values())
        for attacker in all_attackers:
            attack_graph.remove_attacker(attacker)

    attacker = Attacker(attacker_name)
    attack_graph.add_attacker(attacker)

    for entry_point_name in entry_points:
        entry_point = attack_graph.get_node_by_full_name(entry_point_name)
        if not entry_point:
            raise LookupError(f"Node {entry_point_name} does not exist")
        attacker.compromise(entry_point)

    attacker.entry_points = attacker.reached_attack_steps.copy()

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

    for agent_name, agent_info in scenario.get('agents', {}).items():
        class_name = agent_info.get('agent_class')
        agent_type = AgentType(agent_info.get('type'))
        agent_dict = {'name': agent_name, 'type': agent_type}
        agent_config = agent_info.get('config', {})

        if agent_type == AgentType.ATTACKER:
            # Attacker has entrypoints
            entry_points = agent_info.get('entry_points')
            attacker = add_attacker_entrypoints(
                attack_graph, agent_name, entry_points
            )
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
        attack_graph: AttackGraph, scenario: dict) -> AttackGraph:
    """Update attack graph according to scenario

    Apply scenario configurations from a loaded scenario file
    to an attack graph and return the attack graph + config dict

    Arguments:
    - attack_graph: The attack graph to apply scenario to
    - scenario: The scenario file loaded into a dict
    """

    # Validate that all necessary keys are in there
    validate_scenario(scenario)

    # Apply rewards to attack graph
    rewards = scenario.get('rewards', {})
    apply_scenario_rewards(attack_graph, rewards)

    # Apply observability and actionability settings to attack graph
    for node_prop in ['observable', 'actionable']:
        node_prop_settings = scenario.get(node_prop + '_steps')
        apply_scenario_node_property_rules(
            attack_graph, node_prop, node_prop_settings)


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

    A wrapper that loads the graph and config from the scenario file
    and returns a MalSimulator object with registered agents according
    to the configuration.

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
