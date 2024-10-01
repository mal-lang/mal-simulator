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
from typing import Optional

import yaml

from maltoolbox.attackgraph import AttackGraph, Attacker
from maltoolbox.wrappers import create_attack_graph

from .agents.searchers import BreadthFirstAttacker, DepthFirstAttacker
from .agents.keyboard_input import KeyboardAgent
from .sims.mal_simulator import MalSimulator

agent_class_name_to_class = {
    'BreadthFirstAttacker': BreadthFirstAttacker,
    'DepthFirstAttacker': DepthFirstAttacker,
    'KeyboardAgent': KeyboardAgent
}

# All required fields in scenario yml file
required_fields = [
    'lang_file',
    'model_file',
    'attacker_agent_class',
    'defender_agent_class',
]

# All allowed fields in scenario yml fild
allowed_fields = required_fields + [
    'rewards',
    'attacker_entry_points',
    'observable_attack_steps'
]


def validate_scenario(scenario_dict):
    """Verify scenario file keys"""

    # Verify that all keys in dict are supported
    for key in scenario_dict.keys():
        if key not in allowed_fields:
            raise SyntaxError(f"The setting '{key}' is not supported")

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


def _validate_scenario_observability_rules(graph: AttackGraph, rules: dict):
    """Verify that observability rules in a scenario contains only valid
    assets, asset types and attack steps"""

    # a way to lookup attack steps for asset types
    asset_type_step_names = {
        asset_type.name: [a.name for a in asset_type.attack_steps]
        for asset_type in graph.lang_graph.assets
    }

    if rules is None:
        # Rules are allowed to be empty
        return

    assert 'by_asset_type' in rules or 'by_asset_name' in rules, (
        "Observability rules in scenario file must contain either"
        "'by_asset_type' or 'by_asset_name' as keys"
    )

    for asset_type in rules.get('by_asset_type', []):
        # Make sure each specified asset type exists
        assert asset_type in asset_type_step_names.keys(), (
            f"Failed to find asset type '{asset_type}' in language "
            "when applying scenario observability rules")

        for step_name in rules['by_asset_type'][asset_type]:
            # Make sure each specified attack step name
            # exists for the specified asset type
            assert step_name in asset_type_step_names[asset_type], (
                f"Attack step '{step_name}' not found for asset type "
                f"'{asset_type}' in language when applying scenario "
                "observability rules"
            )

    for asset_name in rules.get('by_asset_name', []):
        # Make sure each specified asset exist
        assert asset_name in graph.model.asset_names, (
            f"Failed to find asset name '{asset_name}' in model "
            f"'{graph.model.name}' when applying scenario" 
            "observability rules")

        for step_name in rules['by_asset_name'][asset_name]:
            # Make sure each specified attack step name exists
            # for the specified asset
            expected_full_name = f"{asset_name}:{step_name}"
            assert graph.get_node_by_full_name(expected_full_name), (
                f"Attack step '{step_name}' not found for asset "
                f"'{asset_name}' when applying scenario observability rules"
            )


def apply_scenario_observability_rules(
        attack_graph: AttackGraph,
        observability_rules: Optional[dict]
    ):
    """Apply the observability rules from a scenario configuration
    
    If no observability rules are given in the scenarios file,
    make all steps observable
    
    If rules are given, make all specified steps observable,
    and all other steps non-observable
    
    Arguments:
    - attack_graph: The attack graph to apply the settings to
    - observability_rules: settings from scenario file
    """

    _validate_scenario_observability_rules(attack_graph, observability_rules)

    if not observability_rules:
        # If no observability rules are given,
        # make all nodes in attagraph as observable
        for step in attack_graph.nodes:
            step.extras['observable'] = 1
    else:
        # If observability rules are given
        # make the matching attack steps observable,
        # and all other unobservable
        for step in attack_graph.nodes:
            observable_attack_steps = (
                observability_rules.get(
                    'by_asset_type', {}).get(step.asset.type, []) +
                observability_rules.get(
                    'by_asset_name', {}).get(step.asset.name, [])
            )

            if step.name in observable_attack_steps:
                step.extras['observable'] = 1
            else:
                step.extras['observable'] = 0


def apply_scenario_attacker_entrypoints(
        attack_graph: AttackGraph, entry_points: dict
) -> None:
    """Apply attacker entrypoints to attackgraph from scenario

    Go through attacker entry points from scenario file and add
    them to the referenced attacker in the attack graph

    Args:
    - attack_graph: the attack graph to apply entry points to
    - entry_points: the entry points to apply
    """

    for attacker_name, entry_point_names in entry_points.items():
        attacker = Attacker(
            attacker_name, entry_points=[], reached_attack_steps=[]
        )
        attack_graph.add_attacker(attacker)

        for entry_point_name in entry_point_names:
            entry_point = attack_graph.get_node_by_full_name(entry_point_name)
            if not entry_point:
                raise LookupError(f"Node {entry_point_name} does not exist")
            attacker.compromise(entry_point)

        attacker.entry_points = list(attacker.reached_attack_steps)


def load_scenario_simulation_config(scenario: dict) -> dict:
    """Load configurations used in MALSimulator
    Load parts of scenario are used for the MALSimulator

    Args:
    - scenario: the scenario in question as a dict
    Return:
    - config: a dict containing config
    """

    # Create config object which is later returned
    config = {}
    config['agents'] = {}

    # Currently only support one defender and attacker
    attacker_id = "attacker"
    defender_id = "defender"

    if a_class := scenario.get('attacker_agent_class'):
        if a_class not in agent_class_name_to_class:
            raise LookupError(f"Agent class '{a_class}' not supported")
        config['agents'][attacker_id] = {}
        config['agents'][attacker_id]['type'] = 'attacker'
        config['agents'][attacker_id]['agent_class'] = \
            agent_class_name_to_class.get(a_class)

    if d_class := scenario.get('defender_agent_class'):
        if d_class not in agent_class_name_to_class:
            raise LookupError(f"Agent class '{d_class}' not supported")
        config['agents'][defender_id] = {}
        config['agents'][defender_id]['type'] = 'defender'
        config['agents'][defender_id]['agent_class'] = \
            agent_class_name_to_class.get(d_class)
    return config


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

    # Apply attacker entrypoints to attack graph
    entry_points = scenario.get('attacker_entry_points', {})
    if entry_points:
        # Override attackers in attack graph if
        # entry points defined in scenario
        for attacker in attack_graph.attackers:
            attack_graph.remove_attacker(attacker)

        # Apply attacker entry points from scenario
        apply_scenario_attacker_entrypoints(attack_graph, entry_points)

    # Apply observability settings to attack graph
    observability_settings = scenario.get('observable_attack_steps')
    apply_scenario_observability_rules(attack_graph, observability_settings)


def load_scenario(scenario_file: str) -> tuple[AttackGraph, dict]:
    """Load a scenario from a scenario file to an AttackGraph"""

    with open(scenario_file, 'r', encoding='utf-8') as s_file:
        scenario = yaml.safe_load(s_file)

        lang_file = path_relative_to_file_dir(scenario['lang_file'], s_file)
        model_file = path_relative_to_file_dir(scenario['model_file'], s_file)

        # Create the attack graph from model + lang and apply scenario
        attack_graph = create_attack_graph(lang_file, model_file)
        apply_scenario_to_attack_graph(attack_graph, scenario)

        # Load the scenario configuration
        scenario_config = load_scenario_simulation_config(scenario)

        return attack_graph, scenario_config


def create_simulator_from_scenario(
        scenario_file: str, **kwargs
    ) -> tuple[MalSimulator, dict]:
    """Creates and returns a MalSimulator created according to scenario file

    A wrapper that loads the graph and config from the scenario file
    and returns a MalSimulator object with registered agents according
    to the configuration.

    Args:
    - scenario_file: the file name of the scenario

    Returns:
    - MalSimulator: the resulting simulator
    """

    attack_graph, conf = load_scenario(scenario_file)

    sim = MalSimulator(
        attack_graph.lang_graph,
        attack_graph.model,
        attack_graph,
        **kwargs
    )

    for agent_id, agent_info in conf['agents'].items():
        if agent_info['type'] == "attacker":
            sim.register_attacker(agent_id, 0)
        elif agent_info['type'] == "defender":
            sim.register_defender(agent_id)

    return sim, conf
