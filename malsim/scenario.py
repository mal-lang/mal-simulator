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

import yaml

from maltoolbox.attackgraph import AttackGraph, Attacker
from maltoolbox.wrappers import create_attack_graph

from .agents.searchers import BreadthFirstAttacker, DepthFirstAttacker
from .agents.keyboard_input import KeyboardAgent

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
]


def verify_scenario(scenario_dict):
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


def apply_scenario_attacker_entrypoints(
        attack_graph: AttackGraph, entry_points: dict
) -> None:
    """Go through attacker entry points from scenario file and add
    them to the referenced attacker in the attack graph"""

    for attacker_name, entry_point_names in entry_points.items():
        attacker = Attacker(
            attacker_name, entry_points=[], reached_attack_steps=[]
        )

        for entry_point_name in entry_point_names:
            entry_point = attack_graph.get_node_by_full_name(entry_point_name)
            if not entry_point:
                raise LookupError(f"Node {entry_point_name} does not exist")
            attacker.compromise(entry_point)

        attacker.entry_points = attacker.reached_attack_steps
        attack_graph.add_attacker(attacker)


def load_scenario(scenario_file: str) -> tuple[AttackGraph, dict]:
    """Load a scenario from a scenario file to an AttackGraph"""

    with open(scenario_file, 'r', encoding='utf-8') as s_file:
        scenario = yaml.safe_load(s_file)
        verify_scenario(scenario)

        lang_file = path_relative_to_file_dir(
            scenario['lang_file'],
            s_file
        )
        model_file = path_relative_to_file_dir(
            scenario['model_file'],
            s_file
        )

        # Create the attack graph from the model + lang
        attack_graph = create_attack_graph(lang_file, model_file)

        # Apply rewards and entrypoints to attack graph
        rewards = scenario.get('rewards', {})
        apply_scenario_rewards(attack_graph, rewards)
        entry_points = scenario.get('attacker_entry_points', {})
        apply_scenario_attacker_entrypoints(attack_graph, entry_points)

        # Create config object which is also returned
        config = {}
        if a_class := scenario.get('attacker_agent_class'):
            if a_class not in agent_class_name_to_class:
                raise LookupError(f"Agent class '{a_class}' not supported")
            config['attacker_agent_class'] = \
                agent_class_name_to_class.get(a_class)

        if d_class := scenario.get('defender_agent_class'):
            if d_class not in agent_class_name_to_class:
                raise LookupError(f"Agent class '{d_class}' not supported")
            config['defender_agent_class'] = \
                agent_class_name_to_class.get(d_class)

        return attack_graph, config