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

from maltoolbox.attackgraph import AttackGraph
from maltoolbox.wrappers import create_attack_graph

from .agents.searchers import BreadthFirstAttacker, DepthFirstAttacker
from .agents.keyboard_input import KeyboardAgent

agent_class_name_to_class = {
    'BreadthFirstAttacker': BreadthFirstAttacker,
    'DepthFirstAttacker': DepthFirstAttacker,
    'KeyboardAgent': KeyboardAgent
}

def path_relative_to_file_dir(rel_path, file):
    """Returns the absolute path of a relative path in  a second file

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
    # Set the attacker entrypoint according to scenario rewards
    for attacker_id, entry_point_names in entry_points.items():
        attacker = attack_graph.get_attacker_by_id(attacker_id)

        if not attacker:
            raise LookupError(f"Attacker {attacker_id} does not exist")

        for entry_point_name in entry_point_names:
            entry_point = attack_graph.get_node_by_full_name(entry_point_name)

            if not entry_point:
                raise LookupError(f"Node {entry_point_name} does not exist")

            if entry_point not in attacker.entry_points:
                attacker.entry_points.append(entry_point)


def load_scenario(scenario_file: str) -> AttackGraph:
    """Load a scenario from a scenario file to an AttackGraph"""

    with open(scenario_file, 'r', encoding='utf-8') as s_file:
        scenario = yaml.safe_load(s_file)
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
        rewards = scenario.get('rewards', [])
        apply_scenario_rewards(attack_graph, rewards)
        entry_points = scenario.get('attacker_entry_points', [])
        apply_scenario_attacker_entrypoints(attack_graph, entry_points)

        # Create config object which is also returned
        config = {}
        if a_class := scenario.get('attacker_class'):
            assert a_class in agent_class_name_to_class
            config['attacker_class'] = agent_class_name_to_class.get(a_class)
        if d_class := scenario.get('defender_class'):
            assert d_class in agent_class_name_to_class
            config['defender_class'] = agent_class_name_to_class.get(d_class)

        return attack_graph, config
