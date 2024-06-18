import os

import yaml

from maltoolbox.attackgraph import AttackGraph
from maltoolbox.wrappers import create_attack_graph


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
        rewards = scenario['rewards']
        # entrypoints = scenario.get('entrypoints')

        attack_graph = create_attack_graph(lang_file, model_file)
        for attack_step_id, reward in rewards.items():
            node = attack_graph.get_node_by_id(attack_step_id)
            node.reward = reward

        return attack_graph
