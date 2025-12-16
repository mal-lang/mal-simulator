"""
Script that takes a scenario file and converts the rewards from the old format
(used until Mal Simulator v0.2.6) to the new format.
"""

import json
from typing import Any
from maltoolbox.model import Model
from malsim.mal_simulator import MalSimulator, run_simulation
from malsim.scenario import Scenario, path_relative_to_file_dir
import yaml

    
def _get_model_attackers(model_file: str) -> dict:
    attackers = {}
    with open(model_file, 'r', encoding='utf-8') as file:
        try:
            model_dict = yaml.safe_load(file)
        except:
            model_dict = json.load(file)
        model_attackers = model_dict.get('attackers', {})

        for _, attacker_info in model_attackers.items():
            attacker_name = attacker_info['name']
            entry_points = []
            for ep_asset, ep_info in attacker_info['entry_points'].items():
                for ep_attack_step in ep_info['attack_steps']:
                    entry_points.append(ep_asset + ':' + ep_attack_step)
            attackers[attacker_name] = {
                'type': 'attacker',
                'name': attacker_name,
                'entry_points': entry_points,
            }
    return attackers

def _convert_scenario_attackers(scenario_file: str) -> dict:
    """
    Convert scenario rewards from the old format to the new format.
    The old format is a dictionary with attack step full names as keys and
    rewards as values.
    The new format is a dictionary with either (or both) of the following:
    - 'by_asset_type': a dictionary with asset types as keys
    - 'by_asset_name': a dictionary with asset names as keys
    """
    with open(scenario_file, 'r', encoding='utf-8') as file:
        scenario_dict = yaml.safe_load(file)

        if 'agents' in scenario_dict:
            msg = "Seems to have already been converted, agents already defined in scenario file"
            print(msg)
            raise RuntimeError(msg)

        if 'model_file' not in scenario_dict:
            msg = "Model file not referenced in scenario"
            print(msg)
            raise RuntimeError(msg)

        model_path = path_relative_to_file_dir(scenario_dict['model_file'], file)
        agents = _get_model_attackers(model_path)

        attacker_class = None
        defender_class = None
        if 'attacker_agent_class' in scenario_dict:
            attacker_class = scenario_dict['attacker_agent_class']
            del scenario_dict['attacker_agent_class']
        if 'defender_agent_class' in scenario_dict:
            defender_class = scenario_dict['defender_agent_class']
            del scenario_dict['defender_agent_class']
        if 'attacker_entry_points' in scenario_dict:
            for attacker, eps in scenario_dict['attacker_entry_points'].items():
                agents[attacker] = {
                    'type': 'attacker',
                    'entry_points': eps
                }
            del scenario_dict['attacker_entry_points']

        for attacker_name in agents:
            agents[attacker_name]['agent_class'] = attacker_class

        if defender_class:
            agents['Defender'] = {
                'type': 'defender',
                'agent_class': defender_class
            }

        # Convert the rewards to the new format
        scenario_dict['agents'] = agents
    return scenario_dict


def convert_scenario_attackers(scenario_file: str) -> str:
    """
    Convert scenario rewards to the new format.
    This function is a wrapper that calls the conversion function.
    Args:

    """
    new_scenario = _convert_scenario_attackers(scenario_file)
    with open(scenario_file, 'w', encoding='utf-8') as out_file:
        yaml.safe_dump(new_scenario, out_file, sort_keys=False)
    return scenario_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert model/scenario agents to the new format.'
    )
    parser.add_argument(
        'scenario_file', type=str, help='Path to the scenario file to convert'
    )
    args = parser.parse_args()
    o_file = convert_scenario_attackers(args.scenario_file)

    scenario = Scenario.load_from_file(o_file)
    sim = MalSimulator.from_scenario(scenario)
    run_simulation(sim, scenario.agents)

    print(f'Created new scenario `{o_file}` with new format for agents.')
