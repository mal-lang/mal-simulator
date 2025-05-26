"""
Script that takes a scenario file and converts the rewards from the old format
(used until Mal Simulator v0.2.6) to the new format.
"""

import yaml

def _get_new_rewards_config(rewards: dict[str, float]) -> dict:
    # Set the rewards according to new format
    new_rewards = {'by_asset_name': {}}
    for attack_step_full_name, reward in rewards.items():
        # Split the attack step full name into asset name and step name
        parts = attack_step_full_name.split(':')
        asset_name = ":".join(parts[:-1])  # All parts except the last one
        step_name = parts[-1]  # The last part is the step name
        new_rewards['by_asset_name'].setdefault(asset_name, {})
        new_rewards['by_asset_name'][asset_name][step_name] = reward

    return new_rewards


def _convert_scenario_rewards_to_0_3_0(scenario_file: str) -> dict:
    """
    Convert scenario rewards from the old format to the new format.
    The old format is a dictionary with attack step full names as keys and
    rewards as values.
    The new format is a dictionary with either (or both) of the following:
    - 'by_asset_type': a dictionary with asset types as keys
    - 'by_asset_name': a dictionary with asset names as keys
    """
    with open(scenario_file, 'r', encoding='utf-8') as file:
        scenario = yaml.safe_load(file)
        if 'rewards' not in scenario:
            return scenario
        # Convert the rewards to the new format
        scenario['rewards'] = _get_new_rewards_config(
            scenario.get('rewards', {})
        )
        return scenario


def convert_scenario_rewards(scenario_file: str) -> str:
    """
    Convert scenario rewards to the new format.
    This function is a wrapper that calls the conversion function.
    Args:

    """
    new_scenario = _convert_scenario_rewards_to_0_3_0(scenario_file)
    # Make new scenario file be in the same directory as the original
    # but different name (old file extension can be .yml or .yaml)
    new_scenario_file = ""

    if scenario_file.endswith('.yml'):
        new_scenario_file = scenario_file.replace('.yml', '_converted.yml')
    elif scenario_file.endswith('.yaml'):
        new_scenario_file = scenario_file.replace('.yaml', '_converted.yaml')
    else:
        raise ValueError("Scenario file must have a .yaml or .yml extension")

    with open(new_scenario_file, 'w', encoding='utf-8') as out_file:
        yaml.safe_dump(new_scenario, out_file, sort_keys=False)
    return new_scenario_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert scenario rewards to the new format.'
    )
    parser.add_argument(
        'scenario_file', type=str, help='Path to the scenario file to convert'
    )
    args = parser.parse_args()

    o_file = convert_scenario_rewards(args.scenario_file)
    print(f"Created new scenario `{o_file}` with new format for rewards.")
