"""Test things related to the CLI module"""

import os
from unittest.mock import patch
from typing import Any

from malsim import MalSimulator, run_simulation, load_scenario

def path_relative_to_tests(filename: str) -> str:
    """Returns the absolute path of a file in ./tests

    Arguments:
    filename    - filename to append to tests path
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_dir, f"{filename}")


@patch("builtins.input", return_value="\n") # to not freeze on input()
def test_run_simulation(mock_input: Any) -> None:
    """Make sure we can run simulation with defender agent
    registered in scenario"""

    scenario_file = path_relative_to_tests(
        './testdata/scenarios/bfs_vs_bfs_scenario.yml'
    )

    scenario = load_scenario(scenario_file)
    sim = MalSimulator.from_scenario(scenario)
    run_simulation(sim, scenario.agents)

@patch("builtins.input", return_value="\n") # to not freeze on input()
def test_run_simulation_without_defender_agent(mock_input: Any) -> None:
    """Make sure we can run simulation without defender agent
    registered in scenario"""

    scenario_file = path_relative_to_tests(
        './testdata/scenarios/no_defender_agent_scenario.yml'
    )
    scenario = load_scenario(scenario_file)
    sim = MalSimulator.from_scenario(scenario)
    run_simulation(sim, scenario.agents)
