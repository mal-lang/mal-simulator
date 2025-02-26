"""Test things related to the CLI module"""

import os
from unittest.mock import patch

from malsim.__main__ import run_simulation
from malsim.scenario import create_simulator_from_scenario
from malsim.mal_simulator import MalSimulator


def path_relative_to_tests(filename):
    """Returns the absolute path of a file in ./tests

    Arguments:
    filename    - filename to append to tests path
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_dir, f"{filename}")


@patch("builtins.input", return_value="\n") # to not freeze on input()
def test_run_simulation(mock_input):
    """Make sure we can run simulation with defender agent
    registered in scenario"""

    scenario_file = path_relative_to_tests(
        './testdata/scenarios/bfs_vs_bfs_scenario.yml'
    )

    sim, agents = create_simulator_from_scenario(
        scenario_file, sim_class=MalSimulator)
    run_simulation(sim, agents)

@patch("builtins.input", return_value="\n") # to not freeze on input()
def test_run_simulation_without_defender_agent(mock_input):
    """Make sure we can run simulation without defender agent
    registered in scenario"""

    scenario_file = path_relative_to_tests(
        './testdata/scenarios/no_defender_agent_scenario.yml'
    )
    sim, agents = create_simulator_from_scenario(
        scenario_file, sim_class=MalSimulator)
    run_simulation(sim, agents)
