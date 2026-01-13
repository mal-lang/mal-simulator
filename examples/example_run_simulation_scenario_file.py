"""
The easiest way to run a scenario is probably through a scenario file.
"""

from malsim import MalSimulator, run_simulation
from malsim.scenario.scenario import Scenario


def test_example_scenario_file() -> None:
    SCENARIO_FILE = 'tests/testdata/scenarios/traininglang_scenario.yml'
    scenario = Scenario.load_from_file(SCENARIO_FILE)
    mal_simulator = MalSimulator.from_scenario(scenario)
    _ = run_simulation(mal_simulator, scenario.agent_settings)


if __name__ == '__main__':
    test_example_scenario_file()
