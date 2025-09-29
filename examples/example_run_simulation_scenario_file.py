"""
The easiest way to run a scenario is probably through a scenario file.
"""
from malsim import MalSimulator, run_simulation
from malsim.scenario import Scenario

SCENARIO_FILE = "tests/testdata/scenarios/traininglang_scenario.yml"
scenario = Scenario.load_from_file(SCENARIO_FILE)
mal_simulator = MalSimulator.from_scenario(scenario)
paths = run_simulation(mal_simulator, scenario.agents)
