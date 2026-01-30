
from malsim.mal_simulator.simulator import MalSimulator
from malsim.mal_simulator import run_simulation
from malsim.scenario.scenario import Scenario


def test_active_defenses() -> None:
    """Verify that active defenses are correctly applied"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/detector_lang_scenario.yml'
    )
    sim = MalSimulator.from_scenario(scenario)
    run_simulation(sim, sim._agent_settings)
    breakpoint()
