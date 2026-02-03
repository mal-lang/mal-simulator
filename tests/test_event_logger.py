from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.defender_state import MalSimDefenderState
from malsim.mal_simulator.event_logger import LogEntry
from malsim.mal_simulator.simulator import MalSimulator
from malsim.mal_simulator import run_simulation
from malsim.scenario.scenario import Scenario


def test_active_defenses() -> None:
    """Verify that active defenses are correctly applied"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/detector_lang_scenario.yml'
    )
    sim = MalSimulator.from_scenario(
        scenario, sim_settings=MalSimulatorSettings(seed=10)
    )
    run_simulation(sim, sim._agent_settings)
    defender_state = sim.agent_states['Defender']
    assert isinstance(defender_state, MalSimDefenderState)
    assert defender_state.logs == (
        LogEntry(
            timestep=2,
            detector_name='logExploit',
            asset_name='Application:1',
            attack_step_name='exploit',
            context_assets={'Computer': 'Computer:0'},
        ),
        LogEntry(
            timestep=4,
            detector_name='logExploit',
            asset_name='Application:5',
            attack_step_name='exploit',
            context_assets={'Computer': 'Computer:0'},
        ),
    )
