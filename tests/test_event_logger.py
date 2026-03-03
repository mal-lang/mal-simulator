from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.defender_state import MalSimDefenderState
from malsim.mal_simulator.event_logger import LogEntry
from malsim.mal_simulator.simulator import MalSimulator
from malsim.mal_simulator import run_simulation
from malsim.scenario.scenario import Scenario


def test_logger_attacks() -> None:
    """Verify that compromised nodes are logged correctly in defender state"""

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
            context_nodes={'computer': sim.get_node('Computer:0:authenticate')},
        ),
        LogEntry(
            timestep=3,
            detector_name='logExploit',
            asset_name='Application:5',
            attack_step_name='exploit',
            context_nodes={},
        ),
    )

def test_logger_attacks_false_negative() -> None:
    """Verify that false negatives can occur"""

    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/detector_lang_scenario.yml'
    )
    sim = MalSimulator.from_scenario(
        scenario, sim_settings=MalSimulatorSettings(seed=10)
    )

    app1_exploit = sim.get_node('Application:1:exploit')
    assert app1_exploit.detectors['logExploit'].tprate, "Detector should have a TPRATE"
    app1_exploit.detectors['logExploit'].tprate['value'] = 0.1

    run_simulation(sim, sim._agent_settings)

    defender_state = sim.agent_states['Defender']
    assert isinstance(defender_state, MalSimDefenderState)
    assert app1_exploit in defender_state.compromised_nodes
    assert defender_state.logs == (
        # No logs for Application 1 since it had too low TPRATE, even though it was exploited
        LogEntry(
            timestep=3,
            detector_name='logExploit',
            asset_name='Application:5',
            attack_step_name='exploit',
            context_nodes={},
        ),
    )