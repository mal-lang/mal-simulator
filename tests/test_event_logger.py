from unittest import mock
from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.defender_state import MalSimDefenderState
from malsim.mal_simulator.event_logger import LogEntry
from malsim.mal_simulator.simulator import MalSimulator
from malsim.mal_simulator import run_simulation
from malsim.scenario.scenario import Scenario
from maltoolbox.attackgraph import Detector

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
            trigger=sim.get_node('Application:1:exploit'),
            context={'computer': sim.get_node('Computer:0:authenticate')},
        ),
        LogEntry(
            timestep=3,
            detector_name='logExploit',
            trigger=sim.get_node('Application:5:exploit'),
            context={},
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
    # Set tprate to 0 to guarantee false negative for this step, even though it is exploited
    app1_exploit.detectors['logExploit'] = Detector(
        name=app1_exploit.detectors['logExploit'].name,
        node=app1_exploit.detectors['logExploit'].node,
        potential_context=app1_exploit.detectors['logExploit'].potential_context,
        tprate=0.1
    )

    run_simulation(sim, sim._agent_settings)

    defender_state = sim.agent_states['Defender']
    assert isinstance(defender_state, MalSimDefenderState)
    assert app1_exploit in defender_state.compromised_nodes
    assert defender_state.logs == (
        # No logs for Application 1 since it had too low TPRATE,
        # even though it was exploited
        LogEntry(
            timestep=3,
            detector_name='logExploit',
            trigger=sim.get_node('Application:5:exploit'),
            context={},
        ),
    )
