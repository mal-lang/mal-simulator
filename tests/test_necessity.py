from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator import run_simulation
from malsim.mal_simulator.simulator import MalSimulator
from malsim.scenario.scenario import Scenario


def test_read_data_no_creds_skip_unnecessary():
    scenario_file = 'tests/testdata/scenarios/data_without_creds_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim = MalSimulator.from_scenario(
        scenario,
        MalSimulatorSettings(
            seed=100,
            attack_surface_skip_unnecessary=True,
        ),
    )
    run_simulation(sim, sim._agent_settings)
    performed_nodes = {
        n.full_name for n in
        sim.agent_states["Attacker"].performed_nodes
    }
    assert "Data1:successfulRead" in performed_nodes


def test_read_data_no_creds_show_unnecessary():
    scenario_file = 'tests/testdata/scenarios/data_without_creds_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim = MalSimulator.from_scenario(
        scenario,
        MalSimulatorSettings(
            seed=100,
            attack_surface_skip_unnecessary=False,
        ),
    )
    run_simulation(sim, sim._agent_settings)
    performed_nodes = {
        n.full_name for n in
        sim.agent_states["Attacker"].performed_nodes
    }
    assert "Data1:successfulRead" in performed_nodes


def test_read_data_creds_skip_unnecessary():
    scenario_file = 'tests/testdata/scenarios/data_with_creds_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim = MalSimulator.from_scenario(
        scenario,
        MalSimulatorSettings(
            seed=100,
            attack_surface_skip_unnecessary=True,
        ),
    )
    run_simulation(sim, sim._agent_settings)
    performed_nodes = {
        n.full_name for n in
        sim.agent_states["Attacker"].performed_nodes
    }
    assert "Data1:successfulRead" not in performed_nodes


def test_read_data_creds_show_unnecessary():
    scenario_file = 'tests/testdata/scenarios/data_with_creds_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim = MalSimulator.from_scenario(
        scenario,
        MalSimulatorSettings(
            seed=100,
            attack_surface_skip_unnecessary=False,
        ),
    )
    run_simulation(sim, sim._agent_settings)
    performed_nodes = {
        n.full_name for n in
        sim.agent_states["Attacker"].performed_nodes
    }
    assert "Data1:successfulRead" not in performed_nodes
