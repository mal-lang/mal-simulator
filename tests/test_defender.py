from malsim.mal_simulator.simulator import MalSimulator


def test_defense_surface() -> None:
    """
    Run defender and check that defense surface is updated accordingly.
    """

    scenario = 'tests/testdata/scenarios/traininglang_scenario.yml'
    sim = MalSimulator.from_scenario(scenario)

    expected_defense_surface = {
        sim.get_node('Host:0:notPresent'),
        sim.get_node('Host:1:notPresent'),
        sim.get_node('Data:2:notPresent'),
        sim.get_node('User:3:notPresent'),
    }
    assert sim.agent_states['Defender1'].action_surface == expected_defense_surface

    expected_defense_surface = {
        sim.get_node('Host:1:notPresent'),
        sim.get_node('Data:2:notPresent'),
        sim.get_node('User:3:notPresent'),
    }
    sim.step({'Defender1': [sim.get_node('Host:0:notPresent')]})
    assert sim.agent_states['Defender1'].action_surface == expected_defense_surface
    assert sim.get_node('Host:0:notPresent') in sim.sim_state.enabled_defenses

    expected_defense_surface = {
        sim.get_node('Host:1:notPresent'),
        sim.get_node('Data:2:notPresent'),
    }
    sim.step({'Defender1': [sim.get_node('User:3:notPresent')]})
    assert sim.agent_states['Defender1'].action_surface == expected_defense_surface
    assert sim.get_node('Host:0:notPresent') in sim.sim_state.enabled_defenses
    assert sim.get_node('User:3:notPresent') in sim.sim_state.enabled_defenses
