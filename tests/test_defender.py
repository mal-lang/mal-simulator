from malsim.mal_simulator.attack_surface import get_attack_surface
from malsim.mal_simulator.attacker_state import AttackerState
from malsim.mal_simulator.defense_surface import get_defense_surface
from malsim.mal_simulator.simulator import MalSimulator


def test_defense_surface():
    """
    Run defender to block attack surface and check that attack surface is updated accordingly.
    """

    scenario = 'tests/testdata/scenarios/traininglang_scenario.yml'
    sim = MalSimulator.from_scenario(scenario)

    attack_surface = get_attack_surface(
        sim.sim_settings.attack_surface,
        sim.sim_state,
        sim.agent_states['Attacker1'].settings.actionable_steps,
        sim.agent_states['Attacker1'].performed_nodes,
    )

    expected_defense_surface = {
        sim.get_node('Host:0:notPresent'),
        sim.get_node('Host:1:notPresent'),
        sim.get_node('Data:2:notPresent'),
        sim.get_node('User:3:notPresent'),
    }
    defense_surface = get_defense_surface(
        sim.sim_state,
        sim.agent_states['Defender1'].settings.actionable_steps,
    )
    assert defense_surface == expected_defense_surface

    # TODO: should defender action surface shrink when nodes are performed?
    sim.step({'Defender1': [sim.get_node('Host:0:notPresent')]})
    assert defense_surface == expected_defense_surface

    sim.step({'Defender1': [sim.get_node('User:3:notPresent')]})
    assert defense_surface == expected_defense_surface
