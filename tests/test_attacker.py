from malsim.mal_simulator.attack_surface import get_attack_surface
from malsim.mal_simulator.attacker_state import AttackerState
from malsim.mal_simulator.defense_surface import get_defense_surface
from malsim.mal_simulator.simulator import MalSimulator


def test_attack_surface():
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
    assert attack_surface == {sim.get_node('User:3:compromise')}

    # This wont help, already compromised
    sim.step({'Defender1': [sim.get_node('Host:0:notPresent')]})
    attack_surface = get_attack_surface(
        sim.sim_settings.attack_surface,
        sim.sim_state,
        sim.agent_states['Attacker1'].settings.actionable_steps,
        sim.agent_states['Attacker1'].performed_nodes,
    )
    assert attack_surface == {sim.get_node('User:3:compromise')}

    # This should block the attack from further propagating
    sim.step({'Defender1': [sim.get_node('User:3:notPresent')]})
    attack_surface = get_attack_surface(
        sim.sim_settings.attack_surface,
        sim.sim_state,
        sim.agent_states['Attacker1'].settings.actionable_steps,
        sim.agent_states['Attacker1'].performed_nodes,
    )
    assert attack_surface == set()
    assert sim.agent_is_terminated('Attacker1')
