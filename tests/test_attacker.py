from collections.abc import Set

from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.attack_surface import get_attack_surface
from malsim.mal_simulator.simulator import MalSimulator
from malsim.scenario.scenario import Scenario


def test_attack_surface_traininglang() -> None:
    """
    Run defender to block attack surface and check
    that attack surface is updated accordingly.
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


def _validate_attack_surface(
    sim: MalSimulator, attack_surface: Set[AttackGraphNode]
) -> None:
    for node in attack_surface:
        one_parent_is_enabled_defense = any(
            sim.node_is_enabled_defense(parent_node) for parent_node in node.parents
        )
        assert not one_parent_is_enabled_defense or node.type == 'or', (
            f'Attack surface node {node} has an enabled defense parent,'
            ' which should not be the case.'
        )

        all_parent_is_enabled_defense = all(
            sim.node_is_enabled_defense(parent_node) for parent_node in node.parents
        )
        assert not all_parent_is_enabled_defense, (
            f'Attack surface node {node} has all enabled defense parents,'
            ' which should not be the case.'
        )

        parent_is_compromised = any(
            sim.node_is_compromised(parent_node) for parent_node in node.parents
        )
        assert parent_is_compromised, (
            f'Attack surface node {node} has a parent that is not compromised,'
            ' which should not be the case.'
        )


def test_attack_surface_coreLang() -> None:
    scenario_file = 'tests/testdata/scenarios/simple_scenario.yml'
    scenario = Scenario.load_from_file(
        scenario_file, sim_settings=MalSimulatorSettings(seed=42)
    )
    sim = MalSimulator.from_scenario(scenario)

    while True:
        attack_surface = get_attack_surface(
            sim.sim_settings.attack_surface,
            sim.sim_state,
            sim.agent_states['Attacker1'].settings.actionable_steps,
            sim.agent_states['Attacker1'].performed_nodes,
        )
        _validate_attack_surface(sim, attack_surface)

        next_choice = next((node for node in attack_surface), None)
        sim.step({'Attacker1': [next_choice] if next_choice else []})

        if sim.agent_is_terminated('Attacker1'):
            break

    assert sim.agent_states['Attacker1'].iteration == 98
