from collections.abc import Set

from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.sim_settings import AttackSurfaceSettings, MalSimulatorSettings
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
    sim: MalSimulator,
    attack_surface: Set[AttackGraphNode],
    performed_nodes: Set[AttackGraphNode],
) -> None:
    def _parent_is_blocking_defense(node: AttackGraphNode) -> bool:
        if node.type == 'and':
            return any(
                sim.node_is_enabled_defense(parent_node) for parent_node in node.parents
            )
        elif node.type == 'or':
            return all(
                sim.node_is_enabled_defense(parent_node) for parent_node in node.parents
            )
        else:
            return False

    def _parent_is_blocking_exists(node: AttackGraphNode) -> bool:
        if node.type == 'and':
            return any(
                parent.existence_status
                if parent.existence_status is not None
                else False
                for parent in node.parents
                if parent.type == 'exist'
            )
        elif node.type == 'or':
            return all(
                parent.existence_status
                if parent.existence_status is not None
                else False
                for parent in node.parents
                if parent.type == 'exist'
            )
        else:
            return False

    def _parent_is_blocking_not_exists(node: AttackGraphNode) -> bool:
        if node.type == 'and':
            return all(
                parent.existence_status
                if parent.existence_status is not None
                else False
                for parent in node.parents
                if parent.type == 'notExist'
            )
        elif node.type == 'or':
            return any(
                parent.existence_status
                if parent.existence_status is not None
                else False
                for parent in node.parents
                if parent.type == 'notExist'
            )
        else:
            return False

    for node in attack_surface:
        assert not _parent_is_blocking_defense(node), (
            f'Attack surface node {node} is blocked by defense, '
            'but is in the attack surface.'
        )
        parent_is_compromised = any(
            sim.node_is_compromised(parent_node) for parent_node in node.parents
        )
        assert parent_is_compromised, (
            f'Attack surface node {node} has a parent that is not compromised,'
            ' which should not be the case.'
        )

    for node in performed_nodes:
        # Go through all nodes that could be in attack surface and find out why not
        assert node not in attack_surface, (
            f'Performed node {node} is in attack surface, which should not be the case.'
        )

        for child in node.children:
            if not (
                child in performed_nodes
                or (_parent_is_blocking_defense(child))
                or (_parent_is_blocking_exists(child))
                or (_parent_is_blocking_not_exists(child))
                or (child in sim.sim_state.graph_state.impossible_attack_steps)
            ):
                assert child in attack_surface, (
                    f'Child node {child} of performed node {node} is not in '
                    'attack surface, but should be. It is not blocked by defense '
                    ' and all parents are compromised.'
                )


def test_attack_surface_coreLang() -> None:
    scenario_file = 'tests/testdata/scenarios/simple_scenario.yml'
    scenario = Scenario.load_from_file(
        scenario_file,
        sim_settings=MalSimulatorSettings(
            seed=42, attack_surface=AttackSurfaceSettings(skip_unnecessary=False)
        ),
    )
    sim = MalSimulator.from_scenario(scenario)

    while True:
        next_choice = next(
            (node for node in sim.agent_states['Attacker1'].action_surface), None
        )
        sim.step({'Attacker1': [next_choice] if next_choice else []})
        _validate_attack_surface(
            sim,
            sim.agent_states['Attacker1'].action_surface,
            sim.agent_states['Attacker1'].performed_nodes,
        )
        if sim.agent_is_terminated('Attacker1'):
            break

    assert sim.agent_states['Attacker1'].iteration == 99
