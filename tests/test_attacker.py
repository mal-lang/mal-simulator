from collections.abc import Set

import pytest
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from maltoolbox.model import Model

from malsim.config.agent_settings import AttackerSettings
from malsim.config.sim_settings import AttackSurfaceSettings, MalSimulatorSettings
from malsim.mal_simulator import run_simulation
from malsim.mal_simulator.attack_surface import get_attack_surface
from malsim.mal_simulator.simulator import MalSimulator
from malsim.policies.attackers.searchers import BreadthFirstAttacker
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


def test_attack_surface_coreLang_include_unnecessary() -> None:
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


@pytest.mark.parametrize('skip_compromised', [True, False])
@pytest.mark.parametrize('skip_unnecessary', [True, False])
@pytest.mark.parametrize(
    'scenario_file',
    [
        'tests/testdata/scenarios/bfs_vs_bfs_scenario.yml',
        'tests/testdata/scenarios/traininglang_scenario.yml',
    ],
)
def test_incremental_attack_surface_matches_full_recompute(
    scenario_file: str, skip_unnecessary: bool, skip_compromised: bool
) -> None:
    attack_surface_settings = AttackSurfaceSettings(
        skip_compromised=skip_compromised, skip_unnecessary=skip_unnecessary
    )
    scenario = Scenario.load_from_file(
        scenario_file,
        sim_settings=MalSimulatorSettings(
            seed=1, attack_surface=attack_surface_settings
        ),
    )
    sim = MalSimulator.from_scenario(scenario)
    states = sim.reset()

    attacker_names = [
        name
        for name, state in states.items()
        if state.__class__.__name__ == 'AttackerState'
    ]

    max_steps = 40
    step = 0
    while not sim.done() and step < max_steps:
        actions: dict[str, list[AttackGraphNode]] = {}
        for name, agent_config in sim.agent_settings.items():
            agent = agent_config.agent
            if agent is None:
                continue
            action = agent.get_next_action(states[name])
            if action:
                actions[name] = [action]
        states = sim.step(actions)
        step += 1

        for name in attacker_names:
            if sim.agent_is_terminated(name):
                continue
            state = states[name]
            full_recompute = get_attack_surface(
                attack_surface_settings,
                sim.sim_state,
                state.settings.actionable_steps,
                state.performed_nodes,
            )
            assert frozenset(state.action_surface) == full_recompute, (
                f'step={step} agent={name}: incremental attack surface '
                'diverged from full recompute '
                f'(extra={state.action_surface - full_recompute}, '
                f'missing={full_recompute - state.action_surface})'
            )


def _make_sim(model: Model, entry_points: set[str]) -> MalSimulator:
    return MalSimulator(
        attack_graph=AttackGraph(model.lang_graph, model),
        sim_settings=MalSimulatorSettings(
            seed=42,
            attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
        ),
        agents=(
            AttackerSettings(
                name='Attacker1',
                entry_points=entry_points,
                policy=BreadthFirstAttacker,
            ),
        ),
    )


def _necessity(sim: MalSimulator, node: str) -> bool:
    return sim.sim_state.graph_state.necessity_per_node[sim.get_node(node)]


def _ran(
    sim: MalSimulator, recording: dict[str, list[AttackGraphNode]], node: str
) -> bool:
    return sim.get_node(node) in recording['Attacker1']


def test_coreLang_necessity_pattern_creds_exist_not_reached(model: Model) -> None:
    data = model.add_asset('Data', 'Data1')
    app = model.add_asset('Application', 'App1')
    creds = model.add_asset('Credentials', 'Creds1')

    app.add_associated_assets('containedData', {data})
    data.add_associated_assets('encryptCreds', {creds})

    sim = _make_sim(model, {'App1:fullAccess'})

    assert _necessity(sim, 'Data1:dataEncrypted')
    assert _necessity(sim, 'Data1:accessUnencryptedData')

    rec = run_simulation(sim)

    assert not _ran(sim, rec, 'Data1:accessDecryptedData')
    assert not _ran(sim, rec, 'Data1:read')


def test_coreLang_necessity_pattern_creds_exist_reached(model: Model) -> None:
    data = model.add_asset('Data', 'Data1')
    app = model.add_asset('Application', 'App1')
    creds = model.add_asset('Credentials', 'Creds1')

    app.add_associated_assets('containedData', {data})
    data.add_associated_assets('encryptCreds', {creds})

    sim = _make_sim(model, {'App1:fullAccess', 'Creds1:read'})

    assert _necessity(sim, 'Data1:dataEncrypted')
    assert _necessity(sim, 'Data1:accessUnencryptedData')

    rec = run_simulation(sim)

    assert _ran(sim, rec, 'Data1:accessDecryptedData')
    assert _ran(sim, rec, 'Data1:read')


def test_coreLang_necessity_pattern_creds_do_not_exist(model: Model) -> None:
    data = model.add_asset('Data', 'Data1')
    app = model.add_asset('Application', 'App1')

    app.add_associated_assets('containedData', {data})

    sim = _make_sim(model, {'App1:fullAccess'})

    assert not _necessity(sim, 'Data1:dataEncrypted')
    assert not _necessity(sim, 'Data1:accessUnencryptedData')

    rec = run_simulation(sim)

    assert not _ran(sim, rec, 'Data1:accessDecryptedData')
    assert _ran(sim, rec, 'Data1:read')
