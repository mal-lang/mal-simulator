from malsim.mal_simulator.defender_state import DefenderState
from malsim.mal_simulator.simulator import MalSimulator
from malsim.mal_simulator import run_simulation
from malsim.scenario.scenario import Scenario
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from malsim import _native

from .conftest import get_node

SCENARIO_FILE = 'tests/testdata/scenarios/detector_lang_scenario.yml'


def _force_detector_rates(
    graph: AttackGraph, node: AttackGraphNode, tprate: float, fprate: float
) -> None:
    """Forces node's 'logExploit' detector's tprate/fprate.

    `node.detectors['logExploit'] = Detector(...)` only mutates a
    Python-side cache `maltoolbox`'s bindings hand out - it's never
    visible to malsim's Rust-native `collect_logs`/`collect_false_positives`
    (PORTING_NOTES.md §5/§10 Phase A9), which read the attack graph's real
    detector data through the shared graph handle. `_native.set_detector_rates`
    mutates that real data directly instead.
    """
    _native.set_detector_rates(graph, node.id, 'logExploit', tprate, fprate)


def test_logger_attacks() -> None:
    """Every compromised node with a detector produces exactly one
    true-positive log, with the expected detector context."""

    scenario = Scenario.load_from_file(SCENARIO_FILE)
    graph = scenario.attack_graph

    # Force certain detection and no false positives, so the logs produced
    # depend only on which nodes get compromised, not on rng draws.
    for name in ('Application:1:exploit', 'Application:5:exploit'):
        _force_detector_rates(graph, get_node(graph, name), tprate=1.0, fprate=0.0)

    sim = MalSimulator.from_scenario(scenario)
    run_simulation(sim)

    defender_state = sim.agent_states['Defender']
    assert isinstance(defender_state, DefenderState)

    app1_exploit = sim.get_node('Application:1:exploit')
    app5_exploit = sim.get_node('Application:5:exploit')
    assert {app1_exploit, app5_exploit} <= defender_state.compromised_nodes

    logs_by_trigger = {log.trigger: log for log in defender_state.logs}
    assert set(logs_by_trigger) == {app1_exploit, app5_exploit}
    assert all(not log.false_positive for log in defender_state.logs)
    assert logs_by_trigger[app1_exploit].context == {
        'computer': sim.get_node('Computer:0:authenticate')
    }
    assert logs_by_trigger[app5_exploit].context == {}


def test_logger_attacks_false_negative() -> None:
    """Verify that false negatives can occur"""

    scenario = Scenario.load_from_file(SCENARIO_FILE)
    graph = scenario.attack_graph

    # Application:1's detector can never trigger (tprate below any rng
    # draw), Application:5's always does -- fprate is forced to 0 on both
    # so the only logs possible are true positives.
    _force_detector_rates(
        graph, get_node(graph, 'Application:1:exploit'), tprate=-1.0, fprate=0.0
    )
    _force_detector_rates(
        graph, get_node(graph, 'Application:5:exploit'), tprate=1.0, fprate=0.0
    )

    sim = MalSimulator.from_scenario(scenario)
    run_simulation(sim)

    defender_state = sim.agent_states['Defender']
    assert isinstance(defender_state, DefenderState)

    app1_exploit = sim.get_node('Application:1:exploit')
    assert app1_exploit in defender_state.compromised_nodes

    triggers = {log.trigger for log in defender_state.logs}
    assert app1_exploit not in triggers
    assert sim.get_node('Application:5:exploit') in triggers


def test_logger_attacks_false_positive() -> None:
    """Verify that false positives can occur"""

    scenario = Scenario.load_from_file(SCENARIO_FILE)
    graph = scenario.attack_graph

    # fprate=1.0 guarantees a false positive every step regardless of the
    # rng draw. Application:5's fprate is silenced so it doesn't also log.
    app1_exploit = get_node(graph, 'Application:1:exploit')
    _force_detector_rates(graph, app1_exploit, tprate=1.0, fprate=1.0)
    _force_detector_rates(
        graph, get_node(graph, 'Application:5:exploit'), tprate=1.0, fprate=0.0
    )

    sim = MalSimulator.from_scenario(scenario)

    for _ in range(5):
        sim.step({})

    defender_state = sim.agent_states['Defender']
    assert isinstance(defender_state, DefenderState)
    assert app1_exploit not in defender_state.compromised_nodes
    assert defender_state.logs
    assert all(log.false_positive for log in defender_state.logs)
    assert all(log.trigger == app1_exploit for log in defender_state.logs)
