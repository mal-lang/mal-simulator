"""Smoke test for `malsim._native` (Phase A1 - see PORTING_NOTES.md §5/A1).

Proves end to end that a `maltoolbox.AttackGraph` built on the Python side
can be read from inside `malsim._native`, a separately-compiled PyO3
extension module, via the shared `Rc<RefCell<AttackGraph>>` handle that
mal-toolbox's `PyAttackGraph.__inner_capsule__()` hands out as a
`PyCapsule` (see PORTING_NOTES.md §10 for why a direct pyclass downcast
across extension modules doesn't work and a capsule is needed instead).
"""

from .test_scenario import path_relative_to_tests

from malsim.scenario.scenario import Scenario
from malsim import _native


def test_native_node_count_matches_python() -> None:
    scenario = Scenario.load_from_file(
        path_relative_to_tests('./testdata/scenarios/simple_scenario.yml')
    )
    attack_graph = scenario.attack_graph

    assert _native.node_count(attack_graph) == len(attack_graph.nodes)
