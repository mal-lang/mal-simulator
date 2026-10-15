"""The committed golden scenario shapes (PORTING_NOTES.md §7 C6) must match
what the Python `Scenario` produces today. The Rust side checks the same
file in `core/malsim-core/tests/scenario_parity.rs`, so together they
assert the two scenario loaders agree."""

import json

from .scenario_parity import GOLDEN_FILE, generate


def test_golden_scenario_shapes_are_current() -> None:
    with open(GOLDEN_FILE, encoding='utf-8') as f:
        golden = json.load(f)
    current = json.loads(json.dumps(generate()))
    assert current.keys() == golden.keys(), (
        'Fixture set changed - regenerate with `python -m tests.scenario_parity`'
    )
    for fixture, shape in current.items():
        assert shape == golden[fixture], (
            f'{fixture}: golden shape is stale - regenerate with '
            '`python -m tests.scenario_parity`'
        )
