"""Type stubs for the compiled PyO3 extension module ``malsim._native``.

This is a PEP 561 stub-only package: there is no real ``malsim/_native/``
package on disk at runtime (the real thing is the compiled
``malsim/_native.*.so`` extension module, built via maturin from
``py-bindings/malsim-pyo3``). These ``.pyi`` files exist purely to give
type checkers (mypy) and editors full type/autocomplete information for
that compiled module's API.

See ``PORTING_NOTES.md`` §5/A1 for what's actually implemented so far -
`node_count` is a Phase A1 smoke-test function only, not a stable API.
`Simulator` (Phase A8, §5/A8) is likewise not a stable/public API: it is
not wired into `MalSimulator` yet (that's A9's job), and
`reset_native`/`step_native`'s dict shapes are deliberately typed as
plain `dict[str, Any]` here rather than a precise `TypedDict`, since the
shape is still expected to grow at A9.
"""

from typing import Any

from maltoolbox.attackgraph import AttackGraph

def node_count(graph: AttackGraph) -> int: ...

class Simulator:
    def __init__(self, graph: AttackGraph) -> None: ...
    def reset_native(
        self,
        settings: dict[str, Any],
        agents: dict[str, dict[str, Any]],
        seed: int,
    ) -> dict[str, Any]: ...
    def step_native(self, actions: dict[str, list[int]]) -> dict[str, Any]: ...
