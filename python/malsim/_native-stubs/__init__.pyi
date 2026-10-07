"""Type stubs for the compiled PyO3 extension module ``malsim._native``.

This is a PEP 561 stub-only package: there is no real ``malsim/_native/``
package on disk at runtime (the real thing is the compiled
``malsim/_native.*.so`` extension module, built via maturin from
``py-bindings/malsim-pyo3``). These ``.pyi`` files exist purely to give
type checkers (mypy) and editors full type/autocomplete information for
that compiled module's API.

See ``PORTING_NOTES.md`` §5/A1 for what's actually implemented so far -
`node_count` is a Phase A1 smoke-test function only, not a stable API.
"""

from maltoolbox.attackgraph import AttackGraph

def node_count(graph: AttackGraph) -> int: ...
