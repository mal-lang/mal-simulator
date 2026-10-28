"""Type stubs for the compiled PyO3 extension module ``malsim._native``.

This stub file sits next to the compiled ``malsim/_native.*.so`` extension
module (built via maturin from ``py-bindings/malsim-pyo3``), which is where
type checkers (mypy, pyright/Pylance) look for a compiled module's types.
It exists purely to give them full type/autocomplete information for that
module's API.

See ``PORTING_NOTES.md`` §5/A1 for what's actually implemented so far -
`node_count` is a Phase A1 smoke-test function only, not a stable API.
`model_asset_count` is its Phase B3 equivalent for `maltoolbox.Model`
(§6/B3) - also not a stable API.
`Simulator` (Phase A8/A9, §5) backs `MalSimulator.reset()`/`.step()` as of
A9, but is still not itself a stable/public API (an implementation
detail of `malsim.mal_simulator.simulator`), and
`reset_native`/`step_native`'s dict shapes are deliberately typed as
plain `dict[str, Any]` here rather than a precise `TypedDict`.
`DynaSimulator` (Phase B4/B5, §6/§11) is the `DynaMalSimulator`-backing
equivalent: a separate pyclass, unrelated to `Simulator`, constructed
from the `AttackGraph` and the `Model` it was built from. The model is
snapshotted natively at construction; `restore_model_native` restores
the model/graph to that snapshot, and its `reset_native`/`step_native`
dicts additionally carry the step's model-effect modification record.
The simulators keep all state natively; `agent_states_native`/
`sim_state_native`/`recording_native` translate the current state into
plain dicts on every call, and `Episode` (from `episode_native`) reads an
earlier iteration of an episode, for the lazy `previous_state`
(PORTING_NOTES.md §14).
"""

from typing import Any

from maltoolbox.attackgraph import AttackGraph
from maltoolbox.model import Model

def node_count(graph: AttackGraph) -> int: ...
def model_asset_count(model: Model) -> int: ...
def model_add_asset_native(model: Model, asset_type: str) -> int: ...
def set_detector_rates(
    graph: AttackGraph,
    node_id: int,
    label: str,
    tprate: float | None,
    fprate: float | None,
) -> None: ...

class Simulator:
    def __init__(self, graph: AttackGraph) -> None: ...
    def reset_native(
        self,
        settings: dict[str, Any],
        agents: dict[str, dict[str, Any]],
        seed: int,
    ) -> dict[str, Any]: ...
    def step_native(self, actions: dict[str, list[int]]) -> dict[str, Any]: ...
    def agent_states_native(self) -> dict[str, Any]: ...
    def sim_state_native(self) -> dict[str, Any]: ...
    def recording_native(self) -> dict[int, dict[str, list[int]]]: ...
    def terminated_native(self) -> dict[str, bool]: ...
    def episode_native(self) -> Episode: ...

class DynaSimulator:
    def __init__(self, graph: AttackGraph, model: Model) -> None: ...
    def restore_model_native(self) -> None: ...
    def reset_native(
        self,
        settings: dict[str, Any],
        agents: dict[str, dict[str, Any]],
        seed: int,
    ) -> dict[str, Any]: ...
    def step_native(self, actions: dict[str, list[int]]) -> dict[str, Any]: ...
    def agent_states_native(self) -> dict[str, Any]: ...
    def sim_state_native(self) -> dict[str, Any]: ...
    def recording_native(self) -> dict[int, dict[str, list[int]]]: ...
    def terminated_native(self) -> dict[str, bool]: ...
    def episode_native(self) -> Episode: ...

class Episode:
    def snapshot_native(
        self, iteration: int, agent: str | None = None
    ) -> dict[str, Any]: ...
