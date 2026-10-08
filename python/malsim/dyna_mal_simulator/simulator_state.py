from collections.abc import Mapping, Set
from dataclasses import dataclass
from typing import Any

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.graph_state import GraphState
from maltoolbox.language.language_graph_model_effect import ModelEffectType
from maltoolbox.model import Model, ModelAsset

from malsim.mal_simulator.simulator_state import MalSimulatorState


@dataclass(frozen=True)
class DetachedAsset:
    """A self-contained snapshot (id/type/name) of a model asset that is
    no longer resolvable via `model.assets` by the time a modification
    record is read - built directly from `malsim._native`'s
    `step_modification_record` output (`AssetRef` on the Rust side),
    never via a `Model` lookup. See `modification_record_from_native`'s
    doc comment (PORTING_NOTES.md §6 Phase B5) for when this happens:
    the native dyna step mutates the shared `Model` via the core Rust
    `Model::remove_asset` directly, which - unlike `maltoolbox`'s
    Python-facing `PyModel.remove_asset` - never records a tombstone, so
    a removed asset's `ModelAsset` Python handle is simply gone, not
    just detached-but-still-readable the way the pure-Python path left
    it.
    """

    id: int
    type: str
    name: str


@dataclass(frozen=True)
class AssetOp:
    """Represents a change to an asset in the model."""

    type: ModelEffectType
    asset: ModelAsset | DetachedAsset


@dataclass(frozen=True)
class AssocOp:
    """Represents a change to the way two assets are associated in the model."""

    type: ModelEffectType
    assoc: tuple[ModelAsset | DetachedAsset, str, ModelAsset | DetachedAsset]


@dataclass(frozen=True)
class DynaMalSimulatorState(MalSimulatorState):
    modification_record: list[AssetOp | AssocOp]


def create_simulator_state(
    attack_graph: AttackGraph,
    graph_state: GraphState,
    sim_settings: MalSimulatorSettings,
) -> DynaMalSimulatorState:
    return DynaMalSimulatorState(
        attack_graph,
        sim_settings,
        graph_state,
        enabled_defenses=graph_state.pre_enabled_defenses,
        modification_record=[],
    )


def update_simulator_state(
    sim_state: DynaMalSimulatorState,
    enabled_defenses: Set[AttackGraphNode],
    model_effects: list[AssetOp | AssocOp],
) -> DynaMalSimulatorState:
    return DynaMalSimulatorState(
        sim_state.attack_graph,
        sim_state.settings,
        sim_state.graph_state,
        enabled_defenses=enabled_defenses | sim_state.enabled_defenses,
        modification_record=sim_state.modification_record + model_effects,
    )


def _resolve_asset_ref(
    model: Model, native_op: Mapping[str, Any], prefix: str
) -> ModelAsset | DetachedAsset:
    """Resolves one `AssetRef`-shaped (`{prefix}_id`/`{prefix}_type`/
    `{prefix}_name`) slice of a native modification-record entry. Prefers
    the live `ModelAsset` handle when the id is still resolvable (keeps
    identity/equality with any other live reference to the same asset -
    e.g. an added asset a caller already holds its own handle to) and
    only falls back to `DetachedAsset`'s self-contained snapshot when
    it's not (a removed asset - see `DetachedAsset`'s doc comment).
    """
    asset_id = native_op[f'{prefix}_id']
    live = model.get_asset_by_id(asset_id)
    if live is not None:
        return live
    return DetachedAsset(
        id=asset_id,
        type=native_op[f'{prefix}_type'],
        name=native_op[f'{prefix}_name'],
    )


def modification_record_from_native(
    model: Model, native_record: list[Mapping[str, Any]]
) -> list[AssetOp | AssocOp]:
    """Resolves `dyna_step_native`'s `step_modification_record` output
    (PORTING_NOTES.md §6 Phase B4/B5 - a list of plain dicts, each
    already carrying every asset's id/type/name snapshot inline, per
    `AssetRef`'s doc comment on the Rust side) into `AssetOp`/`AssocOp`
    objects, for `DynaMalSimulatorState.modification_record`.
    """
    record: list[AssetOp | AssocOp] = []
    for native_op in native_record:
        effect_type = ModelEffectType(native_op['type'])
        if native_op['kind'] == 'asset':
            record.append(
                AssetOp(
                    type=effect_type,
                    asset=_resolve_asset_ref(model, native_op, 'asset'),
                )
            )
        else:
            record.append(
                AssocOp(
                    type=effect_type,
                    assoc=(
                        _resolve_asset_ref(model, native_op, 'left_asset'),
                        native_op['field_name'],
                        _resolve_asset_ref(model, native_op, 'right_asset'),
                    ),
                )
            )
    return record
