from collections.abc import Set
from dataclasses import dataclass

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.graph_state import GraphState
from maltoolbox.language.language_graph_model_effect import ModelEffectType
from maltoolbox.model import ModelAsset

from malsim.mal_simulator.simulator_state import MalSimulatorState


@dataclass(frozen=True)
class AssetOp:
    """Represents a change to an asset in the model."""

    type: ModelEffectType
    asset: ModelAsset


@dataclass(frozen=True)
class AssocOp:
    """Represents a change to the way two assets are associated in the model."""

    type: ModelEffectType
    assoc: tuple[ModelAsset, str, ModelAsset]


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
