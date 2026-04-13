from collections.abc import Set
from dataclasses import dataclass

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.graph_state import GraphState


@dataclass(frozen=True)
class MalSimulatorState:
    attack_graph: AttackGraph
    settings: MalSimulatorSettings
    graph_state: GraphState
    enabled_defenses: Set[AttackGraphNode]


def create_simulator_state(
    attack_graph: AttackGraph,
    graph_state: GraphState,
    sim_settings: MalSimulatorSettings,
) -> MalSimulatorState:
    return MalSimulatorState(
        attack_graph,
        sim_settings,
        graph_state,
        enabled_defenses=graph_state.pre_enabled_defenses,
    )


def update_simulator_state(
    sim_state: MalSimulatorState,
    enabled_defenses: Set[AttackGraphNode],
) -> MalSimulatorState:
    return MalSimulatorState(
        sim_state.attack_graph,
        sim_state.settings,
        sim_state.graph_state,
        enabled_defenses=enabled_defenses | sim_state.enabled_defenses,
    )
