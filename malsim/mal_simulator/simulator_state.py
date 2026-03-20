from dataclasses import dataclass


from maltoolbox.attackgraph import AttackGraph


from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.graph_state import GraphState


@dataclass(frozen=True)
class MalSimulatorState:
    attack_graph: AttackGraph
    settings: MalSimulatorSettings
    graph_state: GraphState


def create_simulator_state(
    attack_graph: AttackGraph,
    graph_state: GraphState,
    sim_settings: MalSimulatorSettings,
) -> MalSimulatorState:
    return MalSimulatorState(
        attack_graph,
        sim_settings,
        graph_state,
    )
