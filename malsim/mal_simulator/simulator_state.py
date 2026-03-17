from dataclasses import dataclass

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
import numpy as np

from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.graph_state import GraphState
from malsim.mal_simulator.node_getters import full_name_dict_to_node_dict


@dataclass(frozen=True)
class MalSimulatorState:
    attack_graph: AttackGraph
    settings: MalSimulatorSettings
    graph_state: GraphState
    global_rewards: dict[AttackGraphNode, float]
    global_false_positive_rates: dict[AttackGraphNode, float]
    global_false_negative_rates: dict[AttackGraphNode, float]
    global_actionability: dict[AttackGraphNode, bool]
    global_observability: dict[AttackGraphNode, bool]


def create_simulator_state(
    attack_graph: AttackGraph,
    graph_state: GraphState,
    sim_settings: MalSimulatorSettings,
    rng: np.random.Generator,
    rewards: dict[str, float] | dict[AttackGraphNode, float] | None = None,
    false_positive_rates: dict[str, float] | dict[AttackGraphNode, float] | None = None,
    false_negative_rates: dict[str, float] | dict[AttackGraphNode, float] | None = None,
    node_actionabilities: dict[str, bool] | dict[AttackGraphNode, bool] | None = None,
    node_observabilities: dict[str, bool] | dict[AttackGraphNode, bool] | None = None,
) -> MalSimulatorState:
    return MalSimulatorState(
        attack_graph,
        sim_settings,
        graph_state,
        full_name_dict_to_node_dict(attack_graph, rewards or {}),
        full_name_dict_to_node_dict(attack_graph, false_positive_rates or {}),
        full_name_dict_to_node_dict(attack_graph, false_negative_rates or {}),
        full_name_dict_to_node_dict(attack_graph, node_actionabilities or {}),
        full_name_dict_to_node_dict(attack_graph, node_observabilities or {}),
    )
