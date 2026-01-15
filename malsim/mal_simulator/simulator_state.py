from dataclasses import dataclass

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.graph_state import GraphState


@dataclass
class MalSimulatorState:
    attack_graph: AttackGraph
    settings: MalSimulatorSettings
    graph_state: GraphState
    global_rewards: dict[AttackGraphNode, float]
    global_false_positive_rates: dict[AttackGraphNode, float]
    global_false_negative_rates: dict[AttackGraphNode, float]
    global_actionability: dict[AttackGraphNode, bool]
    global_observability: dict[AttackGraphNode, bool]
