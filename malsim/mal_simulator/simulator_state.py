from dataclasses import dataclass

from maltoolbox.attackgraph import AttackGraph

from malsim.mal_simulator.settings import MalSimulatorSettings
from malsim.mal_simulator.graph_state import GraphState

@dataclass
class MalSimulatorState:
    attack_graph: AttackGraph
    settings: MalSimulatorSettings
    graph_state: GraphState
    global_rewards: dict
    global_false_positive_rates: dict
    global_false_negative_rates: dict
    global_actionability: dict
    global_observability: dict
