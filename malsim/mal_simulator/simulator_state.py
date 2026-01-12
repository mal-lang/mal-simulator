from dataclasses import dataclass
from .graph_state import GraphState
from maltoolbox.attackgraph import AttackGraph


@dataclass
class MalSimulatorState:
    attack_graph: AttackGraph
    graph_state: GraphState
    global_rewards: dict
    global_false_positive_rates: dict
    global_false_negative_rates: dict
    global_actionability: dict
    global_observability: dict
