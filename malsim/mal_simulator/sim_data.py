from maltoolbox.attackgraph import AttackGraphNode


from typing import NamedTuple


class SimData(NamedTuple):
    rewards: dict[AttackGraphNode, float] = {}
    false_positive_rates: dict[AttackGraphNode, float] = {}
    false_negative_rates: dict[AttackGraphNode, float] = {}
    node_actionabilities: dict[AttackGraphNode, bool] = {}
    node_observabilities: dict[AttackGraphNode, bool] = {}