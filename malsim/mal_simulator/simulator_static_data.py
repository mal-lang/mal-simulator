from typing import NamedTuple
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim.config.sim_settings import MalSimulatorSettings


class MALSimulatorStaticData(NamedTuple):
    attack_graph: AttackGraph
    sim_settings: MalSimulatorSettings
    rewards: dict[str, float] | dict[AttackGraphNode, float] | None = None
    false_positive_rates: dict[str, float] | dict[AttackGraphNode, float] | None = None
    false_negative_rates: dict[str, float] | dict[AttackGraphNode, float] | None = None
    node_actionabilities: dict[str, bool] | dict[AttackGraphNode, bool] | None = None
    node_observabilities: dict[str, bool] | dict[AttackGraphNode, bool] | None = None
