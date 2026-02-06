from typing import NamedTuple, Optional
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim.config.sim_settings import MalSimulatorSettings


class MALSimulatorStaticData(NamedTuple):
    attack_graph: AttackGraph
    sim_settings: MalSimulatorSettings
    rewards: Optional[dict[str, float] | dict[AttackGraphNode, float]] = None
    false_positive_rates: Optional[dict[str, float] | dict[AttackGraphNode, float]] = (
        None
    )
    false_negative_rates: Optional[dict[str, float] | dict[AttackGraphNode, float]] = (
        None
    )
    node_actionabilities: Optional[dict[str, bool] | dict[AttackGraphNode, bool]] = None
    node_observabilities: Optional[dict[str, bool] | dict[AttackGraphNode, bool]] = None
