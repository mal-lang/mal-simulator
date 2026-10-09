"""Dataclass storing the graph state used in the simulator.

Computed natively (`malsim._native.Simulator.reset_native`) and mirrored
here - see PORTING_NOTES.md.
"""

from __future__ import annotations
from collections.abc import Set, Mapping

from dataclasses import dataclass
from maltoolbox.attackgraph import AttackGraphNode


@dataclass
class GraphState:
    """Dataclass containing simulator specific graph state"""

    ttc_values: Mapping[AttackGraphNode, float]
    pre_enabled_defenses: Set[AttackGraphNode]
    impossible_attack_steps: Set[AttackGraphNode]
    necessity_per_node: Mapping[AttackGraphNode, bool]
