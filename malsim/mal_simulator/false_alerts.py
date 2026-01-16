"""Functions to generate false negatives/positives in the simulator"""

from __future__ import annotations
from typing import Optional
from collections.abc import Set

import numpy as np
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from malsim.config.node_property_rule import NodePropertyRule


def node_false_negative_rate(
    false_negative_rates_rule: Optional[NodePropertyRule],
    global_false_negative_rates: dict[AttackGraphNode, float],
    node: AttackGraphNode,
) -> float:
    if false_negative_rates_rule:
        # FNR from agent settings
        return float(false_negative_rates_rule.value(node, 0.0))
    if global_false_negative_rates:
        # FNR from global settings
        return global_false_negative_rates.get(node, 0.0)
    return 0.0


def generate_false_negatives(
    false_negative_rate_rule: Optional[NodePropertyRule],
    global_false_negative_rates: dict[AttackGraphNode, float],
    observed_nodes: Set[AttackGraphNode],
    rng: np.random.Generator,
) -> set[AttackGraphNode]:
    """Return a set of false negative attack steps from observed nodes"""
    if false_negative_rate_rule or global_false_negative_rates:
        return {
            node
            for node in observed_nodes
            if rng.random()
            < node_false_negative_rate(
                false_negative_rate_rule, global_false_negative_rates, node
            )
        }
    else:
        return set()


def node_false_positive_rate(
    false_positive_rates_rule: Optional[NodePropertyRule],
    global_false_positive_rates: dict[AttackGraphNode, float],
    node: AttackGraphNode,
) -> float:
    if false_positive_rates_rule:
        # FPR from agent settings
        return float(false_positive_rates_rule.value(node, 0.0))
    if global_false_positive_rates:
        # FPR from global settings
        return global_false_positive_rates.get(node, 0.0)
    return 0.0


def generate_false_positives(
    false_positive_rates_rule: Optional[NodePropertyRule],
    global_false_positive_rates: dict[AttackGraphNode, float],
    attack_graph: AttackGraph,
    rng: np.random.Generator,
) -> set[AttackGraphNode]:
    """Return a set of false positive attack steps from attack graph"""
    if false_positive_rates_rule or global_false_positive_rates:
        return {
            node
            for node in attack_graph.attack_steps
            if rng.random()
            < node_false_positive_rate(
                false_positive_rates_rule, global_false_positive_rates, node
            )
        }
    else:
        return set()
