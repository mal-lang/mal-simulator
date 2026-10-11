"""Per-node false negative/positive rate lookups.

False alert generation itself runs natively (`malsim._native`); these
helpers resolve the rate rules for the settings flattening and the
`MalSimulator` query API.
"""

from __future__ import annotations

from maltoolbox.attackgraph import AttackGraphNode
from malsim.config.node_property_rule import NodePropertyRule


def node_false_negative_rate(
    node: AttackGraphNode,
    false_negative_rates_rule: NodePropertyRule[float] | None = None,
) -> float:
    if false_negative_rates_rule:
        return false_negative_rates_rule.value(node, 0.0)
    return 0.0


def node_false_positive_rate(
    node: AttackGraphNode,
    false_positive_rates_rule: NodePropertyRule[float] | None = None,
) -> float:
    if false_positive_rates_rule:
        # FPR from agent settings
        return float(false_positive_rates_rule.value(node, 0.0))
    return 0.0
