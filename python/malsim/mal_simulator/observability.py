from __future__ import annotations

from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.node_property_rule import NodePropertyRule


def node_is_observable(
    agent_observability_rule: NodePropertyRule[bool],
    node: AttackGraphNode,
) -> bool:
    return bool(agent_observability_rule.value(node, False))
