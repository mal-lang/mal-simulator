from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

@dataclass
class NodePropertyRule:
    """
    Defines a mapping from nodes to values based on:
    - asset_type filters
    - asset_name filters
    """

    by_asset_type: dict[str, Any] = field(default_factory=dict)
    by_asset_name: dict[str, Any] = field(default_factory=dict)
    default: Any = None

    def __post_init__(self) -> None:
        if not self.by_asset_type and not self.by_asset_name:
            raise ValueError('Expected either "by_asset_type" or "by_asset_name"')

    def value(self, node: AttackGraphNode, default: Any = None) -> Any:
        """Get value for `node` based on this node property config"""
        if not node.model_asset:
            return default

        asset_name = node.model_asset.name
        asset_type = node.model_asset.type

        # precedence: asset_name > asset_type > default
        by_asset_name = self.by_asset_name.get(asset_name, {})
        if node.name in by_asset_name:
            if isinstance(by_asset_name, list):
                return True
            return by_asset_name[node.name]
        by_asset_type = self.by_asset_type.get(asset_type, {})
        if node.name in by_asset_type:
            if isinstance(by_asset_type, list):
                return True
            return by_asset_type[node.name]

        return default

    def per_node(self, attack_graph: AttackGraph) -> dict[str, Any]:
        """Return a dict mapping from each step full name to value given by config"""
        per_node_dict = dict()
        for n in attack_graph.nodes.values():
            value = self.value(n)
            if value is not None:
                per_node_dict[n.full_name] = value
        return per_node_dict

    @classmethod
    def _validate_dict(cls, node_property_dict: dict[str, Any]) -> None:
        allowed_fields = {'by_asset_type', 'by_asset_name'}
        present_allowed_fields = allowed_fields & node_property_dict.keys()
        forbidden_fields = node_property_dict.keys() - allowed_fields
        if not present_allowed_fields:
            raise ValueError(
                "Node property dict need at least 'by_asset_type' or 'by_asset_name'"
            )
        if forbidden_fields:
            raise ValueError(f'Node property fields not allowed: {forbidden_fields}')

    @classmethod
    def from_optional_dict(
        cls, node_property_dict: dict[str, dict[str, Any]] | None
    ) -> Optional[NodePropertyRule]:
        if node_property_dict is None:
            return None

        cls._validate_dict(node_property_dict)
        return NodePropertyRule(
            node_property_dict.get('by_asset_type', {}),
            node_property_dict.get('by_asset_name', {}),
        )

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            'by_asset_type': self.by_asset_type,
            'by_asset_name': self.by_asset_name,
        }
