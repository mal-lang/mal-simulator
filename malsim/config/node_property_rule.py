from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Mapping, Optional, TypeVar

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode


T = TypeVar('T')
X = TypeVar('X')
Y = TypeVar('Y')
DefaultType = TypeVar('DefaultType')


def _lookup(subkey: str, key: str, by_dict: Mapping[str, Any]) -> Optional[Any]:
    sub_dict = by_dict.get(key, {})
    if subkey in sub_dict:
        if isinstance(sub_dict, list):
            return True
        return sub_dict[subkey]
    return None


def optional(f: Callable[[X], Y]) -> Callable[[Optional[X]], Optional[Y]]:
    def wrapper(x: X | None) -> Optional[Y]:
        if x is None:
            return None
        return f(x)

    return wrapper


@dataclass
class NodePropertyRule(Generic[T]):
    """
    Defines a mapping from nodes to values based on:
    - asset_type filters
    - asset_name filters
    """

    by_asset_type: Mapping[str, Mapping[str, T]] | None = None
    by_asset_name: Mapping[str, Mapping[str, T]] | None = None
    default: Any = None

    def __len__(self) -> int:
        return len(self.by_asset_type or {}) + len(self.by_asset_name or {})

    def __getitem__(self, key: AttackGraphNode) -> T:
        x = self.value(key, None)
        if x is None:
            raise KeyError(f'No value found for node {key}')
        return x

    def __post_init__(self) -> None:
        if self.by_asset_type is None and self.by_asset_name is None:
            raise ValueError('Expected either "by_asset_type" or "by_asset_name"')

    def __contains__(self, item: AttackGraphNode) -> bool:
        return self.value(item, None) is not None

    @classmethod
    def from_attack_step_dict(
        cls, attack_step_dict: dict[AttackGraphNode, T]
    ) -> NodePropertyRule[T]:
        by_asset_type = {
            asset_name: {
                node.name: value
                for node, value in attack_step_dict.items()
                if node.model_asset and node.model_asset.type == asset_name
            }
            for asset_name in {
                node.model_asset.type for node in attack_step_dict if node.model_asset
            }
        }
        by_asset_name = {
            asset_name: {
                node.name: value
                for node, value in attack_step_dict.items()
                if node.model_asset and node.model_asset.name == asset_name
            }
            for asset_name in {
                node.model_asset.name for node in attack_step_dict if node.model_asset
            }
        }

        return cls(by_asset_type=by_asset_type, by_asset_name=by_asset_name)

    def value(self, node: AttackGraphNode, default: T | DefaultType) -> T | DefaultType:
        """Get value for `node` based on this node property config"""
        if not node.model_asset:
            return default

        node_name = node.name
        asset_name = node.model_asset.name
        asset_type = node.model_asset.type

        by_asset_val = (
            _lookup(node_name, asset_name, self.by_asset_name)
            if self.by_asset_name
            else None
        )
        asset_type_val = (
            _lookup(node_name, asset_type, self.by_asset_type)
            if self.by_asset_type
            else None
        )

        # precedence: asset_name > asset_type > default
        return by_asset_val or asset_type_val or default

    def per_node(self, attack_graph: AttackGraph) -> dict[str, Any]:
        """Return a dict mapping from each step full name to value given by config"""
        per_node_dict = {}
        for n in attack_graph.nodes.values():
            value = self.value(n, None)
            if value is not None:
                per_node_dict[n.full_name] = value
        return per_node_dict

    @classmethod
    def _validate_dict(
        cls, node_property_dict: Mapping[str, Mapping[str, Mapping[str, T]]]
    ) -> NodePropertyRule[T]:
        allowed_fields = {'by_asset_type', 'by_asset_name'}
        present_allowed_fields = allowed_fields & node_property_dict.keys()
        forbidden_fields = node_property_dict.keys() - allowed_fields
        if not present_allowed_fields:
            raise ValueError(
                "Node property dict need at least 'by_asset_type' or 'by_asset_name'"
            )
        if forbidden_fields:
            raise ValueError(f'Node property fields not allowed: {forbidden_fields}')

        return NodePropertyRule(
            by_asset_type=node_property_dict.get('by_asset_type', None),
            by_asset_name=node_property_dict.get('by_asset_name', None),
        )

    @classmethod
    def from_optional_dict(
        cls, node_property_dict: dict[str, dict[str, Any]] | None
    ) -> Optional[NodePropertyRule[T]]:
        return optional(cls.from_dict)(node_property_dict)

    @classmethod
    def from_dict(
        cls,
        node_property_dict: Mapping[str, Mapping[str, Any]],
    ) -> NodePropertyRule[T]:
        return cls._validate_dict(node_property_dict)

    def to_dict(self) -> Mapping[str, Mapping[str, Mapping[str, T]] | None]:
        return {
            'by_asset_type': self.by_asset_type,
            'by_asset_name': self.by_asset_name,
        }
