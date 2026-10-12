"""Build the native (Rust/PyO3) attack surface traversability index.

The attack graph's topology (node kind, parents, existence status) plus
necessity/impossibility are static for the lifetime of one simulation
episode. `AttackGraphIndex` uploads that once into `malsim_native` so that
the hot per-step traversability check in `attack_surface.get_attack_surface`
can run natively instead of walking Python `AttackGraphNode` objects.
"""

from __future__ import annotations
from collections.abc import Mapping, Set

import malsim_native
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

_KIND_CODES = {
    'and': 0,
    'or': 1,
    'exist': 2,
    'notExist': 3,
    'defense': 4,
}


def build_attack_graph_index(
    graph: AttackGraph,
    impossible_attack_steps: Set[AttackGraphNode],
    necessity_per_node: Mapping[AttackGraphNode, bool],
) -> malsim_native.AttackGraphIndex:
    """Build a native traversability index for the given graph/episode state"""

    nodes = list(graph.nodes.values())
    ids = [node.id for node in nodes]
    kinds = [_KIND_CODES.get(node.type, 5) for node in nodes]
    existence_status = [bool(node.existence_status) for node in nodes]
    necessary = [necessity_per_node.get(node, False) for node in nodes]
    impossible = [node in impossible_attack_steps for node in nodes]
    parents = [[parent.id for parent in node.parents] for node in nodes]

    return malsim_native.AttackGraphIndex(
        ids=ids,
        kinds=kinds,
        existence_status=existence_status,
        necessary=necessary,
        impossible=impossible,
        parents=parents,
    )
