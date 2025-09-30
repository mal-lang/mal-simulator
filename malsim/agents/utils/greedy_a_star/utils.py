from typing import Collection, Dict, Iterable, List, Optional
from maltoolbox.attackgraph import AttackGraphNode, AttackGraph
from maltoolbox.model import Model
import networkx as nx

def ttc_map(ttc: Optional[Dict]):
    if not ttc:
        return 0
    elif ttc["name"] == "VeryHardAndUncertain":
        return 50
    elif ttc["name"] == "VeryHardAndCertain":
        return 25
    elif ttc["name"] == "HardAndUncertain":
        return 5
    elif ttc["name"] == "EasyAndCertain":
        return 1
    elif ttc["name"] == "Exponential":
        return 1 / ttc["arguments"][0]
    
def filter_defense(c: Iterable[AttackGraphNode]) -> List[AttackGraphNode]:
    return [node for node in c if node.type != "defense"]


class NoPath(Exception):
    def __init__(self, node: AttackGraphNode, *args: object) -> None:
        super().__init__(*args)
        self.node = node

def merge_paths(
    a: List[AttackGraphNode], b: List[AttackGraphNode], index: int
) -> List[AttackGraphNode]:
    path_extension = [node for node in b if node not in a]
    return a[:index] + path_extension + a[index:]

def to_nx(nodes: AttackGraph | Iterable[AttackGraphNode]):
    if isinstance(nodes, AttackGraph):
        nodes = list(nodes.nodes.values())
    G = nx.DiGraph()

    for node in nodes:
        G.add_node(node.id, **node.to_dict())
        G.nodes[node.id]["full_name"] = node.full_name

    edges = [(node.id, child.id) for node in nodes for child in node.children]
    edges += [(parent.id, node.id) for node in nodes for parent in node.parents]
    G.add_edges_from(edges)

    return G


def model_to_nx(model: Model) -> nx.Graph:
    d = model._to_dict()
    assets = d["assets"]

    G = nx.Graph()

    for id, vals in assets.items():
        G.add_node(id, **vals)

    for id, vals in assets.items():
        neighbour_ids = [id for d in vals["associated_assets"].values() for id in d.keys() ]
        for neighbour_id in neighbour_ids:
            G.add_edge(id, neighbour_id)

    return G