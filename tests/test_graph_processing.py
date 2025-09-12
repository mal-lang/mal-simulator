"""Tests for analyzers"""

from malsim.graph_processing import (
    _propagate_viability_from_node,
    _propagate_necessity_from_node,
    prune_unviable_and_unnecessary_nodes,
    calculate_necessity,
    calculate_viability
)
from maltoolbox.language import LanguageGraph
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from maltoolbox.model import Model

# Tests

def test_viability_viable_nodes(dummy_lang_graph: LanguageGraph) -> None:
    """Make sure expected viable nodes are actually viable"""

    ttc_values = {}
    attack_graph = AttackGraph(dummy_lang_graph)
    dummy_attack_steps = dummy_lang_graph.assets['DummyAsset'].attack_steps

    # Exists + existance -> viable
    exist_attack_step_type = dummy_attack_steps['DummyExistAttackStep']
    exist_node = attack_graph.add_node(exist_attack_step_type)
    exist_node.existence_status = True

    # NotExists + nonexistance -> viable
    not_exist_attack_step_type = dummy_attack_steps['DummyNotExistAttackStep']
    not_exist_node = attack_graph.add_node(not_exist_attack_step_type)
    not_exist_node.existence_status = False

    # defense not enabled -> viable
    defense_step_type = dummy_attack_steps['DummyDefenseAttackStep']
    defense_step_node = attack_graph.add_node(defense_step_type)

    # or-node with viable parent -> viable
    or_attack_step_type = dummy_attack_steps['DummyOrAttackStep']
    or_node = attack_graph.add_node(or_attack_step_type)
    or_node_parent = attack_graph.add_node(or_attack_step_type)
    or_node.parents.add(or_node_parent)

    # and-node with no parents -> viable
    and_attack_step_type = dummy_attack_steps['DummyAndAttackStep']
    and_node = attack_graph.add_node(and_attack_step_type)
    ttc_values[and_node] = 1.0

    # and-node with viable parents -> viable
    and_node2 = attack_graph.add_node(and_attack_step_type)
    and_node_parent1 = attack_graph.add_node(and_attack_step_type)
    and_node_parent2 = attack_graph.add_node(and_attack_step_type)
    and_node2.parents = {and_node_parent1, and_node_parent2}

    # Make sure viable
    enabled_defenses: set[AttackGraphNode] = set()
    viable_nodes = calculate_viability(
        attack_graph, enabled_defenses, set()
    )
    assert exist_node in viable_nodes
    assert not_exist_node in viable_nodes
    assert defense_step_node in viable_nodes
    assert or_node in viable_nodes
    assert and_node in viable_nodes
    assert and_node2 in viable_nodes


def test_viability_unviable_nodes(dummy_lang_graph: LanguageGraph) -> None:
    """Make sure expected unviable nodes are actually unviable"""

    impossible_attack_steps = set()
    attack_graph = AttackGraph(dummy_lang_graph)
    dummy_attack_steps = dummy_lang_graph.assets['DummyAsset'].attack_steps

    # exists, existance_status = False -> not viable
    exist_attack_step_type = dummy_attack_steps['DummyExistAttackStep']
    exist_node = attack_graph.add_node(exist_attack_step_type)
    exist_node.existence_status = False

    # notExists, existence_status = True -> not viable
    not_exist_attack_step_type = dummy_attack_steps['DummyNotExistAttackStep']
    not_exist_node = attack_graph.add_node(not_exist_attack_step_type)
    not_exist_node.existence_status = True

    # Defense status on -> not viable
    defense_step_type = dummy_attack_steps['DummyDefenseAttackStep']
    defense_step_node = attack_graph.add_node(defense_step_type)

    # or-node with no viable parent -> non viable
    or_attack_step_type = dummy_attack_steps['DummyOrAttackStep']
    or_node = attack_graph.add_node(or_attack_step_type)
    unviable_or_node_parent = attack_graph.add_node(or_attack_step_type)
    or_node.parents.add(unviable_or_node_parent)
    unviable_or_node_parent.children.add(or_node)
    impossible_attack_steps.add(unviable_or_node_parent)

    # and-node with two non-viable parents -> non viable
    and_attack_step_type = dummy_attack_steps['DummyAndAttackStep']
    and_node = attack_graph.add_node(and_attack_step_type)

    unviable_and_node_parent1 = attack_graph.add_node(and_attack_step_type)
    unviable_and_node_parent2 = attack_graph.add_node(and_attack_step_type)
    and_node.parents = {unviable_and_node_parent1, unviable_and_node_parent2}
    unviable_and_node_parent1.children.add(and_node)
    unviable_and_node_parent1.children.add(and_node)
    impossible_attack_steps.add(unviable_and_node_parent1)
    impossible_attack_steps.add(unviable_and_node_parent2)

    # Make sure unviable
    enabled_defenses = {defense_step_node}
    viability_per_node = calculate_viability(
        attack_graph, enabled_defenses, impossible_attack_steps
    )

    assert not viability_per_node[unviable_or_node_parent]
    assert not viability_per_node[unviable_and_node_parent1]
    assert not viability_per_node[unviable_and_node_parent2]

    assert not viability_per_node[exist_node]
    assert not viability_per_node[not_exist_node]
    assert not viability_per_node[defense_step_node]
    assert not viability_per_node[or_node]
    assert not viability_per_node[and_node]

# def test_necessity_necessary(dummy_lang_graph: LanguageGraph) -> None:
#     """Make sure expected necessary nodes are necessary"""
# 
#     attack_graph = AttackGraph(dummy_lang_graph)
#     dummy_attack_steps = dummy_lang_graph.assets['DummyAsset'].attack_steps
# 
#     # exists node, existance_status = False -> necessary
#     exist_attack_step_type = dummy_attack_steps['DummyExistAttackStep']
#     exist_node = attack_graph.add_node(exist_attack_step_type)
#     exist_node.existence_status = False
# 
#     # notExists, existance_status = True -> necessary
#     not_exist_attack_step_type = dummy_attack_steps['DummyNotExistAttackStep']
#     not_exist_node = attack_graph.add_node(not_exist_attack_step_type)
#     not_exist_node.existence_status = True
# 
#     # Defense status on -> necessary
#     defense_step_type = dummy_attack_steps['DummyDefenseAttackStep']
#     defense_step_node = attack_graph.add_node(defense_step_type)
#     # defense_step_node.defense_status = True
# 
#     # or-node with necessary parents -> necessary
#     or_attack_step_type = dummy_attack_steps['DummyOrAttackStep']
#     or_node = attack_graph.add_node(or_attack_step_type)
#     or_node_parent = attack_graph.add_node(or_attack_step_type)
#     # or_node_parent.is_necessary = True
#     or_node.parents.add(or_node_parent)
#     or_node_parent.children.add(or_node)
# 
#     # and-node with at least one necessary parents -> necessary
#     and_attack_step_type = dummy_attack_steps['DummyAndAttackStep']
#     and_node = attack_graph.add_node(and_attack_step_type)
# 
#     and_node_parent1 = attack_graph.add_node(and_attack_step_type)
#     # and_node_parent1.is_necessary = True
#     and_node_parent2 = attack_graph.add_node(and_attack_step_type)
#     # and_node_parent2.is_necessary = False
# 
#     and_node.parents = {and_node_parent1, and_node_parent2}
#     and_node.parents = {and_node_parent1, and_node_parent2}
#     and_node_parent1.children = {and_node}
#     and_node_parent2.children = {and_node}
# 
#     enabled_defenses = {defense_step_node}
#     _, necessary_nodes = calculate_viability_and_necessity(
#         attack_graph, enabled_defenses
#     )
# 
#     # Make unnecessary
#     necessary_nodes.remove(or_node_parent)
#     necessary_nodes.remove(and_node_parent2)
# 
#     # Calculate necessety and make sure neccessary
#     assert exist_node in necessary_nodes
#     assert not_exist_node in necessary_nodes
#     assert defense_step_node in necessary_nodes
#     assert or_node in necessary_nodes
#     assert and_node in necessary_nodes


# def test_necessity_unnecessary(dummy_lang_graph):
#     """Make sure expected unnecessary nodes are unnecessary"""
#     pass


def test_analyzers_apriori_prune_unviable_and_unnecessary_nodes(
        model: Model
    ) -> None:

    example_attackgraph = AttackGraph(model.lang_graph, model)

    # Pick out an or node and make it non-necessary
    node_to_make_unnecessary = next(
        node for node in example_attackgraph.nodes.values()
        if node.type == 'or'
    )
    node_to_make_unviable = next(
        node for node in example_attackgraph.nodes.values()
        if node.type == 'and'
    )

    viability_per_node = calculate_viability(example_attackgraph, set(), set())
    necessity_per_node = calculate_necessity(example_attackgraph, set())
    necessity_per_node[node_to_make_unnecessary] = False
    viability_per_node[node_to_make_unviable] = False

    prune_unviable_and_unnecessary_nodes(
        example_attackgraph, viability_per_node, necessity_per_node
    )

    # Make sure the node was pruned
    assert node_to_make_unviable.id not in example_attackgraph.nodes
    assert node_to_make_unnecessary.id not in example_attackgraph.nodes


def test_analyzers_apriori_propagate_viability(dummy_lang_graph: LanguageGraph) -> None:
    r"""Create a graph from nodes
    """

    dummy_or_attack_step = dummy_lang_graph.assets['DummyAsset'].\
        attack_steps['DummyOrAttackStep']
    dummy_and_attack_step = dummy_lang_graph.assets['DummyAsset'].\
        attack_steps['DummyAndAttackStep']
    dummy_defense_attack_step = dummy_lang_graph.assets['DummyAsset'].\
        attack_steps['DummyDefenseAttackStep']
    attack_graph = AttackGraph(dummy_lang_graph)

    # Create a graph of nodes
    vp1 = attack_graph.add_node(
        lg_attack_step = dummy_defense_attack_step
    )
    vp2 = attack_graph.add_node(
        lg_attack_step = dummy_defense_attack_step
    )
    uvp1 = attack_graph.add_node(
        lg_attack_step = dummy_defense_attack_step
    )
    uvp2 = attack_graph.add_node(
        lg_attack_step = dummy_defense_attack_step
    )

    or_1vp = attack_graph.add_node(
        lg_attack_step = dummy_or_attack_step
    )
    or_2uvp = attack_graph.add_node(
        lg_attack_step = dummy_or_attack_step
    )
    and_1uvp = attack_graph.add_node(
        lg_attack_step = dummy_and_attack_step
    )
    and_2vp = attack_graph.add_node(
        lg_attack_step = dummy_and_attack_step
    )

    or_1vp.parents = {vp1, uvp1}
    or_2uvp.parents = {uvp1, uvp2}
    and_1uvp.parents = {vp1, uvp1}
    and_2vp.parents = {vp1, vp2}

    vp1.children = {or_1vp, and_1uvp, and_2vp}
    vp2.children = {and_2vp}
    uvp1.children = {or_1vp, or_2uvp, and_1uvp}
    uvp2.children = {or_2uvp}

    viability_per_node = calculate_viability(attack_graph, set(), set())

    # Make unviable
    viability_per_node[uvp1] = False
    viability_per_node[uvp2] = False

    changed_nodes = set()
    for parent in [vp1, vp2, uvp1, uvp2]:
        changed_nodes |= _propagate_viability_from_node(
            parent, viability_per_node, set()
        )

    assert changed_nodes == {or_2uvp, and_1uvp}

    for node in [vp1, vp2, or_1vp, and_2vp]:
        assert viability_per_node[node]

    for node in [uvp1, uvp2, or_2uvp, and_1uvp]:
        assert not viability_per_node[node]

def test_analyzers_apriori_propagate_necessity(dummy_lang_graph: LanguageGraph) -> None:
    r"""Create a graph from nodes
    """

    dummy_or_attack_step = dummy_lang_graph.assets['DummyAsset'].\
        attack_steps['DummyOrAttackStep']
    dummy_and_attack_step = dummy_lang_graph.assets['DummyAsset'].\
        attack_steps['DummyAndAttackStep']
    attack_graph = AttackGraph(dummy_lang_graph)

    # Create a graph of nodes
    np1 = attack_graph.add_node(
        lg_attack_step = dummy_or_attack_step
    )
    np2 = attack_graph.add_node(
        lg_attack_step = dummy_or_attack_step
    )
    unp1 = attack_graph.add_node(
        lg_attack_step = dummy_or_attack_step
    )
    unp2 = attack_graph.add_node(
        lg_attack_step = dummy_or_attack_step
    )

    or_1unp = attack_graph.add_node(
        lg_attack_step = dummy_or_attack_step
    )
    or_2np = attack_graph.add_node(
        lg_attack_step = dummy_or_attack_step
    )
    and_1np = attack_graph.add_node(
        lg_attack_step = dummy_and_attack_step
    )
    and_2unp = attack_graph.add_node(
        lg_attack_step = dummy_and_attack_step
    )

    or_1unp.parents = {np1, unp1}
    or_2np.parents = {np1, np2}
    and_1np.parents = {np1, unp1}
    and_2unp.parents = {unp1, unp2}

    np1.children = {or_1unp, or_2np, and_1np}
    np2.children = {or_2np}
    unp1.children = {or_1unp, and_1np, and_2unp}
    unp2.children = {and_2unp}

    necessity_per_node = calculate_necessity(attack_graph, set())
    # Make unnecessary
    necessity_per_node[unp1] = False
    necessity_per_node[unp2] = False

    changed_nodes = set()
    for parent in [np1, np2, unp1, unp2]:
        changed_nodes |= (
            _propagate_necessity_from_node(parent, necessity_per_node)
        )
    assert changed_nodes == {or_1unp, and_2unp}

    for node in [np1, np2, or_2np, and_1np]:
        assert necessity_per_node[node]

    for node in [unp1, unp2, or_1unp, and_2unp]:
        assert not necessity_per_node[node]
