from maltoolbox.attackgraph import AttackGraphNode, AttackGraph, Attacker
from maltoolbox.wrappers import create_attack_graph
from maltoolbox.attackgraph.analyzers import apriori
from maltoolbox.attackgraph.query import is_node_traversable_by_attacker

from malsim.sims.mal_simulator import MalSimulator

# Helpers
def add_viable_defense_node(attack_graph):
    """defense node with defense status off -> viable"""
    defense_node = AttackGraphNode(
        'defense', 'DefenseNode', existence_status=True, defense_status=False)
    attack_graph.add_node(defense_node)
    return defense_node


def add_viable_and_node(attack_graph):
    """and-node with viable parents -> viable"""
    and_node = AttackGraphNode('and', 'AndNode', existence_status=True)
    and_node_parent1 = AttackGraphNode('or', 'AndNodeParent1', children=[and_node])
    and_node_parent2 = AttackGraphNode('or', 'AndNodeParent2', children=[and_node])
    and_node.parents = [and_node_parent1, and_node_parent2]
    attack_graph.add_node(and_node)
    attack_graph.add_node(and_node_parent1)
    attack_graph.add_node(and_node_parent2)
    apriori.evaluate_viability(and_node)
    return and_node


def add_viable_or_node(attack_graph):
    """or-node with viable parent -> viable"""
    or_node = AttackGraphNode('or', 'OrNode')
    or_node_parent = AttackGraphNode('or', 'OrNodeParent', children=[or_node])
    or_node.parents.append(or_node_parent)
    attack_graph.add_node(or_node)
    attack_graph.add_node(or_node_parent)
    apriori.evaluate_viability(or_node)
    return or_node


def add_traversable_and_node(attack_graph):
    """Add traversable and node to AG"""
    attack_graph = AttackGraph()
    attacker = Attacker('Attacker1')

    # Viable and-node where all necessary parents are compromised
    viable_and_node = add_viable_and_node(attack_graph)
    for parent in viable_and_node.parents:
        parent.is_necessary = True
        attacker.compromise(parent)
    return viable_and_node


def add_non_viable_and_node(attack_graph):
    """and-node with two non-viable parents -> non viable"""
    and_node = AttackGraphNode(
        'and', 'AndNode', existence_status=True)
    and_node_parent1 = AttackGraphNode(
        'and', 'AndNodeParent1', is_viable=False, children=[and_node])
    and_node_parent2 = AttackGraphNode(
        'and', 'AndNodeParent2', is_viable=False, children=[and_node])
    attack_graph.add_node(and_node)
    attack_graph.add_node(and_node_parent1)
    attack_graph.add_node(and_node_parent2)
    and_node.parents = [and_node_parent1, and_node_parent2]
    apriori.evaluate_viability(and_node)
    return and_node


def add_non_viable_or_node(attack_graph):
    """or-node with no viable parent -> non viable"""
    or_node = AttackGraphNode(
        'or', 'OrNode', existence_status=True)
    or_node_parent = AttackGraphNode(
        'and', 'OrNodeParent', is_viable=False, children=[or_node])
    attack_graph.add_node(or_node)
    attack_graph.add_node(or_node_parent)
    or_node.parents = [or_node_parent]
    apriori.evaluate_viability(or_node)
    return or_node

# Tests

def test_existance():
    pass


def test_viability_viable_nodes():
    """Make sure expected viable nodes are actually viable"""
    
    attack_graph = AttackGraph()

    # Exists + existance -> viable
    exist_node = AttackGraphNode(
        'exist', 'ExistsNode', existence_status=True)
    attack_graph.add_node(exist_node)

    # NotExists + nonexistance -> viable
    not_exist_node = AttackGraphNode(
        'notExist', 'NotExistsNode', existence_status=False)
    attack_graph.add_node(not_exist_node)

    # defense status = false -> viable
    defense_node = AttackGraphNode(
        'defense', 'DefenseNode', existence_status=True, defense_status=False)
    attack_graph.add_node(defense_node)

    # or-node with viable parent -> viable
    or_node = AttackGraphNode('or', 'OrNode')
    or_node_parent = AttackGraphNode('or', 'OrNodeParent', children=[or_node])
    or_node.parents.append(or_node_parent)
    attack_graph.add_node(or_node)
    attack_graph.add_node(or_node_parent)

    # and-node with no parents -> viable
    and_node = AttackGraphNode('and', 'AndNode', existence_status=True)

    # and-node with viable parents -> viable
    and_node2 = AttackGraphNode('and', 'AndNode', existence_status=True)
    and_node_parent1 = AttackGraphNode('or', 'AndNodeParent1', children=[and_node2])
    and_node_parent2 = AttackGraphNode('or', 'AndNodeParent2', children=[and_node2])
    and_node2.parents = [and_node_parent1, and_node_parent2]
    attack_graph.add_node(and_node)
    attack_graph.add_node(and_node_parent1)
    attack_graph.add_node(and_node_parent2)

    # Make sure viable
    apriori.evaluate_viability(exist_node)
    assert exist_node.is_viable
    apriori.evaluate_viability(not_exist_node)
    assert not_exist_node.is_viable
    apriori.evaluate_viability(defense_node)
    assert defense_node.is_viable
    apriori.evaluate_viability(or_node)
    assert or_node.is_viable
    apriori.evaluate_viability(and_node)
    assert and_node.is_viable
    apriori.evaluate_viability(and_node2)
    assert and_node2.is_viable


def test_viability_unviable_nodes():
    """Make sure expected unviable nodes are actually unviable"""

    attack_graph = AttackGraph()

    # not exists -> not viable
    exist_node = AttackGraphNode(
        'exist', 'ExistsNode', existence_status=False)
    attack_graph.add_node(exist_node)

    # exists -> not viable
    not_exist_node = AttackGraphNode(
        'notExist', 'NotExistsNode', existence_status=True)
    attack_graph.add_node(not_exist_node)

    # Defense status on -> not viable
    defense_node = AttackGraphNode(
        'defense', 'DefenseNode', existence_status=True, defense_status=True)
    attack_graph.add_node(defense_node)

    # or-node with no viable parent -> non viable
    or_node = AttackGraphNode(
        'or', 'OrNode', existence_status=True)
    or_node_parent = AttackGraphNode(
        'and', 'OrNodeParent', is_viable=False, children=[or_node])
    attack_graph.add_node(or_node)
    attack_graph.add_node(or_node_parent)
    or_node.parents = [or_node_parent]

    # and-node with two non-viable parents -> non viable
    and_node = AttackGraphNode(
        'and', 'AndNode', existence_status=True)
    and_node_parent1 = AttackGraphNode(
        'and', 'AndNodeParent1', is_viable=False, children=[and_node])
    and_node_parent2 = AttackGraphNode(
        'and', 'AndNodeParent2', is_viable=False, children=[and_node])
    attack_graph.add_node(and_node)
    attack_graph.add_node(and_node_parent1)
    attack_graph.add_node(and_node_parent2)
    and_node.parents = [and_node_parent1, and_node_parent2]

    # Calculate viability and make sure unviable
    apriori.evaluate_viability(exist_node)
    assert not exist_node.is_viable
    apriori.evaluate_viability(not_exist_node)
    assert not not_exist_node.is_viable
    apriori.evaluate_viability(defense_node)
    assert not defense_node.is_viable
    apriori.evaluate_viability(or_node)
    assert not or_node.is_viable
    apriori.evaluate_viability(and_node)
    assert not and_node.is_viable


def test_necessity_necessary():
    """Make sure expected necessary nodes are necessary"""

    attack_graph = AttackGraph()

    # exists, existance_status = False -> necessary
    exist_node = AttackGraphNode(
        'exist', 'ExistsNode', existence_status=False)
    attack_graph.add_node(exist_node)

    # notExists, existance_status = True -> necessary
    not_exist_node = AttackGraphNode(
        'notExist', 'NotExistsNode', existence_status=True)
    attack_graph.add_node(not_exist_node)

    # Defense status on -> necessary
    defense_node = AttackGraphNode(
        'defense', 'DefenseNode', existence_status=True, defense_status=True)
    attack_graph.add_node(defense_node)

    # or-node with necessary parents -> necessary
    or_node = AttackGraphNode(
        'or', 'OrNode', existence_status=True)
    or_node_parent = AttackGraphNode(
        'and', 'OrNodeParent', is_viable=False,
        is_necessary=True, children=[or_node]
    )
    attack_graph.add_node(or_node)
    attack_graph.add_node(or_node_parent)
    or_node.parents = [or_node_parent]

    # and-node with at least one necessary parents -> necessary
    and_node = AttackGraphNode(
        'and', 'AndNode', existence_status=True)
    and_node_parent1 = AttackGraphNode(
        'and', 'AndNodeParent1', is_viable=False,
        is_necessary=True, children=[and_node]
    )
    and_node_parent2 = AttackGraphNode(
        'and', 'AndNodeParent2', is_viable=False,
        is_necessary=False, children=[and_node]
    )
    attack_graph.add_node(and_node)
    attack_graph.add_node(and_node_parent1)
    attack_graph.add_node(and_node_parent2)
    and_node.parents = [and_node_parent1, and_node_parent2]

    # Calculate necessety and make sure neccessary
    apriori.evaluate_necessity(exist_node)
    assert exist_node.is_necessary
    apriori.evaluate_necessity(not_exist_node)
    assert not_exist_node.is_necessary
    apriori.evaluate_necessity(defense_node)
    assert defense_node.is_necessary
    apriori.evaluate_necessity(or_node)
    assert or_node.is_necessary
    apriori.evaluate_necessity(and_node)
    assert and_node.is_necessary



def test_necessity_unnecessary():
    """Make sure expected unnecessary nodes are unnecessary"""

    attack_graph = AttackGraph()

    # exists, existance_status = False -> necessary
    exist_node = AttackGraphNode(
        'exist', 'ExistsNode', existence_status=False)
    attack_graph.add_node(exist_node)

    # notExists, existance_status = True -> necessary
    not_exist_node = AttackGraphNode(
        'notExist', 'NotExistsNode', existence_status=True)
    attack_graph.add_node(not_exist_node)

    # Defense status on -> necessary
    defense_node = AttackGraphNode(
        'defense', 'DefenseNode', existence_status=True, defense_status=True)
    attack_graph.add_node(defense_node)

    # or-node with necessary parents -> necessary
    or_node = AttackGraphNode(
        'or', 'OrNode', existence_status=True)
    or_node_parent = AttackGraphNode(
        'and', 'OrNodeParent', is_viable=False,
        is_necessary=True, children=[or_node]
    )
    attack_graph.add_node(or_node)
    attack_graph.add_node(or_node_parent)
    or_node.parents = [or_node_parent]

    # and-node with at least one necessary parents -> necessary
    and_node = AttackGraphNode(
        'and', 'AndNode', existence_status=True)
    and_node_parent1 = AttackGraphNode(
        'and', 'AndNodeParent1', is_viable=False,
        is_necessary=True, children=[and_node]
    )
    and_node_parent2 = AttackGraphNode(
        'and', 'AndNodeParent2', is_viable=False,
        is_necessary=False, children=[and_node]
    )
    attack_graph.add_node(and_node)
    attack_graph.add_node(and_node_parent1)
    attack_graph.add_node(and_node_parent2)
    and_node.parents = [and_node_parent1, and_node_parent2]

    # Calculate necessety and make sure neccessary
    apriori.evaluate_necessity(exist_node)
    assert exist_node.is_necessary
    apriori.evaluate_necessity(not_exist_node)
    assert not_exist_node.is_necessary
    apriori.evaluate_necessity(defense_node)
    assert defense_node.is_necessary
    apriori.evaluate_necessity(or_node)
    assert or_node.is_necessary
    apriori.evaluate_necessity(and_node)
    assert and_node.is_necessary


def test_traversability_traversable():
    attack_graph = AttackGraph()
    attacker = Attacker('Attacker1')

    # Viable and-node where all necessary parents are compromised
    viable_and_node = add_viable_and_node(attack_graph)
    for parent in viable_and_node.parents:
        parent.is_necessary = True
        attacker.compromise(parent)
    assert is_node_traversable_by_attacker(viable_and_node, attacker)

    # Viable or-node where at least one parent is compromised
    viable_or_node = add_viable_or_node(attack_graph)
    parent = viable_or_node.parents[0]
    parent.is_necessary = True
    attacker.compromise(parent)
    assert is_node_traversable_by_attacker(viable_or_node, attacker)


def test_traversability_not_traversable():
    """Make sure nodes that shouldn't be traversable aren't"""
    attack_graph = AttackGraph()
    attacker = Attacker('Attacker1')

    # Viable and-node where not all necessary parents are compromised
    # -> not traversable
    non_viable_and_node = add_non_viable_and_node(attack_graph)
    assert not is_node_traversable_by_attacker(non_viable_and_node, attacker)

    # Nonviable and-node where all necessary parents are compromised
    # -> not traversabel
    viable_and_node = add_non_viable_and_node(attack_graph)
    for parent in viable_and_node.parents:
        parent.is_necessary = True
        attacker.compromise(parent)
    assert not is_node_traversable_by_attacker(viable_and_node, attacker)

    # Viable or-node where no parent is compromised -> not traversable
    non_viable_or_node = add_non_viable_or_node(attack_graph)
    assert not is_node_traversable_by_attacker(non_viable_or_node, attacker)

    # Nonviable or-node where parent is compromised -> not traversable
    viable_or_node = add_viable_or_node(attack_graph)
    parent = viable_or_node.parents[0]
    parent.is_necessary = True
    attacker.compromise(parent)
    assert not is_node_traversable_by_attacker(viable_or_node, attacker)


def test_state_transitions_attack_step():
    """Show the definitions and verify the observation values"""

    lang_file = "tests/org.mal-lang.coreLang-1.0.0.mar"
    model_file = "tests/example_model.yml"
    attack_graph = create_attack_graph(lang_file, model_file)

    env = MalSimulator(
        attack_graph.lang_graph,
        attack_graph.model,
        attack_graph
    )

    attack_step_full_name = 'OS App:localConnect'
    attacker_entry_point = attack_graph.get_node_by_full_name(
        attack_step_full_name
    )

    assert attacker_entry_point.is_viable
    assert attacker_entry_point.is_necessary

    attacker1 = Attacker(
        "attacker1",
        id=0
    )
    attacker1.compromise(attacker_entry_point)

    attack_node_index = env._index_to_full_name.index(attack_step_full_name)
    env.register_attacker(attacker1.name, attacker1.id)
    env.reset()

    action_dict = {attacker1.name: (1, attack_node_index)}
    observations, rewards, terminations, truncations, infos = env.step(action_dict)

    # Currently all TTCs are 0, this will change
    attacker_obs = observations[attacker1.name]
    for ttc in attacker_obs['remaining_ttc']:
        assert ttc == 0
    assert attacker_obs['observed_state'][attack_node_index] == 1
    for child in attacker_entry_point.children:
        child_node_index = env._index_to_full_name.index(child.full_name)
        if not is_node_traversable_by_attacker(child, attacker1):
            assert attacker_obs['observed_state'][child_node_index] == -1
        else:
            assert attacker_obs['observed_state'][child_node_index] == 0
    assert attacker_entry_point.is_compromised()


def test_state_transitions_defense_step():
    """
    """

    # Attack step will occur if:
    # - IsAttackStep(x)
    # - AttackerSelected(x)
    # - TTC=0
    # - Traversable(x)

    lang_file = "tests/org.mal-lang.coreLang-1.0.0.mar"
    model_file = "tests/example_model.yml"
    attack_graph = create_attack_graph(lang_file, model_file)

    env = MalSimulator(
        attack_graph.lang_graph,
        attack_graph.model,
        attack_graph
    )

    # We need an attacker so the simulation doesn't terminate
    attack_step_full_name = 'OS App:localConnect'
    attacker_entry_point = attack_graph.get_node_by_full_name(
        attack_step_full_name
    )

    attacker1 = Attacker(
        "attacker1",
        id=0
    )
    attacker1.compromise(attacker_entry_point)
    attack_node_index = env._index_to_full_name.index(attack_step_full_name)
    env.register_attacker(attacker1.name, attacker1.id)

    defense_step_full_name = 'OS App:notPresent'
    defenders_entry_point = attack_graph.get_node_by_full_name(
        defense_step_full_name
    )

    assert defenders_entry_point.is_viable

    defender1 = "defender1"
    defense_node_index = env._index_to_full_name.index(defense_step_full_name)
    env.register_defender(defender1)
    assert defender1 in env.possible_agents
    env.reset()
    assert defender1 in env.agents

    action_dict = {
        defender1: (1, defense_node_index),
        attacker1.name: (1, attack_node_index)
    }
    observations, rewards, terminations, truncations, infos = env.step(action_dict)
    # Currently all TTCs are 0, this will change
    defender_obs = observations[defender1]
    for node in attack_graph.nodes:
        if node.type == "defense":
            node_index = env._index_to_full_name.index(node.full_name)
            default_defense_value = node.asset._properties[node.name].default()
            if node_index == defense_node_index:
                assert defender_obs['observed_state'][node_index] == 1
            else:
                assert defender_obs['observed_state'][node_index] == default_defense_value


def test_ttc():
    pass


def test_actions():
    pass


def test_cost_and_reward():
    pass


def test_observation():
    pass


def test_termination():
    pass

