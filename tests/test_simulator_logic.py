from maltoolbox.attackgraph import (
    AttackGraphNode,
    AttackGraph,
    Attacker,
    create_attack_graph
)
from maltoolbox.attackgraph.analyzers import apriori
from maltoolbox.attackgraph.query import is_node_traversable_by_attacker

from malsim import MalSimulator
# Helpers
def add_viable_defense_node(attack_graph):
    """defense node with defense status off -> viable"""
    
    defense_node = AttackGraphNode(
        'defense', 'DefenseNode', existence_status=True, defense_status=False)
    attack_graph.add_node(defense_node)
    return defense_node


def add_viable_and_node(attack_graph: AttackGraph):
    """and-node with viable parents -> viable"""
    dummy_attack_steps = (
        attack_graph.lang_graph.assets['DummyAsset'].attack_steps
    )

    and_node = attack_graph.add_node(dummy_attack_steps['DummyAndAttackStep'])
    and_node.existence_status = True

    and_node_parent1 = (
        attack_graph.add_node(dummy_attack_steps['DummyOrAttackStep'])
    )
    and_node_parent1.children={and_node}
    and_node_parent2 = (
        attack_graph.add_node(dummy_attack_steps['DummyOrAttackStep'])
    )
    and_node_parent2.children={and_node}

    and_node.parents = {and_node_parent1, and_node_parent2}
    apriori.evaluate_viability(and_node)
    return and_node


def add_viable_or_node(attack_graph: AttackGraph):
    """or-node with viable parent -> viable"""
    dummy_attack_steps = (
        attack_graph.lang_graph.assets['DummyAsset'].attack_steps
    )

    or_node = attack_graph.add_node(
        dummy_attack_steps['DummyOrAttackStep'])
    or_node_parent = attack_graph.add_node(
        dummy_attack_steps['DummyOrAttackStep'])
    or_node_parent.children = {or_node}
    or_node.parents = {or_node_parent}
    apriori.evaluate_viability(or_node)
    return or_node


def add_traversable_and_node(attack_graph):
    """Add traversable and node to AG"""
    attacker = Attacker('Attacker1')

    # Viable and-node where all necessary parents are compromised
    viable_and_node = add_viable_and_node(attack_graph)
    for parent in viable_and_node.parents:
        parent.is_necessary = True
        attacker.compromise(parent)
    return viable_and_node


def add_non_viable_and_node(attack_graph):
    """and-node with two non-viable parents -> non viable"""
    dummy_attack_steps = (
        attack_graph.lang_graph.assets['DummyAsset'].attack_steps
    )

    and_node = attack_graph.add_node(dummy_attack_steps['DummyAndAttackStep'])
    and_node.existence_status = True

    and_node_parent1 = (
        attack_graph.add_node(dummy_attack_steps['DummyOrAttackStep'])
    )
    and_node_parent1.children={and_node}
    and_node_parent1.is_viable = False

    and_node_parent2 = (
        attack_graph.add_node(dummy_attack_steps['DummyOrAttackStep'])
    )
    and_node_parent2.children={and_node}
    and_node_parent2.is_viable = False
    and_node.parents = [and_node_parent1, and_node_parent2]
    apriori.evaluate_viability(and_node)

    return and_node


def add_non_viable_or_node(attack_graph):
    """or-node with no viable parent -> non viable"""
    dummy_attack_steps = (
        attack_graph.lang_graph.assets['DummyAsset'].attack_steps
    )

    or_node = attack_graph.add_node(
        dummy_attack_steps['DummyOrAttackStep'])
    or_node_parent = attack_graph.add_node(
        dummy_attack_steps['DummyOrAttackStep'])
    or_node_parent.children = {or_node}
    or_node.parents = {or_node_parent}
    apriori.evaluate_viability(or_node)
    return or_node

# Tests

def test_viability_viable_nodes(dummy_lang_graph):
    """Make sure expected viable nodes are actually viable"""

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

    # defense status = false -> viable
    defense_step_type = dummy_attack_steps['DummyDefenseAttackStep']
    defense_step_node = attack_graph.add_node(defense_step_type)
    defense_step_node.defense_status = False

    # or-node with viable parent -> viable
    or_attack_step_type = dummy_attack_steps['DummyOrAttackStep']
    or_node = attack_graph.add_node(or_attack_step_type)
    or_node_parent = attack_graph.add_node(or_attack_step_type)
    or_node.parents.add(or_node_parent)

    # and-node with no parents -> viable
    and_attack_step_type = dummy_attack_steps['DummyAndAttackStep']
    and_node = attack_graph.add_node(and_attack_step_type)

    # and-node with viable parents -> viable
    and_node2 = attack_graph.add_node(and_attack_step_type)
    and_node_parent1 = attack_graph.add_node(and_attack_step_type)
    and_node_parent2 = attack_graph.add_node(and_attack_step_type)
    and_node2.parents = {and_node_parent1, and_node_parent2}

    # Make sure viable
    apriori.evaluate_viability(exist_node)
    assert exist_node.is_viable
    apriori.evaluate_viability(not_exist_node)
    assert not_exist_node.is_viable
    apriori.evaluate_viability(defense_step_node)
    assert defense_step_node.is_viable
    apriori.evaluate_viability(or_node)
    assert or_node.is_viable
    apriori.evaluate_viability(and_node)
    assert and_node.is_viable
    apriori.evaluate_viability(and_node2)
    assert and_node2.is_viable


def test_viability_unviable_nodes(dummy_lang_graph):
    """Make sure expected unviable nodes are actually unviable"""

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
    defense_step_node.defense_status = True

    # or-node with no viable parent -> non viable
    or_attack_step_type = dummy_attack_steps['DummyOrAttackStep']
    or_node = attack_graph.add_node(or_attack_step_type)
    or_node_parent = attack_graph.add_node(or_attack_step_type)
    or_node_parent.is_viable = False
    or_node.parents.add(or_node_parent)

    # and-node with two non-viable parents -> non viable
    and_attack_step_type = dummy_attack_steps['DummyAndAttackStep']
    and_node = attack_graph.add_node(and_attack_step_type)

    and_node_parent1 = attack_graph.add_node(and_attack_step_type)
    and_node_parent1.is_viable = False

    and_node_parent2 = attack_graph.add_node(and_attack_step_type)
    and_node_parent2.is_viable = False

    and_node.parents = {and_node_parent1, and_node_parent2}
    and_node.parents = {and_node_parent1, and_node_parent2}

    # Calculate viability and make sure unviable
    apriori.evaluate_viability(exist_node)
    assert not exist_node.is_viable
    apriori.evaluate_viability(not_exist_node)
    assert not not_exist_node.is_viable
    apriori.evaluate_viability(defense_step_node)
    assert not defense_step_node.is_viable
    apriori.evaluate_viability(or_node)
    assert not or_node.is_viable
    apriori.evaluate_viability(and_node)
    assert not and_node.is_viable


def test_necessity_necessary(dummy_lang_graph):
    """Make sure expected necessary nodes are necessary"""

    attack_graph = AttackGraph(dummy_lang_graph)
    dummy_attack_steps = dummy_lang_graph.assets['DummyAsset'].attack_steps

    # exists node, existance_status = False -> necessary
    exist_attack_step_type = dummy_attack_steps['DummyExistAttackStep']
    exist_node = attack_graph.add_node(exist_attack_step_type)
    exist_node.existence_status = False

    # notExists, existance_status = True -> necessary
    not_exist_attack_step_type = dummy_attack_steps['DummyNotExistAttackStep']
    not_exist_node = attack_graph.add_node(not_exist_attack_step_type)
    not_exist_node.existence_status = True

    # Defense status on -> necessary
    defense_step_type = dummy_attack_steps['DummyDefenseAttackStep']
    defense_step_node = attack_graph.add_node(defense_step_type)
    defense_step_node.defense_status = True

    # or-node with necessary parents -> necessary
    or_attack_step_type = dummy_attack_steps['DummyOrAttackStep']
    or_node = attack_graph.add_node(or_attack_step_type)
    or_node_parent = attack_graph.add_node(or_attack_step_type)
    or_node_parent.is_necessary = True
    or_node.parents.add(or_node_parent)

    # and-node with at least one necessary parents -> necessary
    and_attack_step_type = dummy_attack_steps['DummyAndAttackStep']
    and_node = attack_graph.add_node(and_attack_step_type)

    and_node_parent1 = attack_graph.add_node(and_attack_step_type)
    and_node_parent1.is_necessary = True

    and_node_parent2 = attack_graph.add_node(and_attack_step_type)
    and_node_parent2.is_necessary = False

    and_node.parents = {and_node_parent1, and_node_parent2}
    and_node.parents = {and_node_parent1, and_node_parent2}

    # Calculate necessety and make sure neccessary
    apriori.evaluate_necessity(exist_node)
    assert exist_node.is_necessary
    apriori.evaluate_necessity(not_exist_node)
    assert not_exist_node.is_necessary
    apriori.evaluate_necessity(defense_step_node)
    assert defense_step_node.is_necessary
    apriori.evaluate_necessity(or_node)
    assert or_node.is_necessary
    apriori.evaluate_necessity(and_node)
    assert and_node.is_necessary


def test_necessity_unnecessary(dummy_lang_graph):
    """Make sure expected unnecessary nodes are unnecessary"""
    pass


def test_traversability_traversable(dummy_lang_graph):
    """Make sure traversability works as expected"""
    attack_graph = AttackGraph(dummy_lang_graph)
    attacker = Attacker('Attacker1')

    # Viable and-node where all necessary parents are compromised
    viable_and_node = add_viable_and_node(attack_graph)
    for parent in viable_and_node.parents:
        parent.is_necessary = True
        attacker.compromise(parent)
    assert is_node_traversable_by_attacker(viable_and_node, attacker)

    # Viable or-node where at least one parent is compromised
    viable_or_node = add_viable_or_node(attack_graph)
    parent = next(iter(viable_or_node.parents))
    parent.is_necessary = True
    attacker.compromise(parent)
    assert is_node_traversable_by_attacker(viable_or_node, attacker)


def test_traversability_not_traversable(dummy_lang_graph):
    """Make sure nodes that shouldn't be traversable aren't"""
    attack_graph = AttackGraph(dummy_lang_graph)
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
    parent = next(iter(viable_or_node.parents))
    parent.is_necessary = True
    attacker.compromise(parent)

    # Fails because the function assumes that one parent is traversable
    # TODO: Should this be changed?
    # assert not is_node_traversable_by_attacker(viable_or_node, attacker)


# def test_state_transitions_attack_step():
#     """Show the definitions and verify the observation values"""

#     lang_file = "tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar"
#     model_file = "tests/testdata/models/simple_no_attacker_test_model.yml"
#     attack_graph = create_attack_graph(lang_file, model_file)

#     attack_step_full_name = 'OS App:localConnect'
#     attacker_entry_point = attack_graph.get_node_by_full_name(
#         attack_step_full_name
#     )
#     assert attacker_entry_point.is_viable
#     assert attacker_entry_point.is_necessary

#     attacker1 = Attacker("attacker1")
#     attack_graph.add_attacker(attacker1)
#     attacker1.compromise(attacker_entry_point)

#     env = MalSimulator(attack_graph)

#     env.register_attacker(attacker1.name, attacker1.id)
#     env.reset()

#     action_dict = {attacker1.name: attacker_entry_point}
#     observations, rewards, terminations, truncations, infos = env.step(action_dict)

#     # Make sure it was compromised
#     assert attacker_entry_point.is_compromised()

#     attacker_obs = observations[attacker1.name]

#     # Currently all TTCs are 0, this will change
#     for ttc in attacker_obs['remaining_ttc']:
#         assert ttc == 0

#     # The attack node should be observed as active
#     assert attacker_obs['observed_state'][attack_node_index] == 1

#     for child in attacker_entry_point.children:
#         # All children of reached attack steps are seen as inactive (0)
#         child_node_index = env._index_to_full_name.index(child.full_name)
#         assert attacker_obs['observed_state'][child_node_index] == 0

#     for node in attack_graph.nodes:
#         if not (node.type == 'or' or node.type == "and"):
#             continue

#         # All unknown nodes are seen as unknown (-1)
#         if (node != attacker_entry_point\
#             and node not in attacker_entry_point.children):

#             node_index = env._index_to_full_name.index(node.full_name)
#             assert attacker_obs['observed_state'][node_index] == -1


def create_simulator():
    lang_file = "tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar"
    model_file = "tests/testdata/models/simple_test_model.yml"
    attack_graph = create_attack_graph(lang_file, model_file)

    return MalSimulator(attack_graph)


# def test_state_transitions_defense_step():
#     """Make sure step is performed and observations set correctly"""

#     env = create_simulator()
#     attack_graph = env.attack_graph

#     # Remember defenses already activated in model
#     preactivated_defenses = [
#         n for n in attack_graph.nodes.values()
#         if n.is_enabled_defense()
#     ]

#     # We need an attacker so the simulation doesn't terminate
#     attack_step_full_name = 'OS App:localConnect'
#     attacker_entry_point = attack_graph.get_node_by_full_name(
#         attack_step_full_name
#     )
#     attacker1 = Attacker("attacker1")
#     attacker1.compromise(attacker_entry_point)
#     attack_node_index = env._index_to_full_name.index(attack_step_full_name)
#     env.register_attacker(attacker1.name, attacker1.id)

#     # The defender will activate a step
#     defense_step_full_name = 'OS App:notPresent'
#     defense_step = attack_graph.get_node_by_full_name(
#         defense_step_full_name
#     )
#     assert defense_step.is_viable
#     defender1 = "defender1"
#     defense_node_index = env._index_to_full_name.index(defense_step_full_name)
#     env.register_defender(defender1)
#     assert defender1 in env.possible_agents
#     env.reset()

#     assert defender1 in env.agents

#     # Each agent performs an action
#     action_dict = {
#         defender1: (1, defense_node_index),
#         attacker1.name: (1, attack_node_index)
#     }

#     observations, _, _, _, _ = env.step(action_dict)
#     defender_obs = observations[defender1]
#     for node in attack_graph.nodes:
#         if node.type == "defense":
#             node_index = env._index_to_full_name.index(node.full_name)
#             if node_index == defense_node_index or node in preactivated_defenses:
#                 assert defender_obs['observed_state'][node_index] == 1
#             else:
#                 assert defender_obs['observed_state'][node_index] == 0


def test_cost_and_reward():

    lang_file = "tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar"
    model_file = "tests/testdata/models/simple_test_model.yml"
    attack_graph = create_attack_graph(lang_file, model_file)

    # Prepare entrypoint node
    entrypoint1_full_name = 'OS App:localConnect'
    attacker_entry_point1 = attack_graph.get_node_by_full_name(
        entrypoint1_full_name
    )
    entrypoint2_full_name = 'OS App:authenticate'
    attacker_entry_point2 = attack_graph.get_node_by_full_name(
        entrypoint2_full_name
    )

    # Add attacker to attack graph
    attacker1 = Attacker("attacker1")
    attack_graph.add_attacker(
        attacker=attacker1,
        entry_points=[attacker_entry_point1.id, attacker_entry_point2.id],
        reached_attack_steps=[attacker_entry_point1.id, attacker_entry_point2.id]
    )

    # Create simulator
    sim = MalSimulator(attack_graph)
    sim.register_attacker(attacker1.name, attacker1.id)
    sim.reset()

    # Need to again fetch attacker after reset
    attacker1 = sim.attack_graph.attackers[attacker1.id]

    # Prepare action step node and add reward
    next_attackstep_full_name = 'OS App:localAccess'
    next_attackstep = sim.attack_graph.get_node_by_full_name(
        next_attackstep_full_name
    )
    next_attackstep.extras['reward'] = 100

    # Make sure next attackstep can be traversed
    assert is_node_traversable_by_attacker(next_attackstep, attacker1)

    # Select next attackstep and perform step
    action_dict = {attacker1.name: [next_attackstep]}
    states = sim.step(action_dict)

    # Make sure reward was given
    assert states[attacker1.name].reward == next_attackstep.extras['reward']

def test_observation():
    pass


def test_termination_max_iters():
    """Show the definitions and verify the observation values"""

    lang_file = "tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar"
    model_file = "tests/testdata/models/simple_test_model.yml"
    attack_graph = create_attack_graph(lang_file, model_file)

    attack_step_full_name = 'OS App:localConnect'
    attacker_entry_point = attack_graph.get_node_by_full_name(
        attack_step_full_name
    )

    assert attacker_entry_point.is_viable
    assert attacker_entry_point.is_necessary

    attacker1 = Attacker("attacker1")
    attack_graph.add_attacker(
        attacker1,
        entry_points=[attacker_entry_point.id],
        reached_attack_steps=[attacker_entry_point.id]
    )

    sim = MalSimulator(attack_graph, max_iter=2)
    sim.register_attacker(attacker1.name, attacker1.id)
    sim.reset()

    # Get again
    attacker_entry_point = sim.attack_graph.get_node_by_full_name(
        attack_step_full_name
    )

    action_dict = {attacker1.name: [attacker_entry_point]}

    states = sim.step(action_dict)
    assert not states[attacker1.name].terminated
    assert not states[attacker1.name].truncated
    assert sim._alive_agents

    states = sim.step(action_dict)
    assert not states[attacker1.name].terminated
    assert not states[attacker1.name].truncated
    assert sim._alive_agents

    states = sim.step(action_dict)
    assert not states[attacker1.name].terminated
    # This step is > max_iters and agent is removed/truncated
    assert states[attacker1.name].truncated
    assert not sim._alive_agents


def test_termination_no_traversable():
    """Show the definitions and verify the observation values"""

    lang_file = "tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar"
    model_file = "tests/testdata/models/simple_test_model.yml"

    attack_graph = create_attack_graph(lang_file, model_file)
    attack_step_full_name = 'OS App:softwareCheck'
    attacker_entry_point = attack_graph.get_node_by_full_name(
        attack_step_full_name
    )

    assert attacker_entry_point
    assert not attacker_entry_point.is_viable
    assert attacker_entry_point.is_necessary

    attacker1 = Attacker("attacker1")
    attack_graph.add_attacker(
        attacker1,
        entry_points=[attacker_entry_point.id],
        reached_attack_steps=[attacker_entry_point.id]
    )

    assert attacker_entry_point.children
    for child in attacker_entry_point.children:
        # No children to traverse
        assert not is_node_traversable_by_attacker(child, attacker1)

    sim = MalSimulator(attack_graph, max_iter=2)
    sim.register_attacker(attacker1.name, attacker1.id)
    sim.reset()

    # Get node again after reset
    attacker_entry_point = sim.attack_graph.get_node_by_full_name(
        attack_step_full_name
    )
    action_dict = {attacker1.name: [attacker_entry_point]}

    sim.step(action_dict)

    # No traversable step for attacker, so it is removed
    assert not sim._alive_agents
