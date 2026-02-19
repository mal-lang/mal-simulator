from collections import Counter
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.language import LanguageGraph
from malsim.mal_simulator import MalSimulator, MalSimDefenderState, run_simulation
from malsim.policies import (
    DefendCompromisedDefender,
    DefendFutureCompromisedDefender,
    RandomAgent,
    RandomDefender,
)
from malsim.scenario.scenario import Scenario
import numpy as np


def test_defend_compromised_defender(dummy_lang_graph: LanguageGraph) -> None:
    r"""
    node0 -------+---------|
     |   node1   |  node2  |
     |   /    \  |  /   \  |
    node3      node4    node5

    """
    dummy_and_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyAndAttackStep'
    ]
    dummy_defense_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyDefenseAttackStep'
    ]

    # Create attack graph with nodes
    ag = AttackGraph(dummy_lang_graph)
    node0 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=0)
    node1 = ag.add_node(lg_attack_step=dummy_defense_attack_step, node_id=1)
    node2 = ag.add_node(lg_attack_step=dummy_defense_attack_step, node_id=2)
    node3 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=3)
    node4 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=4)
    node5 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=5)

    # Connect nodes (Node1 -> Node3, Node4, Node5)
    node0.children.add(node3)
    node3.parents.add(node0)
    node0.children.add(node4)
    node4.parents.add(node0)
    node0.children.add(node5)
    node5.parents.add(node0)

    # Connect nodes (Node1 -> Node3, Node4)
    node1.children.add(node3)
    node3.parents.add(node1)
    node1.children.add(node4)
    node4.parents.add(node1)

    # Connect nodes (Node2 -> Node4, Node5)
    node2.children.add(node4)
    node4.parents.add(node2)
    node2.children.add(node5)
    node5.parents.add(node2)

    sim = MalSimulator(ag)

    # Set up an attacker
    sim.register_attacker('bfs', {node4})

    # Set up a defender
    sim.register_defender('def_comp')
    agent_state = sim.agent_states['def_comp']

    # Configure BreadthFirstAttacker
    agent_config = {'seed': 42, 'randomize': False}
    defender_ai = DefendCompromisedDefender(agent_config)

    # Should pick cheapest one
    sim.sim_state.global_rewards[node1] = 100
    sim.sim_state.global_rewards[node2] = 10

    # Get next action
    assert isinstance(agent_state, MalSimDefenderState)
    action_node = defender_ai.get_next_action(agent_state)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node2.id

    # Should pick cheapest one
    sim.sim_state.global_rewards[node1] = 10
    sim.sim_state.global_rewards[node2] = 100

    # Get next action
    action_node = defender_ai.get_next_action(agent_state)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node1.id


def test_defend_future_compromised_defender(dummy_lang_graph: LanguageGraph) -> None:
    r"""
    node0 -------+-----------|
     |   node1   |    node2  |
     |   /    \  |    /   \  |
    node3      node4  |   node5
                 |    |
                 \    /
                 node 6
    """

    dummy_and_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyAndAttackStep'
    ]
    dummy_defense_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyDefenseAttackStep'
    ]

    # Create attack graph with nodes
    ag = AttackGraph(dummy_lang_graph)
    node0 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=0)
    node1 = ag.add_node(lg_attack_step=dummy_defense_attack_step, node_id=1)
    node2 = ag.add_node(lg_attack_step=dummy_defense_attack_step, node_id=2)
    node3 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=3)
    node4 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=4)
    node5 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=5)
    node6 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=6)

    # Connect nodes (Node1 -> Node3, Node4, Node5)
    node0.children.add(node3)
    node3.parents.add(node0)
    node0.children.add(node4)
    node4.parents.add(node0)
    node0.children.add(node5)
    node5.parents.add(node0)

    # Connect nodes (Node1 -> Node3, Node4)
    node1.children.add(node3)
    node3.parents.add(node1)
    node1.children.add(node4)
    node4.parents.add(node1)

    # Connect nodes (Node2 -> Node5, Node6)
    node2.children.add(node5)
    node5.parents.add(node2)
    node2.children.add(node6)
    node6.parents.add(node2)

    # Connect nodes (Node4 -> Node6)
    node4.children.add(node6)
    node6.parents.add(node4)

    sim = MalSimulator(ag)

    # Set up an attacker
    sim.register_attacker('attacker', {node4})

    # Set up a defender
    sim.register_defender('def_future_comp')
    agent_state = sim.agent_states['def_future_comp']

    # Configure BreadthFirstAttacker
    agent_config = {'seed': 42, 'randomize': False}
    defender_ai = DefendFutureCompromisedDefender(agent_config)

    # Should pick node 2 either way
    assert isinstance(agent_state, MalSimDefenderState)
    action_node = defender_ai.get_next_action(agent_state)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node2.id


def test_random_defender(dummy_lang_graph: LanguageGraph) -> None:
    r"""
    | node1 -> & node2 -> | node3
                    ^
                    |
                # node0
    """

    dummy_and_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyAndAttackStep'
    ]
    dummy_or_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyOrAttackStep'
    ]
    dummy_defense_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyDefenseAttackStep'
    ]

    # Create attack graph with nodes
    ag = AttackGraph(dummy_lang_graph)
    node0 = ag.add_node(lg_attack_step=dummy_defense_attack_step, node_id=0)
    node1 = ag.add_node(lg_attack_step=dummy_or_attack_step, node_id=1)
    node2 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=2)
    node3 = ag.add_node(lg_attack_step=dummy_or_attack_step, node_id=3)

    node0.children.add(node2)
    node1.children.add(node2)
    node2.children.add(node3)
    sim = MalSimulator(ag)

    # Set up a defender
    sim.register_defender('def_random')

    agent_config = {'action_prob': 0.05}
    defender_ai = RandomDefender(agent_config)

    actions = []
    for _ in range(2_000):
        sim.reset()
        while len(sim.agent_states['def_random'].action_surface) > 0:
            agent_state = sim.agent_states['def_random']
            assert isinstance(agent_state, MalSimDefenderState)
            action = defender_ai.get_next_action(agent_state)
            actions.append(action)
            sim.step({'def_random': [action] if action else []})

    counter = Counter(actions)
    node0_frequency = counter[node0] / len(actions)

    assert np.isclose(node0_frequency, 0.05, atol=0.1)


def test_random_defender_scenario() -> None:
    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/random_defender_simple_scenario.yml'
    )
    sim = MalSimulator.from_scenario(scenario)
    run_simulation(sim, scenario.agent_settings)


def test_random_agent(dummy_lang_graph: LanguageGraph) -> None:
    r"""
    node0 -------+---------|
     |   node1   |  node2  |
     |   /    \  |  /   \  |
    node3      node4    node5

    """
    dummy_and_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyAndAttackStep'
    ]
    dummy_defense_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyDefenseAttackStep'
    ]

    # Create attack graph with nodes
    ag = AttackGraph(dummy_lang_graph)
    node0 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=0)
    node1 = ag.add_node(lg_attack_step=dummy_defense_attack_step, node_id=1)
    node2 = ag.add_node(lg_attack_step=dummy_defense_attack_step, node_id=2)
    node3 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=3)
    node4 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=4)
    node5 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=5)

    # Connect nodes (Node1 -> Node3, Node4, Node5)
    node0.children.add(node3)
    node3.parents.add(node0)
    node0.children.add(node4)
    node4.parents.add(node0)
    node0.children.add(node5)
    node5.parents.add(node0)

    # Connect nodes (Node1 -> Node3, Node4)
    node1.children.add(node3)
    node3.parents.add(node1)
    node1.children.add(node4)
    node4.parents.add(node1)

    # Connect nodes (Node2 -> Node4, Node5)
    node2.children.add(node4)
    node4.parents.add(node2)
    node2.children.add(node5)
    node5.parents.add(node2)

    sim = MalSimulator(ag)

    # Set up an attacker
    sim.register_attacker('attacker', {node4})

    # Set up a defender
    defender_name = 'random'
    sim.register_defender(defender_name)
    agent_state = sim.agent_states[defender_name]

    # Configure BreadthFirstAttacker
    agent_config = {'seed': 42}
    defender_ai = RandomAgent(agent_config)

    action_node = defender_ai.get_next_action(agent_state)
    assert action_node == node1

    agent_config = {'seed': 1334}
    defender_ai = RandomAgent(agent_config)

    action_node = defender_ai.get_next_action(agent_state)
    assert action_node == node2

    defender_ai = RandomAgent({})

    prev_action_node = defender_ai.get_next_action(agent_state)

    max_iters = 1000
    i = 0
    while True:
        i += 1
        if prev_action_node != defender_ai.get_next_action(agent_state):
            break
        if i == max_iters:
            assert False, 'Never got any other node as action'
