from maltoolbox.attackgraph import AttackGraph
from maltoolbox.language import LanguageGraph
from malsim.mal_simulator import MalSimulator
from malsim.agents import BreadthFirstAttacker, DepthFirstAttacker


def test_breadth_first_traversal_simple(
        dummy_lang_graph: LanguageGraph
    ) -> None:
    """
                    node0
                      |
                    node1
                      |
                    node2
                      |
                    node3
    """
    dummy_or_attack_step = (
        dummy_lang_graph.assets['DummyAsset']
        .attack_steps['DummyOrAttackStep']
    )

    # Create attack graph with nodes
    ag = AttackGraph(dummy_lang_graph)
    node0 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node1 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node2 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node3 = ag.add_node(lg_attack_step = dummy_or_attack_step)

    # Connect nodes (Node1 -> Node2 -> Node3 -> Node4)
    node0.children.add(node1)
    node1.parents.add(node0)
    node1.children.add(node2)
    node2.parents.add(node1)
    node2.children.add(node3)
    node3.parents.add(node2)

    sim = MalSimulator(ag)

    sim.register_attacker('bfs', {node0})
    agent_state = sim.agent_states['bfs']

    # Configure BreadthFirstAttacker
    agent_config = {"seed": 42, "randomize": False}
    attacker_ai = BreadthFirstAttacker(agent_config)

    # Expected traversal order
    expected_order = [1, 2, 3]

    actual_order = []
    for _ in expected_order:
        # Get next action
        action_node = attacker_ai.get_next_action(agent_state)
        assert action_node

        # Get next action
        states = sim.step({'bfs': [action_node]})
        agent_state = states['bfs']

        # Store the ID for verification
        actual_order.append(next(iter(agent_state.step_performed_nodes)).id)

    assert actual_order == expected_order, \
        "Traversal order does not match expected breadth-first order"

def test_breadth_first_traversal_complicated(
        dummy_lang_graph: LanguageGraph
    ) -> None:
    r"""
                    node0 ______________
                  /       \             \
            node1          node2        node7
            /   \           /   \
        node3   node4    node5  node6

    """

    dummy_or_attack_step = (
        dummy_lang_graph.assets['DummyAsset']
        .attack_steps['DummyOrAttackStep']
    )

    # Create attack graph with nodes
    ag = AttackGraph(dummy_lang_graph)
    node0 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node1 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node2 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node3 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node4 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node5 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node6 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node7 = ag.add_node(lg_attack_step = dummy_or_attack_step)

    # Connect nodes (Node0 -> Node1, Node2, Node7)
    node0.children.add(node1)
    node1.parents.add(node0)
    node0.children.add(node2)
    node2.parents.add(node0)
    node0.children.add(node7)
    node7.parents.add(node0)

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

    sim = MalSimulator(ag)

    sim.register_attacker('bfs', {node0})
    agent_state = sim.agent_states['bfs']

    # Configure BreadthFirstAttacker
    agent_config = {"seed": 42, "randomize": False}
    attacker_ai = BreadthFirstAttacker(agent_config)

    # Expected traversal order
    expected_order = [1, 2, 7, 3, 4, 5, 6]

    actual_order = []
    for _ in expected_order:
        # Get next action
        action_node = attacker_ai.get_next_action(agent_state)
        assert action_node

        # Get next action
        states = sim.step({'bfs': [action_node]})
        agent_state = states['bfs']

        # Store the ID for verification
        actual_order.append(next(iter(agent_state.step_performed_nodes)).id)

    for level in (0, 3), (3, 7):
        assert set(expected_order[level[0]:level[1]]) == set(actual_order[level[0]:level[1]]), \
            "Traversal order does not match expected breadth-first order"


def test_depth_first_traversal_complicated(
        dummy_lang_graph: LanguageGraph
    ) -> None:
    r"""
                    node0 ______________
                  /       \             \
            node1          node2        node7
            /   \           /   \
        node3   node4    node5  node6

    """
    dummy_or_attack_step = (
        dummy_lang_graph.assets['DummyAsset']
        .attack_steps['DummyOrAttackStep']
    )

    # Create attack graph with nodes
    ag = AttackGraph(dummy_lang_graph)
    node0 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node1 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node2 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node3 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node4 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node5 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node6 = ag.add_node(lg_attack_step = dummy_or_attack_step)
    node7 = ag.add_node(lg_attack_step = dummy_or_attack_step)

    # Connect nodes (Node0 -> Node1, Node2, Node7)
    node0.children.add(node1)
    node1.parents.add(node0)
    node0.children.add(node2)
    node2.parents.add(node0)
    node0.children.add(node7)
    node7.parents.add(node0)

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

    sim = MalSimulator(ag)

    sim.register_attacker('dfs', {node0})
    agent_state = sim.agent_states['dfs']

    # Configure BreadthFirstAttacker
    agent_config = {"seed": 42, "randomize": False}
    attacker_ai = DepthFirstAttacker(agent_config)

    actual_order = []
    for _ in range(0,7):
        # Get next action
        action_node = attacker_ai.get_next_action(agent_state)
        assert action_node

        # Get next action
        states = sim.step({'dfs': [action_node]})
        agent_state = states['dfs']

        # Store the ID for verification
        actual_order.append(next(iter(agent_state.step_performed_nodes)).id)



    # All children of 1 must come directly after it
    index1 = actual_order.index(1)
    assert (
        3 in actual_order[index1 + 1: index1 + 3] and
        4 in actual_order[index1 + 1: index1 + 3]
    )

    # All children of 2 must come directly after it
    index2 = actual_order.index(2)
    assert (
        5 in actual_order[index2 + 1: index2 + 3] and
        6 in actual_order[index2 + 1: index2 + 3]
    )
