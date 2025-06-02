from unittest.mock import MagicMock
from maltoolbox.attackgraph import AttackGraphNode, Attacker
from maltoolbox.attackgraph.query import calculate_attack_surface
from maltoolbox.language import LanguageGraph
from malsim.mal_simulator import MalSimAgentStateView
from malsim.agents import BreadthFirstAttacker, DepthFirstAttacker


def test_breadth_first_traversal_simple(
        dummy_lang_graph: LanguageGraph
    ) -> None:
    """
                    node1
                      |
                    node2
                      |
                    node3
                      |
                    node4
    """
    dummy_or_attack_step = (
        dummy_lang_graph.assets['DummyAsset']
        .attack_steps['DummyOrAttackStep']
    )

    # Create nodes
    node1 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=1)
    node2 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=2)
    node3 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=3)
    node4 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=4)

    # Connect nodes (Node1 -> Node2 -> Node3 -> Node4)
    node1.children.add(node2)
    node2.parents.add(node1)
    node2.children.add(node3)
    node3.parents.add(node2)
    node3.children.add(node4)
    node4.parents.add(node3)

    # Set up an attacker
    attacker = Attacker(
        name = "TestAttacker",
        entry_points = {node1},
        reached_attack_steps = set(),
        attacker_id = 100)

    # Set up a mock MalSimAgentState
    agent = MagicMock()
    agent.action_surface = [node1]

    # Set up MalSimAgentStateView
    agent_view = MalSimAgentStateView(agent)

    # Configure BreadthFirstAttacker
    agent_config = {"seed": 42, "randomize": False}
    attacker_ai = BreadthFirstAttacker(agent_config)

    # Expected traversal order
    expected_order = [1, 2, 3, 4]

    actual_order = []
    for _ in expected_order:
        # Get next action
        action_node = attacker_ai.get_next_action(agent_view)
        assert action_node is not None, "Action node shouldn't be None"

        # Mark node as compromised
        attacker.compromise(action_node)
        agent.step_action_surface_additions = calculate_attack_surface(
            attacker, from_nodes=[action_node]
        )

        # Store the ID for verification
        actual_order.append(action_node.id)

    assert actual_order == expected_order, \
        "Traversal order does not match expected breadth-first order"

def test_breadth_first_traversal_complicated(
        dummy_lang_graph: LanguageGraph
    ) -> None:
    r"""
                    node1 ______________
                  /       \             \
            node2          node3        node8
            /   \           /   \
        node4   node5    node6  node7

    """

    dummy_or_attack_step = (
        dummy_lang_graph.assets['DummyAsset']
        .attack_steps['DummyOrAttackStep']
    )

    # Create nodes
    node1 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=1)
    node2 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=2)
    node3 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=3)
    node4 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=4)
    node5 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=5)
    node6 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=6)
    node7 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=7)
    node8 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=8)

    # Connect nodes (Node1 -> Node2 -> Node3 -> Node4)
    node1.children.add(node2)
    node2.parents.add(node1)
    node1.children.add(node3)
    node3.parents.add(node1)
    node1.children.add(node8)
    node8.parents.add(node1)

    node2.children.add(node4)
    node4.parents.add(node2)
    node2.children.add(node5)
    node5.parents.add(node2)

    node3.children.add(node6)
    node6.parents.add(node3)
    node3.children.add(node7)
    node7.parents.add(node3)

    # Set up an attacker
    attacker = Attacker(
        name = "TestAttacker",
        entry_points = {node1},
        reached_attack_steps = set(),
        attacker_id = 100)

    # Set up a mock MalSimAgentState
    agent = MagicMock()
    agent.action_surface = [node1]

    # Set up MalSimAgentStateView
    agent_view = MalSimAgentStateView(agent)

    # Configure BreadthFirstAttacker
    agent_config = {"seed": 42, "randomize": False}
    attacker_ai = BreadthFirstAttacker(agent_config)

    # Expected traversal order
    expected_order = [1, 2, 3, 8, 4, 5, 6, 7]

    actual_order = []
    for _ in expected_order:
        # Get next action
        action_node = attacker_ai.get_next_action(agent_view)
        assert action_node is not None, "Action node shouldn't be None"

        # Mark node as compromised
        attacker.compromise(action_node)
        agent.step_action_surface_additions = calculate_attack_surface(
            attacker, from_nodes=[action_node]
        )

        # Store the ID for verification
        actual_order.append(action_node.id)

    for level in (0, 1), (1, 4), (4, 8):
        assert set(expected_order[level[0]:level[1]]) == set(actual_order[level[0]:level[1]]), \
            "Traversal order does not match expected breadth-first order"


def test_depth_first_traversal_complicated(
        dummy_lang_graph: LanguageGraph
    ) -> None:
    r"""
                    node1 ______________
                  /       \             \
            node2          node3        node8
            /   \           /   \
        node4   node5    node6  node7

    """
    dummy_or_attack_step = (
        dummy_lang_graph.assets['DummyAsset']
        .attack_steps['DummyOrAttackStep']
    )

    # Create nodes
    node1 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=1)
    node2 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=2)
    node3 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=3)
    node4 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=4)
    node5 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=5)
    node6 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=6)
    node7 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=7)
    node8 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=8)

    # Connect nodes (Node1 -> Node2 -> Node3 -> Node4)
    node1.children.add(node2)
    node2.parents.add(node1)
    node1.children.add(node3)
    node3.parents.add(node1)
    node1.children.add(node8)
    node8.parents.add(node1)

    node2.children.add(node4)
    node4.parents.add(node2)
    node2.children.add(node5)
    node5.parents.add(node2)

    node3.children.add(node6)
    node6.parents.add(node3)
    node3.children.add(node7)
    node7.parents.add(node3)

    # Set up an attacker
    attacker = Attacker(
        name = "TestAttacker",
        entry_points = {node1},
        reached_attack_steps = set(),
        attacker_id = 100)

    # Set up a mock MalSimAgentState
    agent = MagicMock()
    agent.action_surface = [node1]

    # Set up MalSimAgentStateView
    agent_view = MalSimAgentStateView(agent)

    # Configure BreadthFirstAttacker
    agent_config = {"seed": 42, "randomize": False}
    attacker_ai = DepthFirstAttacker(agent_config)

    # Expected traversal order
    expected_order =  [1, 8, 3, 7, 6, 2, 5, 4]

    actual_order = []
    for _ in expected_order:
        # Get next action
        action_node = attacker_ai.get_next_action(agent_view)
        assert action_node is not None, "Action node shouldn't be None"

        # Mark node as compromised
        attacker.compromise(action_node)
        agent.step_action_surface_additions = calculate_attack_surface(
            attacker, from_nodes=[action_node]
        )

        # Store the ID for verification
        actual_order.append(action_node.id)

    assert actual_order == expected_order, \
        "Traversal order does not match expected breadth-first order"

    # All children of 1 must come after 1
    assert actual_order.index(8) > actual_order.index(1)
    assert actual_order.index(2) > actual_order.index(1)
    assert actual_order.index(3) > actual_order.index(1)

    # All children of 3 must come after 3
    assert actual_order.index(7) > actual_order.index(3)
    assert actual_order.index(6) > actual_order.index(3)

    # All children of 2 must come after 2
    assert actual_order.index(4) > actual_order.index(2)
    assert actual_order.index(5) > actual_order.index(2)
