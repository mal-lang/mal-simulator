from unittest.mock import MagicMock
from maltoolbox.attackgraph import AttackGraphNode, Attacker
from maltoolbox.attackgraph.query import get_attack_surface
from malsim.sims import MalSimAgentView
from malsim.agents import BreadthFirstAttacker, DepthFirstAttacker


def test_breadth_first_traversal_simple():
    """
                    node1
                      |
                    node2
                      |
                    node3
                      |
                    node4
    """
    # Create nodes
    node1 = AttackGraphNode(name="Node1", id=1, type='or')
    node2 = AttackGraphNode(name="Node2", id=2, type='or')
    node3 = AttackGraphNode(name="Node3", id=3, type='or')
    node4 = AttackGraphNode(name="Node4", id=4, type='or')

    # Connect nodes (Node1 -> Node2 -> Node3 -> Node4)
    node1.children.append(node2)
    node2.parents.append(node1)
    node2.children.append(node3)
    node3.parents.append(node2)
    node3.children.append(node4)
    node4.parents.append(node3)

    # Set up an attacker
    attacker = Attacker(name="TestAttacker")
    attacker.entry_points = [node1]

    # Set up a mock MalSimAgent
    agent = MagicMock()
    agent.action_surface = [node1]

    # Set up MalSimAgentView
    agent_view = MalSimAgentView(agent)

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
        agent.action_surface = get_attack_surface(attacker)

        # Store the ID for verification
        actual_order.append(action_node.id)

    assert actual_order == expected_order, \
        "Traversal order does not match expected breadth-first order"

def test_breadth_first_traversal_complicated():
    r"""
                    node1 ______________
                  /       \             \
            node2          node3        node8
            /   \           /   \
        node4   node5    node6  node7

    """
    # Create nodes
    node1 = AttackGraphNode(name="Node1", id=1, type='or')
    node2 = AttackGraphNode(name="Node2", id=2, type='or')
    node3 = AttackGraphNode(name="Node3", id=3, type='or')
    node4 = AttackGraphNode(name="Node4", id=4, type='or')
    node5 = AttackGraphNode(name="Node5", id=5, type='or')
    node6 = AttackGraphNode(name="Node6", id=6, type='or')
    node7 = AttackGraphNode(name="Node7", id=7, type='or')
    node8 = AttackGraphNode(name="Node8", id=8, type='or')

    # Connect nodes (Node1 -> Node2 -> Node3 -> Node4)
    node1.children.append(node2)
    node2.parents.append(node1)
    node1.children.append(node3)
    node3.parents.append(node1)
    node1.children.append(node8)
    node8.children.append(node1)

    node2.children.append(node4)
    node4.parents.append(node2)
    node2.children.append(node5)
    node5.parents.append(node5)

    node3.children.append(node6)
    node6.parents.append(node3)
    node3.children.append(node7)
    node7.parents.append(node3)

    # Set up an attacker
    attacker = Attacker(name="TestAttacker")
    attacker.entry_points = [node1]

    # Set up a mock MalSimAgent
    agent = MagicMock()
    agent.action_surface = [node1]

    # Set up MalSimAgentView
    agent_view = MalSimAgentView(agent)

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
        agent.action_surface = get_attack_surface(attacker)

        # Store the ID for verification
        actual_order.append(action_node.id)

    assert actual_order == expected_order, \
        "Traversal order does not match expected breadth-first order"


def test_depth_first_traversal_complicated():
    r"""
                    node1 ______________
                  /       \             \
            node2          node3        node8
            /   \           /   \
        node4   node5    node6  node7

    """
    # Create nodes
    node1 = AttackGraphNode(name="Node1", id=1, type='or')
    node2 = AttackGraphNode(name="Node2", id=2, type='or')
    node3 = AttackGraphNode(name="Node3", id=3, type='or')
    node4 = AttackGraphNode(name="Node4", id=4, type='or')
    node5 = AttackGraphNode(name="Node5", id=5, type='or')
    node6 = AttackGraphNode(name="Node6", id=6, type='or')
    node7 = AttackGraphNode(name="Node7", id=7, type='or')
    node8 = AttackGraphNode(name="Node8", id=8, type='or')

    # Connect nodes (Node1 -> Node2 -> Node3 -> Node4)
    node1.children.append(node2)
    node2.parents.append(node1)
    node1.children.append(node3)
    node3.parents.append(node1)
    node1.children.append(node8)
    node8.children.append(node1)

    node2.children.append(node4)
    node4.parents.append(node2)
    node2.children.append(node5)
    node5.parents.append(node5)

    node3.children.append(node6)
    node6.parents.append(node3)
    node3.children.append(node7)
    node7.parents.append(node3)

    # Set up an attacker
    attacker = Attacker(name="TestAttacker")
    attacker.entry_points = [node1]

    # Set up a mock MalSimAgent
    agent = MagicMock()
    agent.action_surface = [node1]

    # Set up MalSimAgentView
    agent_view = MalSimAgentView(agent)

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
        agent.action_surface = get_attack_surface(attacker)

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
