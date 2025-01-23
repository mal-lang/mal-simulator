from unittest.mock import MagicMock
from maltoolbox.attackgraph import AttackGraphNode, Attacker
from maltoolbox.attackgraph.query import get_attack_surface
from malsim.sims import MalSimAgentStateView
from malsim.agents import ShutdownCompromisedMachinesDefender

def test_breadth_first_traversal_complicated():
    r"""
            node1          node2
            /    \         /   \
        node3       node4        node5

    """
    # Create nodes
    node1 = AttackGraphNode(name="Node1", id=1, type='defense')
    node2 = AttackGraphNode(name="Node2", id=2, type='defense')

    node3 = AttackGraphNode(name="Node3", id=3, type='defense')
    node4 = AttackGraphNode(name="Node4", id=4, type='or')
    node5 = AttackGraphNode(name="Node5", id=5, type='or')

    # Connect nodes (Node1 -> Node2 -> Node3 -> Node4)
    node1.children.append(node3)
    node3.parents.append(node1)
    node1.children.append(node4)
    node4.parents.append(node1)

    node2.children.append(node4)
    node4.parents.append(node2)
    node2.children.append(node5)
    node5.parents.append(node5)

    # Set up an attacker
    attacker = Attacker(name="TestAttacker")
    attacker.compromise(node4)

    # Set up a mock MalSimAgentState
    agent = MagicMock()
    agent.action_surface = [node1, node2]

    # Set up MalSimAgentStateView
    agent_view = MalSimAgentStateView(agent)

    # Configure BreadthFirstAttacker
    agent_config = {"seed": 42, "randomize": False}
    defender_ai = ShutdownCompromisedMachinesDefender(agent_config)

    # Should pick cheapest one
    node1.extras['reward'] = 100
    node2.extras['reward'] = 10

    # Get next action
    action_node = defender_ai.get_next_action(agent_view)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node2.id

    # Should pick cheapest one
    node1.extras['reward'] = 10
    node2.extras['reward'] = 100

    # Get next action
    action_node = defender_ai.get_next_action(agent_view)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node1.id
