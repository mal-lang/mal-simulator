from unittest.mock import MagicMock
from maltoolbox.attackgraph import AttackGraphNode, Attacker
from maltoolbox.language import LanguageGraph
from malsim.mal_simulator import MalSimAgentStateView
from malsim.agents.path_finder import PathFindingAttacker

def test_path_finding_attacker(dummy_lang_graph: LanguageGraph):
    r"""
            node1          node2
            /    \         /   \
        node3       node4        node5

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

    # Connect nodes (Node1 -> Node2 -> Node3 -> Node4)
    node1.children.add(node3)
    node3.parents.add(node1)
    node1.children.add(node4)
    node4.parents.add(node1)

    node2.children.add(node4)
    node4.parents.add(node2)
    node2.children.add(node5)
    node5.parents.add(node2)

    # Set up an attacker
    attacker = Attacker(name="TestAttacker")
    attacker.compromise(node4)

    # Set up a mock MalSimAgentState
    agent = MagicMock()
    agent.action_surface = [node1, node2]
    agent_view = MalSimAgentStateView(agent)

    # Configure PathFindingAttacker
    agent_config = {"seed": 42, "randomize": False}
    attacker_ai = PathFindingAttacker(agent_config)

    # Assign rewards
    node1.extras['reward'] = 100
    node3.extras['reward'] = 10

    # Get next action
    action_node = attacker_ai.get_next_action(agent_view)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node1.id

    agent = MagicMock()
    agent.action_surface = [node2, node3]
    agent_view = MalSimAgentStateView(agent)

    # Get next action
    action_node = attacker_ai.get_next_action(agent_view)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node3.id
